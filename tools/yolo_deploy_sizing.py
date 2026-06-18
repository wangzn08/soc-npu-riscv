from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Layer:
    idx: int
    oc: int
    ic: int
    kh: int
    kw: int
    stride: int
    pad: int


@dataclass(frozen=True)
class TensorShape:
    c: int
    h: int
    w: int
    name: str


@dataclass(frozen=True)
class LayerShape:
    layer: Layer
    role: str
    source: str
    input_name: str
    output_name: str
    in_c: int
    in_h: int
    in_w: int
    out_c: int
    out_h: int
    out_w: int
    act_in_bytes: int
    act_out_bytes: int
    weight_bytes_raw: int
    weight_bytes_aligned: int
    macs: int


@dataclass(frozen=True)
class YoloGraph:
    shapes: list[LayerShape]
    by_idx: dict[int, LayerShape]


@dataclass(frozen=True)
class BlockPlan:
    idx: int
    source: str
    input_name: str
    output_name: str
    input_ddr: int
    output_ddr: int
    weight_ddr: int
    act_base: int
    wgt_base: int
    out_base: int
    strip_rows: int
    strip_count: int
    input_words: int
    output_words: int
    weight_words: int
    ctrl_flags: tuple[str, ...]


@dataclass(frozen=True)
class StripPlan:
    idx: int
    strip_idx: int
    out_y: int
    out_rows: int
    in_y: int
    in_rows: int
    top_pad_rows: int
    bottom_pad_rows: int


def align16(value: int) -> int:
    return ((value + 15) // 16) * 16


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def conv_out_dim(size: int, kernel: int, stride: int, pad: int) -> int:
    return (size + 2 * pad - kernel) // stride + 1


def parse_layers(path: Path) -> list[Layer]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\{(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),"
        r"\"weights/conv(\d+)_w\.bin\""
    )
    layers: list[Layer] = []
    for row_idx, match in enumerate(pattern.finditer(text)):
        oc, ic, kh, kw, stride, pad, conv_idx = (int(v) for v in match.groups())
        if conv_idx != row_idx:
            raise ValueError(f"conv row order mismatch: row {row_idx} names conv{conv_idx}")
        layers.append(Layer(row_idx, oc, ic, kh, kw, stride, pad))
    return layers


def build_yolov8n_graph(layers: list[Layer], input_h: int, input_w: int) -> YoloGraph:
    layer_by_idx = {layer.idx: layer for layer in layers}
    shapes: list[LayerShape] = []

    def tensor(c: int, h: int, w: int, name: str) -> TensorShape:
        return TensorShape(c=c, h=h, w=w, name=name)

    def record_conv(idx: int, src: TensorShape, source: str, role: str = "npu_conv") -> TensorShape:
        layer = layer_by_idx[idx]
        if src.c != layer.ic:
            raise ValueError(f"conv{idx} expects IC={layer.ic}, got {src.c} from {src.name}")
        out_h = conv_out_dim(src.h, layer.kh, layer.stride, layer.pad)
        out_w = conv_out_dim(src.w, layer.kw, layer.stride, layer.pad)
        out = tensor(layer.oc, out_h, out_w, f"conv{idx}")
        shapes.append(
            LayerShape(
                layer=layer,
                role=role,
                source=source,
                input_name=src.name,
                output_name=out.name,
                in_c=src.c,
                in_h=src.h,
                in_w=src.w,
                out_c=out.c,
                out_h=out.h,
                out_w=out.w,
                act_in_bytes=src.h * src.w * align16(src.c),
                act_out_bytes=out.h * out.w * align16(out.c),
                weight_bytes_raw=layer.oc * layer.ic * layer.kh * layer.kw,
                weight_bytes_aligned=layer.oc * align16(layer.ic) * layer.kh * layer.kw,
                macs=out.h * out.w * layer.oc * layer.ic * layer.kh * layer.kw,
            )
        )
        return out

    def c2f_block(
        src: TensorShape,
        cv1: int,
        m_cv1: list[int],
        m_cv2: list[int],
        cv2: int,
        shortcut: bool,
        name: str,
    ) -> TensorShape:
        cv1_out = record_conv(cv1, src, f"{name}.cv1")
        half_c = cv1_out.c // 2
        s0 = tensor(half_c, cv1_out.h, cv1_out.w, f"{name}.split0")
        prev = tensor(half_c, cv1_out.h, cv1_out.w, f"{name}.split1")
        pieces = [s0, prev]
        for i, (a_idx, b_idx) in enumerate(zip(m_cv1, m_cv2)):
            t = record_conv(a_idx, prev, f"{name}.m{i}.cv1")
            t = record_conv(b_idx, t, f"{name}.m{i}.cv2")
            if shortcut:
                prev = tensor(t.c, t.h, t.w, f"{name}.m{i}.add")
            else:
                prev = t
            pieces.append(prev)
        concat_c = sum(piece.c for piece in pieces)
        concat = tensor(concat_c, cv1_out.h, cv1_out.w, f"{name}.concat")
        return record_conv(cv2, concat, f"{name}.cv2")

    def sppf(src: TensorShape, cv1: int, cv2: int) -> TensorShape:
        cv1_out = record_conv(cv1, src, "sppf.cv1")
        concat = tensor(cv1_out.c * 4, cv1_out.h, cv1_out.w, "sppf.concat")
        return record_conv(cv2, concat, "sppf.cv2")

    def upsample2x(src: TensorShape, name: str) -> TensorShape:
        return tensor(src.c, src.h * 2, src.w * 2, name)

    def concat2(a: TensorShape, b: TensorShape, name: str) -> TensorShape:
        if (a.h, a.w) != (b.h, b.w):
            raise ValueError(f"{name} concat shape mismatch: {a} vs {b}")
        return tensor(a.c + b.c, a.h, a.w, name)

    inp = tensor(3, input_h, input_w, "input")

    t = record_conv(0, inp, "stem.0")
    t = record_conv(1, t, "stem.1")
    t = c2f_block(t, 2, [3], [4], 5, shortcut=True, name="c2f_2")

    t = record_conv(6, t, "backbone.down3")
    t = c2f_block(t, 7, [8, 10], [9, 11], 12, shortcut=True, name="c2f_4")
    p4 = t

    t = record_conv(13, t, "backbone.down5")
    t = c2f_block(t, 14, [15, 17], [16, 18], 19, shortcut=True, name="c2f_6")
    p5 = t

    t = record_conv(20, t, "backbone.down7")
    sppf_in = c2f_block(t, 21, [22], [23], 24, shortcut=True, name="c2f_8")
    t = sppf(sppf_in, 25, 26)

    up1 = upsample2x(t, "fpn.up1")
    cat1 = concat2(up1, p5, "fpn.cat1")
    fpn_mid = c2f_block(cat1, 27, [28], [29], 30, shortcut=False, name="c2f_12")

    up2 = upsample2x(fpn_mid, "fpn.up2")
    cat2 = concat2(up2, p4, "fpn.cat2")
    pan_p3 = c2f_block(cat2, 31, [32], [33], 34, shortcut=False, name="c2f_15")

    t = record_conv(35, pan_p3, "pan.down_p3")
    cat3 = concat2(t, fpn_mid, "pan.cat3")
    pan_p4 = c2f_block(cat3, 40, [43], [44], 45, shortcut=False, name="c2f_18")

    t = record_conv(46, pan_p4, "pan.down_p4")
    cat4 = concat2(t, sppf_in, "pan.cat4")
    pan_p5 = c2f_block(cat4, 51, [54], [55], 56, shortcut=False, name="c2f_21")

    d0_bbox = record_conv(36, pan_p3, "detect.p3.bbox0")
    d0_bbox = record_conv(38, d0_bbox, "detect.p3.bbox1")
    record_conv(41, d0_bbox, "detect.p3.bbox2")
    d0_cls = record_conv(37, pan_p3, "detect.p3.cls0")
    d0_cls = record_conv(39, d0_cls, "detect.p3.cls1")
    record_conv(42, d0_cls, "detect.p3.cls2")

    d1_bbox = record_conv(47, pan_p4, "detect.p4.bbox0")
    d1_bbox = record_conv(49, d1_bbox, "detect.p4.bbox1")
    record_conv(52, d1_bbox, "detect.p4.bbox2")
    d1_cls = record_conv(48, pan_p4, "detect.p4.cls0")
    d1_cls = record_conv(50, d1_cls, "detect.p4.cls1")
    record_conv(53, d1_cls, "detect.p4.cls2")

    d2_bbox = record_conv(57, pan_p5, "detect.p5.bbox0")
    d2_bbox = record_conv(59, d2_bbox, "detect.p5.bbox1")
    record_conv(61, d2_bbox, "detect.p5.bbox2")
    d2_cls = record_conv(58, pan_p5, "detect.p5.cls0")
    d2_cls = record_conv(60, d2_cls, "detect.p5.cls1")
    record_conv(62, d2_cls, "detect.p5.cls2")

    dfl = layer_by_idx[63]
    dfl_sites = 4 * (80 * 80 + 40 * 40 + 20 * 20)
    shapes.append(
        LayerShape(
            layer=dfl,
            role="cpu_dfl",
            source="cpu.decode.dfl",
            input_name="detect.bbox_logits",
            output_name="decode.dfl",
            in_c=dfl.ic,
            in_h=1,
            in_w=dfl_sites,
            out_c=dfl.oc,
            out_h=1,
            out_w=dfl_sites,
            act_in_bytes=dfl_sites * align16(dfl.ic),
            act_out_bytes=dfl_sites,
            weight_bytes_raw=dfl.oc * dfl.ic * dfl.kh * dfl.kw,
            weight_bytes_aligned=dfl.oc * align16(dfl.ic) * dfl.kh * dfl.kw,
            macs=dfl_sites * dfl.oc * dfl.ic * dfl.kh * dfl.kw,
        )
    )

    by_idx = {shape.layer.idx: shape for shape in shapes}
    if len(by_idx) != len(layers):
        missing = sorted(set(layer_by_idx) - set(by_idx))
        raise ValueError(f"missing layer shapes for conv indices: {missing}")
    return YoloGraph(shapes=shapes, by_idx=by_idx)


def strip_act_bytes(shape: LayerShape, out_rows: int) -> int:
    if shape.role != "npu_conv":
        return 0
    layer = shape.layer
    in_rows = min(shape.in_h, max(1, (out_rows - 1) * layer.stride + layer.kh))
    in_bytes = in_rows * shape.in_w * align16(shape.in_c)
    out_bytes = out_rows * shape.out_w * align16(shape.out_c)
    return in_bytes + out_bytes


def choose_strip_rows(shape: LayerShape, act_sram_bytes: int = 256 * 1024) -> int:
    if shape.role != "npu_conv":
        return 0
    best = 1
    for rows in range(1, shape.out_h + 1):
        if strip_act_bytes(shape, rows) <= act_sram_bytes:
            best = rows
        else:
            break
    if best >= 16:
        return (best // 16) * 16
    if best >= 8:
        return 8
    return best


def summarize(shapes: list[LayerShape]) -> dict[str, int]:
    npu_shapes = [shape for shape in shapes if shape.role == "npu_conv"]
    cpu_shapes = [shape for shape in shapes if shape.role != "npu_conv"]
    return {
        "npu_conv_layers": len(npu_shapes),
        "cpu_tail_layers": len(cpu_shapes),
        "total_macs": sum(shape.macs for shape in npu_shapes),
        "cpu_tail_macs": sum(shape.macs for shape in cpu_shapes),
        "total_weight_bytes_raw": sum(shape.weight_bytes_raw for shape in npu_shapes),
        "total_weight_bytes_aligned": sum(shape.weight_bytes_aligned for shape in npu_shapes),
        "activation_read_bytes_aligned": sum(shape.act_in_bytes for shape in npu_shapes),
        "activation_write_bytes_aligned": sum(shape.act_out_bytes for shape in npu_shapes),
    }


def build_block_plan(
    graph: YoloGraph,
    input_ddr_base: int = 0x40000000,
    activation_ddr_base: int = 0x41000000,
    weight_ddr_base: int = 0x48000000,
    act_base: int = 0,
    wgt_base: int = 0,
    out_base: int = 0,
    act_sram_bytes: int = 256 * 1024,
) -> list[BlockPlan]:
    """Build a deterministic CPU/NPU block schedule skeleton.

    The plan is still per-layer, not a fused whole-network runtime. It assigns
    DDR tensor slots and SRAM bases so firmware can later iterate over strips
    without hand-writing addresses per YOLO layer.
    """

    tensor_ddr: dict[str, int] = {"input": input_ddr_base}
    next_act_ddr = activation_ddr_base
    next_weight_ddr = weight_ddr_base
    plans: list[BlockPlan] = []

    for shape in sorted(graph.shapes, key=lambda item: item.layer.idx):
        if shape.role != "npu_conv":
            continue

        if shape.input_name not in tensor_ddr:
            next_act_ddr = align_up(next_act_ddr, 16)
            tensor_ddr[shape.input_name] = next_act_ddr
            next_act_ddr += shape.act_in_bytes

        next_act_ddr = align_up(next_act_ddr, 16)
        output_ddr = next_act_ddr
        tensor_ddr[shape.output_name] = output_ddr
        next_act_ddr += shape.act_out_bytes

        next_weight_ddr = align_up(next_weight_ddr, 16)
        weight_ddr = next_weight_ddr
        next_weight_ddr += shape.weight_bytes_aligned

        ctrl_flags: list[str] = []
        if shape.layer.kh == 1 and shape.layer.kw == 1:
            ctrl_flags.append("PW_EN")
        if shape.layer.pad != 0:
            ctrl_flags.append("HW_PAD")
        if shape.out_c > 16:
            ctrl_flags.append("OC_SINGLE")
        ctrl_flags.extend(["SILU", "SILU_REQUANT"])

        strip_rows = choose_strip_rows(shape, act_sram_bytes=act_sram_bytes)
        strip_count = (shape.out_h + strip_rows - 1) // strip_rows
        plans.append(
            BlockPlan(
                idx=shape.layer.idx,
                source=shape.source,
                input_name=shape.input_name,
                output_name=shape.output_name,
                input_ddr=tensor_ddr[shape.input_name],
                output_ddr=output_ddr,
                weight_ddr=weight_ddr,
                act_base=act_base,
                wgt_base=wgt_base,
                out_base=out_base,
                strip_rows=strip_rows,
                strip_count=strip_count,
                input_words=shape.act_in_bytes // 16,
                output_words=shape.act_out_bytes // 16,
                weight_words=shape.weight_bytes_aligned // 16,
                ctrl_flags=tuple(ctrl_flags),
            )
        )

    return plans


def build_layer_strip_plan(shape: LayerShape, strip_rows: int) -> list[StripPlan]:
    if shape.role != "npu_conv":
        return []
    if strip_rows <= 0:
        raise ValueError("strip_rows must be positive")

    strips: list[StripPlan] = []
    strip_idx = 0
    out_y = 0
    while out_y < shape.out_h:
        out_rows = min(strip_rows, shape.out_h - out_y)
        raw_in_y0 = out_y * shape.layer.stride - shape.layer.pad
        raw_in_y1 = (out_y + out_rows - 1) * shape.layer.stride - shape.layer.pad + shape.layer.kh
        in_y0 = max(0, raw_in_y0)
        in_y1 = min(shape.in_h, raw_in_y1)
        strips.append(
            StripPlan(
                idx=shape.layer.idx,
                strip_idx=strip_idx,
                out_y=out_y,
                out_rows=out_rows,
                in_y=in_y0,
                in_rows=max(0, in_y1 - in_y0),
                top_pad_rows=max(0, -raw_in_y0),
                bottom_pad_rows=max(0, raw_in_y1 - shape.in_h),
            )
        )
        strip_idx += 1
        out_y += out_rows
    return strips


def build_strip_plan(graph: YoloGraph) -> dict[int, list[StripPlan]]:
    plans = build_block_plan(graph)
    return {
        plan.idx: build_layer_strip_plan(graph.by_idx[plan.idx], plan.strip_rows)
        for plan in plans
    }


PLAN_FLAG_PW_EN = 1 << 0
PLAN_FLAG_HW_PAD = 1 << 1
PLAN_FLAG_OC_SINGLE = 1 << 2
PLAN_FLAG_SILU = 1 << 3
PLAN_FLAG_SILU_REQUANT = 1 << 4


def encode_ctrl_flags(flags: tuple[str, ...]) -> int:
    value = 0
    for flag in flags:
        if flag == "PW_EN":
            value |= PLAN_FLAG_PW_EN
        elif flag == "HW_PAD":
            value |= PLAN_FLAG_HW_PAD
        elif flag == "OC_SINGLE":
            value |= PLAN_FLAG_OC_SINGLE
        elif flag == "SILU":
            value |= PLAN_FLAG_SILU
        elif flag == "SILU_REQUANT":
            value |= PLAN_FLAG_SILU_REQUANT
        else:
            raise ValueError(f"unknown block-plan flag: {flag}")
    return value


def render_block_plan_header(graph: YoloGraph) -> str:
    plans = build_block_plan(graph)
    conv0_strips = build_strip_plan(graph)[0]
    lines = [
        "#ifndef YOLO_BLOCK_PLAN_H",
        "#define YOLO_BLOCK_PLAN_H",
        "",
        "#include <stdint.h>",
        "",
        "#define YOLO_BLOCK_PLAN_COUNT 63u",
        "#define YOLO_PLAN_FLAG_PW_EN 0x00000001u",
        "#define YOLO_PLAN_FLAG_HW_PAD 0x00000002u",
        "#define YOLO_PLAN_FLAG_OC_SINGLE 0x00000004u",
        "#define YOLO_PLAN_FLAG_SILU 0x00000008u",
        "#define YOLO_PLAN_FLAG_SILU_REQUANT 0x00000010u",
        "#define YOLO_CONV0_STRIP_PLAN_COUNT 40u",
        "",
        "typedef struct {",
        "    uint8_t idx;",
        "    uint16_t in_w;",
        "    uint16_t in_h;",
        "    uint16_t in_c;",
        "    uint16_t out_w;",
        "    uint16_t out_h;",
        "    uint16_t out_c;",
        "    uint8_t kernel_h;",
        "    uint8_t kernel_w;",
        "    uint8_t stride;",
        "    uint8_t pad;",
        "    uint16_t strip_rows;",
        "    uint16_t strip_count;",
        "    uint32_t input_ddr;",
        "    uint32_t output_ddr;",
        "    uint32_t weight_ddr;",
        "    uint32_t input_words;",
        "    uint32_t output_words;",
        "    uint32_t weight_words;",
        "    uint32_t flags;",
        "} yolo_block_plan_entry_t;",
        "",
        "typedef struct {",
        "    uint16_t strip_idx;",
        "    uint16_t out_y;",
        "    uint16_t out_rows;",
        "    uint16_t in_y;",
        "    uint16_t in_rows;",
        "    uint8_t top_pad_rows;",
        "    uint8_t bottom_pad_rows;",
        "} yolo_strip_plan_entry_t;",
        "",
        "static const yolo_block_plan_entry_t yolo_block_plan[YOLO_BLOCK_PLAN_COUNT] = {",
    ]
    for plan in plans:
        shape = graph.by_idx[plan.idx]
        flags = encode_ctrl_flags(plan.ctrl_flags)
        lines.append(
            "    "
            f"{{{plan.idx}u, {shape.in_w}u, {shape.in_h}u, {shape.in_c}u, "
            f"{shape.out_w}u, {shape.out_h}u, {shape.out_c}u, "
            f"{shape.layer.kh}u, {shape.layer.kw}u, {shape.layer.stride}u, {shape.layer.pad}u, "
            f"{plan.strip_rows}u, {plan.strip_count}u, "
            f"0x{plan.input_ddr:08X}u, 0x{plan.output_ddr:08X}u, 0x{plan.weight_ddr:08X}u, "
            f"{plan.input_words}u, {plan.output_words}u, {plan.weight_words}u, "
            f"0x{flags:08X}u}},"
        )
    lines.extend(
        [
            "};",
            "",
            "static const yolo_strip_plan_entry_t yolo_conv0_strip_plan[YOLO_CONV0_STRIP_PLAN_COUNT] = {",
        ]
    )
    for strip in conv0_strips:
        lines.append(
            "    "
            f"{{{strip.strip_idx}u, {strip.out_y}u, {strip.out_rows}u, "
            f"{strip.in_y}u, {strip.in_rows}u, "
            f"{strip.top_pad_rows}u, {strip.bottom_pad_rows}u}},"
        )
    lines.extend(
        [
            "};",
            "",
            "#endif",
            "",
        ]
    )
    return "\n".join(lines)


def render_report(graph: YoloGraph) -> str:
    summary = summarize(graph.shapes)
    block_plan = build_block_plan(graph)
    lines = [
        "# YOLOv8n RTL Deploy Sizing",
        "",
        "Generated from `yolov8n_int8/yolov8n_layers.h` and the C golden topology in",
        "`yolov8n_int8/yolov8n_infer.c`.",
        "",
        "## Summary",
        "",
        f"- NPU conv layers: {summary['npu_conv_layers']}",
        f"- CPU decode/DFL tail layers: {summary['cpu_tail_layers']}",
        f"- NPU MACs: {summary['total_macs']:,}",
        f"- CPU DFL MACs: {summary['cpu_tail_macs']:,}",
        f"- Raw NPU weight bytes: {summary['total_weight_bytes_raw']:,}",
        f"- 16-channel-aligned NPU weight bytes: {summary['total_weight_bytes_aligned']:,}",
        f"- Aligned activation reads if every conv streams from memory: {summary['activation_read_bytes_aligned']:,} bytes",
        f"- Aligned activation writes if every conv streams to memory: {summary['activation_write_bytes_aligned']:,} bytes",
        "",
        "The activation numbers are a conservative per-conv streaming upper bound for",
        "Milestone 0. Later strip/layer fusion work should reduce external DDR traffic.",
        "",
        "## Layer Table",
        "",
        "| idx | role | source | input | output | k/s/p | MACs | strip_rows@256KB | strip_bytes |",
        "|---:|:---|:---|---:|---:|:---:|---:|---:|---:|",
    ]
    for shape in sorted(graph.shapes, key=lambda item: item.layer.idx):
        rows = choose_strip_rows(shape)
        strip_bytes = strip_act_bytes(shape, rows)
        lines.append(
            f"| {shape.layer.idx} | {shape.role} | {shape.source} | "
            f"{shape.in_h}x{shape.in_w}x{shape.in_c} | "
            f"{shape.out_h}x{shape.out_w}x{shape.out_c} | "
            f"{shape.layer.kh}x{shape.layer.kw}/{shape.layer.stride}/{shape.layer.pad} | "
            f"{shape.macs:,} | {rows} | {strip_bytes:,} |"
        )
    lines.extend(
        [
            "",
            "## Block Plan Preview",
            "",
            "This is the first deterministic CPU scheduler skeleton. It assigns DDR",
            "tensor slots, weight slots, SRAM bases, strip rows, and control flags for",
            "each NPU conv layer. It is intentionally conservative: tensors are kept",
            "in separate DDR ranges first, then later firmware work can add reuse or",
            "fusion once correctness is stable.",
            "",
            "| idx | input tensor | output tensor | input DDR | output DDR | weight DDR | strips | flags |",
            "|---:|:---|:---|---:|---:|---:|---:|:---|",
        ]
    )
    for plan in block_plan:
        lines.append(
            f"| {plan.idx} | {plan.input_name} | {plan.output_name} | "
            f"0x{plan.input_ddr:08X} | 0x{plan.output_ddr:08X} | 0x{plan.weight_ddr:08X} | "
            f"{plan.strip_count}x{plan.strip_rows} | {'+'.join(plan.ctrl_flags)} |"
        )
    strip_plan = build_strip_plan(graph)
    conv0_strips = strip_plan[0]
    lines.extend(
        [
            "",
            "### Conv0 Strip/Halo Example",
            "",
            "| strip | out_y | out_rows | input_y | input_rows | top_pad | bottom_pad |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for strip in conv0_strips[:4]:
        lines.append(
            f"| {strip.strip_idx} | {strip.out_y} | {strip.out_rows} | "
            f"{strip.in_y} | {strip.in_rows} | {strip.top_pad_rows} | {strip.bottom_pad_rows} |"
        )
    lines.extend(
        [
            "",
            "## Gate For RTL Work",
            "",
            "- The next RTL change must start with a directed test for the single-conv strip path.",
            "- `conv0` is the first stress case: 640-wide input, stride 2, and only 8 output rows fit in a 256KB Act/Out strip budget.",
            "- After every RTL milestone, run `bash run_all.sh sim` and preserve MNIST 10/10.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    layers = parse_layers(root / "yolov8n_int8" / "yolov8n_layers.h")
    graph = build_yolov8n_graph(layers, input_h=640, input_w=640)
    out = root / "docs" / "notes" / "yolov8n-deploy-sizing.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(graph), encoding="utf-8")
    print(f"wrote {out}")
    header = root / "firmware" / "yolo_block_plan.h"
    header.write_text(render_block_plan_header(graph), encoding="utf-8")
    print(f"wrote {header}")


if __name__ == "__main__":
    main()
