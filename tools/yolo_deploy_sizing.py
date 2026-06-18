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


def align16(value: int) -> int:
    return ((value + 15) // 16) * 16


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


def render_report(graph: YoloGraph) -> str:
    summary = summarize(graph.shapes)
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


if __name__ == "__main__":
    main()
