"""Extract per-conv input/output activation (scale, zero_point) from the
INT8 QDQ ONNX graph. Does NOT touch weights/*.bin or yolov8n_layers.h.
Cross-checks conv ordering against the existing yolov8n_layers.h dims."""
import json
import os
import re
import onnx
from onnx import numpy_helper

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(OUT_DIR, "yolov8n_int8.onnx")
LAYERS_H_PATH = os.path.join(OUT_DIR, "yolov8n_layers.h")


def parse_existing_dims():
    """Parse {oc,ic,kh,kw,stride,pad,...} rows out of yolov8n_layers.h,
    in declaration order (= conv index order used by yolov8n_infer.c)."""
    with open(LAYERS_H_PATH) as f:
        text = f.read()
    rows = re.findall(r"\{(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),", text)
    return [tuple(int(x) for x in r) for r in rows]


def get_quant_param(node, init_map):
    """For a QuantizeLinear or DequantizeLinear node, return (scale, zero_point)."""
    scale = float(numpy_helper.to_array(init_map[node.input[1]]).flat[0])
    zp = 0
    if len(node.input) > 2 and node.input[2] in init_map:
        zp = int(numpy_helper.to_array(init_map[node.input[2]]).flat[0])
    return scale, zp


def main():
    model = onnx.load(ONNX_PATH)
    init_map = {i.name: i for i in model.graph.initializer}

    nodes_by_output = {}
    for n in model.graph.node:
        for o in n.output:
            nodes_by_output[o] = n
    nodes_by_input = {}
    for n in model.graph.node:
        for i in n.input:
            nodes_by_input.setdefault(i, []).append(n)

    convs = [n for n in model.graph.node if n.op_type == "Conv"]
    print(f"Found {len(convs)} Conv nodes in ONNX graph")

    expected_dims = parse_existing_dims()
    assert len(expected_dims) == len(convs), (
        f"yolov8n_layers.h has {len(expected_dims)} conv rows but ONNX has "
        f"{len(convs)} Conv nodes -- ordering assumption is broken, stop.")

    meta = []
    for idx, conv in enumerate(convs):
        dq_w = nodes_by_output.get(conv.input[1])
        oc = ic = kh = kw = None
        if dq_w is not None and dq_w.input[0] in init_map:
            w_arr = numpy_helper.to_array(init_map[dq_w.input[0]])
            oc, ic, kh, kw = w_arr.shape

        exp_oc, exp_ic, exp_kh, exp_kw, exp_s, exp_p = expected_dims[idx]
        if (oc, ic, kh, kw) != (exp_oc, exp_ic, exp_kh, exp_kw):
            print(f"  WARNING conv{idx}: ONNX dims ({oc},{ic},{kh},{kw}) != "
                  f"yolov8n_layers.h dims ({exp_oc},{exp_ic},{exp_kh},{exp_kw})")

        dq_in = nodes_by_output.get(conv.input[0])
        assert dq_in is not None and dq_in.op_type == "DequantizeLinear", (
            f"conv{idx} input is not fed by DequantizeLinear: {conv.input[0]}")
        in_scale, in_zp = get_quant_param(dq_in, init_map)

        # In this QDQ graph a conv output is ALWAYS first consumed by a
        # QuantizeLinear (Q1). SiLU, if present, sits AFTER the matching
        # DequantizeLinear: Conv -> Q1 -> DQ1 -> {Sigmoid, Mul} -> Q2.
        out_name = conv.output[0]
        q1 = next((c for c in nodes_by_input.get(out_name, [])
                   if c.op_type == "QuantizeLinear"), None)
        assert q1 is not None, f"conv{idx} output not consumed by QuantizeLinear"
        dq1 = next((c for c in nodes_by_input.get(q1.output[0], [])
                    if c.op_type == "DequantizeLinear"), None)
        post = nodes_by_input.get(dq1.output[0], []) if dq1 is not None else []
        has_silu = any(c.op_type == "Sigmoid" for c in post)

        if has_silu:
            # post-SiLU output quantizer = QuantizeLinear consuming the Mul output
            mul_node = next(c for c in post if c.op_type == "Mul")
            q_out = next(c for c in nodes_by_input[mul_node.output[0]]
                         if c.op_type == "QuantizeLinear")
        else:
            q_out = q1
        out_scale, out_zp = get_quant_param(q_out, init_map)

        meta.append({
            "idx": idx,
            "oc": oc, "ic": ic, "kh": kh, "kw": kw,
            "in_scale": in_scale, "in_zp": in_zp,
            "out_scale": out_scale, "out_zp": out_zp,
            "has_silu": has_silu,
        })
        print(f"  conv{idx}: in=({in_scale:.6f},{in_zp}) out=({out_scale:.6f},{out_zp}) silu={has_silu}")

    for a, b in [(0, 1), (1, 2)]:
        if abs(meta[a]["out_scale"] - meta[b]["in_scale"]) > 1e-9 or meta[a]["out_zp"] != meta[b]["in_zp"]:
            print(f"  WARNING adjacency: conv{a}.out=({meta[a]['out_scale']:.6f},{meta[a]['out_zp']}) "
                  f"!= conv{b}.in=({meta[b]['in_scale']:.6f},{meta[b]['in_zp']})")
        else:
            print(f"  OK adjacency conv{a}.out == conv{b}.in")

    glue_ops = []
    for n in model.graph.node:
        if n.op_type not in ("Add", "Concat", "MaxPool", "Resize"):
            continue
        out_name = n.output[0]
        consumers = nodes_by_input.get(out_name, [])
        q_out = next((c for c in consumers if c.op_type == "QuantizeLinear"), None)
        if q_out is None:
            continue  # output feeds straight into something else (e.g. final reshape), skip
        out_scale, out_zp = get_quant_param(q_out, init_map)

        in_scales = []
        for inp in n.input:
            dq = nodes_by_output.get(inp)
            if dq is not None and dq.op_type == "DequantizeLinear":
                s, z = get_quant_param(dq, init_map)
                in_scales.append((s, z))
        glue_ops.append({
            "name": n.name, "op_type": n.op_type,
            "out_scale": out_scale, "out_zp": out_zp,
            "in_scales": in_scales,
        })
        same_as_input = len(in_scales) > 0 and all(s == in_scales[0] for s in in_scales) and (out_scale, out_zp) == in_scales[0]
        print(f"  {n.op_type} {n.name}: in={in_scales} out=({out_scale:.6f},{out_zp}) "
              f"{'[SAME AS INPUT]' if same_as_input else '[RESCALE NEEDED]'}")

    with open(os.path.join(OUT_DIR, "act_quant_meta.json"), "w") as f:
        json.dump({"convs": meta, "glue_ops": glue_ops}, f, indent=2)
    print(f"\nWrote act_quant_meta.json with {len(meta)} conv entries, {len(glue_ops)} glue op entries")


if __name__ == "__main__":
    main()
