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

        out_name = conv.output[0]
        consumers = nodes_by_input.get(out_name, [])
        has_silu = any(c.op_type == "Sigmoid" for c in consumers)

        if has_silu:
            sigmoid_node = next(c for c in consumers if c.op_type == "Sigmoid")
            mul_node = nodes_by_input[sigmoid_node.output[0]][0]
            assert mul_node.op_type == "Mul"
            q_out = nodes_by_input[mul_node.output[0]][0]
        else:
            q_out = consumers[0]
        assert q_out.op_type == "QuantizeLinear", (
            f"conv{idx} output consumer is not QuantizeLinear: {q_out.op_type}")
        out_scale, out_zp = get_quant_param(q_out, init_map)

        meta.append({
            "idx": idx,
            "oc": oc, "ic": ic, "kh": kh, "kw": kw,
            "in_scale": in_scale, "in_zp": in_zp,
            "out_scale": out_scale, "out_zp": out_zp,
            "has_silu": has_silu,
        })
        print(f"  conv{idx}: in=({in_scale:.6f},{in_zp}) out=({out_scale:.6f},{out_zp}) silu={has_silu}")

    with open(os.path.join(OUT_DIR, "act_quant_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote act_quant_meta.json with {len(meta)} entries")


if __name__ == "__main__":
    main()
