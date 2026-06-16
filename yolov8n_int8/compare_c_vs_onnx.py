"""Compare C-engine debug dump against onnx_checkpoints.json first-8 values."""
import json
import re

CONV_TO_ONNX = {
    "conv0":  "/model.0/act/Mul_output_0",
    "conv2":  "/model.2/cv1/act/Mul_output_0",
    "conv26": "/model.9/cv2/act/Mul_output_0",
    "conv61": "/model.22/cv2.2/cv2.2.2/Conv_output_0",
}

onnx_data = json.load(open("onnx_checkpoints.json"))
log = open("debug_dump.log").read()

for conv, onnx_name in CONV_TO_ONNX.items():
    m = re.search(rf"DEBUG {conv}\s.*?scale=([\d.]+) zp=(-?\d+) first8= ([-\d.\s]+)", log)
    if not m:
        print(f"{conv}: NOT FOUND in debug_dump.log")
        continue
    scale = float(m.group(1))
    c_vals = [float(x) for x in m.group(3).split()][:8]
    o_vals = onnx_data[onnx_name]["values_flat_first_64"][:8]
    print(f"=== {conv}  <->  {onnx_name}   (out_scale={scale:.5f}) ===")
    print(f"  onnx: {[round(v,4) for v in o_vals]}")
    print(f"  c   : {[round(v,4) for v in c_vals]}")
    diffs = [abs(a - b) for a, b in zip(o_vals, c_vals)]
    lsb = [d / scale for d in diffs]
    print(f"  |diff|/out_scale (LSB): {[round(x,2) for x in lsb]}")
    print(f"  max |diff| = {max(diffs):.4f}  ({max(lsb):.2f} LSB)")
    print()
