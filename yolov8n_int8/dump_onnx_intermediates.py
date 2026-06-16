"""Run yolov8n_int8.onnx on bus.ppm and dump dequantized float values for
a handful of checkpoint tensors, for comparison against the C engine."""
import json
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

# Real node output names verified to exist in yolov8n_int8.onnx graph.
CHECKPOINTS = [
    "/model.0/act/Mul_output_0",              # conv0 output (after SiLU)  -> C conv idx 0
    "/model.2/cv1/act/Mul_output_0",          # first C2f cv1 output       -> C conv idx 2
    "/model.9/cv2/act/Mul_output_0",          # SPPF cv2 output            -> C conv idx 26
    "/model.22/cv2.2/cv2.2.2/Conv_output_0",  # scale2 bbox head output    -> C conv idx 61
]


def main():
    model = onnx.load("yolov8n_int8.onnx")
    existing_outputs = {o.name for o in model.graph.output}

    for name in CHECKPOINTS:
        if name not in existing_outputs:
            model.graph.output.append(onnx.ValueInfoProto(name=name))

    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    img = Image.open("bus.ppm").convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None]  # NCHW

    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: arr})
    output_names = [o.name for o in sess.get_outputs()]

    dump = {}
    for name, val in zip(output_names, outputs):
        if name in CHECKPOINTS:
            val = np.asarray(val)
            dump[name] = {
                "shape": list(val.shape),
                "values_flat_first_64": val.flatten()[:64].tolist(),
                "min": float(val.min()),
                "max": float(val.max()),
                "mean": float(val.mean()),
            }

    with open("onnx_checkpoints.json", "w") as f:
        json.dump(dump, f, indent=2)
    print("Wrote onnx_checkpoints.json with", list(dump.keys()))


if __name__ == "__main__":
    main()
