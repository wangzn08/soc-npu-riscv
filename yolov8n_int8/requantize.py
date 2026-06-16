"""Proper static INT8 quantization of yolov8n.onnx using real COCO images
(coco128) for calibration. The pre-existing yolov8n_int8.onnx had a collapsed
classification head (all-zero class scores); this regenerates a correct QDQ
int8 model: per-channel symmetric int8 weights + asymmetric int8 activations.
"""
import glob
import os
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FP32 = os.path.join(OUT_DIR, "yolov8n.onnx")
PREP = os.path.join(OUT_DIR, "yolov8n_prep.onnx")
INT8 = os.path.join(OUT_DIR, "yolov8n_int8_new.onnx")
CALIB_DIR = os.path.join(OUT_DIR, "datasets", "coco128", "images", "train2017")


def load_img(path):
    img = np.asarray(Image.open(path).convert("RGB").resize((640, 640)),
                     dtype=np.float32) / 255.0
    return img.transpose(2, 0, 1)[None]  # NCHW


class CocoReader(CalibrationDataReader):
    def __init__(self, files, input_name):
        self.files = files
        self.input_name = input_name
        self.i = 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        d = {self.input_name: load_img(self.files[self.i])}
        self.i += 1
        return d


def main():
    files = sorted(glob.glob(os.path.join(CALIB_DIR, "*.jpg")))
    assert files, f"no calibration images in {CALIB_DIR}"
    print(f"calibration images: {len(files)}")

    # symbolic shape inference + optimization for better static quant
    quant_pre_process(FP32, PREP, skip_symbolic_shape=False)

    sess = ort.InferenceSession(PREP, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print("input name:", input_name)

    # Exclude the Detect-head decode tail from quantization. The final output
    # concatenates bbox (range 0..640) with class scores (range 0..1); a single
    # shared output quantizer scale (~640/127) rounds every class score to 0 and
    # collapses the classification head. Keeping the decode (DFL projection, box
    # math, sigmoid, final concat) in float fixes that, while all 64 Conv layers
    # stay int8. The decode is float on the SoC CPU side anyway.
    pm = onnx.load(PREP)
    convs = [i for i, n in enumerate(pm.graph.node) if n.op_type == "Conv"]
    last_feature_conv = max(convs[:-1]) if len(convs) > 1 else convs[-1]
    # exclude every non-Conv node topologically after the last head feature conv
    exclude = [n.name for i, n in enumerate(pm.graph.node)
               if i > last_feature_conv and n.op_type != "Conv"]
    print(f"excluding {len(exclude)} decode-tail nodes from quantization")

    reader = CocoReader(files, input_name)
    quantize_static(
        PREP, INT8, reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        nodes_to_exclude=exclude,
    )
    print("wrote", INT8)


if __name__ == "__main__":
    main()
