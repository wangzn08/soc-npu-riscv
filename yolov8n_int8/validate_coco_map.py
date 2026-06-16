"""COCO val2017 mAP for FP32 (yolov8n.pt) vs INT8 QDQ ONNX (yolov8n_int8.onnx).

Uses ultralytics predict for letterbox/decode/NMS (identical pipeline for both
models, so the comparison is apples-to-apples), then scores with pycocotools
COCOeval. Avoids ultralytics' coco.yaml auto-download (which pulls all splits,
~25GB) by evaluating directly against val2017 images + instances_val2017.json.
"""
import json
import os
import sys
import zipfile
import glob
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(OUT_DIR, "datasets")
VAL_DIR = os.path.join(DS, "val2017")
ANN = os.path.join(DS, "annotations", "instances_val2017.json")

# COCO 80-class index -> 91-class category_id
C80_91 = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,
          31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,
          55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,
          82,84,85,86,87,88,89,90]


def ensure_extracted():
    if not os.path.isdir(VAL_DIR):
        z = os.path.join(DS, "val2017.zip")
        print("extracting", z)
        with zipfile.ZipFile(z) as f:
            f.extractall(DS)
    if not os.path.exists(ANN):
        z = os.path.join(DS, "annotations_trainval2017.zip")
        print("extracting", z)
        with zipfile.ZipFile(z) as f:
            f.extractall(DS)
    imgs = sorted(glob.glob(os.path.join(VAL_DIR, "*.jpg")))
    print(f"val2017 images: {len(imgs)}")
    return imgs


def run_model(model_path, imgs, device, tag):
    print(f"\n[{tag}] predicting with {model_path} on {device} ...")
    model = YOLO(model_path, task="detect")
    dets = []
    results = model.predict(imgs, imgsz=640, conf=0.001, iou=0.7, max_det=300,
                            device=device, stream=True, verbose=False)
    for img_path, r in zip(imgs, results):
        image_id = int(os.path.splitext(os.path.basename(img_path))[0])
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xywh = b.xywh.cpu().numpy()  # center x,y,w,h (orig coords)
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        for (cx, cy, w, h), sc, c in zip(xywh, conf, cls):
            dets.append({
                "image_id": image_id,
                "category_id": C80_91[c],
                "bbox": [float(cx - w / 2), float(cy - h / 2), float(w), float(h)],
                "score": float(sc),
            })
    print(f"[{tag}] total detections: {len(dets)}")
    return dets


def coco_eval(coco_gt, dets, tag):
    if not dets:
        print(f"[{tag}] no detections -> mAP 0")
        return {"map50": 0.0, "map": 0.0, "mp": 0.0, "mr": 0.0}
    dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"map": float(ev.stats[0]), "map50": float(ev.stats[1])}


def main():
    imgs = ensure_extracted()
    coco_gt = COCO(ANN)
    # restrict to images that have annotations entries (standard: all val2017)
    fp32 = run_model("yolov8n.pt", imgs, 0, "FP32")
    int8 = run_model("yolov8n_int8.onnx", imgs, "cpu", "INT8")

    json.dump(fp32, open(os.path.join(OUT_DIR, "dets_fp32.json"), "w"))
    json.dump(int8, open(os.path.join(OUT_DIR, "dets_int8.json"), "w"))

    print("\n===== FP32 =====")
    r_fp = coco_eval(coco_gt, fp32, "FP32")
    print("\n===== INT8 =====")
    r_i8 = coco_eval(coco_gt, int8, "INT8")

    print("\n" + "=" * 56)
    print(f"{'model':8} {'mAP50':>8} {'mAP50-95':>10}")
    print(f"{'FP32':8} {r_fp['map50']:8.4f} {r_fp['map']:10.4f}")
    print(f"{'INT8':8} {r_i8['map50']:8.4f} {r_i8['map']:10.4f}")
    print(f"{'drop':8} {r_fp['map50']-r_i8['map50']:8.4f} {r_fp['map']-r_i8['map']:10.4f}")
    json.dump({"fp32": r_fp, "int8": r_i8},
              open(os.path.join(OUT_DIR, "coco_map_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
