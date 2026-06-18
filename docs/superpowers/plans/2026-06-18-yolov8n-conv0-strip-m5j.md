# YOLOv8n M5j Conv0 Full-Width Strip Smoke

## Goal

Verify the first generated YOLO strip shape on the real CPU+NPU RTL path:
`conv0` top strip, 640 input columns by 16 input rows, producing 320 by 8 output
pixels with 16 output channels.

## Scope

- Use real YOLOv8n `conv0` weights, bias, and per-channel scales.
- Use the shared descriptor runner from M5h.
- Exercise stride-2 3x3 convolution, hardware padding with input zero-point,
  SiLU, and SiLU-output requantization.
- Keep the smoke focused on strip execution. This is not full-image YOLO
  correctness yet.

## Result

- Added `tools/gen_yolo_conv0_strip_real_smoke.py`.
- Added generated fixture `firmware/yolo_conv0_strip_real_data.h`.
- Added `firmware/yolo_conv0_strip_smoke.c`.
- `bash run_all.sh sim yolo_conv0_strip_smoke.c yolo_ops.c yolo_plan.c` passes:
  `YOLO CONV0 FULL-WIDTH STRIP RTLINT CPU SMOKE PASS`.

## Caveat

The smoke compares against an RTL-integer reference with a 40-LSB tolerance. The
float C-reference and the current integer/LUT approximation can differ
substantially on this first layer (`YOLO_CONV0_STRIP_FLOAT_CREF_MAX_DIFF` is
emitted in the generated header). A later milestone should tighten the conv0
integer reference or adjust the hardware/model approximation before claiming
full YOLO numerical closure.

## Verification

- Conv0 full-width strip RTL smoke passes, TRAP 7,432,360.
- Default MNIST deploy still passes with `10/10 correct`, `DEPLOY SUCCESS`, and
  `TRAP 941155`.
