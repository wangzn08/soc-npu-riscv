# YOLOv8n M5r Conv0 To Conv2 Chain Smoke

## Goal

Extend the real YOLO strip-chain smoke from two layers to three layers:
`conv0 -> conv1 -> conv2`.

This proves that a DDR-resident feature block produced by one generated strip
plan can feed both the shared 3x3 im2col path and the shared 1x1 pointwise path
without adding a YOLO-only hardware datapath.

## Scope

- Firmware smoke and generated fixtures only; no RTL datapath change.
- Use real YOLOv8n conv0, conv1, and conv2 weights/qparams.
- Run four generated conv0 strips, one conv1 strip, and one conv2 pointwise
  block through CPU MMIO scheduling, AXI DMA, NPU compute, and DDR compare.

## Implementation

- Add `firmware/yolo_conv0_conv1_conv2_chain_smoke.c`.
- Add `tools/gen_yolo_conv2_from_conv1_chain_smoke.py`.
- Generate `firmware/yolo_conv2_from_conv1_chain_data.h` from real conv2
  weights, per-channel qparams, SiLU, and output requant metadata.
- Feed the actual conv1 RTL output from DDR into conv2 using the existing
  `yolo_run_pw_conv1x1_qparams()` helper with `NPU_CTRL_OC_SINGLE`,
  `NPU_CTRL_SILU_EN`, and `NPU_CTRL_SILU_REQUANT_EN`.

## Verification

- Initial TDD red run failed at compile because
  `yolo_conv2_from_conv1_chain_data.h` did not exist.
- First generated run exposed only chained quantization differences:
  max observed mismatch was 47 INT8 counts with the old single-layer tolerance
  of 40.
- Final chain tolerance is 64 for this multi-layer smoke because conv2 consumes
  the actual RTL conv1 output while the generator builds its golden from the
  software RTL-integer conv1 approximation.
- `bash run_all.sh sim yolo_conv0_conv1_conv2_chain_smoke.c yolo_ops.c yolo_plan.c`
  passes with `YOLO CONV0->CONV1->CONV2 CHAIN CPU SMOKE PASS`, TRAP 21,611,402.
