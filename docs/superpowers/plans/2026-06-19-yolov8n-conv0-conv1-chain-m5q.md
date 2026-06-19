# YOLOv8n M5q Conv0 To Conv1 Chain Smoke

## Goal

Verify a real two-layer YOLO chain through RTL: conv0 produces a DDR-resident
feature block, then conv1 consumes that block directly as its input.

## Scope

- Firmware smoke and generated fixtures only; no RTL datapath change.
- Use real YOLOv8n conv0 and conv1 weights/qparams.
- Run conv0 through the generated flat strip table, then run a conv1 strip from
  the conv0 output layout.

## Implementation

- Expand `tools/gen_yolo_conv0_strip_real_smoke.py` to emit a 64-row input,
  covering four conv0 strips.
- Update `firmware/yolo_conv0_strip_plan_smoke.c` so it loops over
  `YOLO_CONV0_STRIP_TEST_COUNT` strips from
  `yolo_strip_plan[yolo_block_plan[0].strip_offset + s]`.
- Add `tools/gen_yolo_conv1_from_conv0_chain_smoke.py` to build conv1 qparams,
  weights, and expected output from the conv0 RTL-integer reference tensor.
- Add `firmware/yolo_conv0_conv1_chain_smoke.c` to run:
  conv0 strips -> DDR feature block -> conv1 -> DDR output -> RTL golden compare.

## Verification

- `bash run_all.sh sim yolo_conv0_strip_plan_smoke.c yolo_ops.c yolo_plan.c`
  passes with `YOLO CONV0 STRIP PLAN CPU SMOKE PASS`, TRAP 27,790,045.
- `bash run_all.sh sim yolo_conv0_conv1_chain_smoke.c yolo_ops.c yolo_plan.c`
  passes with `YOLO CONV0->CONV1 CHAIN CPU SMOKE PASS`, TRAP 21,077,118.
- Default MNIST deploy still passes with `10/10 correct`, `DEPLOY SUCCESS`, and
  `TRAP 941155`.
