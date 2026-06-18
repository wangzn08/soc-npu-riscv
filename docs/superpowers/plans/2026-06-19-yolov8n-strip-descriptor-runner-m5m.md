# YOLOv8n M5m Strip Descriptor Runner

## Goal

Move from firmware-visible strip metadata to a firmware helper that can execute
conv0 strips directly from the generated strip table.

## Scope

- Firmware scheduling only; no RTL datapath change.
- Add separate `pad_h/pad_w` support in the YOLO conv helper so middle strips can
  use horizontal padding without vertical padding.
- Add a conv0-specific strip runner for the first two generated strips.

## Implementation

- Add `yolo_run_conv2d_qparams_pads()` in `firmware/yolo_ops.c`.
- Keep the old `yolo_run_conv2d_qparams()` API as a compatibility wrapper with
  equal `pad_h/pad_w`.
- Add `yolo_run_conv0_strip_from_plan()` in `firmware/yolo_plan.c`.
- Add `firmware/yolo_conv0_strip_plan_smoke.c` to run strip0 and strip1 using
  `yolo_conv0_strip_plan[]` instead of hand-written strip dimensions.

## Verification

- RED: `bash run_all.sh fw yolo_conv0_strip_plan_smoke.c yolo_ops.c yolo_plan.c`
  failed because `yolo_run_conv0_strip_from_plan()` did not exist.
- GREEN: `bash run_all.sh sim yolo_conv0_strip_plan_smoke.c yolo_ops.c yolo_plan.c`
  passes with `YOLO CONV0 STRIP PLAN CPU SMOKE PASS`, TRAP 1,296,188.
- Default MNIST deploy still passes with `10/10 correct`, `DEPLOY SUCCESS`, and
  `TRAP 941155`.
