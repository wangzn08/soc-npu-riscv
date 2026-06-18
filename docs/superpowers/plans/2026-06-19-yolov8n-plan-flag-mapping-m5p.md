# YOLOv8n M5p Plan Flag Mapping

## Goal

Map generated YOLO block-plan flags to real NPU control bits in firmware.

## Scope

- Firmware-only change; no RTL datapath change.
- Keep generated compact `YOLO_PLAN_FLAG_*` values separate from the hardware
  `NPU_CTRL_*` register bits.
- Verify the mapping under CPU RTL simulation.

## Implementation

- Add `yolo_ctrl_from_plan_flags()` in `firmware/yolo_plan.c`.
- Declare the helper in `firmware/yolo_plan.h`.
- Extend `firmware/yolo_block_plan_smoke.c` to verify conv0 and conv5 mappings:
  hardware padding, pointwise mode, `oc_single`, and SiLU requant.

## Verification

- RED: firmware build failed with missing `yolo_ctrl_from_plan_flags()`.
- GREEN: `bash run_all.sh sim yolo_block_plan_smoke.c yolo_plan.c yolo_ops.c`
  passes with `YOLO BLOCK PLAN CPU SMOKE PASS`, TRAP 4,906.
- Default MNIST deploy still passes with `10/10 correct`, `DEPLOY SUCCESS`, and
  `TRAP 941155`.
