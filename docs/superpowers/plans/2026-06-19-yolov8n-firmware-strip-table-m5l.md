# YOLOv8n M5l Firmware Strip Table

## Goal

Expose conv0 strip/halo metadata directly to PicoRV32 firmware so later runtime
code can iterate strips from generated data instead of hard-coded row ranges.

## Scope

- Tool and firmware smoke only; no RTL datapath change.
- Generate a conv0 strip table alongside the existing per-layer block plan.
- Verify the generated table under CPU RTL simulation.

## Implementation

- Add `yolo_strip_plan_entry_t` to generated `firmware/yolo_block_plan.h`.
- Add `YOLO_CONV0_STRIP_PLAN_COUNT` and
  `yolo_conv0_strip_plan[YOLO_CONV0_STRIP_PLAN_COUNT]`.
- Extend `firmware/yolo_block_plan_smoke.c` to check conv0 strip 0, 1, and 39.
- Extend `tests/test_yolo_deploy_sizing.py` to require the generated strip
  table in the header.

## Verification

- `python tests/test_yolo_deploy_sizing.py`
- `python tools/yolo_deploy_sizing.py`
- `bash run_all.sh sim yolo_block_plan_smoke.c`
- Default MNIST deploy still passes with `10/10 correct`, `DEPLOY SUCCESS`, and
  `TRAP 941155`.
