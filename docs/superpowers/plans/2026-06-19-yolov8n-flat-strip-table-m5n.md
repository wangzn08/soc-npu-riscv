# YOLOv8n M5n Flat Strip Table

## Goal

Generalize strip metadata from conv0-only to a firmware-visible flat table for
all 63 NPU conv layers.

## Scope

- Tool/header/firmware-smoke change; no RTL datapath change.
- Add `strip_offset` to each block-plan entry.
- Emit one `yolo_strip_plan[YOLO_STRIP_PLAN_COUNT]` table with all layer strips.
- Keep `yolo_conv0_strip_plan` as a compatibility macro for existing conv0
  smokes.

## Implementation

- `tools/yolo_deploy_sizing.py` now emits `YOLO_STRIP_PLAN_COUNT` and the flat
  strip table.
- `yolo_block_plan_entry_t` now includes `strip_offset`.
- Tests require the total strip count, conv1 offset, and representative strip
  rows in the generated header.

## Verification

- `python tests/test_yolo_deploy_sizing.py`
- `python tools/yolo_deploy_sizing.py`
- `bash run_all.sh sim yolo_block_plan_smoke.c`
- `bash run_all.sh sim yolo_conv0_strip_plan_smoke.c yolo_ops.c yolo_plan.c`
- Default MNIST deploy still passes with `10/10 correct`, `DEPLOY SUCCESS`, and
  `TRAP 941155`.
