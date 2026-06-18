# YOLOv8n M5k Strip/Halo Plan

## Goal

Make strip scheduling explicit enough for firmware to load only the source rows
needed by each output strip.

## Scope

- Tool/report only; no RTL datapath change.
- Add per-strip row metadata for each NPU conv layer:
  output y, output rows, input y, input rows, top pad rows, and bottom pad rows.
- Keep the existing per-layer block plan and generated firmware table unchanged.

## Implementation

- Add `StripPlan`, `build_layer_strip_plan()`, and `build_strip_plan()` to
  `tools/yolo_deploy_sizing.py`.
- Add a conv0 strip/halo example to `docs/notes/yolov8n-deploy-sizing.md`.
- Extend sizing tests so conv0 top/middle/last strips have the expected halo
  rows.

## Verification

- `python tests/test_yolo_deploy_sizing.py`
- `python tools/yolo_deploy_sizing.py`
- `bash run_all.sh sim yolo_block_plan_smoke.c`
- Default MNIST deploy still passes with `10/10 correct` and `TRAP 941155`.
