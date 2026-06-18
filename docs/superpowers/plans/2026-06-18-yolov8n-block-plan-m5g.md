# YOLOv8n M5g Block Plan Skeleton

## Goal

Move from hand-written YOLO smoke scheduling toward a deterministic per-layer
CPU scheduler table for the full graph.

## Scope

- Keep this milestone tool-side only: no RTL datapath change.
- Emit per-layer NPU conv plan entries with DDR input/output addresses, weight
  DDR addresses, SRAM bases, strip rows, strip counts, tensor names, and control
  flags.
- Preserve the existing strip/block policy: full YOLO tensors are not placed
  wholly on chip.

## Implementation

- Add `BlockPlan` and `build_block_plan()` in `tools/yolo_deploy_sizing.py`.
- Extend `LayerShape` with `input_name` and `output_name` so generated plans can
  name producer/consumer tensors.
- Add a "Block Plan Preview" section to `docs/notes/yolov8n-deploy-sizing.md`.
- Extend `tests/test_yolo_deploy_sizing.py` to check `conv0` strip planning,
  `conv5` concat-channel flags, aligned word counts, and non-overlapping output
  DDR ranges.

## Verification

- `python tests/test_yolo_deploy_sizing.py`
- `python tools/yolo_deploy_sizing.py`
- Default MNIST deploy regression before commit.
