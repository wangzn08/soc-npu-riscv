# YOLOv8n M5o Activation Quant Plan

## Goal

Expose YOLO per-layer activation quantization metadata to PicoRV32 firmware so a
generated scheduler can set pad values, output zero-points, and SiLU controls
without hand-coded constants.

## Scope

- Tool/header/firmware-smoke change; no RTL datapath change.
- Parse `yolo_act_quant[64]` from `yolov8n_int8/yolov8n_layers.h`.
- Generate a firmware-visible `yolo_act_quant_plan[]` table.

## Implementation

- Add `ActQuant` and `parse_act_quant()` in `tools/yolo_deploy_sizing.py`.
- Emit `YOLO_ACT_QUANT_COUNT` and `yolo_act_quant_entry_t` in
  `firmware/yolo_block_plan.h`.
- Extend `firmware/yolo_block_plan_smoke.c` to check conv0 and conv5 quant rows.
- Extend `tests/test_yolo_deploy_sizing.py` to test parser output and generated
  header rows.

## Verification

- `python tests/test_yolo_deploy_sizing.py`
- `python tools/yolo_deploy_sizing.py`
- `bash run_all.sh sim yolo_block_plan_smoke.c`
- Default MNIST deploy still passes with `10/10 correct`, `DEPLOY SUCCESS`, and
  `TRAP 941155`.
