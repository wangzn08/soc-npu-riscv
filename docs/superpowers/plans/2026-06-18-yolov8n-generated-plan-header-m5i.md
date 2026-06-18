# YOLOv8n M5i Generated Firmware Block Plan Header

## Goal

Bridge the Python block-plan tool and PicoRV32 firmware by generating a C header
that firmware can include directly.

## Scope

- Tool and firmware smoke only; no RTL datapath change.
- Generated table contains all 63 NPU conv layers.
- The table records dimensions, kernel/stride/pad, strip rows/counts,
  DDR tensor/weight addresses, word counts, and compact control flags.

## Implementation

- Add `render_block_plan_header()` to `tools/yolo_deploy_sizing.py`.
- Generate `firmware/yolo_block_plan.h` from the same graph model that produces
  the sizing report.
- Add `firmware/yolo_block_plan_smoke.c` to verify the generated table under the
  real PicoRV32 RTL simulation flow.
- Extend `tests/test_yolo_deploy_sizing.py` to lock down header shape and flag
  encoding.

## Verification

- `python tests/test_yolo_deploy_sizing.py`
- `python tools/yolo_deploy_sizing.py`
- `bash run_all.sh sim yolo_block_plan_smoke.c`
- Default MNIST deploy still passes with `10/10 correct` and `TRAP 941155`.
