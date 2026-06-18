# YOLOv8n M5f Concat-Channel Pointwise Conv Plan

## Goal

Prove the shared CPU+NPU path can execute a YOLO pointwise layer after concat,
where the input has multiple 16-channel groups and the output has multiple OC
tiles.

## Scope

- Target a small real-weight block from YOLOv8n `conv5`: IC48, OC32, 1x1.
- Keep the existing shared NPU datapath; do not add a YOLO-specific engine.
- Use the CPU only for scheduling, DMA setup, and C-reference comparison.
- Preserve MNIST default behavior.

## Implementation

- Add `tools/gen_yolo_concat_channel_pwconv_real_smoke.py` to generate a tiny
  real `conv5` block with tile-major IC48 activation packing, packed OC32
  weights, per-channel qparams, SiLU, and final requantization.
- Add `firmware/yolo_concat_channel_pwconv_cref_smoke.c` to run:
  CPU -> DDR preload -> DMA to Act/Wgt SRAM -> shared NPU pointwise conv ->
  DMA drain -> CPU C-reference compare.
- Extend `run_pw_conv1x1_common()` so pointwise conv can program up to 64 output
  channels when `NPU_CTRL_OC_SINGLE` is explicitly requested.

## Verification

- `bash run_all.sh sim yolo_concat_channel_pwconv_cref_smoke.c yolo_ops.c`
  passes with `YOLO CONCAT-CHANNEL PWCONV CREF CPU SMOKE PASS`.
- Default MNIST deploy must still pass before committing the milestone.
