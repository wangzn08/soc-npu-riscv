# YOLOv8n M5h Firmware Conv Descriptor Runner

## Goal

Start replacing hand-written YOLO smoke scheduling with a reusable CPU-side
descriptor runner that can consume generated block plans.

## Scope

- Firmware-only change; no RTL datapath change.
- Keep one shared NPU path for MNIST and YOLO.
- First user is the real conv5 concat-channel pointwise smoke from M5f.

## Implementation

- Add `yolo_conv_desc_t` in `firmware/yolo_plan.h`.
- Add `yolo_run_conv_desc()` in `firmware/yolo_plan.c`.
- The descriptor runner performs Act/Wgt DMA preload, pad-value setup, SiLU
  requant setup, shared NPU conv launch, and Out-SRAM DMA drain.
- Convert `firmware/yolo_concat_channel_pwconv_cref_smoke.c` to call the
  descriptor runner instead of open-coding the conv sequence.
- Export `YOLO_CONCAT_CH_IN_ZP` from the conv5 data generator for descriptor
  pad-value programming.

## Verification

- `bash run_all.sh sim yolo_concat_channel_pwconv_cref_smoke.c yolo_ops.c yolo_plan.c`
  passes with `YOLO CONCAT-CHANNEL PWCONV CREF CPU SMOKE PASS`.
- Default MNIST deploy still passes: `10/10 correct`, `DEPLOY SUCCESS`, and
  `TRAP after 941155 clock cycles`.
