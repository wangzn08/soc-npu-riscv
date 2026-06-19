# YOLOv8n M5t Conv0 To Conv4 C2f-Bottleneck Chain Smoke

## Goal

Extend the early real YOLO RTL chain through the first complete C2f bottleneck:
`conv0 -> conv1 -> conv2 -> slice(s1) -> conv3 -> conv4`.

This proves the shared CPU+NPU path can run the first C2f branch's two 3x3
convolutions back-to-back after selecting the correct split channel group.

## Scope

- Firmware smoke and generated fixtures only; no RTL datapath change.
- Use real YOLOv8n conv0..conv4 weights/qparams.
- Keep the test strip/block-local: four conv0 strips, one conv1/conv2/conv3/conv4
  block chain, and DDR-visible final compare.

## Implementation

- Add `firmware/yolo_conv0_conv4_chain_smoke.c`.
- Add `tools/gen_yolo_conv4_from_conv3_chain_smoke.py`.
- Generate `firmware/yolo_conv4_from_conv3_chain_data.h`.
- Reuse the existing channel-slice helper for the C2f `s1` branch and then run
  real conv3 and conv4 through the shared 3x3 NPU path.

## Verification

- Initial TDD red run failed because `yolo_conv4_from_conv3_chain_data.h` did
  not exist.
- `bash run_all.sh sim yolo_conv0_conv4_chain_smoke.c yolo_ops.c yolo_plan.c`
  passes with `YOLO CONV0->CONV4 C2F-BOTTLENECK CPU SMOKE PASS`,
  TRAP 11,110,398.
