# YOLOv8n M5s Conv0 To Conv3 C2f-Branch Chain Smoke

## Goal

Extend the real YOLO strip-chain smoke through the first C2f branch:
`conv0 -> conv1 -> conv2 -> split/slice(s1) -> conv3`.

This verifies that the same CPU+NPU SoC can run a generated strip chain, then
select the correct C2f channel branch from a DDR-resident tile-major tensor
before launching the next 3x3 NPU conv.

## Scope

- Firmware helpers, CPU smokes, and generated fixtures only; no RTL datapath
  change.
- Use real YOLOv8n conv0, conv1, conv2, and conv3 weights/qparams.
- Model the first C2f split by slicing conv2 output channel group 1 (`s1`) into
  Act SRAM, matching the C reference's `prev = s1` before conv3.

## Implementation

- Add `yolo_slice_ddr_to_act()` to `firmware/yolo_ops.c/.h`.
- Add `firmware/yolo_slice_smoke.c` to verify channel-group slicing through
  CPU, AXI DMA, Act SRAM, and DDR readback.
- Add `tools/gen_yolo_conv3_from_conv2_chain_smoke.py`.
- Generate `firmware/yolo_conv3_from_conv2_chain_data.h`.
- Add `firmware/yolo_conv0_conv3_chain_smoke.c`.

## Verification

- Initial TDD red run for `firmware/yolo_slice_smoke.c` failed because
  `yolo_slice_ddr_to_act()` was not declared.
- `bash run_all.sh sim yolo_slice_smoke.c yolo_ops.c` passes with
  `YOLO SLICE CPU SMOKE PASS`, TRAP 10,062.
- Initial TDD red run for `firmware/yolo_conv0_conv3_chain_smoke.c` failed
  because `yolo_conv3_from_conv2_chain_data.h` did not exist.
- `bash run_all.sh sim yolo_conv0_conv3_chain_smoke.c yolo_ops.c yolo_plan.c`
  passes with `YOLO CONV0->CONV3 C2F-SLICE CPU SMOKE PASS`, TRAP 12,205,588.
