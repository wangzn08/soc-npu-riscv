# YOLOv8n Real-Weight Pointwise Conv Block Milestone 4b Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move from synthetic pointwise weights to a tiny real YOLOv8n INT8 weight block running through the shared CPU+NPU SoC path.

**Architecture:** Use `yolov8n_int8/weights/conv2_w.bin` as the real model weight source. Generate a 2x2 synthetic activation block with 32 input channels, pack the first 16 output channels of conv2 into the NPU pointwise weight layout, and compute a golden output using the RTL post-process formula `(psum + bias) * scale_mul >>> scale_shift`, ReLU, clip. The CPU loads generated arrays into DDR, uses existing `yolo_ops` DMA/conv helpers, drains Out SRAM, and compares all output bytes.

**Result:** Complete. The real-weight smoke exposed two shared `pw_en` RTL issues that IC=16 synthetic tests could not see: pointwise Act SRAM addressing used pixel-major order, and the FSM did not re-read Act SRAM when advancing to IC group 1+. Both were fixed in `rtl/top_controller_fsm.v` while keeping the same shared NPU datapath.

**Tech Stack:** Python generator, generated C header, PicoRV32 C smoke, ModelSim through `bash run_all.sh`.

## Global Constraints

- Use real YOLOv8n INT8 weights, but keep activation input synthetic and tiny.
- Do not add a separate YOLO-only hardware path. Fix shared pointwise RTL if the real-weight smoke exposes a generality bug.
- CPU must call the same `yolo_ops` helper path used by synthetic M4.
- Run real-weight smoke and default MNIST regression.

## Task 1: RED Real-Weight Smoke

- [x] Create `firmware/yolo_pwconv_real_smoke.c`.
- [x] Include generated `yolo_pwconv_real_data.h`.
- [x] Load generated activation and conv2 weight words into DDR.
- [x] Call `yolo_dma_ddr_to_act`, `yolo_dma_ddr_to_wgt`, `yolo_run_pw_conv1x1`, and `yolo_dma_out_to_ddr`.
- [x] Compare each output byte to `yolo_real_pw_expected`.
- [x] Run `bash run_all.sh fw yolo_pwconv_real_smoke.c yolo_ops.c`.
- [x] Expected RED: compile failure because `yolo_pwconv_real_data.h` is missing.

## Task 2: GREEN Data Generator

- [x] Create `tools/gen_yolo_pwconv_real_smoke.py`.
- [x] Read `yolov8n_int8/weights/conv2_w.bin` as `[32,32,1,1]` int8.
- [x] Pack first 16 output channels into NPU Wgt SRAM layout: `word = oc * ic_groups + ic_group`, each word holds 16 IC bytes.
- [x] Generate deterministic positive activation bytes for a 2x2 block and 32 channels.
- [x] Compute golden output for `scale_mul=1`, `scale_shift=7`, ReLU enabled.
- [x] Emit `firmware/yolo_pwconv_real_data.h`.

## Task 3: GREEN Shared Pointwise RTL

- [x] Confirm RED mismatch pattern from `bash run_all.sh sim yolo_pwconv_real_smoke.c yolo_ops.c`.
- [x] Trace pointwise Act SRAM address generation in `rtl/top_controller_fsm.v`.
- [x] Change pointwise Act addressing to the existing tile-major tensor contract: `addr = ic_group * spatial + pixel`.
- [x] Reproduce the remaining mismatch as "IC group 1 reused IC group 0 activation data".
- [x] Route pointwise IC-tile advances back through `S_PREFETCH_WGT` so each IC group performs a fresh Act SRAM read without re-prefetching resident weights.

## Task 4: Verification

- [x] Run `python tools/gen_yolo_pwconv_real_smoke.py`.
- [x] Run `bash run_all.sh sim yolo_pwconv_real_smoke.c yolo_ops.c`.
- [x] Expected: `YOLO REAL PWCONV CPU SMOKE PASS` and `ALL TESTS PASSED.`
- [x] Run `bash run_all.sh sim`.
- [x] Expected MNIST: `=== Result: 10/10 correct ===`, `DEPLOY SUCCESS.`, and `ALL TESTS PASSED.`
