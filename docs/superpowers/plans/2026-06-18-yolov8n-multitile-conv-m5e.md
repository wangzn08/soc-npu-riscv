# YOLOv8n Real Multi-Tile Conv Milestone 5e Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove a real YOLO 3x3 layer block can use multiple input-channel groups and multiple output-channel tiles on the shared NPU.

**Architecture:** Use `conv8`: `OC32 x IC32 x 3x3`, `stride=1`, `pad=1`. The CPU loads a 4x4x32 input tile, real packed weights, per-channel qparams for all 32 output channels, SiLU/requant settings, and `NPU_PAD_VALUE=input_zp`. The firmware runs one NPU start with `NPU_CTRL_OC_SINGLE`, drains two output-channel tiles, and compares against a C-reference golden using signed INT8 tolerance.

## Task 1: RED Multi-Tile Smoke

- [x] Add `firmware/yolo_conv3x3_multitile_cref_smoke.c`.
- [x] Reference missing `yolo_conv3x3_multitile_real_data.h`.
- [x] Run `bash run_all.sh fw yolo_conv3x3_multitile_cref_smoke.c yolo_ops.c`.
- [x] Expected RED: missing generated data header.

## Task 2: Real Conv8 Fixture

- [x] Add `tools/gen_yolo_conv3x3_multitile_real_smoke.py`.
- [x] Read real `conv8_w.bin`, `conv8_b.bin`, and `conv8_s.bin`.
- [x] Generate deterministic 4x4x32 input near the layer input zero-point.
- [x] Pack weights as `oc -> ic_group -> kh*kw`, matching `wgt_reader`.
- [x] Generate qparams and C-reference output for all 32 output channels.

## Task 3: Firmware Helper

- [x] Extend `yolo_run_conv2d_qparams()` to support `out_c<=64`.
- [x] Require `NPU_CTRL_OC_SINGLE` when `out_c>16`.
- [x] Load per-channel qparams for every active output channel.

## Task 4: Verification

- [x] Run `bash run_all.sh clean && bash run_all.sh sim yolo_conv3x3_multitile_cref_smoke.c yolo_ops.c`.
- [x] Expected: `YOLO REAL CONV3X3 MULTITILE CREF CPU SMOKE PASS` and `ALL TESTS PASSED.`
- [x] Run `bash run_all.sh clean && bash run_all.sh sim`.
- [x] Expected MNIST: `=== Result: 10/10 correct ===`, `DEPLOY SUCCESS.`, and `ALL TESTS PASSED.`

## Result

Complete. The real YOLO conv8 multi-tile C-reference smoke passes in 161,104 cycles. The default MNIST deploy remains 10/10 with the same 941,155-cycle trap count.

Remaining gap: this proves 32x32 3x3. Full YOLO still needs non-16-aligned concat channels and generated whole-model layer plans.
