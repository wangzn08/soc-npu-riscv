# YOLOv8n Real Stride-2 Conv Milestone 5d Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the shared NPU's ordinary im2col convolution path can execute a real YOLO stride-2 downsample block. This is the next required step after stride-1 3x3 support.

**Architecture:** Use the first output-channel tile of `conv1`: `OC16 x IC16 x 3x3`, `stride=2`, `pad=1`. The full `conv1` layer has OC32, but this milestone intentionally verifies stride-2 behavior first; multi-OC-tile coverage is the next milestone. The CPU loads a 6x6x16 input tile, real packed weights, per-channel qparams, SiLU/requant settings, and `NPU_PAD_VALUE=input_zp`, then compares the 3x3x16 output against a C-reference golden with a 10-LSB signed INT8 tolerance.

## Task 1: RED Stride-2 Smoke

- [x] Add `firmware/yolo_conv3x3_stride2_cref_smoke.c`.
- [x] Reference a missing generated `yolo_conv3x3_stride2_real_data.h`.
- [x] Run `bash run_all.sh fw yolo_conv3x3_stride2_cref_smoke.c yolo_ops.c`.
- [x] Expected RED: missing generated data header.

## Task 2: Real Conv1 Fixture

- [x] Add `tools/gen_yolo_conv3x3_stride2_real_smoke.py`.
- [x] Read real `conv1_w.bin`, `conv1_b.bin`, and `conv1_s.bin`.
- [x] Generate deterministic 6x6x16 input near the layer input zero-point.
- [x] Pack the first 16 output channels as `oc -> kh*kw -> ic_group` words for `wgt_reader`.
- [x] Generate per-channel qparams, SiLU requant config, and C-reference expected output.
- [x] Set `YOLO_CONV3X3_S2_CREF_TOL=10`, matching the observed LUT/requant approximation envelope.

## Task 3: Verification

- [x] Run `bash run_all.sh clean && bash run_all.sh sim yolo_conv3x3_stride2_cref_smoke.c yolo_ops.c`.
- [x] Expected: `YOLO REAL CONV3X3 STRIDE2 CREF CPU SMOKE PASS` and `ALL TESTS PASSED.`
- [x] Run `bash run_all.sh clean && bash run_all.sh sim`.
- [x] Expected MNIST: `=== Result: 10/10 correct ===`, `DEPLOY SUCCESS.`, and `ALL TESTS PASSED.`

## Result

Complete for one real stride-2 OC tile. The real YOLO conv1 stride-2 C-reference smoke passes in 48,146 cycles. The default MNIST deploy remains 10/10 with the same 941,155-cycle trap count.

Remaining gap: this proves one stride-2 OC tile. The next milestone should cover multi-OC-tile / multi-IC-group execution for a real YOLO layer.
