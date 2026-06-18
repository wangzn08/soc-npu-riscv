# YOLOv8n Real 3x3 Conv Milestone 5c Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the shared NPU's ordinary im2col convolution path can execute a real YOLO 3x3 block with YOLO-style quantization semantics, not only pointwise conv.

**Architecture:** Use `conv3` from `yolov8n_int8`: `OC16 x IC16 x 3x3`, `stride=1`, `pad=1`. The CPU loads a 4x4x16 tile and real packed weights, programs per-channel qparams, sets SiLU/requant, sets the new hardware padding fill byte to the layer input zero-point, runs the NPU conv path, drains Out SRAM, and compares against a C-reference golden.

## Task 1: RED 3x3 Smoke

- [x] Add `firmware/yolo_conv3x3_cref_smoke.c`.
- [x] Reference a missing generated `yolo_conv3x3_real_data.h`.
- [x] Call missing `yolo_set_pad_value` and `yolo_run_conv2d_qparams`.
- [x] Run `bash run_all.sh fw yolo_conv3x3_cref_smoke.c yolo_ops.c`.
- [x] Expected RED: missing header, then missing helper declarations.

## Task 2: Real Conv3 Fixture

- [x] Add `tools/gen_yolo_conv3x3_real_smoke.py`.
- [x] Read real `conv3_w.bin`, `conv3_b.bin`, and `conv3_s.bin`.
- [x] Generate deterministic 4x4x16 input near the layer input zero-point.
- [x] Pack 3x3 weights as `oc -> ic_group -> kh*kw` for `wgt_reader`.
- [x] Generate per-channel qparams, SiLU requant config, and C-reference expected output.
- [x] Use signed INT8 tolerance `YOLO_CONV3X3_CREF_TOL=10` for LUT/requant approximation.

## Task 3: Shared Hardware/Firmware Support

- [x] Add `NPU_PAD_VALUE` at `0x3D0`, defaulting to `0`.
- [x] Route `NPU_PAD_VALUE[7:0]` from `param_regfile` into `npu_top`.
- [x] Replace hard-coded hardware-border zero injection with the configured pad byte.
- [x] Add `yolo_set_pad_value()` to firmware helpers.
- [x] Add `yolo_run_conv2d_qparams()` for small shared-NPU conv blocks.

## Task 4: Verification

- [x] Run `bash run_all.sh clean && bash run_all.sh sim yolo_conv3x3_cref_smoke.c yolo_ops.c`.
- [x] Expected: `YOLO REAL CONV3X3 CREF CPU SMOKE PASS` and `ALL TESTS PASSED.`
- [x] Run `bash run_all.sh clean && bash run_all.sh sim`.
- [x] Expected MNIST: `=== Result: 10/10 correct ===`, `DEPLOY SUCCESS.`, and `ALL TESTS PASSED.`

## Result

Complete for a tiny real stride-1 3x3 block. The real YOLO conv3 3x3 C-reference smoke passes in 66,662 cycles. The default MNIST deploy remains 10/10 with the same 941,155-cycle trap count.

Remaining gap: this proves stride=1 3x3 with pad=1. YOLO still needs full strip/block scheduling and later stride=2 layer handling.
