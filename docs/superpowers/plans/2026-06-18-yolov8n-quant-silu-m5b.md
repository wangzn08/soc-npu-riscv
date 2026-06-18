# YOLOv8n Quantized SiLU Pointwise Milestone 5b Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move beyond raw RTL integer pointwise checks by validating the YOLO-style quantization pieces in the shared NPU: input zero-point correction folded into bias, per-channel scale/shift, post-process SiLU LUT, and optional final output-scale/zero-point requantization.

**Architecture:** Use real `yolov8n_int8/weights/conv2_{w,b,s}.bin` assets. The host generator folds the C-reference conv2 constants into the current RTL contract:

```text
q4_4 = ((acc + bias_int[oc]) * scale_mul[oc]) >>> scale_shift[oc]
out  = silu_lut_q4_4[clamp_s8(q4_4)]
```

The CPU smoke programs those per-channel qparams, enables `NPU_CTRL_SILU_EN`, drains Out SRAM, and compares exact output bytes. A second smoke enables `CTRL[19]` and `NPU_SILU_REQUANT_CFG` to requantize the SiLU Q4.4 output to the layer output zero-point/scale.

## Task 1: RED Smoke

- [x] Add `firmware/yolo_pwconv_silu_real_smoke.c`.
- [x] Reference a generated `yolo_pwconv_silu_real_data.h`.
- [x] Call a missing `yolo_run_pw_conv1x1_qparams` helper.
- [x] Run `bash run_all.sh fw yolo_pwconv_silu_real_smoke.c yolo_ops.c`.
- [x] Expected RED: missing generated header, then missing qparam helper.

## Task 2: Data Generator

- [x] Add `tools/gen_yolo_pwconv_silu_real_smoke.py`.
- [x] Read real conv2 weights, float bias, and per-channel weight scales.
- [x] Generate deterministic int8 activations near conv2 input zero-point.
- [x] Fold input zero-point correction and float bias into `bias_int`.
- [x] Generate per-channel `scale_mul` and `scale_shift` for Q4.4 preact.
- [x] Generate exact RTL SiLU LUT golden bytes.

## Task 3: Firmware Helper

- [x] Add `yolo_run_pw_conv1x1_qparams` to `firmware/yolo_ops.h`.
- [x] Implement it in `firmware/yolo_ops.c`.
- [x] Preserve the existing uniform-qparam `yolo_run_pw_conv1x1` API.

## Task 4: Verification

- [x] Run `bash run_all.sh clean && bash run_all.sh sim yolo_pwconv_silu_real_smoke.c yolo_ops.c`.
- [x] Expected: `YOLO REAL PWCONV SILU CPU SMOKE PASS` and `ALL TESTS PASSED.`
- [x] Run `bash run_all.sh clean && bash run_all.sh sim`.
- [x] Expected MNIST: `=== Result: 10/10 correct ===`, `DEPLOY SUCCESS.`, and `ALL TESTS PASSED.`

## Task 5: SiLU Output Requant

- [x] Add optional `CTRL[19]` / `NPU_CTRL_SILU_REQUANT_EN`.
- [x] Add `NPU_SILU_REQUANT_CFG` at `0x3CC`: `[31:24]zp [21:16]shift [15:0]mul`.
- [x] Extend `post_process_top` so `i_silu_en && i_silu_requant_en` maps SiLU Q4.4 output through `((silu * mul) >>> shift) + zp` and signed INT8 clamp.
- [x] Add `yolo_set_silu_requant` firmware helper.
- [x] Add `firmware/yolo_pwconv_silu_requant_real_smoke.c`.
- [x] Extend `tools/gen_yolo_pwconv_silu_real_smoke.py` to generate requant golden bytes.
- [x] Run `bash run_all.sh clean && bash run_all.sh sim yolo_pwconv_silu_requant_real_smoke.c yolo_ops.c`.
- [x] Expected: `YOLO REAL PWCONV SILU REQUANT CPU SMOKE PASS` and `ALL TESTS PASSED.`
- [x] Re-run `bash run_all.sh clean && bash run_all.sh sim yolo_pwconv_silu_real_smoke.c yolo_ops.c` to prove default-off behavior preserves the Q4.4 SiLU path.
- [x] Run default MNIST regression.

## Task 6: C-Reference Tolerance Smoke

- [x] Add `firmware/yolo_pwconv_cref_real_smoke.c`.
- [x] Extend `tools/gen_yolo_pwconv_silu_real_smoke.py` to generate `yolo_silu_real_expected_cref` from the YOLO-style float reference:
  input zero-point correction, real input/weight scales, float bias, float SiLU, output scale/zero-point, and C-style rounding.
- [x] Compare RTL output against the C reference with a +/-1 signed INT8 LSB tolerance.
- [x] Run `bash run_all.sh fw yolo_pwconv_cref_real_smoke.c yolo_ops.c` before generator support and observe RED on the missing C-reference array.
- [x] Run `bash run_all.sh clean && bash run_all.sh sim yolo_pwconv_cref_real_smoke.c yolo_ops.c`.
- [x] Expected: `YOLO REAL PWCONV CREF CPU SMOKE PASS` and `ALL TESTS PASSED.`

## Result

Complete. The real conv2 SiLU/qparam smoke passes in 19,566 cycles. The real conv2 SiLU+requant smoke passes in 20,074 cycles. The real conv2 C-reference tolerance smoke passes in 23,425 cycles. The default MNIST deploy remains 10/10 with the same 941,155-cycle trap count.

Remaining gap: this is still a tiny pointwise block smoke. The next step is to carry the C-reference golden through a multi-op real subgraph, then scale the scheduler beyond one block.
