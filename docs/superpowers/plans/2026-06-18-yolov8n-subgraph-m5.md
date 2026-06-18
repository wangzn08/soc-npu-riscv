# YOLOv8n Tiny Subgraph Milestone 5 Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the CPU can schedule multiple shared NPU primitives as a YOLO-style block graph, not only isolated operators.

**Architecture:** Run a tiny deterministic subgraph through the existing CPU+NPU SoC:

```text
2x2 input
  -> pointwise conv, IC16->OC16
  -> upsample2x, 2x2->4x4
  -> concat with a 4x4 skip tensor, IC16+IC16=IC32
  -> pointwise conv, IC32->OC16
  -> CPU byte-compare
```

All tensor movement goes through DDR, AXI DMA, Act/Wgt/Out SRAM, MMIO control, and the shared NPU datapath. The first smoke uses synthetic weights; the second smoke uses real `yolov8n_int8/weights/conv2_w.bin` pointwise weights with a tiny deterministic tensor and RTL-integer golden outputs. A third smoke upgrades the same tiny real subgraph to YOLO-style C-reference quantization semantics: input zero-point correction, per-channel weight scales, float bias, SiLU, and output requantization.

## Task 1: RED Subgraph Smoke

- [x] Add `firmware/yolo_subgraph_smoke.c`.
- [x] Use existing `yolo_ops` APIs for DMA, concat, and pointwise conv.
- [x] Call a missing `yolo_run_upsample2x` helper.
- [x] Run `bash run_all.sh fw yolo_subgraph_smoke.c yolo_ops.c`.
- [x] Expected RED: compile failure due to missing `yolo_run_upsample2x`.

## Task 2: GREEN Upsample Helper

- [x] Add `yolo_run_upsample2x` declaration to `firmware/yolo_ops.h`.
- [x] Implement `yolo_run_upsample2x` in `firmware/yolo_ops.c`.
- [x] Reuse the existing upsample MMIO sequence: source Act base, destination Act base, shape, IC groups, trigger, and done status.

## Task 3: Verification

- [x] Run `bash run_all.sh clean && bash run_all.sh sim yolo_subgraph_smoke.c yolo_ops.c`.
- [x] Expected: `YOLO SUBGRAPH CPU SMOKE PASS` and `ALL TESTS PASSED.`
- [x] Run `bash run_all.sh clean && bash run_all.sh sim`.
- [x] Expected MNIST: `=== Result: 10/10 correct ===`, `DEPLOY SUCCESS.`, and `ALL TESTS PASSED.`

## Task 4: Real-Weight Subgraph

- [x] Add `firmware/yolo_subgraph_real_smoke.c`.
- [x] Add `tools/gen_yolo_subgraph_real_smoke.py`.
- [x] Generate `firmware/yolo_subgraph_real_data.h` from real YOLOv8n `conv2_w.bin`.
- [x] Run a real-weight tiny subgraph: `conv2-style pointwise -> upsample2x -> concat skip -> conv2-style pointwise`.
- [x] Compare final output against exact RTL-integer golden values.
- [x] Run `bash run_all.sh clean && bash run_all.sh sim yolo_subgraph_real_smoke.c yolo_ops.c`.
- [x] Expected: `YOLO REAL SUBGRAPH CPU SMOKE PASS` and `ALL TESTS PASSED.`
- [x] Run `bash run_all.sh clean && bash run_all.sh sim`.
- [x] Expected MNIST: `=== Result: 10/10 correct ===`, `DEPLOY SUCCESS.`, and `ALL TESTS PASSED.`

## Task 5: C-Reference Real Subgraph

- [x] Add `firmware/yolo_subgraph_cref_smoke.c`.
- [x] Extend `tools/gen_yolo_subgraph_real_smoke.py` to emit per-channel qparams, SiLU requant config, and `yolo_real_sub_expected_cref`.
- [x] Keep the old `yolo_real_sub_expected` exact RTL-integer golden so `yolo_subgraph_real_smoke.c` remains a compatibility check.
- [x] Run the tiny real subgraph as `pointwise + SiLU + requant -> upsample2x -> concat -> pointwise + SiLU + requant`.
- [x] Compare final output against the C-reference golden with `YOLO_REAL_SUB_CREF_TOL=3` signed INT8 LSB tolerance, accounting for two stages of LUT/requant approximation.
- [x] Run `bash run_all.sh clean && bash run_all.sh sim yolo_subgraph_cref_smoke.c yolo_ops.c`.
- [x] Expected: `YOLO REAL SUBGRAPH CREF CPU SMOKE PASS` and `ALL TESTS PASSED.`
- [x] Re-run `bash run_all.sh clean && bash run_all.sh sim yolo_subgraph_real_smoke.c yolo_ops.c` to preserve the older exact RTL-integer smoke.

## Result

Complete for synthetic, real-weight RTL-integer, and tiny real C-reference subgraph smoke. The synthetic subgraph passes in 49,986 cycles. The real-weight RTL-integer subgraph passes in 55,925 cycles after the C-reference fixture update. The real C-reference subgraph passes in 71,654 cycles. The default MNIST deploy remains 10/10 with the same 941,155-cycle trap count.

Remaining gap: this is still a tiny pointwise-only subgraph. The next correctness milestone is a real 3x3 YOLO conv block with the same C-reference qparam/SiLU/requant contract, then per-layer strip/block scheduling.
