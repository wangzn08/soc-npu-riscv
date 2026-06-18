# YOLOv8n Block Concat Milestone 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first YOLO FPN/PAN concat capability at block level through CPU scheduling and shared SoC/NPU data movement.

**Architecture:** Do not add a second hardware datapath. Use the existing CPU, shared DDR, NPU AXI DMA, and Act SRAM layout. For two tensors with the same spatial block shape and tile-major layout `addr = base + group * spatial + pos`, concat is implemented by placing source0 groups at `dst + 0` and source1 groups at `dst + src0_groups * spatial`. This is enough for YOLO channel concat when channel counts are multiples of 16; later milestones can add partial-group packing if a graph layer requires it.

**Tech Stack:** PicoRV32 C firmware, `firmware/firmware.h` MMIO macros, ModelSim through `bash run_all.sh`.

## Global Constraints

- Keep MNIST default firmware unchanged.
- CPU must call the feature through C, not by testbench poking SRAM.
- Use shared NPU DMA/Act SRAM; do not create YOLO-only hardware.
- Run concat CPU smoke and default MNIST regression after implementation.

## Task 1: RED CPU Concat Smoke

- [x] Create `firmware/yolo_concat_smoke.c`.
- [x] Include `yolo_ops.h` and call `yolo_concat2_ddr_to_act`.
- [x] Fill two DDR source tensors:
  - source0: 2 channel groups, spatial 2x3.
  - source1: 3 channel groups, spatial 2x3.
- [x] Expect Act output layout groups `[src0.g0, src0.g1, src1.g0, src1.g1, src1.g2]`.
- [x] DMA the Act output back to DDR and compare all 128-bit words from CPU C.
- [x] Run `bash run_all.sh fw yolo_concat_smoke.c` before adding `yolo_ops.h`.
- [x] Expected RED: compile failure because `yolo_ops.h` is missing.

## Task 2: GREEN Shared Firmware YOLO Ops

- [x] Create `firmware/yolo_ops.h`.
- [x] Create `firmware/yolo_ops.c`.
- [x] Implement:
  - `int yolo_dma_ddr_to_act(uint32_t ddr_addr, uint32_t act_base, uint32_t words);`
  - `int yolo_dma_act_to_ddr(uint32_t ddr_addr, uint32_t act_base, uint32_t words);`
  - `int yolo_concat2_ddr_to_act(uint32_t src0_ddr, uint32_t src1_ddr, uint32_t dst_act_base, uint32_t spatial_words, uint32_t src0_groups, uint32_t src1_groups);`
- [x] Split DMA requests into chunks of at most 256 128-bit words.
- [x] Poll `NPU_DMA_STATUS_RD_DONE` / `NPU_DMA_STATUS_WR_DONE`.

## Task 3: Build Hook For Smoke Helpers

- [x] Extend `run_all.sh` so extra positional arguments after the user C file are appended as firmware C sources.
- [x] Verify `bash run_all.sh fw yolo_concat_smoke.c yolo_ops.c` compiles.

## Task 4: Verification

- [x] Run `bash run_all.sh sim yolo_concat_smoke.c yolo_ops.c`.
- [x] Expected: `YOLO CONCAT CPU SMOKE PASS` and `ALL TESTS PASSED.`
- [x] Run `bash run_all.sh sim`.
- [x] Expected MNIST: `=== Result: 10/10 correct ===`, `DEPLOY SUCCESS.`, and `ALL TESTS PASSED.`

## Observed Verification

- RED: `bash run_all.sh fw yolo_concat_smoke.c` failed with `fatal error: yolo_ops.h: No such file or directory`.
- GREEN compile: `bash run_all.sh fw yolo_concat_smoke.c yolo_ops.c` completed and linked `yolo_ops.c`.
- CPU/SoC smoke: `bash run_all.sh sim yolo_concat_smoke.c yolo_ops.c` printed `YOLO CONCAT CPU SMOKE PASS`, `TRAP after 16215 clock cycles`, and `ALL TESTS PASSED.`
- MNIST gate: `bash run_all.sh sim` printed `=== Result: 10/10 correct ===`, `DEPLOY SUCCESS.`, `TRAP after 941155 clock cycles`, and `ALL TESTS PASSED.`
