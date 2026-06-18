# YOLOv8n Pointwise Conv Block Milestone 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the CPU can schedule a YOLO-style 1x1 pointwise conv block through the shared NPU compute datapath.

**Architecture:** Use the existing PicoRV32 CPU, DDR, NPU DMA, Act/Wgt/Out SRAMs, `CTRL[14] pw_en`, systolic array, and post-process path. The smoke uses synthetic deterministic INT8 activations/weights so CPU can check exact output bytes. This is a scheduler primitive for later real YOLO layers, not a second hardware path.

**Tech Stack:** PicoRV32 C firmware, existing NPU MMIO register map, ModelSim through `bash run_all.sh`.

## Global Constraints

- No new RTL for this milestone unless the existing shared NPU path cannot run the block.
- CPU must call the pointwise conv helper from C.
- Keep default MNIST firmware unchanged.
- Run pointwise conv CPU smoke and default MNIST regression.

## Task 1: RED CPU Pointwise Conv Smoke

- [x] Create `firmware/yolo_pwconv_smoke.c`.
- [x] Fill a 2x3 Act tensor in DDR: one 16-channel group, all channels at pixel `pos` equal `pos + 1`.
- [x] Fill 16 pointwise weight words in DDR: output channel `oc` sums the first `(oc % 4) + 1` input channels.
- [x] Call `yolo_dma_ddr_to_act`, `yolo_dma_ddr_to_wgt`, `yolo_run_pw_conv1x1`, and `yolo_dma_out_to_ddr`.
- [x] Expect output byte `out[pos][oc] = (pos + 1) * ((oc % 4) + 1)`.
- [x] Run `bash run_all.sh fw yolo_pwconv_smoke.c yolo_ops.c`.
- [x] Expected RED: compile failure because `yolo_ops.h` does not yet declare the Wgt/Out/conv helpers.

## Task 2: GREEN YOLO Conv Helpers

- [x] Add prototypes to `firmware/yolo_ops.h`:
  - `int yolo_dma_ddr_to_wgt(uint32_t ddr_addr, uint32_t wgt_base, uint32_t words);`
  - `int yolo_dma_out_to_ddr(uint32_t ddr_addr, uint32_t out_base, uint32_t words, uint32_t out_pong);`
  - `int yolo_run_pw_conv1x1(...);`
- [x] Implement Wgt DMA helper using `NPU_DMA_SRAM_SEL=1`.
- [x] Implement Out DMA helper using `NPU_DMA_PATH_CTL=0x1`.
- [x] Implement pointwise conv helper using `NPU_CTRL_START | NPU_CTRL_PW_EN | caller_flags`.
- [x] Wait for the existing firmware IRQ flag `npu_irq_flag`, matching MNIST's CPU/NPU completion semantics.

## Task 3: Verification

- [x] Run `bash run_all.sh sim yolo_pwconv_smoke.c yolo_ops.c`.
- [x] Expected: `YOLO PWCONV CPU SMOKE PASS` and `ALL TESTS PASSED.`
- [x] Run `bash run_all.sh sim`.
- [x] Expected MNIST: `=== Result: 10/10 correct ===`, `DEPLOY SUCCESS.`, and `ALL TESTS PASSED.`

## Observed Verification

- RED: `bash run_all.sh fw yolo_pwconv_smoke.c yolo_ops.c` failed with implicit declarations for `yolo_dma_ddr_to_wgt`, `yolo_run_pw_conv1x1`, and `yolo_dma_out_to_ddr`.
- Initial GREEN attempt exposed a completion-semantics bug: polling `NPU_STATUS_DONE_IRQ` missed the IRQ handler's immediate `NPU_CTRL_CLEAR_DONE`; fixed by waiting on `npu_irq_flag`, matching `deepnet_deploy.c`.
- CPU/SoC smoke: `bash run_all.sh sim yolo_pwconv_smoke.c yolo_ops.c` printed `YOLO PWCONV CPU SMOKE PASS`, `TRAP after 29516 clock cycles`, and `ALL TESTS PASSED.`
- MNIST gate: `bash run_all.sh sim` printed `=== Result: 10/10 correct ===`, `DEPLOY SUCCESS.`, `TRAP after 941155 clock cycles`, and `ALL TESTS PASSED.`
