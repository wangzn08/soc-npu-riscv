# YOLOv8n Shared Upsample2x Milestone 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-idle 2x nearest-neighbor upsample engine to the shared NPU SRAM datapath so YOLOv8n FPN/PAN upsample operators can run in RTL without adding a second model-specific hardware path.

**Architecture:** Reuse the existing Act SRAM Port-B data-movement lane. The `upsample2x` engine reads tile-major Act words from `NPU_DMA_RD_SRAM_BASE` and writes the 2x-expanded tensor to `NPU_DMA_WR_SRAM_BASE`. Firmware programs dimensions through new registers at `0x3C0..0x3C8`. Port-B priority becomes `img_expand > upsample2x > sram_copy > DMA`; firmware is responsible for not overlapping these engines. Reset/default behavior leaves MNIST unchanged.

**Tech Stack:** Verilog/SystemVerilog RTL, ModelSim directed tests, existing `bash run_all.sh sim` MNIST regression.

## Global Constraints

- MNIST and YOLO share the same `npu_top`, SRAMs, AXI DMA, and post-process datapath.
- Add a directed integration test before production wiring.
- Keep all new triggers/configs default zero and idle.
- Do not change MNIST firmware scheduling.
- Run directed tests and full MNIST regression after implementation.

## Task 1: RED npu_top Upsample Integration Test

- [x] Create `tests/tb_npu_upsample2x.v`.
- [x] Instantiate `npu_top` directly and drive AXI-Lite register writes.
- [x] Preload Act SRAM ping bank with two IC groups of a 2x3 tile-major tensor.
- [x] Program `NPU_DMA_RD_SRAM_BASE`, `NPU_DMA_WR_SRAM_BASE`, `NPU_UPSAMPLE_CFG0`, and `NPU_UPSAMPLE_CFG1`.
- [x] Trigger `NPU_DMA_UPSAMPLE_TRIG` and expect `NPU_DMA_STATUS[4]`.
- [x] Check every output word obeys `dst(g,2y+dy,2x+dx)=src(g,y,x)`.
- [x] Run before wiring and verify failure because no `npu_top` register/Port-B integration exists.

## Task 2: GREEN Register Map

- [x] Add upsample config outputs to `rtl/param_regfile.v`.
- [x] Add `o_upsample_trig`, `i_upsample_done`, `o_upsample_in_w`, `o_upsample_in_h`, and `o_upsample_ic_groups`.
- [x] Decode:
  - `0x3C0`: `NPU_UPSAMPLE_CFG0 = {in_h[31:16], in_w[15:0]}`
  - `0x3C4`: `NPU_UPSAMPLE_CFG1 = {16'd0, ic_groups[15:0]}`
  - `0x3C8`: write pulse `NPU_DMA_UPSAMPLE_TRIG`
- [x] Expose `NPU_DMA_STATUS[4] = upsample_done`.
- [x] Add firmware macros in `firmware/firmware.h`.

## Task 3: GREEN npu_top Wiring

- [x] Add `rtl/upsample2x.v` to `axi_sys.f`.
- [x] Instantiate `upsample2x` in `rtl/npu_top.v`.
- [x] Drive source/destination bases from existing DMA SRAM base registers.
- [x] Insert the engine into Act SRAM Port-B priority between `img_expand` and `sram_copy`.
- [x] Feed `act_sram_dob` to the engine read-data input.

## Task 4: Verification

- [x] Run `tb_upsample2x`.
- [x] Run `tb_npu_upsample2x`.
- [x] Run CPU/NPU cooperative smoke: `bash run_all.sh sim upsample2x_smoke.c`.
- [x] Run `bash run_all.sh sim`.
- [x] Confirm MNIST still reports `=== Result: 10/10 correct ===`, `DEPLOY SUCCESS.`, and `ALL TESTS PASSED.`

## Observed Verification

- `tb_upsample2x`: `TB_UPSAMPLE2X PASS`.
- `tb_npu_upsample2x`: `TB_NPU_UPSAMPLE2X PASS`.
- `bash run_all.sh sim upsample2x_smoke.c`: `UPSAMPLE2X CPU SMOKE PASS`, `ALL TESTS PASSED.`
- `bash run_all.sh sim`: MNIST `10/10`, `DEPLOY SUCCESS.`, `ALL TESTS PASSED.`, `TRAP after 941155 clock cycles`.
