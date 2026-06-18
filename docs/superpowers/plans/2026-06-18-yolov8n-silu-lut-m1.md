# YOLOv8n SiLU LUT Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-off SiLU LUT path to the shared `post_process_top` so YOLOv8n activations use the same NPU hardware as MNIST without changing the MNIST default behavior.

**Architecture:** Add `i_silu_en` to the existing shared `post_process_top`. When low, the existing ReLU/clip path is byte-identical. When high, stage 3 saturates the signed stage-2 quantized value to INT8 and maps it through a 256-entry Q4.4 SiLU ROM loaded from `rtl/silu_lut_q4_4.hex`. Wire the control as `CTRL[18]` through `param_regfile`, `npu_top`, and `firmware.h`; default reset value is 0.

**Tech Stack:** Verilog/SystemVerilog RTL, ModelSim directed test, existing `bash run_all.sh sim` MNIST regression.

## Global Constraints

- Write the failing directed test before RTL changes.
- Keep `CTRL[18]` default off.
- Do not change existing ReLU, ReLU6, pooling, int32_out, or MNIST firmware behavior.
- Run the directed SiLU test and full MNIST regression after implementation.

---

## Task 1: RED SiLU Directed Test

- [ ] Create `tests/tb_silu_lut.v`.
- [ ] Instantiate `post_process_top` with `.i_silu_en(i_silu_en)`.
- [ ] Drive `scale_mul=1`, `scale_shift=0`, `bias=0`, pool disabled.
- [ ] Feed representative stage-2 values `-64, -16, 0, 16, 32, 64, 127`.
- [ ] Expect Q4.4 SiLU ROM outputs `ff, fc, 00, 0c, 1c, 3f, 7f`.
- [ ] Compile before implementation and verify failure because `i_silu_en` is not a valid port.

## Task 2: GREEN SiLU RTL

- [ ] Add `rtl/silu_lut_q4_4.hex`.
- [ ] Add `i_silu_en` port to `rtl/post_process_top.v`.
- [ ] Load the ROM with `$readmemh("rtl/silu_lut_q4_4.hex", silu_lut);`.
- [ ] In stage 3, when `i_silu_en=1`, bypass ReLU/clip and output `silu_lut[saturated_signed_s2[7:0]]`.
- [ ] Leave the existing path unchanged when `i_silu_en=0`.

## Task 3: Control Wiring

- [ ] Add `o_silu_en` to `rtl/param_regfile.v`, backed by `CTRL[18]`.
- [ ] Add `cfg_silu_en` in `rtl/npu_top.v` and connect it to `post_process_top`.
- [ ] Add `#define NPU_CTRL_SILU_EN (1 << 18)` to `firmware/firmware.h`.

## Task 4: Verification

- [ ] Compile and run `tb_silu_lut`.
- [ ] Run `bash run_all.sh sim`.
- [ ] Confirm MNIST still reports 10/10, `DEPLOY SUCCESS.`, and `ALL TESTS PASSED.`
