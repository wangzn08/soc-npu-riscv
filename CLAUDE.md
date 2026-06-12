# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

PicoRV32 RISC-V CPU + 16×16 systolic-array NPU SoC, verified in ModelSim/Questa.
Firmware (C + asm, rv32imc) runs on the simulated CPU, drives the NPU through MMIO registers.
Target workload: DeepConvNet (15-layer MNIST CNN) inference offloaded to the NPU.

## Build & Simulation

All commands run from an MSYS/Git Bash shell on Windows (not PowerShell/cmd).

```bash
bash run_all.sh              # full flow: fw + RTL compile + simulation
bash run_all.sh fw           # firmware only
bash run_all.sh compile      # RTL compile only (vlib + vlog)
bash run_all.sh sim          # full flow, headless
bash run_all.sh waves        # full flow + GUI waveform
bash run_all.sh clean        # remove sim artifacts
bash run_all.sh distclean    # also remove firmware/build
```

The Makefile is deprecated; use `run_all.sh` instead.

Tool paths (override via environment variables):
- `RISCV_PREFIX` — RISC-V GCC toolchain (default: `E:/Riscv_Tools/xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-`)
- `MGC_LICENSE_FILE` — ModelSim license (default: `E:/modelsim/LICENSE.TXT`)
- `PYTHON` — Python interpreter (default: `python`)

### How testing works

There is no unit-test framework. "The test" is one simulation run:
- The testbench (`rtl/axi_sys_tb.v`) prints `ALL TESTS PASSED.` when firmware writes `123456789` to MMIO `0x2000_0000`, or `ERROR!`/`TIMEOUT` otherwise.
- Firmware `print_*` output appears on the simulator console via UART MMIO.
- The current firmware (`deepnet_deploy.c`) runs 10 MNIST images and prints per-image predictions and an overall accuracy count.
- `vsim` must run from the repo root — firmware hex path is resolved by `$readmemh` relative to the simulation working directory.
- After editing RTL, run `bash run_all.sh clean` before recompiling (or vlog the changed file manually).

## Architecture

### Memory Map (`rtl/axi_sys.v`)

```
picorv32_axi (AXI-Lite master)
  ├─ 0x0000_0000–0x00FF_FFFF → axi_lite_ram (private mem, firmware loaded via $readmemh)
  ├─ 0x1000_0000             → UART tx (write char → simulator $write)
  ├─ 0x2000_0000             → test-pass register (write 123456789 → tests_passed)
  ├─ 0x3000_0000–0x3000_0FFF → NPU registers (pulse interface to npu_axi_wrapper)
  └─ 0x4000_0000–...         → shared memory ("DDR"):
        axi_lite_to_axi_full bridge → axi_arbiter_2to1 ← npu DMA master
                                          └→ axi_full_slave_v1_0_S00_AXI
```

### NPU Register Interface

NPU registers at `0x3000_0000` are special-cased in `axi_sys.v`: MMIO writes/reads are converted to `reg_wr_en/reg_rd_en` pulses into `npu_axi_wrapper` (byte addr → word addr via `addr[11:2]`), with handshake stretching until `reg_wr_done`/`rd_data_valid`.

Register definitions: `rtl/param_regfile.v` (hardware) ↔ `firmware/firmware.h` (firmware `NPU_*` defines). Changes must touch both files.

### NPU Dataflow

```
axi_dma (DDR ↔ SRAM)
  → ping-pong Act/Wgt SRAMs (sram_models.v, 128-bit wide)
  → im2col_line_buffer + wgt_reader
  → systolic_16x16 (gp_4x4 → pe_core, INT8 MAC, output-stationary)
  → post_process_top: bias → quantization (INT32→INT8) → ReLU → max_pooling_2x2 / vector_alu (eltwise add)
  → output SRAM → DMA back to DDR
```

`top_controller_fsm.v` sequences the computation; `param_regfile.v` holds config/status.

### Firmware Architecture

**Entry point:** `usercode7()` in `deepnet_deploy.c`, called from `start7.S`.

**Linked objects** (fixed in `run_all.sh`):
- `start7.S` — reset vector, stack setup, calls `usercode7()`
- `irq.c` — interrupt handler, sets `npu_irq_flag` on NPU IRQ (bit 3)
- `print.c` — UART console output (`print_str`, `print_dec`, `print_hex`)
- `libgcc_stub.c` — GCC runtime helpers (freestanding)
- `deepnet_deploy.c` — main firmware (15-layer CNN inference)

**Build flags:** `-march=rv32imc -O2 -ffreestanding -nostdlib -Werror -Wall -Wextra -pedantic` (strict warning-clean)

**Memory layout** (`sections.lds`): code/data at 0x0 in private RAM, stack grows down from top.

### DeepConvNet (15-layer MNIST CNN)

```
Input: 28×28×1 (INT8, 10 test images from mnist_test_images.h)

Conv1:  30×30×16 (pad 28→30) → 28×28×16, 3×3, stride 1, ReLU        [NPU]
Conv2:  30×30×16 (pad 28→30) → 28×28×16, 3×3, stride 1, ReLU        [NPU]
Pool1:  28×28×16 → 14×14×16, 2×2 max, stride 2                      [CPU]
Conv3:  16×16×16 (pad 14→16) → 14×14×32, 3×3, stride 1, ReLU        [NPU]
Conv4:  18×18×32 (pad 14→18) → 16×16×32, 3×3, stride 1, ReLU        [NPU]
Pool2:  16×16×32 → 8×8×32                                            [CPU]
Conv5:  10×10×32 (pad 8→10)  → 8×8×64, 3×3, stride 1, ReLU          [NPU]
Conv6:  10×10×64 (pad 8→10)  → 8×8×64, 3×3, stride 1, ReLU          [NPU]
Pool3:  8×8×64 → 4×4×64                                              [CPU]
Reorder: spatial-first-per-tile → channels-first                      [CPU]
Affine1: 1024 → 50, ReLU                                             [NPU GEMM]
Affine2: 50 → 10, raw INT32 scores                                   [CPU]
Argmax: find predicted digit                                          [CPU]
```

**NPU operations:** 6 conv layers (all 3×3 INT8 with ReLU) via `npu_conv_pass()`, plus **Affine1 (1024→50) via `npu_gemm_pass()`** (GEMM/FC mode, decision F). Conv weights are preloaded into the Wgt SRAM **PING** bank once (`preload_conv_weights`); FC weights into the **PONG** bank once (`preload_fc_weights`); both stay resident across all images. Each NPU op configures registers, starts the NPU, waits for IRQ (`npu_irq_flag`), then DMA-reads results back to DDR.

**CPU operations:** activation padding (`pad_activation`), data reorder, **Affine2 (50→10, raw INT32 logits)**, argmax. All in `deepnet_deploy.c`. (Max-pool is fused into the NPU conv pass via `pool_en`, see below — `cpu_max_pool_2x2` is unused.)

**Note:** NPU pooling (`pool_en`) IS used — Conv2/Conv4/Conv6 fuse 2×2 max-pool into the same NPU pass. **Affine2 stays on CPU** because the NPU's INT8 post-process output saturates the final logits (argmax would be wrong); FC1 (the bottleneck) and any INT8-activation FC layer benefit from the NPU, the final logits layer does not.

### DDR Buffer Layout (firmware/deepnet_deploy.c)

```
0x4000_1000  ACT_BUF_A    48 KB   activation buffer A
0x4000_4000  ACT_BUF_B    48 KB   activation buffer B
0x4000_7000  WGT_BUF      64 KB   weight staging buffer
0x4001_0000  NPU_OUT_BUF  16 KB   NPU output staging
0x4001_2000  PAD_BUF      12 KB   padded activation buffer
0x4001_5000  SCORES       40 B    10× INT32 classification scores
0x4001_6000  AFFINE_SCR   16 KB   affine layer scratch / data reorder
```

### NPU Register Map (`firmware/firmware.h`, `rtl/param_regfile.v`)

| Offset | Name | Description |
|--------|------|-------------|
| `0x000` | CTRL | `[0]`start, `[1]`ping_pong, `[2]`pool_en, `[3]`eltwise_en, `[4]`clear_done, `[5]`relu_en, `[6]`out_ping, `[7]`gemm_en |
| `0x004` | STATUS | `[0]`done_irq, `[1]`busy, `[2]`dma_rd_err, `[3]`dma_wr_err (RO) |
| `0x008–0x01C` | SRAM addrs | Act/Wgt/Out SRAM base addresses (ping/pong) |
| `0x020–0x02C` | Dimensions | IN_W, IN_H, IC, OC |
| `0x030–0x034` | Kernel/Stride | `[15:8]=KH/SX, [7:0]=KW/SY` |
| `0x040–0x07C` | BIAS | 16× per-OC 32-bit bias values |
| `0x080–0x0BC` | SCALE_MUL | 16× per-OC 32-bit scale multipliers |
| `0x0C0–0x0FC` | SCALE_SHIFT | 16× per-OC 6-bit shift amounts |
| `0x120–0x148` | DMA | Read/write triggers, DDR addr, length, SRAM base, status, sel |

### Interrupt Handling

NPU `irq_done` → latched in `axi_sys.v` → PicoRV32 IRQ bit 3 → `irq.c` sets `npu_irq_flag`.
Firmware polls `npu_irq_flag` (RAM flag, fast) rather than reading MMIO `NPU_STATUS`.

## Architecture Decisions (explicit, do not change without understanding)

### D: Multi-OC-tile — firmware-driven 16-OC-per-pass

Hardware `param_regfile.v` has exactly 16 bias/scale/shift registers (indices 0..15). One NPU start always processes exactly 16 output channels. For layers with OC > 16, firmware calls `npu_conv_pass()` in a loop, one pass per 16-OC tile, each reconfiguring bias/scale registers and triggering a separate NPU start. This is the intended architecture — do **not** try to issue a single start for OC=32/64.

### E: CPU-side padding — intentional, SRAM residency not feasible

Activation-padding (`pad_activation`) runs on the CPU, reading/writing DDR. This is a deliberate architectural choice:
- CPU cannot access the NPU's internal SRAMs (Act/Wgt/Out) — only the NPU DMA engine can.
- Every inter-layer transfer therefore requires Out SRAM → DDR (DMA write), then CPU reads DDR, then DDR → Act SRAM (DMA read).
- "SRAM residency" (skipping the DDR round-trip) would require either adding a CPU-accessible SRAM port or removing CPU-side padding — not in scope.
- **Pooling is NPU-fused** (not CPU-side): Conv2/Conv4/Conv6 set `pool_en=1` so the 2×2 max-pool happens in the same NPU pass. (`cpu_max_pool_2x2` exists but is unused.)

### F: General GEMM / fully-connected mode (`gemm_en`, CTRL[7])

Fully-connected layers run on the systolic array as a degenerate 1×1 conv. `gemm_en` bypasses the im2col line buffer: the FSM feeds the input vector's IC-tile word straight from Act SRAM, **replicated to all 16 array rows** (identical to how im2col replicates a conv window), so every PE in a column computes the same dot product and the existing drain/post-process/write path is reused unchanged. The reduction streams `ceil(IC/16)` IC-tiles as k-steps; OC is firmware-tiled by 16 (decision D); `out_w=out_h=1`. `wgt_reader`'s kernel-offset count is the runtime `kh*kw` (9 for conv, 1 for GEMM).
- **General-purpose:** any `IC ≤ 1024` (HW `ic_group` counter width) and any `OC` (16-tiled), configured via registers — not hardcoded to the MNIST model.
- **Bank convention:** GEMM runs with `ping_pong=1` → input vector in Act **PONG**, FC weights resident in Wgt **PONG** (packed by `pack_fc_tile`, layout `word(o,g)=o*icg+g`); conv keeps PING. Output to Out PING (`CTRL[6]=0`).
- **Firmware:** `npu_gemm_pass(in_dim,out_dim,scale,bias,relu,in_ddr,out_ddr,wgt_base)`.
- **Limitation:** the NPU post-process clamps output to INT8, which saturates final-classifier logits → Affine2 (50→10) stays on CPU. Intermediate FC layers with INT8 activations (Affine1) run on the NPU. This moved Affine1 from ~22.3M to ~0.6M cycles; full 10-image run 40.87M → 22.62M.

### G: Weight-prefetch reuse — per-OC-tile, general (`ICG_BUF`)

Conv weights are position-invariant, but the FSM used to re-prefetch an OC-tile's
weights from Wgt SRAM **once per output pixel** (≈144 SRAM reads vs ~9 compute
cycles — the array wants 16 OC-words/cycle, the single-port SRAM gives 1/cycle).
Fix: prefetch each OC-tile's weights **once** and reuse across the whole spatial
sweep. `wgt_reader`'s buffer is `wgt_buf[ko][oc][ICG_BUF]` (`ICG_BUF=4`); the FSM
`reuse_mode = !gemm_en && ic_groups ≤ ICG_BUF` prefetches all IC-groups once
(`o_prefetch_all`), then loops IC-tiles in CALC selecting from the buffer
(`o_wgt_ic_sel`) with no re-prefetch; `wgt_loaded` tracks residency, cleared on
layer start / OC-tile change.
- **General, not model-specific:** any layer with `ic_groups ≤ ICG_BUF` benefits
  (all current conv layers do); `ic_groups > ICG_BUF` and GEMM fall back to the
  per-IC-tile prefetch — correct, just unaccelerated. `ICG_BUF` is a hardware
  capacity knob (like the 16×16 array / 16-OC tiling), not a model dimension.
- Result: conv (NPU) 8.64M → 2.15M cycles (all 6 layers −64..83%); full 10-image
  run **22.62M → 16.14M**, still bit-identical 10/10. Output is byte-for-byte
  unchanged (same weights, only their timing/source moves).
- Next bottleneck is now CPU `pad_activation` (~6.4M).

### IRQ map (PicoRV32 `irq[]` bits)

| Bit | Source | Set by | Cleared by |
|-----|--------|--------|------------|
| 3 | NPU compute done | `axi_sys.v` latch on `npu_irq_done` | firmware writes `NPU_CTRL_CLEAR_DONE` in handler |
| 4 | DMA read done | `axi_sys.v` latch on rising edge of `npu_dma_rd_done` | firmware reads `NPU_DMA_STATUS` in handler (ack), or writes `NPU_DMA_RD_TRIG` |
| 5 | DMA write done | `axi_sys.v` latch on rising edge of `npu_dma_wr_done` | firmware reads `NPU_DMA_STATUS` in handler (ack), or writes `NPU_DMA_WR_TRIG` |

**Critical:** all three IRQ lines are level-sensitive latches. The handler MUST drop the line before `retirq` or it re-fires forever. **The IRQ mask is set in `start7.S`** via `picorv32_maskirq_insn` — a `1` bit means *disabled*. It currently enables bits 3/4/5 (`~((1<<3)|(1<<4)|(1<<5))`); adding a new IRQ source requires un-masking its bit there, otherwise the latch sets but the CPU never responds.

### Out SRAM bank decoupling (Issue C)

`param_regfile.v` register `0x14C`:
- bit 0: `dma_act_ping_sel` — which Act SRAM bank DMA writes to
- bit 1: `dma_wgt_ping_sel` — which Wgt SRAM bank DMA writes to
- bit 2: `dma_out_ping_sel` — which Out SRAM bank DMA **reads from** (independent of NPU write bank `cfg_ping_pong_sel`)

Decoupling bit 2 from `cfg_ping_pong_sel` allows DMA to drain a completed Out SRAM bank while NPU simultaneously writes to the other bank, enabling future ping-pong overlap optimizations.

### Debug prints

Verbose per-layer dumps (max-abs, channel spot-checks) are guarded under `#ifdef DEBUG_VERBOSE`. Compile with `-DDEBUG_VERBOSE` to re-enable. Error/timeout messages (DMA rd/wr timeout, NPU IRQ timeout, DMA error) are always active.

## Conventions

- Comments in RTL and firmware are bilingual (Chinese/English); keep matching surrounding style.
- Register-map changes require touching both sides: `rtl/param_regfile.v` and `firmware/firmware.h`.
- New RTL files must be added to `axi_sys.f`.
- Firmware C code must be warning-clean under strict CFLAGS.
- Multiple test programs exist (`deepnet_run.c`, `deepnet_deploy.c`), each defining `usercode7()`. Only one may be linked at a time; switch by editing `FW_C_SRCS` in `run_all.sh`.
- `makehex.py` pads firmware binary to 524288 words for `$readmemh`.
- DeepConvNet model assets: `deepnet.h` (dimensions), `deepnet_weights.h` (weights/biases), `mnist_test_images.h` (test images, generated by `extract_images.py`).
