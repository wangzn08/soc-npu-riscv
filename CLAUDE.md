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
  └─ 0x4000_0000–...         → shared memory ("DDR", 128-bit AXI4 data path):
        axi_lite_to_axi_full bridge (32b) → axi_upsizer_32_128 ┐
                                                               ├→ axi_arbiter_2to1 (128b) → axi_full_slave_v1_0_S00_AXI (128b)
        npu_top native 128-bit DMA master ─────────────────────┘
```

### 128-bit AXI shared-memory path (AXI utilization)

The shared-memory data path is **128-bit end to end** so the NPU's native 128-bit
DMA drives the bus directly — the old 128→32 width converter inside
`npu_axi_wrapper` is removed (NPU writes/reads full beats, no 4× down-conversion,
so bus utilization is no longer the bottleneck). The low-traffic 32-bit CPU reaches
the 128-bit fabric through `axi_upsizer_32_128` (single-beat 32→128 adapter:
partial writes via WSTRB on the lane selected by `addr[3:2]`; reads mux that lane
back out). `axi_arbiter_2to1` and `axi_full_slave_v1_0_S00_AXI` are parameterized
to `DATA_WIDTH=128` (wstrb = `DATA_WIDTH/8`); `npu_axi_wrapper`'s `SOC_AXI_DATA_W`
is 128. NPU DMA writes tie WSTRB all-ones (full-beat). Directed testbenches:
`tests/tb_axi_upsizer.v`, `tests/tb_axi_read_backpressure.v`.

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
Reorder: spatial-first-per-tile → channels-first                      [NPU HW transpose, decision L]
Affine1: 1024 → 50, ReLU                                             [NPU GEMM]
Affine2: 50 → 10, raw INT32 scores                                   [CPU]
Argmax: find predicted digit                                          [CPU]
```

**NPU operations:** 6 conv layers (all 3×3 INT8 with ReLU) via `npu_conv_pass()`, plus **Affine1 (1024→50) via `npu_gemm_pass()`** (GEMM/FC mode, decision F). Conv weights are preloaded into the Wgt SRAM **PING** bank once (`preload_conv_weights`); FC weights into the **PONG** bank once (`preload_fc_weights`); both stay resident across all images. Each NPU op configures registers, starts the NPU, waits for IRQ (`npu_irq_flag`), then DMA-reads results back to DDR.

**CPU operations:** Conv1 image formatting (image → tile-major DDR words), data reorder, **Affine2 (50→10, raw INT32 logits)**, argmax. All in `deepnet_deploy.c`. (Max-pool is fused into the NPU conv pass via `pool_en`; **activation padding is now done in hardware** — decision H — so `pad_activation` and `cpu_max_pool_2x2` are both unused.)

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
| `0x000` | CTRL | `[0]`start, `[1]`ping_pong, `[2]`pool_en, `[3]`eltwise_en, `[4]`clear_done, `[5]`relu_en, `[6]`out_ping, `[7]`gemm_en, `[8]`hw_pad, `[9]`row_par_en, `[10]`gemm_reduce, `[11]`row_block_en, `[12]`oc_single |
| `0x150` | PAD | `[15:8]`pad_h, `[7:0]`pad_w (hardware padding border, with CTRL[8]) |
| `0x004` | STATUS | `[0]`done_irq, `[1]`busy, `[2]`dma_rd_err, `[3]`dma_wr_err (RO) |
| `0x008–0x01C` | SRAM addrs | Act/Wgt/Out SRAM base addresses (ping/pong) |
| `0x020–0x02C` | Dimensions | IN_W, IN_H, IC, OC |
| `0x030–0x034` | Kernel/Stride | `[15:8]=KH/SX, [7:0]=KW/SY` |
| `0x040–0x07C` | BIAS | per-OC 32-bit bias, channels 0..15 (decision O: ch 16..63 at `0x160–0x21C`) |
| `0x080–0x0BC` | SCALE_MUL | per-OC 32-bit scale mul, ch 0..15 (ch 16..63 at `0x220–0x2DC`) |
| `0x0C0–0x0FC` | SCALE_SHIFT | per-OC 6-bit shift, ch 0..15 (ch 16..63 at `0x2E0–0x39C`) |
| `0x120–0x148` | DMA | Read/write triggers, DDR addr, length, SRAM base, status, sel |
| `0x140` | DMA_STATUS | `[0]`rd_done `[1]`wr_done `[2]`copy_done (RO) |
| `0x154` | COPY_TRIG | write → start on-chip Out→Act copy (`sram_copy`, decision J); src/dst/len reuse DMA RD_SRAM_BASE/WR_SRAM_BASE/RD_LEN |
| `0x15C` | TRANSPOSE_TRIG | write → start Conv6→FC1 transpose (`transpose_engine`, decision L); done = DMA_STATUS[4]; src/dst/n_pos reuse RD_SRAM_BASE/WR_SRAM_BASE/RD_LEN |

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

### H: Hardware padding (`hw_pad`, CTRL[8] + `NPU_PAD`)

CPU `pad_activation` used to transpose (tile-major→position-major) AND zero-pad the
previous layer's output in DDR (~6.4M cycles, memory-bound). Replaced by hardware
padding: the FSM reads the previous output **tile-major directly from Act SRAM**
and **injects border zeros** so im2col sees the same padded stream — no CPU pad,
no transpose, no PAD_BUF round-trip. `NPU_IN_W/H` stay the padded dims (geometry
unchanged); `NPU_PAD = {pad_h, pad_w}` (0x150) gives the per-side zero border;
`CTRL[8] hw_pad` enables it. The border flag is delayed one cycle in `npu_top`
(`fsm_border_d`) to match the SRAM read, muxing a zero into `im2col.i_pixel_data`.
im2col/systolic/post are **unchanged**.
- **General:** any pad (Conv4 uses pad=2, others pad=1), any dims/tiles — driven by
  `NPU_PAD`. Firmware DMAs the previous layer's unpadded output straight in.
- Result: `pad` 6.4M → ~1.0M (residual = Conv1 image-to-DDR formatting); inference
  10.1M → 4.6M; full run **16.14M → 10.91M**, bit-identical 10/10. `pad_activation`
  is removed (unused). When task E rewrites the im2col front-end, padding can
  migrate fully into im2col.
- Next bottleneck: the ~6M of sim UART prints / one-time preload (outside the
  per-image inference path).

### I: 16-row spatial parallelism (`row_par_en`, CTRL[9])

The 16×16 array ran conv at **1/16 utilization**: im2col replicated **one** window
to all 16 rows, so 15 rows computed the same thing (256 useful MAC/cycle of 4096).
`row_par_en` makes the 16 rows compute **16 different output pixels** (a horizontal
group of ≤16 adjacent output columns): one K_END now produces 16 pixels × 16 OC.
Primary goal is **not wasting the silicon**; cycles are a secondary win.
- **im2col (`im2col_line_buffer.v`):** in row-par mode `o_act_window[r]` is a
  **16-wide combinational slice** of the line buffer — array row `r` reads input
  column `group_base + off_col + r` from the bank selected by `bank_for_offrow`
  (mirrors the legacy `win[]`-fill mapping). Legacy replicate path kept under the
  `i_row_par_en` mux (byte-identical when off). No new storage (fan-out of `lb_bankX`).
- **FSM (`top_controller_fsm.v`):** the sweep advances `cur_ox` by
  `group_size = min(16, out_w - cur_ox)` per step (=1 when row-par off ⇒ identical).
- **Drain is REVERSE:** `pe_core` drains row 15 first → row 0 last, so drain valid
  `k` (0..15) = output column `group_base + 15 - k`.
- **Non-pool write (`npu_top.v`):** a sequencer **armed at DRAIN start** (not S_POST —
  the post-process pipeline emits most valids during S_DRAIN) counts `pp_feat_vld`
  and writes each to column `group_base + 15 - rp_vld_cnt`, guarded to
  `[group_base, group_base+group_size)`; `fsm_pp_done = rp_done` holds S_POST until
  all 16 captured.
- **Pool path (`post_process_top.v`):** the drain's reverse order breaks the pooler's
  row-major 2×2 windowing, so a **16-deep reorder buffer** captures the drained
  pixels by column-within-group (`rp_buf[15-k]`) then **replays `group_size`
  row-major** into `max_pooling_2x2` (group boundaries fall on even columns, so no
  2×2 pair is split). `fsm_pp_done = o_rp_pool_done` holds S_POST until replay done.
- **General:** `group_size` is runtime from `out_w` (28→16+12, 16, 14, 8 all handled);
  partial groups use ≤16 rows. Like the 16×16 array and 16-OC tiling, "16 pixels/group"
  is a hardware width, not a model dimension. `systolic/gp/pe/wgt_reader/dma/sram`
  unchanged (weights are shared across the 16 pixels).
- Result: full run **10.91M → 10.05M**, bit-identical 10/10 (per-layer spot checks
  byte-match: Conv1(7,10), Conv3(7,7), Pool1/Pool2 nz). Brought up incrementally
  behind the mode bit: Conv1 → Conv3/5 (non-pool) → Conv2/4/6 (pool). PAD migration
  into im2col (decision H follow-on) remains future work — currently row-par still
  consumes the FSM's border-zero-injected stream.

### J: On-chip SRAM residency for conv→conv (`sram_copy`, 0x154)

Every conv layer used to round-trip its result through DDR: layer N drains Out SRAM
→ DDR, then layer N+1 loads DDR → Act SRAM. After hardware padding (decision H) the
CPU does nothing between conv layers, so this DDR bounce was pure transport overhead.
A small `sram_copy` engine (`rtl/sram_copy.v`) copies Out SRAM → Act SRAM **on-chip**,
eliminating the round-trip.
- **Engine:** time-shares the SRAM **Port B** paths (Out Port B read → Act Port B
  write), isolated from `axi_dma` (DDR path untouched). Out Port B read is
  **combinational** (`out_sram_wrapper COMB_B=1`), so it reads `Out[src+cnt]` and
  writes `Act[dst+cnt]` in the **same cycle** (one word/cycle, no pipeline — a 1-cycle
  read-latency version copies off-by-one). `copy_busy` gives the engine Port-B priority
  over the DMA muxes in `npu_top`.
- **Registers:** trigger `NPU_DMA_COPY_TRIG` (0x154) → regfile `o_copy_trig`; completion
  polled via `NPU_DMA_STATUS[2]` (`copy_done`) — **no IRQ** (avoids start7.S mask
  plumbing). src/dst/len **reuse** the DMA `RD_SRAM_BASE`/`WR_SRAM_BASE`/`RD_LEN`
  registers; banks reuse `dma_out_ping_sel`/`dma_act_ping_sel`. `i_len` = full word count.
- **Two Act regions (firmware), NOT Act ping-pong:** the global `ping_pong_sel` couples
  the Act and Wgt read banks (conv weights are resident in Wgt PING), so it can't be
  flipped to alternate Act banks. Instead two **address regions** in the PING bank
  ping-pong via `NPU_ACT_ADDR_A`: R0=word 0, R1=word 1024 (`ACT_RES_B`). A conv reads
  region A while its copy writes region B; the next conv reads B. Different addresses ⇒
  no input corruption even though the copy is serial.
- **Serial per-pass copy:** resident layers copy each OC-pass's output right after that
  pass (NPU idle), `act_dst + pass*out_words`, reading that pass's Out bank. This frees
  the per-pass Out bank before the next pass reuses it (works for >2-pass layers,
  Conv5/6) and sidesteps any same-bank dual-port concern. The DDR path keeps the
  OC-pass overlap; resident layers don't use it (the copy is on the critical path but
  far cheaper than the DDR round-trip).
- **Firmware:** `dma_out_to_act(act_dst_word, out_src_word, nwords, out_bank)`;
  `npu_conv_pass(... , act_in, act_dst)` (`act_dst>=0` ⇒ resident copy; `<0` ⇒ DDR).
- **Scope:** the 5 conv→conv boundaries (Conv1→…→Conv6). Conv6→reorder→FC stays DDR
  (CPU channels-first reorder + FC2/argmax need DDR). Conv1's input (the image) loads
  from DDR.
- **General:** the copy is a layout-agnostic word copy (Out tile-major == next-layer Act
  tile-major); works for any dims/IC/OC/tiles, pooled or not. With `act_dst<0` the
  behavior is byte-identical to the DDR path.
- Result: per-image inference **374K → 289K (−23%)**; full run **9.86M → 8.83M**
  (−1.03M), bit-identical 10/10 (Pool1 nz 711/379/784/838, Pool2 618/458/673 byte-match).
  `prof_npu` dropped (the output-DDR-drain it included is gone) and `prof_load` dropped
  (input DDR reads gone, copy folded in). Next bottleneck: `pad` (Conv1 image→DDR
  formatting, ~101K/image).

### L: Hardware reorder — Conv6 → FC1 transpose (`transpose_engine`, 0x15C)

The CPU used to transpose Pool3's output from tile-major (pass×pos×ch) to the
channel-major layout FC1 (Affine1 GEMM) expects (~43K cycles/image, ~20% of
inference) via 3 DDR round-trips (Conv6 Out→DDR, CPU DDR→DDR transpose, GEMM
DDR→Act). Replaced by an on-chip **transpose engine** (`rtl/transpose_engine.v`,
decision-J-style, time-shares SRAM Port B): Conv6 transposes each OC-pass's Out
SRAM output **directly into Act PONG** (the GEMM input bank), channel-major, on
chip — no CPU reorder, no DDR round-trip.
- **Engine:** per OC-pass, a register transpose buffer `M[16*MAX_NPOS]` —
  **LOAD** reads `n_pos` Out words (word p = 16 channels at position p) and
  scatters byte ch_in to `M[ch_in*n_pos + p]` (stride n_pos); **DRAIN** reads M
  sequentially, packs 16 bytes/word, writes `n_pos` Act words. = a 16×n_pos byte
  transpose. `MAX_NPOS` is a capacity knob (like ICG_BUF); `n_pos ≤ MAX_NPOS`
  in one pass (multi-Act-word channels, n_pos>16, handled by the sequential
  drain). `n_pos > MAX_NPOS` → CPU fallback.
- **Control:** trigger `NPU_DMA_TRANSPOSE_TRIG` (0x15C) → `o_transpose_trig`;
  polled via `NPU_DMA_STATUS[4]` (`transpose_done`) — no IRQ. src/dst/n_pos reuse
  `RD_SRAM_BASE`/`WR_SRAM_BASE`/`RD_LEN`; Out read bank = `dma_out_ping_sel`, Act
  write bank = `dma_act_ping_sel` (= PONG). **Serial per-pass** (decision-J
  pattern): Conv6's alternating Out banks (`pass&1`) don't coexist, so transpose
  each pass right after it computes, before the next overwrites the bank.
- **Firmware:** `dma_out_transpose_to_act(act_dst, out_src, n_pos, out_bank)`;
  `npu_conv_pass(... , transpose_npos)` (>0 ⇒ per-pass transpose to Act PONG);
  `npu_gemm_pass(... , in_resident)` (1 ⇒ input already in Act PONG, skip DMA).
  The CPU reorder loop is removed.
- **General:** any conv→FC boundary; `n_pos`/`n_ch` runtime. (NPU_GEMM_PARITY's
  CPU oracle is not wired in this build since Conv6 no longer drains to DDR.)
- Result: per-image inference **195,823 → 150,471 (−23%)**; full run
  **8,041,665 → 7,618,571 (−423K, −5.3%)**, bit-identical (SCORE_CHK match) 10/10.
  `reorder` 429,680 → 7,040 (just the per-pass trigger MMIO); `load` 75,110 →
  50,670 (GEMM-input DDR round-trip gone).

### M: GEMM array utilization — 16-row IC-reduction (`gemm_reduce`, CTRL[10])

Legacy GEMM (decision F) replicated the input IC-tile to all 16 array rows, so
15/16 rows were redundant — **array utilization 6.25%**. `gemm_reduce` makes the
16 rows compute **16 distinct IC-tiles** and reduce them **down each column**
(spatial reduction), so one super-step consumes 256 IC (16 rows × 16) and the
array runs at **~100% utilization**. Primary goal (like decision I) is **not
wasting the silicon**, not cycles.
- **pe_core (`i_reduce`):** during drain, `o_psum_casc = psum_shift + i_psum_casc`
  forms a combinational adder chain down the column; the bottom PE's output =
  sum of all 16 rows' latched accumulators. Legacy shift-drain unchanged when
  `i_reduce=0` (byte-identical).
- **Weights (16×16 plane):** PE(r,c) needs OC-c's weights for IC-tile r.
  `wgt_reader` prefetches a **256-word plane** per super-step into `gemm_plane`
  (`plane[r][c]=Wgt[wgt_base + c*icg + s*16 + r]`, reusing `pack_fc_tile`'s
  `word(o,g)=o*icg+g` layout — no repack); `systolic_16x16`/`gp_4x4` slice it
  per-PE and select it under `i_reduce`. New wide `i_wgt_plane` bus; legacy
  `i_wgt` path untouched.
- **Activations:** `npu_top` `act_row[0:15]` register holds 16 distinct IC-tile
  words, loaded from Act SRAM by the FSM super-step sequencer.
- **FSM (`top_controller_fsm`):** GEMM-reduce states `S_RDC_WLOAD`(plane prefetch)
  → `S_RDC_ALOAD`(16 act loads) → `S_RDC_VLD`(1 accumulate); `ceil(IC/256)` super-
  steps (FC1: 4 vs the legacy 64 k-steps), then `S_K_END` → `S_DRAIN`. The reduce
  drain still runs **16 cycles** (combinational sum stable while `psum_shift`
  holds) to match the legacy GEMM post-process pipeline-fill timing — a single
  drain valid mis-times the S_POST write (`pp_start` feeds zeros once `drain_en`
  drops). Output path reuses the legacy non-pool/non-rowpar write (1 word/pass).
- **General:** any GEMM/FC layer (`IC≤1024`, OC 16-tiled), runtime dims; FC2 stays
  CPU (INT8 saturation, decision F). Conv/legacy-GEMM byte-identical (mode-gated).
- **Firmware:** `npu_gemm_pass(..., reduce)`; FC1 passes `reduce=1`.
- Result: array util **6.25% → ~100%**, bit-identical `SCORE_CHK=D30179DF` 10/10.
  Cycle win is **marginal** — FC1 19,638 → 18,966/img (−3.4%); full run
  7,687,157 → 7,680,437 (−6,720, −0.09%). FC1 is **weight-bandwidth-bound**
  (single-port Wgt SRAM): the 256-word/super-step plane prefetch reads the same
  1024 words/pass as the legacy 64-k-step path, so the single-port SRAM floor —
  not compute — dominates. The redesign fixes utilization, not the bottleneck.
  (Consistent with Phase 0: FC1 was ~5× the weight floor, overhead/bandwidth-
  bound, not compute-bound.)

### N: Row-block packing — fill idle array rows for narrow layers (`row_block_en`, CTRL[11])

Row-parallel (decision I) makes the 16 rows compute 16 columns of ONE output row;
when `out_w < 16` the rest idle (Conv5/6 out_w=8 → **50% utilization**). `row_block_en`
packs **R output rows** into the array (R = ⌊16/group_size⌋, capped MAX_R=2):
array row `gi` → block `b=gi/group_size` (output row `cur_oy+b`), col `c=gi%group_size`.
Conv5/6 (group_size=8) → R=2 → 100% utilization. Like decisions I/M, the goal is
**not wasting the silicon**; cycles are a secondary (here large) win.
- **im2col (`im2col_line_buffer.v`):** a **4th line-buffer bank** (R+2 rows), mod-4
  `row_sel` rotation, and a `(block,col)` window read — block b's 3-row window is
  the base window slid down b input rows: `bank = (row_sel + b + off_row + 1) mod 4`.
  All gated by `i_row_block_en`; **byte-identical when off** (R=1 = decision I).
  Exact only when `2*group_size==16` (group_size==8), so R=2 engages only there.
- **FSM (`top_controller_fsm.v`):** loads R+2 rows/block (`lr_target = cur_oy+R+kh-2`),
  sweeps `cur_oy += R`. `rows_per_grp` (=2 iff `row_block_en && group_size==8`).
- **Drain/Out-SRAM (`npu_top.v` rp sequencer):** drained array row r → `(b,c)`,
  writes `out_base + oc_off + (cur_oy+b)*out_row_stride + group_base + c`.
- **Pool path (`post_process_top.v`):** R=2 **aligns with the 2×2 pool row-pair**
  (oy even). `rp_buf[r]` is already row-major (`b*group_size+c = r`), so the pool
  replay just streams `R*group_size` pixels (both rows) into `max_pooling_2x2`
  in one drain; the pooler pairs rows oy/oy+1. Contiguous `pool_out_addr_cnt`
  still yields row-major 4×4.
- **Firmware:** auto-engages on `row_par && out_w==8` (Conv5 non-pool + Conv6 pool).
- **General:** any narrow layer with out_w==8; runtime `group_size`/`R`. (out_w in
  9..15 stays R=1 — packing needs 2*group_size≤16; out of scope, see spec.)
- Result: array util **50% → ~100%** (Conv5/6); full run **7,680,437 → 7,427,997
  (−252,440, −3.3%)**, bit-identical `SCORE_CHK=D30179DF` 10/10. (The Phase-0
  `prof_busy_layer` IRQ-wait proxy under-estimated this at ~0.2%; the clean
  row_block on/off full-run A/B is the ground truth — measure, don't extrapolate.)

### O: One-start-all-OC — OC-inner loop with all-OC-resident weights (`oc_single`, CTRL[12])

For layers with OC > 16 the legacy path (decision D) issues one NPU start per
16-OC tile, each **re-sweeping the spatial output and reloading the im2col line
buffer** — an O(OC/16) redundancy that grows with output-channel count.
`oc_single` computes **all OC tiles in ONE start**: per spatial group the im2col
window is loaded once and **reused across every OC tile** (OC-inner loop), so the
per-OC im2col reload is removed. The primary goal is **generality** — removing an
O(OC) redundancy that bigger models pay proportionally more — not MNIST cycles.
- **Weights (decision G interaction):** OC-inner reuse would re-prefetch weights
  per group unless all OC tiles' weights are resident. So `wgt_reader` prefetches
  **all OC tiles** once into `wgt_buf[ko][MAX_OC_RESIDENT=64][ICG_BUF]` (outer
  `pf_oct` tile loop); during CALC `i_oc_tile_sel` selects the active tile's 16
  channels. Weight-read count is unchanged vs decision G.
- **Regfile:** bias/scale/shift expanded to 64 entries (ch 0..15 legacy blocks,
  ch 16..63 at `0x160/0x220/0x2E0`); `i_oc_tile_sel` (from FSM `oc_t`) presents
  the active tile's 16-window. `i_oc_tile_sel==0` ⇒ legacy low-16 ⇒ byte-identical.
- **FSM (`top_controller_fsm`):** after S_POST, if more OC tiles remain it advances
  `oc_t` and re-enters S_PREFETCH_WGT (reuse settle path — weights already resident)
  to re-CALC against the **same frozen im2col window** with the next tile's weights;
  no spatial/window advance, no re-prefetch. `active_oc_idx` (= `oc_t` in oc_single,
  else `oc_tile[9:4]`) drives the Out-SRAM write base and OC-tile selects. Spatial
  exhaustion ⇒ S_DONE (one IRQ). oc_single off ⇒ `oc_t=0`, byte-identical.
- **Firmware:** `npu_conv_pass(..., oc_single)` — one start writes all tiles'
  bias/scale/shift, `NPU_OC`=full OC, CTRL[12]=1, one IRQ, then a single tile-major
  copy/transpose/DDR-drain of `out_words*oc_passes`.
- **Scope:** initially non-pool conv (**Conv3** 2 tiles, **Conv5** 4 tiles). Pooled
  layers (Conv4/Conv6) were extended in **decision P** (per-OC-tile pooler state);
  all four multi-tile conv layers are now oc_single.
- **General:** any non-pool conv with OC ≤ `MAX_OC_RESIDENT` (64); `OC > 64`
  falls back to multi-start (decision-D path), like decision G's ICG gate.
- Result (bit-identical `SCORE_CHK=D30179DF`, 10/10): full run **7,427,997 →
  7,376,666 (-51,331, -0.69%)**. Per-layer A/B confirms the O(OC) thesis — the
  4-tile Conv5 wins big (**-76,600**, im2col reused across 3 extra tiles/group)
  while the 2-tile Conv3 is the floor case (**+25,269**: the narrow-layer im2col
  saving — FSM busy -3,160, load -2,560 — is outweighed by per-start CPU/MMIO
  overhead). MNIST is the small-model floor; the win scales with OC for bigger
  models. (Consistent with [[soc-npu-spec-estimate-caution]]: measure the full-run
  A/B, the per-phase proxy misleads.)

### P: Pooled oc_single — per-OC-tile pooler state (decision O follow-on)

Decision O's `oc_single` was limited to **non-pool** conv because the 2×2 pooler
(`max_pooling_2x2`) holds cross-ROW state (previous-row line buffer + column/row
phase + neighbour regs) that the OC-inner loop's interleaved tile drains corrupt.
Decision P makes the pooler state **per-OC-tile** so pooled layers (Conv4, Conv6)
also run oc_single. Chosen over the row-pair-loop-nesting alternative because it's
**general for any out_w** (no need to pack 2 rows into the 16-wide array — infeasible
for out_w=16) and needs **no FSM loop change** — only localized pooler/Out-write edits.
- **`max_pooling_2x2`:** `line_buf`/`col`/`row_odd`/`cur_left`/`above_left` →
  `[NUM_TILES=4]`, indexed by new `i_tile`; `i_start` clears all tiles. `i_tile==0`
  (oc_single off) touches only tile 0 ⇒ byte-identical.
- **Out-write counter:** the OC-inner loop revisits each tile once per group-row
  (interleaved), so the within-tile pooled counter is **per-tile** (`pool_out_cnt[tile]`,
  reset only on op start — not on tile switch) plus a tile-major base
  `tile*pool_tile_words`. (A single counter / reset-on-switch overwrote a tile's
  later pool-rows — the Conv4 2/10 bug.)
- **Tile-id alignment:** the pooler emits a **registered** `o_pool_tile` aligned with
  `o_pool_vld`, used for the write tile/base — the FSM `oc_t` may already have advanced
  at the last pooled pixel of a drain (the Conv4 9/10 bug). Routed
  `pooler → post_process_top → npu_top`.
- **Timing safety:** `fsm_pp_done = rp_pool_done` holds S_POST until a tile's replay
  completes and `oc_t` only advances at S_POST exit, so `i_tile` is stable for the
  whole of a tile's drain/replay; the registered `o_pool_tile` covers the final pixel.
- **General:** any pooled conv, any out_w; composes with row-par (I), row-block (N,
  R=2 replay), and the per-tile HW transpose (L). `NUM_TILES`/`POOL_NTILES` is a
  capacity knob (= MAX_OC_RESIDENT/16).
- Result (bit-identical `SCORE_CHK=D30179DF`, 10/10): **Conv4** (2-tile pool)
  7,376,666 → 7,343,116 (**-33,550**); **Conv6** (4-tile pool+transpose+row-block)
  7,343,116 → 7,266,176 (**-76,940**, mirrors the 4-tile Conv5 non-pool -76,600).
  All four multi-tile conv layers (Conv3/4/5/6) now oc_single. **Full run vs the
  pre-decision-O baseline: 7,427,997 → 7,266,176 (-161,821, -2.18%)** (O -51,331 +
  P -110,490). Per-image inference 123,492 → 112,042 (npu phase 594,680 → 480,180);
  ≈ **0.56 ms/image @200MHz**.

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
