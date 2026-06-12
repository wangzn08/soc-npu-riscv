# NPU General GEMM / Fully-Connected Mode — Design

## Context

Cycle profiling of the 10-image DeepConvNet run (added under `NPU_PROFILE`)
showed the CPU fully-connected (affine) layers dominate runtime: **42.5M of
58.6M inference cycles (72.6%)**. Two quick fixes were applied and verified
(10/10 preserved):

- RTL `ENABLE_FAST_MUL=1` ([rtl/axi_sys.v:55](../../../rtl/axi_sys.v)): slow
  serial multiplier → fast multiplier. 62.67M → 45.02M total cycles.
- Firmware: affine operands cached in private RAM instead of DDR. 45.02M →
  40.87M total. Affine itself dropped 42.5M → 22.3M.

After those, affine is **still 58% of inference** (~22.3M). The remaining cost
is the matrix-vector multiply running on a multi-cycle scalar CPU. The systolic
array is a 4096-MAC/cycle GEMM engine sitting idle during this phase.

This design adds a **general-purpose GEMM / fully-connected mode** to the NPU so
the systolic array performs fully-connected layers. Per project principle
([memory] soc-npu-general-purpose), this must be general for any IC/OC
configured via registers — not hardcoded to DeepConvNet's 1024→50 / 50→10.

## Goal & Non-Goals

**Goal:** A new NPU mode computing `out[oc] = Σ_ic act[ic]·W[oc][ic] (+bias, scale,
ReLU)` for arbitrary `IC` (≤65535) and `OC` (tiled by 16), driven through the
existing register interface, reusing the systolic array and post-process
pipeline.

**Non-goals (YAGNI):**
- Multi-sample batching across the 16 array rows (only one input vector per pass;
  15/16 rows idle — acceptable because GEMM is weight-bandwidth bound, not
  compute bound).
- Changing the conv datapath behavior (GEMM is purely additive, gated by a flag).

## Why this is feasible (datapath facts)

- `pe_core` ([rtl/pe_core.v](../../../rtl/pe_core.v)): each `PE[row][col]` is an
  **independent** accumulator `Σ_k act[row]·wgt[col]` across valid cycles, latched
  on `i_k_end`. The column cascade is used **only during drain** to shift the 16
  rows out serially — there is no cross-row sum during compute.
- `systolic_16x16`: row `r` ← activation group `r`; column `oc` ← weight group
  `oc` (16 INT8 = one IC-tile of one OC). So a single input vector mapped to one
  row, with 16 OC across the columns, computes 16 FC outputs in parallel,
  accumulating over IC-tiles streamed as k-steps. This is exactly a GEMM row.
- The only blockers are front-end: `im2col_line_buffer` is 3×3/spatial-shaped and
  caps IC at `ICG_MAX=4`; the FSM/im2col path is conv-specific. GEMM bypasses
  im2col entirely.

## Architecture (Approach B: dedicated GEMM mode, bypass im2col)

Reused unchanged: `systolic_16x16`, `post_process_top` (bias/scale/ReLU),
`out_sram`, `axi_dma`, the Act/Wgt/Out SRAM wrappers.

Modified/added:

### 1. Register: `gemm_en` (CTRL[7])
- `param_regfile` ([rtl/param_regfile.v](../../../rtl/param_regfile.v)): decode
  CTRL bit 7 → `o_gemm_en`.
- `firmware.h`: `#define NPU_CTRL_GEMM_EN (1<<7)`.
- Wired through `npu_top` → `top_controller_fsm`.

### 2. GEMM activation feed (mux on `systolic.i_act`)
- In `npu_top`, `systolic.i_act` becomes
  `gemm_en ? gemm_act_bus : im2col_act_window`.
- `gemm_act_bus` places the current IC-tile's 128-bit Act SRAM word in **one
  active row** (the row chosen to align with the captured drain phase — see FSM
  branch / risk notes) and zeros the other 15 rows. Zeroed rows accumulate 0
  (weights are column-broadcast, `0·w=0`) and are ignored at drain — clean and
  harmless.
- The Act SRAM Port-A address in GEMM mode is FSM-driven:
  `act_base + ic_tile` (input vector laid out as `ceil(IC/16)` consecutive
  128-bit words). No im2col, no spatial loop.

### 3. wgt_reader: runtime kernel-offset count
- Today `KERNEL_OFFSETS = KH*KW` is a **compile-time** parameter (always 9). Add a
  runtime input `i_kernel_offsets` (= `kh*kw`, =1 for GEMM) so the prefetch loops
  `pf_ko` over `0..i_kernel_offsets-1` and the address uses the runtime kernel
  size. This is a generality improvement (any kernel size) and makes GEMM read
  exactly `16 OC × ceil(IC/16)` weight groups with no 9× waste.
- GEMM weight SRAM layout (KH=KW=1) collapses the existing formula to:
  `addr = oc*ic_groups + ic_group` (oc-major, ic-group inner) — firmware packs FC
  weights this way (same packer shape as conv with KH=KW=1).

### 4. FSM GEMM branch
The existing sequence `S_PREFETCH_WGT → S_CALC_KERNEL → S_K_END → S_DRAIN →
S_POST → S_NEXT_TILE` is reused with `gemm_en` gating:
- Skip `S_LOAD_ROW`/`S_WAIT_WIN` (no im2col line load).
- Per OC-tile, loop IC-tiles (`ic_tile = 0..ic_groups-1`): prefetch that
  IC-group's 16-OC weights, drive Act SRAM addr, pulse `o_array_vld` for one cycle
  with `o_im2col_offset_sel=0`; assert `o_array_k_end` on the last IC-tile.
- `S_DRAIN`: shift out; enable the Out-SRAM write for **only** the drain phase
  carrying the active row (one 16-OC output word per OC-tile). `post_process_top`
  applies per-OC bias/scale/ReLU unchanged.
- `S_NEXT_TILE`: advance `oc_tile` by 16 until `oc_tiles_total`; no spatial
  advance. Output address = `oc_tile/16` (one word per OC-tile).
- Output geometry forced to 1×1 in GEMM mode (`out_w=out_h=1`,
  `out_spatial_size=1`) so existing address math degenerates correctly.

### 5. Firmware: general `npu_gemm_pass`
New helper in `deepnet_deploy.c` (general, dimension-parameterized):
```
npu_gemm_pass(in_dim, out_dim, scale_mul, biases, in_ddr_addr, out_ddr_addr,
              wgt_base)
```
- DMA input vector (`ceil(in_dim/16)` words) DDR→Act SRAM.
- For each OC-tile (16): set BIAS/SCALE/SHIFT regs, set `gemm_en`, dims
  (`IN_W=IN_H=1`, `IC=in_dim`, `OC=16`, `KH=KW=1`), start, wait IRQ, advance.
- DMA the `ceil(out_dim/16)` output words back to DDR.
- Weights: packed by a general FC weight packer (oc-major, 16-IC-group inner),
  loaded resident at a `*_WGT_BASE` if Wgt SRAM has room, else DMA per pass.
- FC1 and FC2 both call `npu_gemm_pass`; the generic argmax stays on CPU.

## Data flow (one OC-tile)

```
Act SRAM[ic_tile] ─(row0)─┐
                          ├─► systolic (accumulate over ic_tile k-steps)
Wgt SRAM ─wgt_reader──────┘        │ k_end on last ic_tile
                                   ▼
                                 drain → post_process(bias/scale/relu)
                                   ▼
                                 Out SRAM[oc_tile/16] (one 16-OC word)
```

## Correctness & edge cases

- **OC not multiple of 16** (e.g. 50): last OC-tile partial; firmware writes 16-OC
  words but only reads back the valid `out_dim` channels from DDR. Extra channels'
  bias/scale set to 0/harmless.
- **IC not multiple of 16**: input vector zero-padded to `ceil(IC/16)*16` in the
  Act buffer; zero activations contribute 0.
- **Signedness**: INT8 signed throughout (matches conv and CPU affine).
- **Quantization parity**: GEMM uses the same `post_process` bias→`(psum*scale)>>shift`
  →ReLU→clamp as conv; firmware passes the same `SCALE_AFFINE*`/`SCALE_SHIFT`
  used by the CPU path, so results must match the CPU affine bit-for-bit (the
  acceptance test).
- **ReLU**: `relu_en` per pass (FC1 ReLU on, FC2 off) — existing CTRL[5].

## Testing / Verification

1. **Functional parity:** keep CPU affine available behind a compile switch;
   compare NPU GEMM output vs CPU affine **per element** for all 10 images (must
   be identical). Then switch the deploy path to NPU GEMM.
2. **End-to-end:** `bash run_all.sh sim` → require `=== Result: 10/10 correct ===`
   and `ALL TESTS PASSED.`
3. **Cycle measurement:** re-run with `NPU_PROFILE=1`; confirm `affine` category
   collapses (~22.3M → <1M) and report new total vs the 40.87M baseline (expected
   ~19M).
4. RTL changed → `bash run_all.sh clean` before recompiling.

## Risk notes

- Highest-risk unit is the FSM GEMM branch + drain-phase output capture (which
  drain cycle carries the active row). Mitigate by feeding the input to the row
  that drains first and enabling Out-write for exactly that one phase.
- `wgt_reader` runtime-offset change must not regress conv (default
  `i_kernel_offsets = kh*kw = 9` reproduces current behavior).
