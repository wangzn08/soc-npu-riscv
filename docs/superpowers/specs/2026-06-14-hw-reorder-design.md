# Spec: #2 Hardware reorder (Conv6 → FC1 transpose engine)

## Context / problem
Between Conv6+Pool3 and Affine1 (FC1 GEMM), the CPU transposes Pool3's output from
tile-major (pass×pos×ch) to channel-major (ch×pos), the layout the FC1 GEMM
expects. This CPU byte-transpose costs **~43K cycles/image (~20% of inference,
prof_reorder 429,680/10)** and forces **3 DDR round-trips** (Conv6 Out→DDR, CPU
DDR→DDR transpose, GEMM DDR→Act SRAM).

Goal: a small on-chip **transpose engine** that reads Conv6's Out SRAM output
(tile-major), transposes to channel-major, and writes Act SRAM PONG (the GEMM
input bank) directly — eliminating the CPU reorder and all 3 DDR round-trips.
Mirrors decision J (on-chip Out→Act copy), adding a transpose.

## Layout facts
- Pool3 output (Conv6, OC=64 → 4 OC-passes, spatial = 4×4 = **n_pos=16**).
- Out SRAM per pass (16 channels): `n_pos` contiguous words, word p = the 16
  channels at position p (16 bytes). `OutWord(pass,p) = base + p`.
- FC1 GEMM input = flattened **channel-major** vector, element index
  `v = global_ch * n_pos + pos`, `global_ch = pass*16 + ch_in`. 16 elements/word.
  Per pass spans Act words `[pass*n_pos .. pass*n_pos + n_pos - 1]` (n_pos words).
- FC1 weights (`pack_fc_tile`) already match this channel-major order — unchanged.

## Engine: register-array transpose (decision-J-style, per-pass serial)
Conv6 uses alternating Out banks (`pass&1`) + per-pass drain, so the 4 passes
don't coexist in Out SRAM. Therefore transpose **per pass**, right after that
pass's compute (NPU idle), before the next pass overwrites the bank.

**Transpose buffer** `M`: `16 * MAX_NPOS` bytes (capacity knob `MAX_NPOS`, like
ICG_BUF). Default `MAX_NPOS = 16` (covers MNIST n_pos=16).

- **Load (scatter, stride n_pos):** read the pass's `n_pos` Out words; for word p,
  write its 16 bytes into `M[ch_in*n_pos + p] = OutWord[p][ch_in]` (ch_in 0..15).
  One Out word read/cycle (Port B), scattered to 16 stride-`n_pos` byte slots.
- **Drain (sequential):** read `M[0 .. 16*n_pos - 1]` in order, pack 16 bytes/word,
  write `n_pos` Act words to `Act[act_dst + 0 .. n_pos-1]` (Port B). A channel's
  `n_pos` bytes land contiguously, spanning `ceil(n_pos/16)` Act words — multi-word
  channels (n_pos>16) handled automatically by the sequential drain (no special
  tiling needed as long as `n_pos ≤ MAX_NPOS`).

For n_pos=16 this is the clean 16×16 byte transpose (drain word w = column w of M).
`n_pos > MAX_NPOS` → firmware CPU fallback (general but bounded).

## Control / registers (reuse decision-J pattern)
- New module `rtl/transpose_engine.v`, time-shares SRAM Port B (Out read → Act
  write), `busy` gives it Port-B priority in `npu_top` muxes (alongside
  `copy_busy`/`expand_busy`).
- New trigger `NPU_DMA_TRANSPOSE_TRIG` (0x15C) → regfile `o_transpose_trig`;
  completion via `NPU_DMA_STATUS[4]` (`transpose_done`) — polled, no IRQ (like
  copy). Two-side reg-map sync (`param_regfile.v` ↔ `firmware.h`).
- Reuse `RD_SRAM_BASE` (Out src base), `WR_SRAM_BASE` (Act dst base = pass*n_pos),
  `RD_LEN` = `n_pos`; banks reuse `dma_out_ping_sel` (Out read bank) and write Act
  **PONG** (GEMM input bank) — add Act write-bank select for the engine.
- Firmware: `dma_out_transpose_to_act(act_dst, out_src, n_pos, out_bank)`; Conv6
  switches from `act_dst=-1` (DDR) to a transpose path: each pass computes →
  transpose its Out bank → Act PONG `[pass*n_pos]`. After Conv6, the 1024-elem
  vector is resident in Act PONG; `npu_gemm_pass` skips its input DMA (new flag:
  input already resident).

## Banks (no conflict)
- Conv6 input: Act PING (resident from Conv5). Conv6 output: Out SRAM.
- Transpose: Out SRAM → Act **PONG**. GEMM reads Act PONG; FC weights in Wgt PONG.
- Conv resident copies (decision J) use Act PING regions → disjoint from PONG.

## Incremental bring-up (each step bit-identical 10/10)
1. `transpose_engine.v` + regs + npu_top Port-B mux, **dormant** (no firmware use):
   compile, 10/10, 8,041,665 cycles unchanged.
2. Firmware: route Conv6 → per-pass transpose → Act PONG; `npu_gemm_pass` reads
   resident input (skip DMA). Remove CPU reorder loop. Verify **GEMM input bytes
   in Act PONG byte-identical** to the old AFFINE_SCR (instrument), then 10/10.
3. Profile: prof_reorder → ~0, load/DDR drops; update CLAUDE.md (decision L) + memory.

## Verification
- Each step `bash run_all.sh clean && bash run_all.sh sim` → `10/10 correct`, `Errors: 0`.
- Bit-identity gate: a tb/firmware checksum of the GEMM input vector (Act PONG
  words 0..63) must match the CPU-reorder baseline before enabling.
- Quantify with `NPU_PROFILE=1` (edit `#define NPU_PROFILE` to 1, distclean+sim;
  revert after): expect `reorder` bucket → ~0, `load` down.

## Risk
- Port-B contention with copy/expand engines — mutually exclusive in time (copy is
  conv→conv, transpose is Conv6→FC, expand is Conv1 input). Guard with busy priority.
- Act PONG addressing for the GEMM input must exactly match `pack_fc_tile` order.
- n_pos tiling beyond MAX_NPOS is CPU fallback (not exercised by MNIST) — keep simple.
