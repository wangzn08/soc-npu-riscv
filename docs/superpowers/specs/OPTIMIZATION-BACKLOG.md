# NPU optimization backlog (index)

Current baseline (branch `feat/npu-hw-reorder`, after #2 HW reorder / decision L):
- **7,618,571 cycles** full run, **150,471 cycles/image** inference (0.75 ms @200MHz)
- 10/10 correct, bit-identical gate: **SCORE_CHK = `D30179DF`**
- preload (one-time weight pack+DMA) ~6.25M = ~82% of full run (per-image excludes it)

## Completed
- Decisions D–M (GEMM/FC mode, weight reuse, hw-padding, row-parallel, SRAM
  residency, 128-bit AXI, **#2 HW reorder**, **GEMM array util**). See CLAUDE.md
  "Architecture Decisions".
- Bus/AXI bandwidth utilization: **100%** (128-bit path, not a bottleneck anymore).
- **GEMM array util (decision M, done 2026-06-14):** array utilization 6.25% →
  ~100%, bit-identical `SCORE_CHK=D30179DF`. **Phase 0 corrected the spec's
  premise**: FC1 is **19,638 cyc/img**, NOT the assumed ~55K (that bucket was
  dominated by CPU FC2's DDR re-reads, since fixed for −11.6K/img). FC1 is only
  ~5× the weight floor → **weight-bandwidth-bound, not compute-bound**, so the
  redesign fixed utilization but the cycle win is marginal (FC1 −3.4%, full run
  −0.09%): the 256-word/super-step plane prefetch reads the same 1024 words/pass
  as the legacy 64-k-step path (single-port Wgt SRAM is the floor). Real cycle
  cuts would need reducing weight reads (dual-port Wgt SRAM / weight reuse) —
  out of scope.

## Ready-to-implement specs (this folder)

### ~~`2026-06-14-gemm-array-util-design.md`~~ — DONE (decision M, see Completed)
Implemented behind `gemm_reduce` (CTRL[10]). Note: the spec's "FC1 ≈ 55K/img,
~45K win" premise was WRONG — Phase 0 found FC1 = 19.6K (weight-bandwidth-bound),
so the realized win was −0.09% full run. The utilization goal (6.25%→~100%) was
met. Plan: `docs/superpowers/plans/2026-06-14-gemm-array-util.md`.

### ~~`2026-06-14-row-block-packing-design.md`~~ — #4 DONE (decision N, 2026-06-14)
Pack R=2 output rows into the array for out_w==8 layers (Conv5/6). Util **50%→100%**.
**Implemented behind `row_block_en` (CTRL[11])**, bit-identical `SCORE_CHK=D30179DF`.
- Result (clean row_block on/off full-run A/B): **7,680,437 → 7,427,997 (−252,440,
  −3.3%)**. Conv5 done first (non-pool), then Conv6 (pool — R=2 aligns with the
  2×2 row-pair, no reorder pain). No debug cycles needed; both bit-identical
  first try.
- **Phase-0 LESSON (important):** Phase 0 first ABANDONED this, estimating ~0.2%
  full run from the firmware `prof_busy_layer` (IRQ-wait) proxy + a per-group-cost
  extrapolation. That proxy **under-counted by ~15×** — the real on/off A/B showed
  −3.3%. So: a Phase-0 *proxy* can be as wrong as a spec estimate. When the change
  is bounded and mode-gated, the cheapest true measurement is to **build it behind
  the bit-identical gate and A/B the mode bit**, not extrapolate from a proxy.
- Probe infra (keep): firmware `prof_busy_layer[6]`, npu_top `FSMDBG` counters.

## Recommended order
1. ~~GEMM array util~~ — DONE (decision M). Util 6.25%→100%; cycle win marginal.
2. ~~#4 row-block packing~~ — DONE (decision N). Util 50%→100%; **−3.3% full run**.
3. **Open (if pursuing more conv cycles): reduce per-OC-pass start/IRQ overhead**
   (~1,683 cyc/start × 4 OC-passes/layer). OC-tiling-in-one-start (big, bumps
   decision D) or firmware MMIO trims (broadcast scale/shift). Phase 0 first — but
   per the #4 lesson, prefer a mode-gated A/B over a proxy extrapolation.
4. **Open (capability generality, not cycles): output bit-width (let final FC run
   on NPU) / depthwise conv.** Not bandwidth-trap-prone; payoff more certain.

## Methodology (do not skip)
- Branch from `feat/npu-hw-reorder`.
- Each increment behind a dormant mode bit; bring up layer-by-layer.
- **SCORE_CHK gate every step**: firmware checksum of all 10 images' int32 scores
  must equal `D30179DF`. (This is what proved #2 and would have caught #3's
  ic_tile hazard early — `(score_chk*31)+scr[i]+(d*10+i)`, printed at the end.)
- Build/sim: MSYS Bash, `cd /e/code/6-10/soc;` each call (cwd resets); env vars
  do NOT propagate to child shells — to profile, edit `#define NPU_PROFILE 1` in
  `firmware/deepnet_deploy.c`, `distclean`, sim, then revert.

## Deprioritized
- #1 CPU/NPU overlap: ~2.6%, high risk (timing-sensitive, see reverted #3). Skipped.
- preload pre-packing: ~5.7M (82% of full run) but one-time; build-time
  pre-packing of weights into SRAM layout would cut it. User deprioritized
  (one-time). Revisit if the scoring metric is total cycles.

## Shelved / reverted
- #3 PSum double-buffer drain overlap (decision K): bit-identical fix found
  (ic_tile reset), NPU −6K, but net +254K non-profile timing artifact. Reverted.
