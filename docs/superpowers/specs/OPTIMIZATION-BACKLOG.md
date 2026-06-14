# NPU optimization backlog (index)

Current baseline (branch `feat/npu-hw-reorder`, after #2 HW reorder / decision L):
- **7,618,571 cycles** full run, **150,471 cycles/image** inference (0.75 ms @200MHz)
- 10/10 correct, bit-identical gate: **SCORE_CHK = `D30179DF`**
- preload (one-time weight pack+DMA) ~6.25M = ~82% of full run (per-image excludes it)

## Completed
- Decisions D–L (GEMM/FC mode, weight reuse, hw-padding, row-parallel, SRAM
  residency, 128-bit AXI, **#2 HW reorder**). See CLAUDE.md "Architecture Decisions".
- Bus/AXI bandwidth utilization: **100%** (128-bit path, not a bottleneck anymore).

## Ready-to-implement specs (this folder)

### `2026-06-14-gemm-array-util-design.md` — GEMM array utilization ⭐ biggest
FC1 GEMM replicates input to 16 rows (15/16 redundant) → array util **6.25%**,
FC1 ≈ 55K/img (overhead-bound). Redesign: 16 rows reduce 16 IC-tiles down each
column → 16× fewer steps, util → ~100%, FC1 ~55K → ~5–10K/img (**~45K/img**).
Deepest/riskiest (changes systolic compute topology + weight feed). Has a
Phase-0 measurement gate — confirm FC1 is overhead-bound before committing.

### `2026-06-14-row-block-packing-design.md` — #4 narrow-layer row packing
Row-parallel idles rows when out_w<16 (Conv5/6 = 50% util). Pack R=⌊16/group_size⌋
output rows into the array (Conv5/6 R=2). Util 50%→100%, ~6–8K/img. High risk
(drain/post/pool machinery, where #3's hazard lived).

## Recommended order
1. **GEMM array util** — biggest win + biggest utilization gap. Do Phase 0 first.
2. **#4 row-block packing** — independent; either order works.

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
