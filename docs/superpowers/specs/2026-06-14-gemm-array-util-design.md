# Spec: GEMM array utilization — row IC-reduction (FC1 speedup)

## Context / problem
The GEMM/FC mode (decision F) runs the FC layers as a degenerate 1×1 conv: the
input IC-tile word is **replicated to all 16 array rows** (`npu_top.v` i_act =
`{ARRAY_ROWS{act_sram_doa}}`), so every PE in a column computes the *same* dot
product — **15 of 16 rows are redundant**. Each column = one OC (16 OC/pass).
- **Array utilization = 16/256 = 6.25%** (far worse than conv's 50–100%).
- IC is reduced over TIME: `ceil(IC/16)` k-steps (FC1: 1024/16 = **64 k-steps**)
  × `ceil(OC/16)` OC-passes (FC1: 64/16 = 4) — many steps, heavy per-step
  overhead (prefetch, fill, drain, per-pass MMIO config + IRQ).
- **FC1 (1024→50) measured ≈ 55K cycles/image** (the `affine` bucket; FC2 is a
  tiny CPU 50→10). The weight-read floor is only ~4,096 words (single-port Wgt
  SRAM, 1 word/cyc) → FC1 is **~13× over the weight floor = overhead-bound**, not
  bandwidth-bound. So cutting steps pays off.

Goal: feed **16 different IC-tiles** to the 16 rows and reduce them **down each
column** (spatial reduction), so one "super-step" consumes 256 IC (16 rows ×
16 IC/row) instead of 16. FC1: `ceil(1024/256) = 4` super-steps instead of 64 —
**16× fewer steps**, array utilization 6.25% → ~100%, FC1 expected 55K → ~5–10K
cycles/image (toward the ~4K weight-read floor + overhead).

## Phase 0 (do first): confirm the bottleneck
Add a firmware probe timing FC1 GEMM alone (separate from FC2). Confirm FC1 ≈
50–60K/img and that the dominant cost is per-k-step/per-pass overhead (not weight
reads). If FC1 turns out weight-bandwidth-bound (near ~4K), STOP — the redesign
won't pay off; revisit only the per-pass MMIO/IRQ overhead instead.

## Design (option A — row IC-reduction, recommended)
GEMM-reduce mode (gated, GEMM-only; conv path untouched and byte-identical):
- **Activation feed (`npu_top.v`):** instead of replicating one IC-tile to all
  rows, feed **row r = IC-tile r** (16 distinct IC-tile words per super-step).
  Read 16 IC-tile words from Act SRAM into a row-feed register (16 cyc) then
  present them in parallel.
- **Weights (`wgt_reader.v` / array i_wgt):** PE(r,c) needs OC-c's weights for
  IC-tile r. The weight bus is currently 16 column-groups (one word/column,
  broadcast to all rows). Extend to a **16×16 per-PE weight matrix** per
  super-step: a weight buffer (like the decision-G reuse buffer) holds 256 words
  (16 OC × 16 IC-tiles), prefetched from Wgt SRAM (256 cyc), consumed in parallel.
  FC weight packing (`pack_fc_tile`) re-laid out to this 16×16 order.
- **Reduction (`pe_core.v` / `systolic_16x16.v`):** in GEMM-reduce mode the column
  forms a **systolic adder chain during compute** (not just drain): PE(r,c)
  outputs `o_psum_casc = tree_sum + i_psum_casc`; the bottom PE's output = the
  column's 256-IC partial sum, accumulated into a per-column accumulator across
  super-steps. (Conv keeps the current local-accumulate + drain-shift topology —
  the new behavior is behind a `gemm_reduce` mode so conv stays byte-identical.)
- **Drain/post:** one result per column (16 OC), not 16 redundant copies. The
  post-process/quant/relu path is unchanged (16 OC results); only the count of
  meaningful rows changes (now 1 column-sum vs 16 identical).
- **FSM (`top_controller_fsm.v`):** GEMM k-step loop streams `ceil(IC/256)`
  super-steps (was `ceil(IC/16)`); the IC-tile counter advances by 16 tiles/step.

## Alternative (option B — rows as more OC), noted not recommended
Use the 16 rows for 16 *additional* OC (16 rows × 16 cols = 256 OC/pass). FC1 has
64 OC → 1 pass instead of 4. But each PE then needs a distinct OC weight, same
256-word/super-step weight feed problem, AND it doesn't cut the 64 IC k-steps
(still time-reduced) — so smaller win than A. Skip.

## Touchpoints
- `rtl/pe_core.v` — GEMM-reduce mode: cascade-sum during compute + per-column accumulator.
- `rtl/systolic_16x16.v` — cascade wiring already vertical; expose the reduce mode.
- `rtl/npu_top.v` — GEMM act feed: 16 distinct IC-tiles (not replicate); array i_wgt 16×16.
- `rtl/wgt_reader.v` — 16×16 weight buffer for GEMM; FC weight prefetch.
- `rtl/top_controller_fsm.v` — GEMM super-step loop (`ceil(IC/256)`); act IC-tile addressing.
- `firmware/deepnet_deploy.c` — `pack_fc_tile` 16×16 layout; `npu_gemm_pass` step config.
- `rtl/param_regfile.v` + `firmware/firmware.h` — `gemm_reduce` mode bit if needed.

## Incremental bring-up (each step bit-identical SCORE_CHK + 10/10)
1. Phase 0 measurement (above). Gate the whole effort on it.
2. Plumb the mode + weight buffer + act feed, **dormant** (conv + legacy GEMM
   byte-identical): 7,618,571 cycles, SCORE_CHK `D30179DF` unchanged.
3. Enable on **FC1** only → verify SCORE_CHK == `D30179DF` (FC1 output feeds FC2
   feeds scores), 10/10; measure FC1 cycle drop.
4. Profile + update CLAUDE.md (decision N) + memory.

## Verification
- `bash run_all.sh clean && bash run_all.sh sim` → `10/10`, `Errors: 0` each step.
- SCORE_CHK gate (firmware checksum of all 10 images' int32 scores == `D30179DF`).
- FC1 cycle probe before/after to quantify (expect ~55K → ~5–10K/img).
- tb array-util counter on the GEMM window to confirm 6.25% → ~100%.

## Risk (highest of all remaining work)
- Changes the systolic array's **compute topology** (cascade-sum during compute)
  + the **weight feed** (16×16) — the deepest dataflow change in the project.
  The conv path MUST stay byte-identical (mode-gated); verify with the conv-only
  SCORE_CHK before touching GEMM.
- Weight buffer 256 words = sizeable registers; Wgt SRAM single-port prefetch
  (256 cyc/super-step) is the new floor — confirm it beats the current 64-k-step path.
- FC2 stays on CPU (INT8 saturation, decision F) — unaffected.

## Relationship to other specs
- Independent of #4 row-block packing (conv narrow-layer rows) — different mode,
  different layers. Can be done in either order.
- Bigger array-utilization gap than #4 (6.25% vs 50% starting point) and likely
  bigger cycle win (~45K/img vs ~6–8K/img), but deepest/riskiest.
