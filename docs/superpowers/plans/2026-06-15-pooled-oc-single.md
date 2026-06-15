# Pooled oc_single via per-OC-tile pooler state — decision P

> Branch: `feat/npu-oc-single-start` (continues decision O). Bit-identical gate:
> firmware `SCORE_CHK == D30179DF` + 10/10 every step. Metric: non-profiled full-run
> TRAP A/B (per-phase proxy MISLEADS — see [[soc-npu-spec-estimate-caution]]).

**Goal:** extend `oc_single` (decision O) to **pooled** conv layers (Conv4, Conv6),
which currently fall back to legacy because the 2×2 pooler holds cross-row state
that is **not per-OC-tile**. The OC-inner loop interleaves OC tiles between rows, so
a single shared pooler line-buffer/phase gets corrupted.

**Design (option A — per-OC-tile pooler state):** give `max_pooling_2x2` an
`i_tile` index and replicate its cross-row state per OC tile (capacity
`NUM_TILES = MAX_OC_RESIDENT/16 = 4`). Each tile pools independently, so the
interleaved-tile drain stream is fine. **General** for any out_w / tile count —
no FSM loop restructure, no geometry constraint (unlike the row-pair-nesting
alternative, which can't pack 2 rows into the 16-wide array for out_w=16).

The pooler's cross-row state (`max_pooling_2x2.v`): `line_buf[0:MAX_WIDTH-1]`,
`col`, `row_odd`, `cur_left`, `above_left`. All become `[NUM_TILES][...]`,
indexed by `i_tile`. `i_tile==0` (oc_single off) ⇒ byte-identical.

**Timing (why i_tile = fsm_oc_tile_sel works):** in oc_single the pooler is fed by
the `post_process_top` rp replay; `fsm_pp_done = rp_pool_done` holds S_POST until a
tile's replay completes, and `oc_t` only advances at S_POST exit — so
`fsm_oc_tile_sel` is **stable for the whole of a tile's drain+replay**, and the
pooler's state[tile] updates (on `i_feat_vld`) all see the correct tile.

## Steps (each: clean build + full sim, SCORE_CHK D30179DF + 10/10, commit)
- [ ] **P1. `max_pooling_2x2` per-tile state** — add `i_tile` input; replicate
  `line_buf`/`col`/`row_odd`/`cur_left`/`above_left` to `[NUM_TILES]`; `above`
  async read = `line_buf[i_tile][col[i_tile]]`; `i_start` clears all tiles.
  Tie `i_tile=0` at the instantiation → byte-identical. Build+sim. Commit.
- [ ] **P2. Route `oc_t` to the pooler** — `post_process_top` gains `i_tile`,
  forwards to `u_pool`; `npu_top` wires `fsm_oc_tile_sel`. oc_single off ⇒ tile 0
  ⇒ byte-identical. Build+sim. Commit.
- [ ] **P3. Enable Conv4 oc_single** (OC=32, 2 tiles, pool, non-row-block) — flip
  the firmware flag. Verify D30179DF + 10/10. Debug per-tile pooler if mismatch.
- [ ] **P4. Enable Conv6 oc_single** (OC=64, 4 tiles, pool + transpose, row-block)
  — flip flag. Verify D30179DF + 10/10. (Per-tile pooler must compose with the
  row-block R=2 replay and the per-tile transpose; tile-major `pool_wr_addr` from
  decision O already in place.)
- [ ] **P5. Measure** non-profiled full-run TRAP A/B (all conv oc_single on vs the
  decision-O non-pool-only baseline 7,376,666). Document decision P in CLAUDE.md +
  memory. Note per-layer deltas.

## Risk / rollback
- Each step mode-gated (`i_tile=0` ⇒ byte-identical). If P3/P4 can't reach
  D30179DF, revert that layer to legacy (oc_single=0) — nothing regresses.
- Conv4 (non-row-block) and Conv6 (row-block + transpose) are different pool
  geometries; bring up Conv4 (simpler) first.
- `NUM_TILES` is a capacity knob (like ICG_BUF / MAX_OC_RESIDENT); OC>64 already
  falls back to multi-start (decision O), which doesn't use oc_single at all.
