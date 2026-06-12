# NPU Weight-Prefetch Reuse (general, per-OC-tile) — Design

## Context

Architecture review found the conv datapath re-prefetches an OC-tile's weights
from Wgt SRAM **once per output pixel**, even though conv weights are
position-invariant. Per output pixel (single IC-tile layer): ~144 weight-SRAM
reads vs only ~9 compute cycles — a 16:1 mismatch (the array consumes 16 OC
words/cycle, the single-port SRAM provides 1/cycle), so weight prefetch dominates
(~144 of ~174 cycles/pixel). This is pure redundancy: the same weights re-read
784× for a 28×28 layer.

A model-specific fix (reuse only when `ic_groups==1`, i.e. IC≤16) was rejected
because it only helps layers that happen to have IC≤16 — it would silently do
nothing if the deployed model changes. This design is **general**: it reuses
weights for any conv layer whose OC-tile weights fit an on-chip buffer of
configurable depth, and falls back to correct (unoptimized) behavior beyond it.

## Goal & principle

Prefetch each OC-tile's weights **once**, hold them on-chip, and reuse them across
the entire spatial sweep — for any model. Governed by a hardware capacity
parameter `ICG_BUF` (number of IC-tiles the weight buffer holds), not by model
dimensions — consistent with decision D (register-driven 16-OC tiling).

- `ICG_BUF = 4` (covers IC ≤ 64). All current conv layers fit (Conv1-6 have
  `ic_groups ∈ {1,2,4}`), so all benefit.
- Layers with `ic_groups > ICG_BUF` fall back to today's per-IC-tile prefetch —
  correct, just not accelerated.

## Non-goals

- Not touching the GEMM/FC path (it uses `gemm_en`; excluded from reuse — its
  `ic_groups` for FC1 is 64 anyway, and FC has a single spatial point).
- Not changing the systolic array, post-process, SRAM, or DMA.
- Not the 16-row-utilization rework (that is task E, orthogonal, separate).

## Components

### 1. `wgt_reader.v` — depth-ICG_BUF buffer + dual-mode prefetch

- **Buffer:** `wgt_buf[KERNEL_OFFSETS][NUM_OC][ICG_BUF]` (was `[KERNEL_OFFSETS][NUM_OC]`).
  Param `ICG_BUF = 4`. Register cost ≈ 9×16×4×128b ≈ 9 KB FF.
- **New input** `i_prefetch_all` (1 = reuse mode: prefetch every IC-group of the
  OC-tile in one operation; 0 = legacy: prefetch the single `i_ic_group`).
- **New input** `i_wgt_ic_sel [3:0]` — during CALC, selects which IC-group's
  weights to present.
- **Prefetch FSM:** add an outer `pf_icg` loop when `i_prefetch_all`:
  `pf_icg = 0..i_ic_groups_total-1`, writing `wgt_buf[pf_ko][pf_oc][pf_icg]`; the
  SRAM address uses `pf_icg` in the IC term (`addr_ic_component = pf_icg *
  i_kernel_offsets`) instead of `i_ic_group`. When `!i_prefetch_all`, behave
  exactly as today (write slot `[..][..][0]`, address uses `i_ic_group`).
- **Output mux:** `o_wgt[gi] = wgt_buf[ko_sel][gi][i_prefetch_all ? i_wgt_ic_sel : 0]`.
- Weight SRAM layout is unchanged (oc-major → ic_group → ko); the existing
  `load_conv_weights` packer already matches, so **no firmware weight-packing
  change**.

### 2. `top_controller_fsm.v` — prefetch once per OC-tile, reuse across sweep

- `reuse_mode = !i_gemm_en && (ic_groups <= ICG_BUF)` (wire; `ICG_BUF`=4 mirrored as a localparam).
- New reg `wgt_loaded`: set when the OC-tile's prefetch completes in reuse mode;
  cleared at `i_start` and whenever `oc_tile` advances.
- Drive `o_prefetch_all = reuse_mode`, `o_wgt_ic_sel = ic_tile[7:4]` to wgt_reader.
- **Prefetch trigger:** in reuse mode, run `S_PREFETCH_WGT` (the all-groups
  prefetch) **only at OC-tile entry** (when `!wgt_loaded`); for subsequent pixels
  skip the SRAM prefetch — `S_PREFETCH_WGT` waits a short settle for the im2col
  window, then goes to `S_CALC_KERNEL`. `o_wgt_start_prefetch` gated with
  `(!reuse_mode || !wgt_loaded)`.
- **IC-tile loop in CALC:** when `S_CALC_KERNEL` finishes an IC-tile and more
  remain, in reuse mode advance `ic_tile` and re-enter `S_CALC_KERNEL` directly
  (select next group from buffer via `o_wgt_ic_sel`) — **no** return to
  `S_PREFETCH_WGT`. In fallback mode, return to `S_PREFETCH_WGT` as today.
- Non-reuse (`ic_groups > ICG_BUF`, or GEMM): identical to current behavior.

### 3. `npu_top.v` — wire the two new wgt_reader ports

Connect `o_prefetch_all`/`o_wgt_ic_sel` from the FSM to wgt_reader's
`i_prefetch_all`/`i_wgt_ic_sel`.

## Data flow (reuse mode, one OC-tile)

```
OC-tile entry: prefetch ALL ic_groups -> wgt_buf[ko][oc][icg]   (once)
for each output pixel (spatial sweep):
    for ic_tile in 0..ic_groups-1:           (no SRAM prefetch)
        CALC ko offsets, weights = wgt_buf[ko][oc][ic_tile]  -> accumulate
    K_END -> DRAIN -> POST -> write
```

## Correctness

- Weights are byte-identical to the per-pixel path; only the *timing/source*
  changes. Therefore conv output must be **bit-identical** → still 10/10. This is
  the acceptance gate, not just "passes".
- `o_wgt_vld` subtlety: after prefetch the wgt_reader returns to `PF_IDLE` where
  `o_wgt_vld` deasserts, but `o_wgt` (combinational from `wgt_buf`) stays valid.
  Confirm the array/FSM gate compute on `fsm_array_vld`, not `wgt_reader`'s
  `o_wgt_vld` (initial read says yes; verify `wgt_reader_wgt_vld` is unused in CALC).
- Settle latency before CALC when skipping prefetch: the im2col window advances in
  `S_NEXT_TILE`; a 1–2 cycle settle in `S_PREFETCH_WGT` covers register/SRAM-comp
  latency. Exact value tuned during bring-up against the bit-identical gate.

## Testing / verification

1. `bash run_all.sh clean && bash run_all.sh sim` → require
   `=== Result: 10/10 correct ===`, `ALL TESTS PASSED.`, and conv results
   bit-identical (the 10/10 + identical predictions is the proof).
2. `NPU_PROFILE=1` run: confirm `npu` total and the per-layer `Conv1..Conv6`
   counts drop (expect the pixel-heavy Conv1/2/3 to fall most); report new total
   vs the 22,624,763 baseline.
3. Generality check (design-level, not a new test now): a layer with
   `ic_groups ∈ {1,2,4}` uses reuse; `>4` falls back — both correct.

## Risks

- Highest risk is the FSM IC-tile-loop-without-prefetch + settle timing. Mitigate
  with the bit-identical gate and waveform debug if a mismatch appears.
- Buffer indexing bugs (`pf_icg` loop / `i_wgt_ic_sel` mux). The fallback path
  must remain byte-identical (regression covers it: GEMM/`>4` layers — though all
  current conv layers are ≤4, so explicitly keep the fallback code exercised by
  GEMM's non-reuse path).
