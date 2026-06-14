# Spec: #4 Row-block packing — fill idle array rows for narrow layers

## Context / problem
Row-parallel (decision I) makes the 16 array rows compute 16 adjacent output
columns of ONE output row. When `out_w < 16` the group covers the whole row
(`group_size = out_w`), so only `out_w` of 16 array rows do useful MACs and the
rest idle. Per-layer row utilization (= group_size/16):

| layer | out_w | group_size | row util |
|-------|-------|------------|----------|
| Conv1/2 | 28 | 16, then 12 | ~87.5% |
| Conv3 | 14 | 14 | 87.5% |
| Conv4 | 16 | 16 | 100% |
| **Conv5/6** | **8** | **8** | **50%** |

Goal: when `out_w < 16`, pack **R output rows** into the 16 array rows
(R output rows × group_size columns ≤ 16), filling the idle rows. Primary value
is **array utilization** (Conv5/6 50%→100%); cycles are a secondary win
(fewer groups/drains/LOAD_ROWs — MAC count is unchanged, per-group fill/drain/
post overhead and per-row activation streaming roughly ÷R).

## Generality
`R = min( floor(16 / group_size), MAX_R )`, `group_size = min(16, out_w)`.
`MAX_R` is a hardware capacity knob (like the 16×16 array, ICG_BUF): the im2col
line buffer holds `MAX_R + 2` row banks. **MAX_R = 2** covers MNIST (Conv5/6
out_w=8 → R=2 → 4 banks); other layers have group_size≥14 → R=1 (byte-identical
to current row-parallel). `R = 1` MUST stay byte-identical to decision I.

## Core mapping
Array row `gi` (0..15) maps to:
- output-row block `b = gi / group_size`  → output row `cur_oy + b`
- column within group `c = gi % group_size` → output column `group_base + c`

(Decision I is the R=1 special case: b=0, c=gi.)

Input rows needed: R output rows, each a 3-row window (stride 1, 3×3):
block b uses input rows `{oy+b, oy+b+1, oy+b+2}`. Union over b=0..R-1 =
`{oy .. oy+R+1}` = **R+2 input rows** → line buffer needs `MAX_R+2` banks.

## Four implementation pieces

### 1. im2col line buffer (`rtl/im2col_line_buffer.v`)
- Storage: `MAX_R+2` row banks (was 3). `row_sel` rotates over `MAX_R+2`.
- Window read (the `gen_window` loop, currently `rp_col`/`rp_bank`):
  - `c = gi % group_size`, `b = gi / group_size` (needs `group_size` as input).
  - `rp_col = group_base + off_col_dec + c`  (was `+ gi`).
  - `rp_bank = bank_for_offrow(off_row_dec, row_sel)` **shifted by b** — block b's
    3-row window is `{oy+b..oy+b+2}`, i.e. the base 3-row window advanced by b
    rows. Generalize `bank_for_offrow` to add the block offset b.
- Needs a new input `i_group_size` (and the block decomposition). `R=1` path
  (group_size≥window width) MUST reduce to the existing mapping byte-identically.

### 2. FSM (`rtl/top_controller_fsm.v`)
- Compute `R` (runtime) from group_size and MAX_R.
- Spatial sweep: advance `cur_oy` by `R` per row-block (was +1). `out_h` loop
  handles the last partial block (R rows may exceed remaining out_h → use ≤R).
- LOAD_ROW: ensure `R+2` input rows are resident before a row-block's CALC
  (load R extra rows per block step, or load R+2 at block start).
- Drives `o_group_size` / a new `o_rows_per_grp = R` to im2col + drain path.

### 3. Drain / Out-SRAM write (`rtl/npu_top.v` rp sequencer)
- The drain emits `R*group_size` valid pixels (reverse order, row 15 first).
  Map drained array-row `k` → block `b=k/group_size`, col `c=k%group_size` (with
  the existing reverse-order `15-cnt` indexing) → Out SRAM addr
  `out_base + oc_off + (cur_oy+b)*out_row_stride + (group_base+c)`.
- Generalize `rp_col_valid` / `rp_wr_addr` to the (b,c) decomposition; guard
  `c < group_size` and `cur_oy+b < out_h`.

### 4. Pool path (`rtl/post_process_top.v` rp_buf)
- Pooled layers (Conv6) currently replay one output row's drained pixels into
  `max_pooling_2x2`. With R rows packed, the drain carries R rows — 2×2 pooling
  needs 2 input rows, so R=2 packing actually aligns with the pool window, but
  the `rp_buf` reorder + replay must group by (block, col) and feed the pooler
  the correct row-major 2×2 windows. This is the most delicate piece; bring up
  AFTER the non-pool path.

## Incremental bring-up (each step bit-identical 10/10 + SCORE_CHK gate)
Methodology = decision I (the one that surfaced the #3 hazard). Behind the
existing `row_par_en` (or a new `CTRL` mode bit), dormant until enabled.
1. Plumb R / group_size / `MAX_R+2` banks; with **R forced 1**, verify
   byte-identical to the post-#2 baseline (7,618,571 cycles, SCORE_CHK `D30179DF`).
2. Enable R=2 on **Conv5** (non-pool) → verify SCORE_CHK == baseline, 10/10.
3. Enable R=2 on **Conv6** (pool path) → verify SCORE_CHK == baseline, 10/10.
4. Profile: Conv5/6 group/drain/LOAD_ROW ÷2; update CLAUDE.md (decision M) + memory.

## Verification
- Per step `bash run_all.sh clean && bash run_all.sh sim` → `10/10`, `Errors: 0`.
- **Bit-identity gate**: a firmware SCORE_CHK (XOR/sum of all 10 images' int32
  scores) must equal the baseline `D30179DF` before/after each enable. (This is
  what proved #2 bit-identical and would have caught #3's hazard early.)
- An RTL array-utilization counter (tb: count useful-MAC cycles vs busy cycles)
  to confirm Conv5/6 row util 50%→~100% — the primary success metric.

## Risk
- Highest-risk area: the drain/post/pool machinery (#3's ic_tile hazard lived
  here). The R=1-byte-identical gate (step 1) and per-layer SCORE_CHK gates are
  the safety net — do NOT skip them.
- Partial last block (out_h not a multiple of R): use ≤R rows, guard writes.
- `bank_for_offrow` block-offset generalization must keep R=1 exact.
- Line-buffer bank count change touches reset/init and row_sel rotation — verify
  the R=1 path first.

## Out of scope
- Packing for `out_w > 16` (Conv1/2 last 12-wide group): R=1 there, idle rows in
  the partial group unaddressed (small: 4 idle rows in 1 group/row).
- MAX_R > 2 (no layer needs R>2); structurally supported by bumping MAX_R + banks.
