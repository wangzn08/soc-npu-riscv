# NPU 16-Row Spatial Parallelism (E) — Design Spec

> **Status:** design only. Plan + implementation deferred to a fresh session
> (this is the largest/riskiest change; a clean context lowers bring-up error
> risk). Pick up from this spec → writing-plans → executing-plans.

## Context & motivation

The 16×16 array (4096 MAC/cycle) is run at **1/16 utilization** for conv: im2col
replicates **one** window to all 16 rows ([im2col_line_buffer.v:239-242]), so the
16 rows compute the *same* thing; only the 16 columns (OC) are useful → 256
useful MAC/cycle. The drain shifts 16 rows out but 15 are redundant.

**E makes the 16 rows compute 16 different output pixels** (a horizontal group of
16 adjacent output columns), so the array does 16 pixels × 16 OC per K_END.

This is primarily about **not wasting the silicon** (the stated goal), not just
cycles. Post-PAD baseline: full run 10.91M, inference (`prof_infer`) 4.60M of
which NPU conv = 2.15M. E targets that 2.15M.

**Expected benefit:** conv compute ~16× on the per-pixel work (bounded by drain
16-cycle/group and per-group overhead, and by out_w for late layers). Estimate
conv 2.15M → ~0.3-0.5M; inference 4.6M → ~2.7-2.9M (**~40% faster inference**).
Full run 10.9M → ~9M (the rest is sim UART prints, outside inference). **Hard
ceiling = the 2.15M conv cost.**

## Confirmed datapath facts (verified this session)

- `pe_core`: `PE[r][c]` is an **independent** accumulator `Σ act[r]·wgt[c]`,
  latched on `i_k_end`; the column cascade is used **only during drain** to shift
  the 16 rows out serially (one row per drain cycle). So the array already
  supports 16 distinct (row,col) results — E just stops feeding identical rows.
- `systolic_16x16`/`gp_4x4`: act group `r` → row `r`; wgt group `c` → col `c`.
- **Weights are shared across the 16 pixels** (same conv weights regardless of
  output position) → `i_wgt` (16 OC broadcast to all rows) and `wgt_reader` are
  **unchanged**.
- **Drain already serializes pixels:** each drain cycle yields 16 OC for ONE row
  (= one pixel). Over 16 drain cycles → 16 pixels, row-major. So
  `post_process_top`, `max_pooling_2x2`, and the Out-SRAM write see the *same
  one-pixel-per-cycle stream* they see today — **largely unchanged** (the pooler
  streams row-major; verify the column-counter continuity across groups).

## Architecture

### What changes
1. **`im2col_line_buffer.v` — the core rewrite.** Replace "3×3 shift-register
   window + 16× replication" with "**16-wide line-buffer slice**": for the
   current kernel offset `(ko_row, ko_col)` and IC tile `t`, output the 16
   consecutive line-buffer entries
   `lb[row = ko_row][col = group_base + ko_col + (0..15)][t]` as the 16 array-row
   activation groups (`o_act_window[r] = slice[r]`). Out-of-range columns/rows
   (left/right/top/bottom borders) emit **zero** — this is where hardware padding
   should now live (migrating PAD's FSM zero-injection into im2col, the "clean"
   architecture deferred from decision H). The line-buffer storage
   (`lb_bank0/1/2[MAX_WIDTH][ICG_MAX]`, registers) is **reused** — 16 parallel
   reads are just combinational fan-out; no new storage.
2. **`top_controller_fsm.v` — spatial loop by groups of 16.** The output sweep
   advances `cur_ox` by the **group size** (≤16) per step instead of 1. Per
   group: stream offsets×IC-tiles (same K-step structure, now 16 pixels at once),
   K_END, drain (16 cycles → up to 16 pixels), write the **valid** pixels.
   `group_size = min(16, out_w - group_base)` handles `out_w` not a multiple of 16
   (e.g. 28 → 16+12) and `out_w < 16` (e.g. 8 → one group of 8). Invalid rows
   (≥ group_size) are drained but **not written**.
3. **Out-SRAM write — 16 words/group.** Non-pool: write `group_size` consecutive
   row-major addresses. Pool: the pooler consumes the drained stream self-timed
   (`pp_feat_vld`), so it likely works if the drained pixel order is row-major and
   continuous across groups — **must verify** the pooler column counter.

### What stays unchanged
`systolic_16x16`, `gp_4x4`, `pe_core`, `wgt_reader`, `post_process_top`,
`max_pooling_2x2`, `vector_alu`, SRAM wrappers, `axi_dma`.

### Interaction with PAD (decision H)
PAD currently zero-injects borders in the FSM (`o_border` + `npu_top` mux) feeding
the *replicated* stream. E rewrites im2col's windowing, so border zeroing should
**move into im2col's 16-wide slice** (a slice column is zero when its absolute
input col/row is outside the unpadded image). When E lands, the FSM
zero-injection + `fsm_border_d` mux in `npu_top` can be **removed** (padding owned
by im2col). The `NPU_PAD` register + `CTRL[8] hw_pad` stay (now consumed by
im2col). This is the planned migration noted in decisions G/H.

## Key design decisions (resolve in planning)
- **Drain→pixel order:** confirm whether drain emits row 0 first or row 15 first,
  and align the Out-write address / pooler order to row-major. (Inspect
  `pe_core` drain shift direction + current `drain_cnt`→addr mapping.)
- **Group geometry across input rows:** a group of 16 output cols (stride 1, 3×3)
  spans 16+2 input cols within 3 input rows — the line buffer already holds full
  rows, so one group reads 3 rows × (16+2) cols. Confirm MAX_WIDTH covers the
  widest layer (28+2 ≤ 256 ✓).
- **Stride > 1:** current convs are stride 1; keep general (group of 16 output
  cols maps to 16·stride input cols). Spec assumes stride 1 is the test case;
  design the addressing generally.
- **Late layers (out_w=8):** only 8 of 16 rows used → 2× not 16×. Acceptable; the
  array fills what it can. No special-casing needed beyond `group_size`.

## Generality (per [[soc-npu-general-purpose]])
Group size derived at runtime from `out_w` (register-driven), not hardcoded. Any
layer benefits up to 16-wide; partial groups handled. Like the 16×16 array and
16-OC tiling, "16 pixels/group" is a hardware width, not a model dimension.

## Verification
- **Bit-identical gate:** conv output must be byte-for-byte identical to the
  current (PAD) build → `10/10` + `ALL TESTS PASSED`, predictions unchanged. E
  only changes *which rows compute what* + timing, not the math.
- Bring-up incrementally: a `CTRL` mode bit (e.g. `row_par_en`) to enable E
  per-start, so it can be validated on one layer (suggest Conv1, out_w=28) before
  all — mirrors the PAD bring-up that worked well.
- `NPU_PROFILE=1`: confirm `npu`/per-layer conv cycles drop; report vs 2.15M.
- RTL change → `bash run_all.sh clean` first.

## Risks
- **Highest-risk change in the project.** im2col windowing rewrite + FSM spatial
  restructure + drain/pixel-order alignment + pooler-stream continuity. Mitigate
  with: the mode bit (dormant checkpoint), one-layer bring-up, bit-identical gate,
  and waveform debug (`run_all.sh waves`).
- Pooler interaction across group boundaries is the subtlest correctness point —
  verify early on a pooled layer (Conv2).
- Don't combine with the PAD-migration in one step: first get E working keeping
  PAD's FSM zero-injection (feed im2col the padded stream as today but
  non-replicated), THEN migrate padding into im2col as a separate, separately-
  verified step.

## Files
- Rewrite: `rtl/im2col_line_buffer.v`
- Modify: `rtl/top_controller_fsm.v` (group loop, out-write), `rtl/npu_top.v`
  (mode bit wiring; later remove `fsm_border_d` mux on PAD migration),
  `rtl/param_regfile.v` + `firmware/firmware.h` (mode bit), `firmware/deepnet_deploy.c`
  (enable per conv).
- Unchanged: systolic/pe/gp/wgt_reader/post_process/pooling/dma/sram.
