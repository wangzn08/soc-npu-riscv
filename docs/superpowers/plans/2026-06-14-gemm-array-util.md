# GEMM Array Utilization (row IC-reduction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make NPU GEMM/FC mode use all 16 array rows to reduce 16 distinct IC-tiles down each column (spatial reduction), raising GEMM array utilization from 6.25% to ~100% and cutting FC1 from ~19.6K to ~5–10K cycles/image — a general-purpose matrix-engine improvement, not a model-specific hack.

**Architecture:** A new, mode-gated `gemm_reduce` path (CTRL[10]) layered on top of the existing GEMM mode (CTRL[7]). The conv path and the legacy GEMM path stay **byte-identical** (the new behavior only activates when `gemm_reduce=1`). Per super-step: array row `r` gets IC-tile `r`'s activation word; PE(r,c) gets OC-`c`'s weight for IC-tile `r` (a 16×16 weight plane); each PE accumulates its IC-tile over `ceil(IC/256)` super-steps (time); a single combinational **reduce-drain** sums the 16 rows down each column (space) → one result per column = 16 OC. FC1 (1024 IC) = 4 super-steps instead of 64 k-steps.

**Tech Stack:** Verilog-2001/SystemVerilog RTL (ModelSim/Questa), rv32imc C firmware. "The test" = one simulation run (`bash run_all.sh sim`) printing `10/10 correct`, `ALL TESTS PASSED`, `Errors: 0`, plus a firmware **SCORE_CHK** checksum that must stay bit-identical across every step.

**Phase 0 (already done):** FC1 GEMM measured at **19,638 cyc/img** (not the spec's assumed ~55K; that bucket was dominated by CPU FC2 DDR re-reads, since fixed). FC1 is ~5× its ~4K weight-read floor → overhead-bound → the redesign still pays off, but the realistic win is ~10K/img. Proceeding as a generality/utilization improvement (array util 6.25% → ~100%).

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `firmware/deepnet_deploy.c` | inference + verification harness | Add SCORE_CHK print (Task 1); `npu_gemm_pass` reduce config (Task 8) |
| `rtl/param_regfile.v` | config/status registers | Add `gemm_reduce` mode bit CTRL[10] → `o_gemm_reduce` (Task 2) |
| `firmware/firmware.h` | firmware register defines | Add `NPU_CTRL_GEMM_REDUCE` (Task 2) |
| `rtl/pe_core.v` | single PE: MAC + accumulate + drain | Add `i_reduce`: combinational column-sum during drain (Task 3) |
| `rtl/gp_4x4.v` | 4×4 PE sub-grid | Thread `i_reduce`; widen weight input to per-PE plane (Task 4) |
| `rtl/systolic_16x16.v` | 16×16 array | Thread `i_reduce`; add 16×16 weight-plane input `i_wgt_plane` (Task 4) |
| `rtl/wgt_reader.v` | weight prefetch buffer | Hold ≥16 IC-tiles for GEMM; expose 16×16 plane output (Task 5) |
| `rtl/top_controller_fsm.v` | computation sequencer | `gemm_reduce` super-step loop: act row-load, weight plane, 1 vld/super-step, reduce-drain (Task 6/7) |
| `rtl/npu_top.v` | datapath glue | Build 16×128 act row register; mux act/weight plane + `i_reduce` into array; single-valid post capture (Task 6/7) |

---

## Task 1: Establish the SCORE_CHK bit-identical gate (firmware only, no behavior change)

The methodology requires a checksum of all 10 images' int32 scores that must stay constant across every RTL change. It is not currently in the firmware. Add it, capture the golden value.

**Files:**
- Modify: `firmware/deepnet_deploy.c` (argmax/score loop region, ~line 955–990; result print region)

- [ ] **Step 1: Add a running checksum accumulator and fold each image's scores**

In `usercode7()`, declare near the per-image loop:

```c
uint32_t score_chk = 0;
```

After argmax for each image (where `scores[0..9]` and predicted digit `pred` are final), fold them in (same recurrence the backlog cites):

```c
for (int i = 0; i < 10; i++)
    score_chk = (score_chk * 31u) + (uint32_t)scores[i] + (uint32_t)(pred * 10 + i);
```

(Place it inside the existing per-image loop, after `scores[]` are computed and `pred` is known. Use the actual variable names present — `pred`/`best`/`argmax` — match the file.)

- [ ] **Step 2: Print the checksum at the end**

After the `=== Result: X/10 correct ===` print:

```c
print_str("SCORE_CHK="); print_hex(score_chk); print_chr('\n');
```

- [ ] **Step 3: Build + run to capture the golden value**

Run: `cd /e/code/6-10/soc; bash run_all.sh distclean >/dev/null 2>&1; bash run_all.sh sim 2>&1 | grep -E "Result:|SCORE_CHK|TESTS PASSED|Errors:"`
Expected: `10/10 correct`, `ALL TESTS PASSED`, `Errors: 0`, and a line `SCORE_CHK=XXXXXXXX`. **Record `XXXXXXXX` here in the plan** — it is the gate for all later steps. (Compare against the backlog's `D30179DF`; if it differs, the firmware diverged since — use the freshly measured value as the new golden and note it.)

**GOLDEN: `SCORE_CHK=D30179DF`** (measured 2026-06-14, matches backlog). This is the gate for all later steps.

- [x] **Step 4: Commit**

```bash
git add firmware/deepnet_deploy.c
git commit -m "test(npu): add SCORE_CHK bit-identical gate for GEMM-reduce bring-up"
```

---

## Task 2: Add the dormant `gemm_reduce` mode bit (CTRL[10]) — no datapath effect yet

**Files:**
- Modify: `rtl/param_regfile.v` (CTRL decode; output port)
- Modify: `firmware/firmware.h` (CTRL bit define)

- [ ] **Step 1: Add the regfile output**

In `param_regfile.v`, alongside `o_row_par_en` (CTRL[9]), add:

```verilog
output wire o_gemm_reduce,    // CTRL[10]: GEMM spatial-reduce mode
...
assign o_gemm_reduce = ctrl_reg[10];
```

- [ ] **Step 2: Add the firmware define**

In `firmware/firmware.h`, next to the other CTRL bits:

```c
#define NPU_CTRL_GEMM_REDUCE (1u << 10)   // GEMM spatial-reduce (16-row IC reduction)
```

- [ ] **Step 3: Thread the wire through `npu_top.v` (declare + connect regfile → unused for now)**

```verilog
wire cfg_gemm_reduce;
// ... in param_regfile instance:
.o_gemm_reduce (cfg_gemm_reduce),
```

Leave `cfg_gemm_reduce` unused (Verilog tolerates it; or add `// verilator lint_off UNUSED` if needed). Compile must be clean.

- [ ] **Step 4: Build + run — must be byte-identical**

Run: `cd /e/code/6-10/soc; bash run_all.sh clean >/dev/null 2>&1; bash run_all.sh sim 2>&1 | grep -E "Result:|SCORE_CHK|TESTS PASSED|Errors:"`
Expected: `10/10`, `SCORE_CHK=<golden from Task 1>`, `Errors: 0`.

- [ ] **Step 5: Commit**

```bash
git add rtl/param_regfile.v rtl/npu_top.v firmware/firmware.h
git commit -m "feat(npu): dormant gemm_reduce mode bit CTRL[10]"
```

---

## Task 3: pe_core reduce-drain (combinational column sum) — gated, dormant

Add `i_reduce`. When `i_drain_en && i_reduce`: instead of shifting, each PE drives `o_psum_casc = psum_shift + i_psum_casc` (combinational adder chain down the column), so the bottom column output = sum of all 16 rows' latched accumulators in one cycle. When `i_reduce=0`, behavior is **bit-identical** to today.

**Files:**
- Modify: `rtl/pe_core.v`

- [ ] **Step 1: Add the port**

```verilog
    input  wire                     i_reduce     // GEMM spatial-reduce: sum down column
```

- [ ] **Step 2: Guard the shift-load so reduce mode holds psum_shift**

In the `always` block, change the drain-load branch:

```verilog
            if (i_k_end) begin
                psum_shift <= psum_acc;
                psum_acc   <= {PSUM_WIDTH{1'b0}};
            end else if (i_drain_en && !i_reduce) begin
                psum_shift <= i_psum_casc;   // legacy shift (unchanged when reduce=0)
            end
            // reduce mode: psum_shift holds its k_end-latched value (combinational sum below)
```

- [ ] **Step 3: Reduce-aware cascade output**

```verilog
    assign o_psum_casc = i_drain_en
                       ? (i_reduce ? (psum_shift + i_psum_casc) : psum_shift)
                       : {PSUM_WIDTH{1'b0}};
```

- [ ] **Step 4: Build (compile only) — verify conv path byte-identical**

Run: `cd /e/code/6-10/soc; bash run_all.sh clean >/dev/null 2>&1; bash run_all.sh sim 2>&1 | grep -E "Result:|SCORE_CHK|TESTS PASSED|Errors:"`
Expected: `10/10`, `SCORE_CHK=<golden>`, `Errors: 0`. (`i_reduce` is tied 0 everywhere until Task 6, so this proves the gate is inert.)

> Note: Steps here require `i_reduce` to be wired through gp_4x4/systolic (Task 4) before it compiles. If doing strictly task-by-task, fold Task 3+4 compile/verify into Task 4's run. Keep the commit boundaries.

- [ ] **Step 5: Commit**

```bash
git add rtl/pe_core.v
git commit -m "feat(npu): pe_core reduce-drain combinational column sum (gated)"
```

---

## Task 4: Thread `i_reduce` + 16×16 weight plane through gp_4x4 and systolic_16x16 (dormant)

The array's weight bus is currently 16 column-groups (broadcast to all rows). Reduce mode needs each PE(r,c) to receive a **distinct** weight (OC-c, IC-tile-r). Add a parallel wide input `i_wgt_plane` [ARRAY_ROWS*ARRAY_COLS*WGT_GROUP_W] selected by `i_reduce`; the legacy `i_wgt` path is untouched when `i_reduce=0`.

**Files:**
- Modify: `rtl/gp_4x4.v` (per-PE weight slicing; thread `i_reduce`)
- Modify: `rtl/systolic_16x16.v` (add `i_wgt_plane`, `i_reduce`; per-GP plane slice)

- [ ] **Step 1: systolic_16x16 — add ports**

```verilog
    input  wire [ARRAY_ROWS*ARRAY_COLS*WGT_GROUP_W-1:0] i_wgt_plane, // 16x16 per-PE weights (reduce mode)
    input  wire                                         i_reduce,
```

- [ ] **Step 2: systolic_16x16 — slice the plane per GP and pass both weight sources + i_reduce to gp_4x4**

For GP[gr][gc], the plane slice covers PE rows `gr*4..gr*4+3` × cols `gc*4..gc*4+3`. Pass `i_reduce` and the plane slice (flattened 4×4×128) plus the legacy `gp_wgt`. (Define the plane index as `((row*ARRAY_COLS)+col)*WGT_GROUP_W` so row-major matches the firmware pack order.)

- [ ] **Step 3: gp_4x4 — select per-PE weight**

In gp_4x4, for each PE(lr,lc): `pe_wgt = i_reduce ? plane[lr][lc] : i_wgt[lc]` (legacy broadcasts column weight to all rows). Thread `i_reduce` to each `pe_core` instance's new `i_reduce` port.

- [ ] **Step 4: npu_top — tie new inputs off (dormant)**

```verilog
    .i_wgt_plane (2048'd0 ... /* ARRAY_ROWS*ARRAY_COLS*128 zeros */),
    .i_reduce    (1'b0),
```

(Use a sized literal `{(ARRAY_ROWS*ARRAY_COLS*128){1'b0}}`.)

- [ ] **Step 5: Build + run — byte-identical**

Run: `cd /e/code/6-10/soc; bash run_all.sh clean >/dev/null 2>&1; bash run_all.sh sim 2>&1 | grep -E "Result:|SCORE_CHK|TESTS PASSED|Errors:"`
Expected: `10/10`, `SCORE_CHK=<golden>`, `Errors: 0`.

- [ ] **Step 6: Commit**

```bash
git add rtl/gp_4x4.v rtl/systolic_16x16.v rtl/npu_top.v
git commit -m "feat(npu): 16x16 weight-plane + i_reduce wiring through array (dormant)"
```

---

## Task 5: wgt_reader — hold 16 IC-tiles for GEMM and expose the 16×16 plane (dormant)

In legacy GEMM, `wgt_buf[0][oc][icg]` holds `ICG_BUF=4` IC-tiles. Reduce mode needs 16 IC-tiles resident per super-step. Two options — pick the simpler that fits:
- **(A) prefetch 16 IC-tiles per super-step** into a dedicated `gemm_plane[oc][0..15]` register (256 words), refilled each super-step.
- **(B) raise `ICG_BUF`** for GEMM to hold more tiles.

Recommended: a dedicated `o_wgt_plane` output `[ARRAY_ROWS*ARRAY_COLS*WGT_GROUP_W]` where `plane[ic_tile r][oc c] = gemm weight word(oc=c, icg = superstep*16 + r)`. Packing already stores `word(o,g)=o*icg+g` (Task 8 confirms), so this is a re-index of the existing prefetch buffer, not a new packing.

**Files:**
- Modify: `rtl/wgt_reader.v`

- [ ] **Step 1: Add a GEMM plane prefetch path** that, on a super-step trigger, reads 16 IC-tiles × 16 OC = 256 words from Wgt SRAM into `gemm_plane`, and outputs them as `o_wgt_plane`. Gate entirely under a new `i_gemm_reduce` input so legacy paths are untouched.

- [ ] **Step 2: Add ports** `input i_gemm_reduce`, `input [3:0] i_superstep`, `output [ARRAY_ROWS*ARRAY_COLS*WGT_GROUP_W-1:0] o_wgt_plane`, `output o_plane_done`.

- [ ] **Step 3: npu_top — tie `i_gemm_reduce=0`, leave `o_wgt_plane` feeding the array's `i_wgt_plane` (still inert because `i_reduce=0`).**

- [ ] **Step 4: Build + run — byte-identical**

Run: same sim command. Expected `10/10`, `SCORE_CHK=<golden>`, `Errors: 0`.

- [ ] **Step 5: Commit**

```bash
git add rtl/wgt_reader.v rtl/npu_top.v
git commit -m "feat(npu): wgt_reader GEMM 16x16 plane prefetch (dormant)"
```

---

## Task 6: FSM + npu_top super-step sequencer for `gemm_reduce` (activate the act/weight feed)

**Files:**
- Modify: `rtl/top_controller_fsm.v` (super-step loop; act row-load addressing; reduce-drain timing)
- Modify: `rtl/npu_top.v` (16×128 act row register; mux act window + i_reduce when `cfg_gemm_reduce`)

- [ ] **Step 1: npu_top — act row register.** When `cfg_gemm_reduce`, the FSM streams 16 Act-SRAM read addresses (IC-tiles `superstep*16 + r`, r=0..15) over 16 cycles into a `reg [127:0] act_row[0:15]`; present `{act_row[15],...,act_row[0]}` (2048b) as the array `i_act` for the 1 vld cycle of the super-step. Mux: `i_act = cfg_gemm_reduce ? act_row_bus : (cfg_gemm_en ? {16{act_sram_doa}} : im2col_act_window)`.

- [ ] **Step 2: npu_top — drive `i_reduce = cfg_gemm_reduce` and `i_wgt_plane = wgt_reader o_wgt_plane`; weight prefetch `i_gemm_reduce = cfg_gemm_reduce`.**

- [ ] **Step 3: FSM super-step loop.** Add states (or extend the GEMM branch): for each super-step `s` in `0..ceil(IC/256)-1`: trigger weight-plane prefetch (256 cyc), load 16 act words (16 cyc), assert `o_array_vld` for 1 cycle (each PE row accumulates its IC-tile; **do NOT** clear psum between super-steps — only the final `S_K_END` latches). After the last super-step: `S_K_END` → `S_DRAIN` with reduce: assert `o_array_drain_en` for **1 cycle** (combinational column sum is ready immediately), capture, → `S_POST`.

- [ ] **Step 4: post capture — single valid.** In reduce mode the 16-column result is valid for 1 drain cycle; ensure `pp_input_vld` pulses once and post_process writes 16 OC results (columns), not 16 serial rows. (The post path already consumes `array_psum_col`; reduce just changes the count of valid cycles from 16 to 1.)

- [ ] **Step 5: Bring up on FC1 only.** Firmware Task 8 sets `gemm_reduce` for FC1. Build + run.

Run: same sim command.
Expected: `10/10`, `SCORE_CHK=<golden>`, `Errors: 0`. If SCORE_CHK differs → debug with `superpowers:systematic-debugging` (likely act/weight index transpose or off-by-one in super-step accumulation). Use `$display` traces on `array_psum_col` at the reduce-drain cycle vs the legacy-GEMM golden.

- [ ] **Step 6: Commit**

```bash
git add rtl/top_controller_fsm.v rtl/npu_top.v
git commit -m "feat(npu): gemm_reduce super-step sequencer + reduce-drain capture"
```

---

## Task 7: Profile FC1 cycle drop + array-util check

**Files:**
- Modify: `firmware/deepnet_deploy.c` (temporarily `NPU_PROFILE 1`)

- [ ] **Step 1:** Set `#define NPU_PROFILE 1`, `distclean`, sim, read `fc1(gemm): N total /img M`. Expect M well below 19,638 (target ~5–10K). Revert `NPU_PROFILE 0`.
- [ ] **Step 2:** (optional) add a tb array-util counter over the GEMM compute window to confirm 6.25% → ~100%.
- [ ] **Step 3: Commit** any probe tweaks (profile reverted to 0).

---

## Task 8: Firmware — enable reduce for FC1 + confirm weight layout

**Files:**
- Modify: `firmware/deepnet_deploy.c` (`npu_gemm_pass` / FC1 call sets `NPU_CTRL_GEMM_REDUCE`)

- [ ] **Step 1:** In `npu_gemm_pass`, OR `NPU_CTRL_GEMM_REDUCE` into the CTRL write when a `reduce` arg is set; pass `reduce=1` from the FC1 deploy call only (FC2 stays CPU). Confirm `pack_fc_tile`'s `word(o,g)=o*icg+g` matches the plane index `plane[r][c]=word(c, s*16+r)` used in Task 5; if not, adjust the plane read index (NOT the packing).
- [ ] **Step 2: Build + run — final gate.**

Run: same sim command.
Expected: `10/10`, `SCORE_CHK=<golden>`, `Errors: 0`.

- [ ] **Step 3: Commit**

```bash
git add firmware/deepnet_deploy.c
git commit -m "feat(npu): enable gemm_reduce for FC1 (decision M)"
```

---

## Task 9: Document the decision + update memory

- [ ] **Step 1:** Add "Decision M: GEMM array utilization (gemm_reduce, CTRL[10])" to `CLAUDE.md` Architecture Decisions, with the register-map row for CTRL[10] and the measured FC1 before/after + full-run cycles + `SCORE_CHK` gate value.
- [ ] **Step 2:** Update `docs/superpowers/specs/OPTIMIZATION-BACKLOG.md`: mark gemm-array-util done; correct the stale "FC1 ≈ 55K" premise (Phase 0 found 19.6K); record the new baseline.
- [ ] **Step 3:** Update memory: `soc-npu-gemm-mode.md` (reduce mode), `project_soc_simulation.md` (new cycle counts), add a `feedback`/`reference` note that the spec's FC1 estimate was wrong (Phase 0 lesson). One-line pointers in `MEMORY.md`.
- [ ] **Step 4: Commit** the docs.

---

## Risk / rollback

- Every step before Task 6 is dormant and must keep `SCORE_CHK=<golden>` — if any dormant step changes it, the gating is leaky; stop and fix before proceeding.
- The deepest risk is Task 6 (compute topology). If FC1 reduce can't reach bit-identical within reasonable debugging, **revert Tasks 6/8** (keep the dormant plumbing) and ship the Phase-0 FC2 win alone — the mode bit stays off, conv + legacy GEMM unaffected.
- Conv path is never on the `gemm_reduce` path; the conv SCORE contribution must never move.
