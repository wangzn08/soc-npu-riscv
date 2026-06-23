# Conv large-IC streaming via INT32 psum accumulation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make any im2col conv with `ic_groups > ICG_MAX` produce bit-identical output to a single-pass conv, by streaming IC in chunks of `<= ICG_MAX` tiles and accumulating INT32 partial sums in an Out-SRAM psum region across chunks, applying bias+requant+SiLU only on the last chunk.

**Architecture:** IC-chunk becomes the OUTER loop (OC tiled to 16, oc_single OFF). Each chunk's partial conv is post-processed in one of three accumulate modes — ACC_FIRST (write INT32), ACC_ADD (read+add+write INT32), ACC_FINAL (read+add+bias+requant+SiLU → INT8). The cross-chunk accumulator lives in an Out-SRAM INT32 region (4×128b words per OC-group per position, like int32_out). OFF (icg<=ICG_MAX) is byte-identical to today.

**Tech Stack:** Verilog-2001 RTL (PicoRV32+NPU SoC), ModelSim/Questa sim (`run_all.sh`), C firmware (riscv-none-elf-gcc). No unit-test framework — "tests" are directed ModelSim testbenches and the stage-checksum integration smoke.

**Spec:** `docs/superpowers/specs/2026-06-22-conv-ic-streaming-psum-design.md`. Read it first.

## STATUS: FEATURE COMPLETE (2026-06-22) — via CPU accumulate, not HW psum
The large-IC 3x3 conv works and is verified (c2f8 exact maxdiff=0, c2f4/c2f6/MNIST
regress clean). It was implemented via **CPU INT32-psum accumulation**, NOT the
HW Out-SRAM-psum readback this plan specced: hands-on, the HW readback must align
4 INT32 words to the post_process INPUT stage across the row_par/multi-cycle drain
(high-risk timing), whereas CPU accumulate needed only a tiny HW change (int32
sequencer i32_base x4 when cfg_ic_stream) + firmware. See commit c03f2b1 and
docs/notes/RESUME-yolov8n-soc.md "3x3 LARGE-IC RESOLUTION". The HW-accumulate
primitives below (Tasks 2-5: post_process ACC modes, acc_mode/psum_rd_base regs,
CTRL[23]) are committed and OFF-by-default, available as a future perf optimization.
The original task breakdown is retained below as the design-of-record for that
optimization.

## STATUS: COMPLETE (2026-06-22) — via CPU-accumulate, not the HW-accumulate spec
The large-IC 3x3 feature is DONE and verified, but implemented with a SIMPLER, lower-risk
mechanism than this plan/spec proposed: **CPU INT32-psum accumulate** instead of HW
post_process accumulate + Out-SRAM readback. Reason: the HW readback timing (4-word INT32
aligned to the post_process input across the row_par-style drain) was high-risk; the deep
layers needing large-IC 3x3 are small-spatial, so CPU accumulation is cheap and reliable.
Implementation: `firmware/yolo_run_conv2d_ic_stream` (yolo_ops.c) + `i32_base` x4 in
npu_top (commit dbe7fc2). The HW-accumulate primitives below (Tasks 2/5 acc_mode,
NPU_ACC_MODE/PSUM_RD_BASE, post_process ACC modes + TB) are committed but UNUSED — they
remain as scaffolding if a future pass wants to move the accumulate into HW for speed.
Verified: c2f8 exact maxdiff=0, c2f4 maxdiff=0, c2f6 PASS, MNIST 10/10 941,155.
Commits: 6772ff6, 74e4224, 0a27a20, 72f104f, dbe7fc2, c03f2b1.

## PROGRESS (2026-06-22, update before resuming)
- **Task 1 DONE** (commit `6772ff6`): IC-group counters widened (`pf_icg`→[4:0] etc.); MNIST 941,155 byte-identical. NOTE the implementer flagged: `load_tile`/`o_im2col_load_tile` (FSM, [3:0]) were NOT widened — widen them WITH the im2col consumer in Task 5 if streaming needs >16 within-column tiles (3x3 chunk tiles are <=ICG_MAX=4, so likely fine).
- **Task 2 DONE** (commit `74e4224`): `rtl/post_process_top.v` has `i_acc_mode[1:0]` (0=NONE/1=FIRST/2=ADD/3=FINAL) + `i_psum_readback`; directed TB `tests/tb_post_process_acc.v` PASS; MNIST byte-identical. In `npu_top.v` these are currently tied off: `.i_acc_mode(2'd0)`, `.i_psum_readback({(16*32){1'b0}})` (~line 1051) — Task 5 replaces with FSM-driven values.
- **Task 5 scaffolding DONE** (this session): `CTRL[23] ic_stream` plumbed — `o_ic_stream` in `param_regfile.v`, `cfg_ic_stream` in `npu_top.v` routed to FSM `i_ic_stream` (currently unused → OFF identical), `NPU_CTRL_IC_STREAM (1<<23)` in `firmware.h`. **CTRL bit is 23, NOT 18** (18-22 are silu_en/silu_requant_en/elt_signed/sigmoid_en/silu_exact_en).
- **MECHANISM CONFIRMED = Out-SRAM INT32 psum (spec approach), NOT array-accumulation.** Hands-on analysis: array-internal accumulation across IC chunks would force per-position line-buffer reloads (loses column-slide reuse) → slower. The Out-SRAM-psum approach keeps strip line-buffer reuse and pays INT32 Out-SRAM traffic — it is the performant one. Task 2's acc modes are the right primitive.
- **npu_top integration map** for the remaining work: out-write mux at `rtl/npu_top.v:1266-1278` (cfg_gpool/int32/pool/row_par/fsm); INT32 write sequencer `~1175-1218` (FC-only today: `!i32_active` gate drops back-to-back, `i32_base=fsm_out_wr_addr` not ×4-spaced — Task 3 fixes); post_process inst `~1013-1052`; Out SRAM `sdp_bram` Port A has read `doa` (Task 4 readback).
- **Remaining: Tasks 3, 4, 5(core), 6, 7.** Gate = `yolo_c2f8_320_exact_smoke.c` (CV1/ADD0/CONCAT/OUT all OK, ~27s) + c2f6/c2f4/MNIST regressions.

**Baseline to preserve:** MNIST `bash run_all.sh sim` = 10/10, TRAP 941,155 cyc. Every task's OFF-path (icg<=ICG_MAX, acc_mode=NONE) must keep this byte-identical.

**Build/run reminders (from CLAUDE.md / RESUME):** Git Bash; prefix Bash-tool commands with `builtin cd /e/code/6-10/soc;`. Remove `.yolo_ddr` before MNIST. New RTL files go in `axi_sys.f`. Register-map changes touch both `rtl/param_regfile.v` and `firmware/firmware.h`. After editing RTL run `bash run_all.sh clean` before recompile. Long sims run in background. Isolation smoke `yolo_c2f8_320_exact_smoke.c` is ~27s.

---

## File Structure

- `rtl/post_process_top.v` — add `i_acc_mode[1:0]` (NONE/FIRST/ADD/FINAL); ACC_FINAL reuses s1/s2/SiLU with `psum_in += psum_readback`; ACC_FIRST/ADD emit raw INT32. (~existing s1/s2 stages.)
- `rtl/npu_top.v` — generalize the INT32 write sequencer for spatial (×4 address spacing, position-aware, no dropped pulses); wire the psum-region readback (Out SRAM Port A `doa`) into post_process; route `i_acc_mode` from FSM.
- `rtl/top_controller_fsm.v` — IC-chunk OUTER loop when `ic_groups > ICG_MAX`; drive `o_acc_mode`; widen IC-group counters; psum-region base addressing.
- `rtl/im2col_line_buffer.v` — only relevant if NOT streaming (kept at ICG_MAX=4; streaming loads <=ICG_MAX tiles per chunk, so the window stays within ICG_MAX). No depth change needed.
- `firmware/yolo_ops.c` — streamed-conv arm in `yolo_run_conv2d_tiled` (icg>ICG_BUF, non-1x1): tile OC<=16, drop oc_single, drive IC-chunk via the new HW mode.
- `firmware/firmware.h` — new CTRL/config bit(s) for ic-stream/acc-mode.
- `tests/tb_npu_integ.v` (extend) or new `tests/tb_ic_stream_conv.v` — directed 3x3 icg=8/16 golden check.

**Key principle:** the FSM loops IC chunks in HW (driven by `ic_groups` + a CTRL enable); the firmware only sets dims + the enable, exactly like oc_single. This keeps per-layer MMIO low and the dataflow in one place.

---

## Task 1: Widen IC-group counters (foundation, OFF-path identical)

**Files:**
- Modify: `rtl/top_controller_fsm.v` (`ic_groups`, `o_wgt_ic_group`, `o_wgt_ic_sel`, `ic_tile` selects)
- Modify: `rtl/wgt_reader.v` (`i_ic_group`, `pf_icg`, `i_ic_groups_total` uses)

- [ ] **Step 1:** Audit every `[3:0]` / 4-bit IC-group field. Grep:

Run: `builtin cd /e/code/6-10/soc; grep -nE "ic_group|pf_icg|ic_sel|ic_groups\[3:0\]|i_ic_groups" rtl/top_controller_fsm.v rtl/wgt_reader.v`
Expected: list of fields capped at 4 bits.

- [ ] **Step 2:** Widen `pf_icg`/`pf_icg_d` to `[4:0]` (or `$clog2(IC_GROUPS_MAX)`) in `rtl/wgt_reader.v`, and the `i_ic_group` port / `eff_icg` math to match. Keep `ICG_BUF` indexing correct (streaming uses index 0 only, so buffer depth unaffected).

- [ ] **Step 3:** In `rtl/top_controller_fsm.v`, ensure `o_wgt_ic_group` and the `ic_tile`→group conversions carry >4 bits (they are already `[9:0]`; verify no truncation to `[3:0]`). Fix the `load_tile >= ic_groups[3:0]-1` compare (LOAD_ROW) to use full-width `ic_groups`.

- [ ] **Step 4:** Build + MNIST regression (OFF-path identical).

Run: `builtin cd /e/code/6-10/soc; rm -f .yolo_ddr; bash run_all.sh clean >/dev/null; bash run_all.sh sim >/dev/null; tail -6 transcript`
Expected: `=== Result: 10/10 correct ===`, `TRAP after 941155 clock cycles`.

- [ ] **Step 5:** Commit.

```bash
git add rtl/top_controller_fsm.v rtl/wgt_reader.v
git commit -m "rtl(npu): widen IC-group counters for >16 IC groups (OFF-path identical)"
```

---

## Task 2: post_process accumulate modes (ACC_FIRST/ADD/FINAL)

**Files:**
- Modify: `rtl/post_process_top.v` (add `i_acc_mode[1:0]`, `i_psum_readback[NUM_OC*PSUM_WIDTH]`; ACC paths)
- Test: `tests/tb_post_process_pool.v` pattern → new `tests/tb_post_process_acc.v`

- [ ] **Step 1: Write the failing TB** `tests/tb_post_process_acc.v`: drive a known `i_psum`, `i_bias`, `i_scale_mul/shift`, SiLU LUT, and:
  - ACC_FIRST: expect `o_feat32 == i_psum` (raw, no bias).
  - ACC_ADD with `i_psum_readback=R`: expect `o_feat32 == R + i_psum`.
  - ACC_FINAL with readback R: expect `o_feat == SiLU_LUT(clamp(((R+i_psum+bias)*scale)>>shift))` (== legacy single-pass with full psum).
  Golden computed in the TB (mirror `gen_yolo_c2f_exact.py conv_exact`).

- [ ] **Step 2: Run TB, verify it fails** (port `i_acc_mode` undefined).

Run: `builtin cd /e/code/6-10/soc; vlib sim/work_pp; vlog -work sim/work_pp rtl/post_process_top.v tests/tb_post_process_acc.v; vsim -c -lib sim/work_pp tb_post_process_acc -do "run -all; quit -f"`
Expected: compile error / FAIL.

- [ ] **Step 3: Implement** in `rtl/post_process_top.v`:
  - Add inputs `i_acc_mode[1:0]` (0=NONE legacy, 1=FIRST, 2=ADD, 3=FINAL) and `i_psum_readback`.
  - In Stage 1: `s1_in = i_psum + ((acc_mode==ADD||acc_mode==FINAL) ? i_psum_readback : 0)`; bias added only when `acc_mode==NONE||FINAL`.
  - `o_feat32` (the INT32 output) = `s1_in` for FIRST/ADD (raw accumulate, pre-bias for FIRST/ADD — NOTE bias only at FINAL); for FINAL run the normal s2/SiLU on `s1_in + bias`.
  - Gate: `acc_mode==NONE` ⇒ every new term is 0/unused ⇒ byte-identical to current.

- [ ] **Step 4: Run TB, verify PASS** (same vsim command). Expected: all three modes match golden.

- [ ] **Step 5:** Commit.

```bash
git add rtl/post_process_top.v tests/tb_post_process_acc.v
git commit -m "rtl(post_process): INT32 psum accumulate modes (FIRST/ADD/FINAL); TB"
```

---

## Task 3: INT32 spatial write sequencer (multi-position)

**Files:**
- Modify: `rtl/npu_top.v` (INT32 write sequencer ~lines 1160-1218)

- [ ] **Step 1: Write failing check** — extend `tests/tb_npu_integ.v` (or a small TB) to run a 2-position int32-out write and read back 2×(4 words), expecting both positions present at ×4-spaced addresses (today the 2nd is dropped).

- [ ] **Step 2: Run, verify fail** (2nd position missing/collided).

Run: `builtin cd /e/code/6-10/soc; bash run_all.sh sim yolo_c2f8_320_exact_smoke.c yolo_c2f.c yolo_ops.c >/dev/null; grep -aE "ck " transcript` (proxy until TB exists)

- [ ] **Step 3: Implement** in `rtl/npu_top.v`:
  - Make `i32_base` track the output position: `i32_base = out_wr_addr_int32` where the FSM provides a ×4-spaced psum address (Task 5 supplies it). Until then, derive `i32_base = fsm_out_wr_addr << 2` in int32 mode.
  - Allow a new capture when `i32_active` is finishing (`i32_last_write`) so back-to-back position pulses (>=4 cyc apart in conv) are not dropped; if a pulse arrives while busy (shouldn't in conv), assert an error/assert.
  - Keep `cfg_int32_out` OFF ⇒ unchanged.

- [ ] **Step 4: Run** the c2f8 smoke as proxy; later validated end-to-end in Task 7. Confirm no regression to MNIST.

Run: `builtin cd /e/code/6-10/soc; rm -f .yolo_ddr; bash run_all.sh clean >/dev/null; bash run_all.sh sim >/dev/null; tail -4 transcript`
Expected: MNIST 10/10 941,155 cyc.

- [ ] **Step 5:** Commit.

```bash
git add rtl/npu_top.v
git commit -m "rtl(npu): spatial-capable INT32 write sequencer (x4 spacing, no dropped pulses)"
```

---

## Task 4: Out-SRAM INT32 psum readback into post_process

**Files:**
- Modify: `rtl/npu_top.v` (Out SRAM Port A read `doa` → `pp_psum_readback`; pipeline align)

- [ ] **Step 1:** Confirm `sdp_bram` Port A read timing (registered `doa` vs COMB). Read `rtl/sram_models.v` `sdp_bram` and note latency.

Run: `builtin cd /e/code/6-10/soc; grep -nE "doa|COMB|always|assign" rtl/sram_models.v | head -30`

- [ ] **Step 2:** Wire a readback path: in ACC_ADD/FINAL, issue an Out-SRAM Port-A read of the psum word(s) for the current position one stage ahead of post_process, deserialize 4×128b → `pp_psum_readback[NUM_OC*32]`, align to `i_psum_vld`. Add the read-address generator keyed off the FSM's psum base + position (Task 5).

- [ ] **Step 3:** Build (no functional gate yet without FSM); ensure compiles and MNIST identical (ACC OFF).

Run: `builtin cd /e/code/6-10/soc; rm -f .yolo_ddr; bash run_all.sh clean >/dev/null; bash run_all.sh sim >/dev/null; tail -4 transcript`
Expected: 10/10 941,155 cyc.

- [ ] **Step 4:** Commit.

```bash
git add rtl/npu_top.v
git commit -m "rtl(npu): Out-SRAM INT32 psum readback feeding post_process ACC modes"
```

---

## Task 5: FSM IC-chunk outer loop + acc_mode (core)

**Files:**
- Modify: `rtl/top_controller_fsm.v` (new `ic_stream` enable, IC-chunk loop, `o_acc_mode`, psum base)
- Modify: `firmware/firmware.h` + `rtl/param_regfile.v` (CTRL bit `NPU_CTRL_IC_STREAM`)

- [x] **Step 1: DONE** — `CTRL[23] ic_stream` plumbed (param_regfile `o_ic_stream`, npu_top `cfg_ic_stream`→FSM `i_ic_stream`, `NPU_CTRL_IC_STREAM (1<<23)`). Bit is 23 not 18. Document in CLAUDE.md CTRL list (later).

- [ ] **Step 2:** In `rtl/top_controller_fsm.v`, when `i_ic_stream`:
  - Compute `ic_chunks = ceil(ic_groups / ICG_MAX)`, `chunk_tiles = min(ICG_MAX, remaining)`.
  - Restructure so an outer `ic_chunk` index wraps the existing spatial sweep: per chunk, S_LOAD_ROW loads only this chunk's `chunk_tiles` IC tiles; CALC iterates those tiles; S_POST drives `o_acc_mode` = FIRST (chunk 0) / ADD (middle) / FINAL (last).
  - The S_CALC_KERNEL IC-tile loop bound becomes `chunk_tiles` (not full ic_groups); the array does NOT drain between chunks of different positions — it drains per position per chunk, accumulation is in Out SRAM.
  - Provide `o_out_psum_base` (×4-spaced) for the int32 write (Task 3) and readback (Task 4).
  - oc_single forced OFF in ic_stream (caller guarantees OC<=16 tiles).

- [ ] **Step 3:** Directed TB: extend `tests/tb_npu_integ.v` with a 3x3 conv, icg=8, ic_stream=1, golden-checked vs a CPU reference in the TB. Expect bit-exact.

Run: `builtin cd /e/code/6-10/soc; <vsim tb_npu_integ command per its header>`
Expected: PASS.

- [ ] **Step 4:** MNIST regression (ic_stream OFF ⇒ identical).

Run: `builtin cd /e/code/6-10/soc; rm -f .yolo_ddr; bash run_all.sh clean >/dev/null; bash run_all.sh sim >/dev/null; tail -4 transcript`
Expected: 10/10 941,155 cyc.

- [ ] **Step 5:** Commit.

```bash
git add rtl/top_controller_fsm.v rtl/param_regfile.v firmware/firmware.h tests/tb_npu_integ.v
git commit -m "rtl(npu): FSM IC-chunk streaming loop + acc_mode for large-IC convs"
```

---

## Task 6: Firmware streamed-conv arm

**Files:**
- Modify: `firmware/yolo_ops.c` (`yolo_run_conv2d_tiled`, the non-1x1 branch)

- [ ] **Step 1:** In the OC-chunk loop, add `stream_conv = (!is_pw) && (icg > YOLO_ICG_BUF_MAX_RESIDENT)` where the resident 3x3 cap is `ICG_MAX` (window) — i.e. stream when `icg > ICG_MAX`. For streamed conv: tile OC<=16, drop `NPU_CTRL_OC_SINGLE`, add `NPU_CTRL_IC_STREAM`, mask `NPU_CTRL_ROW_PAR` (HW loops IC chunks internally; row_par interaction deferred). Provide the psum Out-SRAM base via existing out-addr (HW computes ×4).

```c
uint32_t stream_conv = (!is_pw) && (icg > YOLO_ICG_MAX);   /* im2col window limit */
...
if (stream_conv) {
    uint32_t cf = (ctrl_flags & ~NPU_CTRL_ROW_PAR) | NPU_CTRL_IC_STREAM;
    if (!yolo_run_conv2d_qparams_pads(0u, wgt_base, 0u, in_w, strip_in_h, in_c, chunk,
                                      kernel_h, kernel_w, stride, 0u, pad,
                                      bias+done, scale_mul+done, scale_shift+done, cf))
        return 0;
}
```
Add `#define YOLO_ICG_MAX 4u` mirror (with comment to keep in sync with `im2col_line_buffer.v ICG_MAX`).

- [ ] **Step 2:** Ensure chunk cap for streamed conv = 16 (one OC tile), like PW stream.

- [ ] **Step 3:** Build firmware only.

Run: `builtin cd /e/code/6-10/soc; bash run_all.sh fw 2>&1 | tail -5`
Expected: warning-clean compile.

- [ ] **Step 4:** Commit.

```bash
git add firmware/yolo_ops.c
git commit -m "fw(yolo): stream large-IC 3x3 conv (IC_STREAM, OC<=16, no oc_single)"
```

---

## Task 7: Integration — c2f8 exact + regressions

**Files:** none (validation only)

- [ ] **Step 1:** Run c2f8 exact stage-checksum smoke.

Run: `builtin cd /e/code/6-10/soc; rm -f .yolo_ddr; bash run_all.sh clean >/dev/null; bash run_all.sh sim yolo_c2f8_320_exact_smoke.c yolo_c2f.c yolo_ops.c >/dev/null; grep -aE "ck |maxdiff|STANDALONE" transcript`
Expected: `[ck CV1] ... OK`, `[ck ADD0] ... OK`, `[ck CONCAT] ... OK`, `[ck OUT] ... OK`, `STANDALONE PASS`.

- [ ] **Step 2:** Regress c2f6 (icg16, must still PASS) and c2f4 (exact maxdiff=0).

Run: `builtin cd /e/code/6-10/soc; bash run_all.sh sim yolo_c2f6_smoke.c yolo_c2f.c yolo_ops.c >/dev/null; grep -a "C2F6.*PASS" transcript; bash run_all.sh sim yolo_c2f4_320_exact_smoke.c yolo_c2f.c yolo_ops.c >/dev/null; grep -a "maxdiff" transcript`
Expected: c2f6 PASS; c2f4 maxdiff=0.

- [ ] **Step 3:** MNIST baseline.

Run: `builtin cd /e/code/6-10/soc; rm -f .yolo_ddr; bash run_all.sh clean >/dev/null; bash run_all.sh sim >/dev/null; tail -4 transcript`
Expected: 10/10, 941,155 cyc.

- [ ] **Step 4:** Update docs: `CLAUDE.md` (CTRL[18] ic_stream, large-IC conv note), `docs/notes/RESUME-yolov8n-soc.md` (c2f8 SOLVED), memories `[[soc-npu-weight-reuse]]` / `[[project_yolov8n_soc_resume]]`.

- [ ] **Step 5:** Commit.

```bash
git add CLAUDE.md docs/notes/RESUME-yolov8n-soc.md
git commit -m "docs: large-IC conv streaming (CTRL[18]); c2f8 RTL-clean"
```

---

## Self-Review notes

- **Spec coverage:** FSM IC-chunk loop (T5), psum Out-SRAM region + INT32 write (T3) + readback (T4), post_process ACC modes (T2), counter widths (T1), firmware arm (T6), tests (T2/T5/T7), OFF byte-identical gate (every task Step 4). Covered.
- **Open sub-decision from spec** (psum in Out SRAM vs dedicated SRAM): resolved here to reuse Out SRAM (T3/T4) with OC<=16 + bounded strip; if Out SRAM proves too small for a target strip, add a Psum SRAM as a follow-up (does not change the FSM/post_process interfaces).
- **Risk order:** T2 (isolated, TB-gated) and T3 are lowest risk; T5 is the highest-risk integration — do it with the directed TB before the c2f8 smoke. row_par + ic_stream is explicitly deferred (masked in T6).
- **Counter-width caveat:** YOLOv8n 3x3 max icg=16; T1 must cover at least 16 (5-bit). 1x1 PW large IC (icg up to 32) is already handled by the committed weight-streaming path (no im2col).
