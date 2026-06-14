# NPU 16-Row Spatial Parallelism (Task E) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 16×16 systolic array compute **16 adjacent output pixels in parallel** (one per array row) per K_END instead of replicating one window to all 16 rows, raising conv utilization from 1/16 toward full and cutting NPU conv cost (~2.15M → ~0.3–0.5M cycles).

**Architecture:** Add a per-start mode bit `row_par_en` (CTRL[9]). In that mode, `im2col_line_buffer` stops replicating one window and instead presents, for the current kernel offset and IC tile, **16 consecutive line-buffer column slices** as the 16 array-row activation groups. The FSM advances the spatial sweep by **groups of ≤16 output columns**. The drained 16-pixel stream (which emerges **row 15 first → row 0 last**, i.e. reverse column order) is post-processed and written to Out SRAM at the correct per-pixel addresses. Bring up incrementally behind the mode bit: non-pool layer first (Conv1), then the pool path (Conv2), then migrate hardware padding into im2col.

**Tech Stack:** Verilog-2001 RTL (ModelSim/Questa), rv32imc firmware (C). No unit-test framework — "the test" is one simulation run via `bash run_all.sh sim`; the gate is **byte-for-byte identical output, 10/10 MNIST, `ALL TESTS PASSED`**.

---

## CRITICAL FINDING — spec correction

The design spec (`docs/superpowers/specs/2026-06-12-npu-row-parallel-design.md`) states `post_process_top`, `max_pooling_2x2`, and the Out-SRAM write are "largely unchanged." **This is wrong and must not be assumed.** Reading the RTL:

1. **The current design deliberately collapses the 16 identical drain outputs into ONE written value.**
   - Pool path: `post_process_top.v:157` latches `pool_gated_data` only on the **first** `s3_vld` of the drain (`!pool_data_latched`), and feeds the pooler exactly once per conv point (`pool_gated_vld = i_in_post && !in_post_d`, the S_POST-entry edge). So 15 of the 16 drained values are discarded by design.
   - Non-pool path: `npu_top.v:794-798` drives a single FSM-timed Out-SRAM write (`fsm_out_wr_en` at one `fsm_out_wr_addr` = `cur_ox,cur_oy`). Only one of the 16 post-processed pixels is written.
   - This is correct **today** because all 16 rows hold identical psum (replicated act). Task E makes them **distinct**, so the drain→write path **must** be reworked to capture all `group_size` pixels.

2. **Drain order is reverse, confirmed from `pe_core.v:88-99`.** At `i_k_end`, `psum_shift <= psum_acc`; during `i_drain_en`, `psum_shift <= i_psum_casc` (loads from PE above; row 0 loads 0 from array top). `o_psum_col[c] = psum_shift[row15][c]`. So drain cycle `d` (0..15) emits the psum for **row `15 - d`**. Array row `r` is assigned output column `group_base + r`. Therefore the drained pixel stream is column `group_base+15` first, down to `group_base+0` last.

These two facts drive Tasks 6–8 (out-write rework) and Task 9 (pool path). They are the highest-risk part of E.

---

## File Structure

| File | Responsibility | Change type |
|------|----------------|-------------|
| `firmware/firmware.h` | `NPU_CTRL_ROW_PAR` define (CTRL[9]) | Modify |
| `rtl/param_regfile.v` | Decode CTRL[9] → `o_row_par_en` | Modify |
| `rtl/npu_top.v` | Wire `row_par_en` to FSM/im2col; row-parallel Out-SRAM write capture; later PAD-into-im2col migration | Modify |
| `rtl/im2col_line_buffer.v` | 16-wide combinational slice read in row-par mode (core rewrite); keep legacy path for `row_par_en=0` | Modify (additive) |
| `rtl/top_controller_fsm.v` | Group-of-16 spatial loop; per-group drain/write sequencing; new im2col group-base output | Modify |
| `firmware/deepnet_deploy.c` | Set `NPU_CTRL_ROW_PAR` per conv pass (staged: one layer, then all) | Modify |

**Reuse rule:** all changes are **additive behind `row_par_en`**. When `row_par_en=0` the datapath must be **bit-identical to the current build** (legacy `win`-shift path, single-pixel write). This guarantees a dormant checkpoint after every task and isolates regressions.

**Unchanged:** `systolic_16x16`, `gp_4x4`, `pe_core`, `wgt_reader`, `vector_alu`, SRAM wrappers, `axi_dma`. (Weights are shared across the 16 pixels — `wgt_reader` and the `i_wgt` broadcast are correct as-is.)

---

## Conventions for every task

- After editing any RTL file: `bash run_all.sh clean` before recompiling (per CLAUDE.md).
- Bilingual (Chinese/English) comments matching surrounding style.
- Firmware must stay warning-clean under strict CFLAGS (`-Werror -Wall -Wextra -pedantic`).
- The verification command is always: `bash run_all.sh sim` (run from repo root). Capture the console output.
- **Bit-identical gate** at each verification step: output identical to the pre-E build → `ALL TESTS PASSED.` and the same 10/10 predictions. Keep a saved golden log from the current `master` build for diffing (Task 0).
- Commit after every green task.

---

## Task 0: Capture golden baseline + create the work branch

**Files:** none (tooling only)

- [ ] **Step 1: Branch from current state**

```bash
git checkout -b feat/npu-row-parallel
```

- [ ] **Step 2: Run the current build and save the golden log**

Run: `bash run_all.sh sim 2>&1 | tee docs/notes/golden_pre_E.log`
Expected: log contains `ALL TESTS PASSED.` and the 10 per-image predictions with the current accuracy count. This file is the byte-for-byte reference for every later diff.

- [ ] **Step 3: Record the baseline NPU conv cycle count**

Run: `NPU_PROFILE=1 bash run_all.sh sim 2>&1 | tee docs/notes/golden_pre_E_prof.log`
Expected: a `prof_infer` / per-layer conv line near the documented ~2.15M conv, ~4.6M inference, ~10.9M full run. Note the numbers — Task 12 compares against them.

- [ ] **Step 4: Commit the baseline logs**

```bash
git add docs/notes/golden_pre_E.log docs/notes/golden_pre_E_prof.log
git commit -m "test(npu): capture golden pre-E baseline logs for bit-identical gate"
```

---

## Task 1: Add the `row_par_en` mode bit (CTRL[9]) — dormant

**Files:**
- Modify: `firmware/firmware.h:86` (after `NPU_CTRL_HW_PAD`)
- Modify: `rtl/param_regfile.v` (port `o_row_par_en`, reg `ctrl_row_par`, decode bit 9, readback)

- [ ] **Step 1: Add the firmware define**

In `firmware/firmware.h`, after the `NPU_CTRL_HW_PAD` line:

```c
#define NPU_CTRL_ROW_PAR    (1 << 9)   // 16-row spatial parallelism (task E): 16 output pixels/group
```

- [ ] **Step 2: Add the regfile output port**

In `rtl/param_regfile.v`, after the `o_hw_pad` output port (around line 85):

```verilog
    output wire                         o_row_par_en,    // CTRL[9]: 16-row spatial parallelism (task E)
```

- [ ] **Step 3: Add the control reg, reset, decode, readback**

After `reg ctrl_hw_pad;` (~line 172):
```verilog
    reg        ctrl_row_par;   // CTRL[9]: 16-row spatial parallelism
```
In the reset block, after `ctrl_hw_pad <= 1'b0;` (~line 281):
```verilog
            ctrl_row_par    <= 1'b0;    // row-parallel off by default
```
In the CTRL write decode, after `ctrl_hw_pad <= s_axi_wdata[8];` (~line 350):
```verilog
                        ctrl_row_par    <= s_axi_wdata[9];
```
Update the CTRL readback (line 455) to include bit 9:
```verilog
                    10'h00: rdata <= {22'd0, ctrl_row_par, ctrl_hw_pad, ctrl_gemm_en, ctrl_out_ping, ctrl_relu_en, ctrl_clear_done, ctrl_eltwise_en, ctrl_pool_en, ctrl_ping_pong, ctrl_start};
```
Add the output assignment near `assign o_hw_pad = ctrl_hw_pad;` (~line 537):
```verilog
    assign o_row_par_en   = ctrl_row_par;
```

- [ ] **Step 4: Wire the new port in npu_top (declare + connect, leave unused this task)**

In `rtl/npu_top.v`, add a wire near `cfg_hw_pad` (~line 129):
```verilog
    wire                            cfg_row_par_en;     // CTRL[9]: 16-row spatial parallelism
```
In the `u_param_regfile` instantiation, after `.o_hw_pad (cfg_hw_pad),`:
```verilog
        .o_row_par_en     (cfg_row_par_en),
```
Add `(* keep *)` is unnecessary — Verilog allows an unconnected wire. To avoid an "unused" lint and prove plumbing, temporarily reference it harmlessly is NOT needed; leave it declared. (It gets consumed in Task 4.)

- [ ] **Step 5: Compile and run — must be bit-identical (bit is dormant)**

Run: `bash run_all.sh clean && bash run_all.sh sim 2>&1 | tee docs/notes/t1.log`
Expected: `ALL TESTS PASSED.`; `diff docs/notes/golden_pre_E.log docs/notes/t1.log` shows only run-timestamp noise (predictions identical). The new bit is decoded but unused, so behavior is unchanged.

- [ ] **Step 6: Commit**

```bash
git add firmware/firmware.h rtl/param_regfile.v rtl/npu_top.v
git commit -m "feat(npu): add row_par_en mode bit (CTRL[9]), dormant"
```

---

## Task 2: Plumb `row_par_en` + group-base into the FSM and im2col (ports only, dormant)

**Files:**
- Modify: `rtl/top_controller_fsm.v` (new input `i_row_par_en`; new output `o_im2col_group_base`)
- Modify: `rtl/im2col_line_buffer.v` (new inputs `i_row_par_en`, `i_group_base`)
- Modify: `rtl/npu_top.v` (connect them)

- [ ] **Step 1: Add FSM ports**

In `top_controller_fsm.v` port list, near `i_hw_pad`:
```verilog
    input  wire                     i_row_par_en,   // 16-row spatial parallelism (task E)
```
Near the im2col control outputs (after `o_im2col_load_tile`):
```verilog
    output wire [15:0]              o_im2col_group_base,  // first output column of the current 16-wide group
```
Drive it from `cur_ox` for now (the group base IS `cur_ox` once the sweep advances by groups — Task 5):
```verilog
    assign o_im2col_group_base = cur_ox;
```

- [ ] **Step 2: Add im2col ports**

In `im2col_line_buffer.v` port list (after `i_offset_sel`):
```verilog
    input  wire                         i_row_par_en,   // task E: 16-wide slice mode
    input  wire [15:0]                  i_group_base,   // first output column of the 16-wide group
```

- [ ] **Step 3: Connect in npu_top**

In `u_controller` instantiation add `.i_row_par_en (cfg_row_par_en),` and a wire `fsm_im2col_group_base`:
```verilog
    wire [15:0] fsm_im2col_group_base;
```
`.o_im2col_group_base (fsm_im2col_group_base),`
In `u_im2col` instantiation add:
```verilog
        .i_row_par_en    (cfg_row_par_en),
        .i_group_base    (fsm_im2col_group_base),
```

- [ ] **Step 4: Compile and run — bit-identical (ports unused)**

Run: `bash run_all.sh clean && bash run_all.sh sim 2>&1 | tee docs/notes/t2.log`
Expected: `ALL TESTS PASSED.`; predictions identical to golden.

- [ ] **Step 5: Commit**

```bash
git add rtl/top_controller_fsm.v rtl/im2col_line_buffer.v rtl/npu_top.v
git commit -m "feat(npu): plumb row_par_en + group_base ports (dormant)"
```

---

## Task 3: im2col 16-wide combinational slice (row-par read path)

**Files:**
- Modify: `rtl/im2col_line_buffer.v` (add a parallel 16-row read; mux `o_act_window` on `i_row_par_en`)

**Design.** In legacy mode (`i_row_par_en=0`) keep the existing `selected_offset` replication **exactly** (the `gen_repl` block). In row-par mode, for array row `r` (0..15), output the line-buffer column slice for the current offset:

- Offset decodes to `(off_row_dec, off_col_dec)` ∈ {0,1,2}² (existing combinational decode, reuse it).
- The three row banks hold input rows `win_y, win_y+1, win_y+2` (padded coords); the existing `case(row_sel)` mapping (`cur_v/p1_v/p2_v` → bank0/1/2 by `row_sel`) tells which physical bank is the window's top/mid/bottom row. Reuse that mapping to pick **`bank_for_offrow(off_row_dec)`**.
- Column index for row `r` = `i_group_base + off_col_dec + r`. (Stride 1; window-left of output col `group_base+r` equals `group_base+r`, and `off_col_dec` selects within the 3-wide kernel.)
- Out of range → zero: when `col_idx >= i_width`, or when the needed bank is not yet valid (`valid_rows` < required), emit zero. (Top/bottom/left/right borders. With PAD still in the FSM for now, the line buffer already holds zero-injected border columns, so the only extra guard here is `col_idx >= i_width`.)

- [ ] **Step 1: Add a bank-select helper for the offset row (combinational)**

After the existing `off_row_dec`/`off_col_dec` decode (~line 233), add:
```verilog
    // Map the kernel-offset row (0=top,1=mid,2=bottom of the 3-row window) to a
    // physical line-buffer bank, mirroring the win[]-fill case(row_sel) mapping.
    // row_sel points at the most-recently-written (bottom) row.
    function [1:0] bank_for_offrow;
        input [1:0] off_r;       // 0=top, 1=mid, 2=bottom
        input [1:0] rsel;        // row_sel
        reg   [1:0] bottom, mid, top;
        begin
            bottom = rsel;                                  // newest row
            mid    = (rsel == 2'd0) ? 2'd2 : rsel - 2'd1;
            top    = (rsel == 2'd1) ? 2'd2 :
                     (rsel == 2'd0) ? 2'd1 : rsel - 2'd2;
            bank_for_offrow = (off_r == 2'd2) ? bottom :
                              (off_r == 2'd1) ? mid : top;
        end
    endfunction
```
Verify this mapping against `win[][][]` fill at lines 188-204: there `cur_v` (bottom, this row) = bank[row_sel], `p1_v` (mid) = bank[row_sel-1], `p2_v` (top) = bank[row_sel-2], with the same wrap. The function must reproduce that exactly — adjust if the bring-up waveform disagrees.

- [ ] **Step 2: Generate the 16-wide row-parallel window combinationally**

Replace the `gen_repl` generate block (lines 237-242) with a muxed version:
```verilog
    wire [1:0] rp_bank = bank_for_offrow(off_row_dec, row_sel);

    genvar gi;
    generate
        for (gi = 0; gi < ARRAY_ROWS; gi = gi + 1) begin : gen_window
            // Legacy (replicate one window) vs row-par (16 distinct columns)
            wire [15:0] rp_col = i_group_base + {14'd0, off_col_dec} + gi[15:0];
            reg  [ACT_GROUP_W-1:0] rp_val;
            always @(*) begin
                if (rp_col >= i_width) begin
                    rp_val = {ACT_GROUP_W{1'b0}};
                end else begin
                    case (rp_bank)
                        2'd0: rp_val = lb_bank0[rp_col[ADDR_W-1:0]][i_win_tile];
                        2'd1: rp_val = lb_bank1[rp_col[ADDR_W-1:0]][i_win_tile];
                        2'd2: rp_val = lb_bank2[rp_col[ADDR_W-1:0]][i_win_tile];
                        default: rp_val = {ACT_GROUP_W{1'b0}};
                    endcase
                end
            end
            assign o_act_window[gi*ACT_GROUP_W +: ACT_GROUP_W] =
                       i_row_par_en ? rp_val : selected_offset;
        end
    endgenerate
```
Note: this reads `lb_bankX` combinationally (a fan-out of the existing registers — no new storage, per spec). Also gate row/top validity: when `valid_rows < 3` and the selected bank corresponds to a not-yet-loaded row, the bank still holds its reset-zero contents (line buffer is zero-initialized at reset and on row rotation borders are zero-injected), so no extra guard is required for the bring-up layer; revisit if a layer reads a stale bank.

- [ ] **Step 3: Compile and run — bit-identical (row_par still 0 everywhere)**

Run: `bash run_all.sh clean && bash run_all.sh sim 2>&1 | tee docs/notes/t3.log`
Expected: `ALL TESTS PASSED.`; predictions identical to golden. (The mux selects `selected_offset` because firmware never sets the bit yet, and the moved-but-unchanged legacy assignment must be byte-identical.)

- [ ] **Step 4: Commit**

```bash
git add rtl/im2col_line_buffer.v
git commit -m "feat(npu): im2col 16-wide row-par slice read (mux behind row_par_en)"
```

---

## Task 4: FSM group-of-16 spatial loop (row-par mode)

**Files:**
- Modify: `rtl/top_controller_fsm.v` (group sizing; `cur_ox` advance by group; per-group drain still 16 cycles)

**Design.** Add `group_size = min(16, out_w - cur_ox)`. In `S_NEXT_TILE`, when `i_row_par_en`, advance `cur_ox` by `group_size` (and `cur_in_col` by `group_size*stride`) instead of 1. The K-step structure (offsets × IC-tiles), `S_K_END`, and the 16-cycle `S_DRAIN` are unchanged — the array now latches 16 distinct pixels in one K_END. The Out-write sequencing change is Task 6; this task only restructures the sweep so `group_base = cur_ox` is correct.

- [ ] **Step 1: Add group-size derivation**

After `out_row_stride` (~line 169):
```verilog
    // Row-parallel: number of valid output columns in the current 16-wide group
    wire [15:0] rp_remaining = out_w - cur_ox;
    wire [15:0] group_size   = (i_row_par_en && rp_remaining > 16'd16) ? 16'd16
                             : (i_row_par_en)                          ? rp_remaining
                             :                                          16'd1;
```

- [ ] **Step 2: Advance the sweep by group_size**

In `S_NEXT_TILE`, replace the column-advance branch (lines 468-474) with:
```verilog
                    if (cur_ox + group_size < out_w) begin
                        // Next group of columns in current row
                        cur_ox <= cur_ox + group_size;
                        cur_in_col <= cur_in_col + group_size * {8'd0, i_stride_sx};
                        state <= S_PREFETCH_WGT;
                        pf_wait_cnt <= 16'd0;
                    end else if (cur_oy + 16'd1 < out_h) begin
```
(For `i_row_par_en=0`, `group_size=1`, so this is identical to the original `cur_ox + 1 < out_w` logic.)

- [ ] **Step 3: Compile and run — bit-identical (row_par still 0)**

Run: `bash run_all.sh clean && bash run_all.sh sim 2>&1 | tee docs/notes/t4.log`
Expected: identical to golden. With `group_size=1` the FSM is unchanged.

- [ ] **Step 4: Commit**

```bash
git add rtl/top_controller_fsm.v
git commit -m "feat(npu): FSM group-of-16 spatial loop (group_size=1 when row_par off)"
```

---

## Task 5: Row-parallel Out-SRAM write capture — NON-POOL path

**Files:**
- Modify: `rtl/top_controller_fsm.v` (sequence `group_size` writes over the drained stream)
- Modify: `rtl/npu_top.v` (row-par write address counter; mux into Out-SRAM Port A for non-pool)

**Design.** The drained stream presents pixels in reverse: drain cycle `d` → output column `cur_ox + (group_size-1) - d_valid`, where only the **last `group_size` drained rows** (rows `0..group_size-1` = the valid pixels) are written. Because drain emits row 15 first, the valid pixels (rows `0..group_size-1`) come out **last** (drain cycles `16-group_size .. 15`). Account for the post-process pipeline latency (S1+S2+S3 = 3 cycles, +3 bypass = 6) by counting valids at the **post-process output** (`pp_feat_vld`) rather than at drain time.

Add a row-par write FSM in npu_top that, while `cfg_row_par_en && !cfg_pool_en`, captures each `pp_feat_vld` into Out SRAM at a decrementing column address, writing only the `group_size` valid pixels.

- [ ] **Step 1: Export group geometry from the FSM to npu_top**

In `top_controller_fsm.v`, add outputs:
```verilog
    output wire [15:0]              o_group_size,
    output wire [15:0]              o_group_base,   // = cur_ox
```
```verilog
    assign o_group_size = group_size;
    assign o_group_base = cur_ox;
```

- [ ] **Step 2: Row-par non-pool write counter in npu_top**

In `npu_top.v`, after the `pool_out_addr_cnt` block (~line 789), add:
```verilog
    // ---- Row-parallel non-pool Out-SRAM write sequencer ----
    // Drain emits row 15 first → row 0 last (pe_core). Valid pixels are rows
    // 0..group_size-1, i.e. the LAST group_size post-processed values of the
    // drain. We count pp_feat_vld pulses within a drain and map the k-th valid
    // (k=0..15) to output column = group_base + (15 - k) when that column is in
    // range (< group_base + group_size).
    wire [15:0] fsm_group_size;
    wire [15:0] fsm_group_base;
    reg  [4:0]  rp_vld_cnt;        // counts pp_feat_vld within a conv point
    reg         rp_active;         // 1 while draining a row-par group
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rp_vld_cnt <= 5'd0;
            rp_active  <= 1'b0;
        end else begin
            if (fsm_pp_start) begin           // S_POST entry: new conv point
                rp_vld_cnt <= 5'd0;
                rp_active  <= cfg_row_par_en & ~cfg_pool_en;
            end else if (rp_active && pp_feat_vld) begin
                rp_vld_cnt <= rp_vld_cnt + 5'd1;
            end
        end
    end
    wire [15:0] rp_col = fsm_group_base + (16'd15 - {11'd0, rp_vld_cnt});
    wire        rp_col_valid = rp_active && pp_feat_vld
                             && (rp_col >= fsm_group_base)
                             && (rp_col <  fsm_group_base + fsm_group_size);
    wire [SRAM_ADDR_W-1:0] rp_wr_addr = fsm_out_wr_addr  // base + oc + oy*stride + cur_ox ...
                             - {{(SRAM_ADDR_W-16){1'b0}}, fsm_group_base}      // strip cur_ox
                             + {{(SRAM_ADDR_W-16){1'b0}}, rp_col};             // add rp_col
```
Wire `fsm_group_size`/`fsm_group_base` from the new FSM outputs in `u_controller`.

> **Verification note for bring-up:** the exact alignment of `pp_feat_vld` to drain cycles (which drained rows have already flushed when the first `pp_feat_vld` arrives) MUST be confirmed on a waveform (Task 7). `rp_vld_cnt` counting **all** `pp_feat_vld` from S_POST entry, combined with the `rp_col in [base, base+group_size)` guard, makes the mapping robust to a constant pipeline offset as long as exactly 16 valids are produced per point. If the array emits valids only during the 16 drain cycles, k maps 1:1 to drain cycle; the guard drops the invalid high columns for partial groups.

- [ ] **Step 3: Mux the row-par write into Out-SRAM Port A (non-pool)**

Replace the non-pool Out-SRAM Port A assigns (lines 794-799) so row-par uses its own counter:
```verilog
    wire rp_wr_en = cfg_row_par_en & ~cfg_pool_en & rp_col_valid;
    assign out_sram_ena   = cfg_pool_en ? pp_feat_vld
                          : cfg_row_par_en ? rp_wr_en
                          : fsm_out_wr_en;
    assign out_sram_wea   = out_sram_ena;
    assign out_sram_addra = cfg_pool_en ? pool_out_addr_cnt[OUT_SRAM_ADDR_W-1:0]
                          : cfg_row_par_en ? rp_wr_addr[OUT_SRAM_ADDR_W-1:0]
                          : fsm_out_wr_addr[OUT_SRAM_ADDR_W-1:0];
    assign out_sram_dia   = alu_res;
```

- [ ] **Step 4: Keep `fsm_pp_done` correct for row-par non-pool**

Row-par non-pool must let the FSM advance once per conv point after the drain fully flushes. Reuse the non-pool `alu_vld`? No — there are now 16 valids. Gate FSM advance on the **last** expected valid. Simplest robust choice: advance on `fsm_pp_start`-relative count. Change `fsm_pp_done`:
```verilog
    // Non-pool row-par: advance after the group's valids have been written.
    wire rp_done = rp_active && (rp_vld_cnt == 5'd15) && pp_feat_vld;
    assign fsm_pp_done = cfg_pool_en   ? fsm_pp_start
                       : cfg_row_par_en ? rp_done
                       :                  alu_vld;
```
(Confirm in bring-up that S_POST holds until `rp_done`; the FSM `S_POST` waits on `i_pp_done`, so this is safe.)

- [ ] **Step 5: Compile and run — bit-identical (row_par still 0)**

Run: `bash run_all.sh clean && bash run_all.sh sim 2>&1 | tee docs/notes/t5.log`
Expected: identical to golden (firmware hasn't enabled the bit; all `cfg_row_par_en` muxes select the legacy arm).

- [ ] **Step 6: Commit**

```bash
git add rtl/top_controller_fsm.v rtl/npu_top.v
git commit -m "feat(npu): row-par non-pool Out-SRAM write sequencer (dormant)"
```

---

## Task 6: Enable row-par on ONE non-pool layer (Conv1) and bring up to bit-identical

**Files:**
- Modify: `firmware/deepnet_deploy.c` (set `NPU_CTRL_ROW_PAR` only for Conv1's `npu_conv_pass`)

Conv1 is non-pooled (`pool_en=0`), `out_w=28`, no skip/eltwise — the ideal first target (spec §verification). `28 = 16 + 12`, exercising one full group + one partial group.

- [ ] **Step 1: Locate Conv1's conv-pass call and add the mode flag**

Find the `npu_conv_pass(...)` invocation(s) for Conv1 in `deepnet_deploy.c` (the first conv, IC=1→16 OC, no pool). Add `NPU_CTRL_ROW_PAR` to the CTRL word written for that pass **only**. If `npu_conv_pass` builds the CTRL value internally, thread a `row_par` parameter or set a file-scope flag consulted for Conv1. Exact mechanism depends on the current `npu_conv_pass` signature — inspect it first; keep the change scoped to Conv1.

- [ ] **Step 2: Compile and run**

Run: `bash run_all.sh clean && bash run_all.sh sim 2>&1 | tee docs/notes/t6.log`
Expected (success): `diff docs/notes/golden_pre_E.log docs/notes/t6.log` shows only timestamp noise — Conv1's output is byte-for-byte identical, predictions 10/10 unchanged.

- [ ] **Step 3: If NOT identical — debug on waveform (systematic-debugging skill)**

Run: `bash run_all.sh waves`. Inspect, in `u_im2col`/`u_controller`/`u_post_process`:
1. `o_act_window` per array row during `S_CALC_KERNEL` for `oy=0,group_base=0` — rows 0..15 must equal input columns `0..15` for offset (mid,mid), etc. Verify `bank_for_offrow` and `rp_col`.
2. `pp_feat_vld` count per conv point = exactly 16; `rp_vld_cnt`/`rp_col`/`rp_wr_addr` map to columns `0..15` (group 0) and `0..11` valid for group `base=16` (rows 12..15 dropped by the guard).
3. Drain order: confirm the first valid corresponds to column 15 (row 15). If the latency offset differs, adjust where `rp_vld_cnt` starts counting (the guard tolerates a constant offset only if 16 valids still bracket the real pixels — otherwise add a fixed `rp_skip` constant).

Fix the smallest root cause, re-run Step 2. Do not proceed until bit-identical.

- [ ] **Step 4: Commit the working one-layer bring-up**

```bash
git add firmware/deepnet_deploy.c
git commit -m "feat(npu): enable row-parallel on Conv1, bit-identical 10/10"
```

---

## Task 7: Enable row-par on all NON-POOL conv layers (Conv3, Conv5)

**Files:**
- Modify: `firmware/deepnet_deploy.c` (add `NPU_CTRL_ROW_PAR` to Conv3, Conv5 passes)

Conv3 (`out_w=14`, one group of 14) and Conv5 (`out_w=8`, one group of 8) exercise `group_size < 16` (only 14 / 8 of 16 rows used — acceptable per spec §late layers).

- [ ] **Step 1: Add the flag to Conv3 and Conv5 conv passes** (non-pool ones only — leave Conv2/4/6 alone; they pool).

- [ ] **Step 2: Compile and run — bit-identical**

Run: `bash run_all.sh clean && bash run_all.sh sim 2>&1 | tee docs/notes/t7.log`
Expected: `diff` vs golden shows only timestamp noise; 10/10.

- [ ] **Step 3: If a partial-group layer fails**, waveform-check the `rp_col in [base, base+group_size)` guard drops exactly the high invalid rows. Fix, re-run.

- [ ] **Step 4: Commit**

```bash
git add firmware/deepnet_deploy.c
git commit -m "feat(npu): enable row-parallel on Conv3/Conv5 (non-pool), bit-identical"
```

---

## Task 8: Row-parallel POOL path — reorder buffer + pooler feed

**Files:**
- Modify: `rtl/post_process_top.v` (row-par drain capture → row-major replay into the pooler)
- Modify: `rtl/npu_top.v` (row-par pool write addressing)

**Design — this is the subtlest part (spec §risks).** The 2×2 pooler needs the group's pixels in **row-major** order (col 0..15) and continuity across groups and across the two input rows it pairs. The drain delivers them reversed (col 15 first). Add a **16-deep reorder buffer** at the post-process input:

1. During drain, write the post-processed pixel for drain cycle `d` into `reorder_buf[15 - d]` (so index = output column within group). Capture all 16 (only `0..group_size-1` are meaningful).
2. After drain, **replay** `reorder_buf[0..group_size-1]` one per cycle as the pooler input (`pool_gated_data`/`pool_gated_vld`), in ascending column order. The existing `max_pooling_2x2` then sees a normal row-major stream and pairs columns/rows exactly as in the single-pixel design — its `i_width` based row-boundary logic is unchanged.
3. Out-SRAM write in pool mode stays self-timed by `pool_vld` with `pool_out_addr_cnt` (unchanged) — pooled outputs remain contiguous.

This replaces the `pool_data_latched`/single-feed logic (lines 143-170) **only when `i_row_par_en`**; legacy pool path stays byte-identical when the bit is 0.

- [ ] **Step 1: Add `i_row_par_en` + `i_group_size` ports to `post_process_top`** and wire from npu_top (`cfg_row_par_en`, `fsm_group_size`).

- [ ] **Step 2: Implement the reorder buffer + row-major replay (row-par only)**

In `post_process_top.v`, behind `i_row_par_en`, add a 16-entry `reg [DATA_W-1:0] reorder_buf [0:15];`, a write index derived from a drain-cycle counter (count `s3_vld` during `i_in_drain`, write `reorder_buf[15 - idx]`), and a replay counter that, on `i_in_post` entry, streams `reorder_buf[0..group_size-1]` into `pool_gated_data` with `pool_gated_vld` high for `group_size` cycles. Keep the legacy single-feed for `!i_row_par_en`. (Full code to be written during bring-up; structure verified on waveform.)

- [ ] **Step 3: Row-par pool write addressing in npu_top** — confirm `pool_out_addr_cnt` still yields contiguous pooled addresses across groups (it increments per `out_sram_ena`); no change expected, verify on waveform.

- [ ] **Step 4: Enable row-par on Conv2 only (pool), compile, run**

Run: `bash run_all.sh clean && bash run_all.sh sim 2>&1 | tee docs/notes/t8.log`
Expected: bit-identical vs golden. Conv2 is the first pooled layer; verify pooled output byte-for-byte.

- [ ] **Step 5: Debug on waveform if needed** — focus on pooler column counter continuity across the 16/12-wide group boundary (the documented subtlest point). Fix, re-run to bit-identical.

- [ ] **Step 6: Commit**

```bash
git add rtl/post_process_top.v rtl/npu_top.v firmware/deepnet_deploy.c
git commit -m "feat(npu): row-parallel pool path (reorder buffer), Conv2 bit-identical"
```

---

## Task 9: Enable row-par on remaining pooled layers (Conv4, Conv6)

**Files:**
- Modify: `firmware/deepnet_deploy.c` (Conv4 `out_w=16` pad=2; Conv6 `out_w=8`)

- [ ] **Step 1: Add `NPU_CTRL_ROW_PAR` to Conv4 and Conv6 passes.**
- [ ] **Step 2: Run — bit-identical.** `bash run_all.sh clean && bash run_all.sh sim 2>&1 | tee docs/notes/t9.log`; `diff` vs golden = timestamp only; 10/10.
- [ ] **Step 3: Commit.** `git commit -am "feat(npu): row-parallel on Conv4/Conv6, all convs bit-identical 10/10"`

---

## Task 10: Profile — confirm the cycle drop

**Files:** none

- [ ] **Step 1:** `NPU_PROFILE=1 bash run_all.sh sim 2>&1 | tee docs/notes/t10_prof.log`
- [ ] **Step 2:** Compare conv / inference / full-run cycles vs `docs/notes/golden_pre_E_prof.log`. Expected per spec: conv ~2.15M → ~0.3–0.5M; inference ~4.6M → ~2.7–2.9M; full ~10.9M → ~9M. Record actuals.
- [ ] **Step 3:** If conv did **not** drop, the array is still under-fed (e.g. group not advancing by 16, or per-group overhead dominating) — investigate before claiming success. The bit-identical gate passing does **not** prove the speedup; the profile does.

---

## Task 11 (optional, separate verification): migrate hardware padding into im2col

> **Do this only after Tasks 0–10 are green and committed** (spec §risks: "Don't combine with the PAD-migration in one step").

**Files:**
- Modify: `rtl/im2col_line_buffer.v` (zero a slice column when its absolute input col/row is outside the unpadded image, using `i_pad_w/h` + `i_width/height`)
- Modify: `rtl/top_controller_fsm.v` / `rtl/npu_top.v` (remove FSM `o_border` zero-inject + `fsm_border_d` mux **for row-par mode**; keep for legacy)

- [ ] **Step 1:** Add pad-aware zeroing to the row-par slice (Task 3 `rp_val`): emit zero when the slice's absolute input coordinate is in the border. Pass `i_pad_w/i_pad_h` into im2col (new ports) and the unpadded geometry, mirroring the FSM `at_border` test.
- [ ] **Step 2:** In row-par mode, feed im2col the **unpadded** tile-major stream directly (no FSM border injection) and let im2col zero the borders.
- [ ] **Step 3:** Run — bit-identical 10/10. Commit. If it regresses, revert this task (E is already complete and committed without it).

---

## Task 12: Finalize — docs, memory, branch completion

**Files:**
- Modify: `CLAUDE.md` (add "Decision I: row-parallel (CTRL[9] row_par_en)" mirroring decisions G/H)
- Memory: update `soc-npu-row-parallel-todo.md` → done; add `soc-npu-row-parallel-done.md`

- [ ] **Step 1:** Write Decision I in CLAUDE.md: what `row_par_en` does, the drain-reverse-order fact, the post_process reorder buffer, measured cycles, generality (group_size runtime from out_w).
- [ ] **Step 2:** Update memory index + the TODO memory to reflect completion with the actual numbers from Task 10.
- [ ] **Step 3:** Use `superpowers:finishing-a-development-branch` to merge/PR.

---

## Self-review notes

- **Spec coverage:** im2col 16-wide slice (Task 3), FSM group loop (Task 4), out-write (Tasks 5–7), pool path (Tasks 8–9, the spec's flagged risk), PAD migration (Task 11), mode-bit incremental bring-up (Tasks 1, 6), bit-identical gate + profile (every task + Task 10), generality via runtime `group_size` (Task 4). All covered.
- **Spec correction surfaced:** post_process / out-write are NOT "largely unchanged" — Tasks 5 and 8 are the real work and the highest risk. This is called out at the top.
- **Type consistency:** `row_par_en` (CTRL[9]) / `cfg_row_par_en` / `i_row_par_en` / `o_row_par_en` consistent across regfile→npu_top→FSM→im2col; `group_size`/`group_base` consistent FSM→npu_top→post_process.
- **Honest placeholders:** Tasks 8 Step 2 and the bring-up debug steps describe *structure* + *waveform verification gates* rather than final Verilog, because the exact pipeline-latency alignment and pooler continuity are only determinable on a waveform (this is real RTL bring-up, not algorithmic code). Every such task has an explicit bit-identical gate that blocks progress until correct. The determinable code (registers, im2col slice, FSM group advance, non-pool write counter) is given in full.
```