# NPU Weight-Prefetch Reuse (D2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Prefetch each conv OC-tile's weights once into an on-chip buffer of depth `ICG_BUF=4` and reuse them across the whole spatial sweep, eliminating the redundant per-pixel weight re-prefetch (~144 cyc/pixel) for any layer with `ic_groups ≤ 4`; fall back to current behavior beyond that.

**Architecture:** Expand `wgt_reader`'s buffer to `[ko][oc][ICG_BUF]` with a dual-mode prefetch (all-groups vs single-group), gated by a new `i_prefetch_all` input and an `i_wgt_ic_sel` output mux. The FSM prefetches once per OC-tile in `reuse_mode` and loops IC-tiles in CALC without returning to prefetch. Reuse is general (any model, `ic_groups ≤ ICG_BUF`), not tied to specific dims.

**Tech Stack:** Verilog (ModelSim), `bash run_all.sh`. No firmware change (weight packing layout unchanged).

**Spec:** `docs/superpowers/specs/2026-06-12-npu-weight-reuse-design.md`

**Project rules (override skill defaults):**
- **No git commits** unless the user asks — "commit" steps are "checkpoint" notes.
- RTL changed → `bash run_all.sh clean` before recompiling.
- Acceptance gate is **bit-identical** conv output: `10/10` AND identical predictions (weights are unchanged, only their timing/source moves — any result change is a bug).
- Baseline to preserve/beat: current deploy build = **22,624,763 cycles, 10/10**.

**Verified facts (from source):**
- `wgt_reader.v` buffer is `wgt_buf[KERNEL_OFFSETS][NUM_OC]`; prefetch loops `pf_ko`(0..ko-1)×`pf_oc`(0..15) for ONE `i_ic_group`; output `o_wgt[gi]=wgt_buf[ko_sel][gi]`, `ko_sel=i_wgt_offset`.
- Packed weight layout (`load_conv_weights`): word(ocl,g,ko) = `ocl*ic_groups*KO + g*KO + ko`. The reuse prefetch address `wgt_base + pf_icg*KO + pf_ko + pf_oc*(ic_groups*KO)` matches it exactly (oc_base=0 since OC=16/start, decision D).
- Activations already hold all IC-tiles: im2col selects by `i_win_tile = fsm_cur_ic_tile[7:4]` ([npu_top.v:584]). So in CALC, advancing `ic_tile` selects both act (im2col) and (after this change) wgt (buffer) by the same index — accumulation across IC-tiles works without re-prefetch.
- All current conv layers: `ic_groups ∈ {1,2,4}` ≤ 4 → all use reuse. GEMM (`gemm_en`, FC1 ic_groups=64) → fallback, keeps the single-group path exercised.
- `wgt_reader`'s `o_wgt_vld` is NOT used to gate compute (systolic `i_vld = fsm_array_vld`); after prefetch the reader idles but `o_wgt` (combinational from `wgt_buf`) stays valid. (Confirm during Task 2.)

---

### Task 1: wgt_reader dual-mode buffer (dormant — regression must stay bit-identical)

**Files:**
- Modify: `rtl/wgt_reader.v`
- Modify: `rtl/npu_top.v` (tie new inputs off for now)

- [ ] **Step 1.1: Add params/ports**

In `rtl/wgt_reader.v` params (after `IC_GROUPS_MAX`):
```verilog
    parameter IC_GROUPS_MAX  = 64,   // max IC groups (1024 ch / 16)
    parameter ICG_BUF        = 4     // IC-tiles held on-chip for per-OC-tile reuse
```
In the port list, after `i_kernel_offsets`:
```verilog
    input  wire                         i_prefetch_all,    // 1=prefetch ALL ic_groups (reuse), 0=single i_ic_group
    input  wire [3:0]                   i_wgt_ic_sel,      // during CALC: which ic_group's weights to present (reuse)
```

- [ ] **Step 1.2: Expand buffer + pf_icg counters**

Change the buffer decl:
```verilog
    reg [WGT_GROUP_W-1:0] wgt_buf [0:KERNEL_OFFSETS-1][0:NUM_OC-1][0:ICG_BUF-1];
```
Add near the other `pf_*` regs (after `pf_oc_d`):
```verilog
    reg [3:0] pf_icg;     // Current IC group being pre-fetched (reuse mode)
    reg [3:0] pf_icg_d;   // Delayed (matches SRAM 1-cycle latency)
```

- [ ] **Step 1.3: Address uses the effective IC group**

Replace the `addr_ic_component` assign with:
```verilog
    // Reuse mode prefetches every ic_group (pf_icg); legacy mode the single i_ic_group.
    wire [9:0] eff_icg = i_prefetch_all ? {6'd0, pf_icg} : i_ic_group;
    assign addr_ic_component = {{(SRAM_ADDR_W-10){1'b0}}, eff_icg}
                             * {{(SRAM_ADDR_W-8){1'b0}}, i_kernel_offsets};
```

- [ ] **Step 1.4: Prefetch FSM — pf_icg loop, write 3rd index, completion**

In reset block add: `pf_icg <= 4'd0; pf_icg_d <= 4'd0;`
Add delayed copy near `pf_ko_d <= pf_ko;`: `pf_icg_d <= pf_icg;`
In `PF_IDLE` start and `PF_DONE` restart blocks (both set `pf_ko<=0; pf_oc<=0;`) add `pf_icg <= 4'd0;`.
Replace the `PF_READING` write + advance body with:
```verilog
                PF_READING: begin
                    if (pf_reading_d)
                        wgt_buf[pf_ko_d][pf_oc_d][i_prefetch_all ? pf_icg_d : 4'd0] <= i_sram_data;

                    if (pf_oc == 5'd15 && pf_ko == (i_kernel_offsets[3:0] - 4'd1) &&
                        (!i_prefetch_all || pf_icg == (i_ic_groups_total[3:0] - 4'd1))) begin
                        pf_state <= PF_WAIT_LAST;
                    end else begin
                        if (pf_oc == 5'd15) begin
                            pf_oc <= 5'd0;
                            if (pf_ko == (i_kernel_offsets[3:0] - 4'd1)) begin
                                pf_ko  <= 4'd0;
                                pf_icg <= pf_icg + 4'd1;   // advances only when i_prefetch_all
                            end else begin
                                pf_ko <= pf_ko + 4'd1;
                            end
                        end else begin
                            pf_oc <= pf_oc + 5'd1;
                        end
                    end
                end
```
In `PF_WAIT_LAST`, change the write to:
```verilog
                    wgt_buf[pf_ko_d][pf_oc_d][i_prefetch_all ? pf_icg_d : 4'd0] <= i_sram_data;
```

- [ ] **Step 1.5: Output mux selects the ic group**

Replace the output generate:
```verilog
    wire [3:0] ic_sel = i_prefetch_all ? i_wgt_ic_sel : 4'd0;
    genvar gi;
    generate
        for (gi = 0; gi < NUM_OC; gi = gi + 1) begin : gen_wgt_out
            assign o_wgt[gi*WGT_GROUP_W +: WGT_GROUP_W] = wgt_buf[ko_sel][gi][ic_sel];
        end
    endgenerate
```

- [ ] **Step 1.6: npu_top — tie new inputs OFF (dormant)**

In `rtl/npu_top.v` `u_wgt_reader` port list add:
```verilog
        .i_prefetch_all    (1'b0),
        .i_wgt_ic_sel      (4'd0),
```

- [ ] **Step 1.7: Regression — bit-identical**

```bash
cd /e/code/6-10/soc && bash run_all.sh clean >/dev/null 2>&1 && bash run_all.sh sim 2>&1 | grep -E "Result:|TRAP|ALL TESTS|Errors:"
```
Required: `10/10`, `ALL TESTS PASSED.`, `TRAP after 22624763 clock cycles` EXACTLY (feature dormant → byte-identical). If cycles differ: the buffer/address refactor changed the single-group path — debug (the `i_prefetch_all=0` path must equal the old code: 3rd index 0, pf_icg never advances, completion at last oc&ko).

**Checkpoint:** wgt_reader generalized, dormant, conv bit-identical.

---

### Task 2: FSM activates reuse + npu_top wiring

**Files:**
- Modify: `rtl/top_controller_fsm.v`
- Modify: `rtl/npu_top.v`

- [ ] **Step 2.1: FSM new outputs + reuse wires**

In `rtl/top_controller_fsm.v` port list, after `o_wgt_base`:
```verilog
    output wire                     o_prefetch_all,  // 1 = reuse mode (prefetch all ic_groups once)
    output wire [3:0]               o_wgt_ic_sel,    // current ic_group for the weight output mux
```
Near `ko_total` (after the `ic_groups`/`ko_total` wires) add:
```verilog
    localparam ICG_BUF = 4;
    // Reuse when this layer's weights fit the on-chip buffer (any model). GEMM excluded.
    wire reuse_mode = !i_gemm_en && (ic_groups <= 16'd ICG_BUF);
```
Add a reg with the other state regs (near `load_tile`): `reg wgt_loaded;`
Output assigns (near `o_wgt_base`):
```verilog
    assign o_prefetch_all = reuse_mode;
    assign o_wgt_ic_sel   = ic_tile[7:4];
```

- [ ] **Step 2.2: Gate the prefetch trigger**

Change `o_wgt_start_prefetch`:
```verilog
    assign o_wgt_start_prefetch = (state == S_PREFETCH_WGT) && (pf_wait_cnt == 16'd0)
                                && (!reuse_mode || !wgt_loaded);
```

- [ ] **Step 2.3: S_PREFETCH_WGT — settle-and-skip when loaded**

Add a settle localparam near ICG_BUF: `localparam WGT_REUSE_SETTLE = 16'd2;`
Replace the `S_PREFETCH_WGT` state body with:
```verilog
                S_PREFETCH_WGT: begin
                    pf_wait_cnt <= pf_wait_cnt + 16'd1;
                    if (reuse_mode && wgt_loaded) begin
                        // Weights already resident from this OC-tile's first prefetch;
                        // just let the im2col window settle, then compute.
                        if (pf_wait_cnt >= WGT_REUSE_SETTLE) begin
                            ko_cnt <= 4'd0;
                            state  <= S_CALC_KERNEL;
                        end
                    end else if (i_wgt_prefetch_done) begin
                        if (reuse_mode) wgt_loaded <= 1'b1;
                        ko_cnt  <= 4'd0;
                        state   <= S_CALC_KERNEL;
                    end
                end
```

- [ ] **Step 2.4: S_CALC_KERNEL — loop IC-tiles without re-prefetch in reuse**

Replace the IC-tile advance branch inside `S_CALC_KERNEL` (`if (ic_tile + 16'd16 < i_dim_in_c)`):
```verilog
                        if (ic_tile + 16'd16 < i_dim_in_c) begin
                            ic_tile <= ic_tile + 16'd16;
                            if (reuse_mode) begin
                                // next ic_group already in wgt_buf; o_wgt_ic_sel follows ic_tile
                                state <= S_CALC_KERNEL;   // ko_cnt reset above
                            end else begin
                                state   <= S_PREFETCH_WGT;
                                pf_wait_cnt <= 16'd0;
                            end
                        end else begin
                            state  <= S_K_END;
                        end
```
(The `ko_cnt <= 4'd0;` already executes in the enclosing `if (ko_cnt == ...)` branch.)

- [ ] **Step 2.5: Clear wgt_loaded at start and OC-tile change**

In `S_IDLE` `i_start` block add: `wgt_loaded <= 1'b0;`
In the reset block add: `wgt_loaded <= 1'b0;`
In `S_NEXT_TILE` OC-advance branch (after `oc_tile <= oc_tile + 16'd16;`) add: `wgt_loaded <= 1'b0;`

- [ ] **Step 2.6: npu_top — wire FSM outputs to wgt_reader**

In `rtl/npu_top.v`: add wires
```verilog
    wire        fsm_prefetch_all;
    wire [3:0]  fsm_wgt_ic_sel;
```
In `u_controller` port list add `.o_prefetch_all(fsm_prefetch_all), .o_wgt_ic_sel(fsm_wgt_ic_sel),`.
Change the `u_wgt_reader` ties from Step 1.6 to:
```verilog
        .i_prefetch_all    (fsm_prefetch_all),
        .i_wgt_ic_sel      (fsm_wgt_ic_sel),
```

- [ ] **Step 2.7: Verify — bit-identical results, fewer cycles**

```bash
cd /e/code/6-10/soc && bash run_all.sh clean >/dev/null 2>&1 && bash run_all.sh sim 2>&1 | grep -E "Result:|TRAP|ALL TESTS|timeout|Errors:"
```
Required: `10/10`, `ALL TESTS PASSED.`, and `TRAP after N` with **N < 22,624,763** (conv layers faster). Predictions must be identical (bit-exact). If FAIL/mismatch: prime suspects — (1) `WGT_REUSE_SETTLE` too small (window not ready → bump to 3-4), (2) `o_wgt_ic_sel`/`i_win_tile` misalignment across the IC-tile boundary, (3) `wgt_loaded` not cleared between starts. Debug with `bash run_all.sh waves`.

- [ ] **Step 2.8: Profile the gain**

Set `#define NPU_PROFILE 1` in `firmware/deepnet_deploy.c`, run, capture `npu` total + per-layer `Conv1..Conv6` (expect large drops, esp. Conv1/2/3). Revert `NPU_PROFILE 0`, rebuild. Record old (8.6M npu / 22.6M total) vs new.

**Checkpoint:** weight reuse active; conv bit-identical, faster; GEMM unaffected (fallback path).

---

### Task 3: Docs + memory

**Files:** `CLAUDE.md`, memory dir.

- [ ] **Step 3.1:** CLAUDE.md — add a short note under the NPU dataflow / decision list: weight-prefetch reuse (per-OC-tile, `ICG_BUF=4` in `wgt_reader`/FSM), general over `ic_groups ≤ 4`, GEMM/`>4` fall back. Update the measured cycle figure.
- [ ] **Step 3.2:** Memory — add/update an entry (e.g. `soc-npu-weight-reuse.md`): mechanism, `ICG_BUF=4` knob, fallback, before/after cycles; link `[[soc-npu-general-purpose]]`, `[[soc-npu-gemm-mode]]`. Update `MEMORY.md` index.
- [ ] **Step 3.3:** Summarize working-tree changes; offer the user a commit (do not commit unprompted).

---

## Self-review

- **Spec coverage:** buffer `[ko][oc][ICG_BUF]` ✓(1.2), dual-mode prefetch ✓(1.4), `i_prefetch_all`/`i_wgt_ic_sel` ✓(1.1/1.5), FSM `reuse_mode`+`wgt_loaded`+prefetch-once ✓(2.1-2.5), npu_top wiring ✓(2.6), fallback (GEMM/`>4`) preserved ✓(`i_prefetch_all=0` path identical), bit-identical gate ✓(1.7/2.7), docs ✓(T3).
- **Placeholders:** none; `WGT_REUSE_SETTLE` has a concrete initial value (2) with a tuning note.
- **Type consistency:** `i_prefetch_all`(1b)/`i_wgt_ic_sel`[3:0] and FSM `o_prefetch_all`/`o_wgt_ic_sel` match; `ICG_BUF=4` in both `wgt_reader` param and FSM localparam; `pf_icg`[3:0] within `ICG_BUF` range (reuse ⇒ ic_groups≤4).
