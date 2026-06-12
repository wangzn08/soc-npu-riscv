# NPU General GEMM / FC Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a general-purpose GEMM/fully-connected mode to the NPU (CTRL[7] `gemm_en`) so FC layers run on the 16×16 systolic array, collapsing the 22.3M-cycle CPU affine bottleneck to <1M.

**Architecture:** Bypass im2col; replicate the current IC-tile activation word across all 16 array rows (exactly like im2col's "replicated 16×" conv behavior, so drain/post/write logic works bit-identically); stream `ceil(IC/16)` IC-tiles as k-steps per 16-OC pass; reuse wgt_reader (generalized to runtime kernel-offset count), post_process, Out SRAM, DMA unchanged. Firmware drives OC tiling per architecture decision D.

**Tech Stack:** Verilog (ModelSim), C firmware (rv32imc), `bash run_all.sh` flow.

**Spec:** `docs/superpowers/specs/2026-06-12-npu-gemm-fc-mode-design.md`
(One refinement vs spec: activation is replicated to all 16 rows instead of one
active row — all PEs in a column compute identical results, so the captured
drain phase is irrelevant. Strictly safer, equally general.)

**Project rules that override skill defaults:**
- **No git commits unless the user explicitly asks** (CLAUDE.md). Commit steps are replaced by "checkpoint" notes.
- RTL changed → `bash run_all.sh clean` before recompiling.
- Firmware must stay warning-clean under strict CFLAGS.
- Register-map changes must touch BOTH `rtl/param_regfile.v` and `firmware/firmware.h`.
- The harness Bash tool does not pass env vars through; toggle `#define` switches in source instead.

**Key facts an implementer needs (verified against source):**
- `pe_core.v`: each PE accumulates independently; column cascade used only during drain. Drain cycle k outputs row (15−k); post_process pipeline + FSM `S_POST` write capture one 16-OC word — with replicated rows, all drained values are equal.
- `top_controller_fsm.v` states: `S_IDLE→S_LOAD_ROW→S_WAIT_WIN→S_PREFETCH_WGT→S_CALC_KERNEL→S_K_END→S_DRAIN→S_POST→S_NEXT_TILE→S_DONE`. GEMM skips `S_LOAD_ROW`/`S_WAIT_WIN`.
- FSM latches `act/wgt/out_base` from the **pong** registers when CTRL[1] (global ping_pong) =1. GEMM runs with ping_pong=1: input vector in Act PONG bank, FC weights resident in Wgt PONG bank — conv data in PING banks untouched.
- `wgt_reader.v` address: `wgt_base + oc_base*icg*KO + ic_group*KO + ko + pf_oc*(icg*KO)`; note `(pf_ko/KW)*KW + pf_ko%KW == pf_ko`. `KO` becomes runtime `i_kernel_offsets` (conv 9, GEMM 1).
- Firmware always programs 3×3 kernels for conv, so runtime `kh*kw=9` reproduces today's behavior exactly; regression gate = full sim still 10/10.
- HW cap: `o_wgt_ic_group` is 10-bit fed from `ic_tile[9:0]>>4` → GEMM `in_dim ≤ 1024`. FC1 = 1024 exactly. Document, don't widen (YAGNI).
- One NPU start = one 16-OC pass (`NPU_OC`=16, decision D): with `out_w=out_h=1`, `S_NEXT_TILE` falls straight to `S_DONE`.
- Out write address = `out_base + 0` → set `NPU_OUT_ADDR_B = pass` so pass p lands at Out SRAM word p; one DMA drain at the end reads all words.
- DDR staging: `WGT_BUF` (0x40007000) usable region is 36KB (next buffer at 0x40010000); FC1 packed total is 64KB → pack & DMA **per 16-OC tile** (16KB each).
- FC1 input must be in DDR (NPU DMA can't read CPU private RAM): reorder writes to `AFFINE_SCR` again. FC1 output → `NPU_OUT_BUF`; padded OCs 50..63 compute to exact 0 in HW (zero weights, zero bias/scale) so no CPU padding needed before FC2.

---

### Task 1: wgt_reader runtime kernel-offset count (generality; conv regression)

**Files:**
- Modify: `rtl/wgt_reader.v`
- Modify: `rtl/npu_top.v` (compute & connect `kernel_offsets`)

- [ ] **Step 1.1: Add runtime port to wgt_reader**

In `rtl/wgt_reader.v`, after the `i_wgt_base` input (~line 42), add:

```verilog
    input  wire [7:0]                   i_kernel_offsets,  // runtime kh*kw (conv 9, GEMM 1)
```

- [ ] **Step 1.2: Replace compile-time KH*KW address math**

Replace lines ~75–93 (`oc_stride` through `sram_rd_addr`) with:

```verilog
    // Address stride between consecutive OCs for the same (ic_group, ko)
    // KO = i_kernel_offsets (runtime kh*kw): conv 3x3 -> 9 (identical to old
    // compile-time KH*KW), GEMM 1x1 -> 1.
    wire [SRAM_ADDR_W-1:0] oc_stride;
    assign oc_stride = i_ic_groups_total[SRAM_ADDR_W-1:0]
                     * {{(SRAM_ADDR_W-8){1'b0}}, i_kernel_offsets};

    // Base address: oc_base*icg*KO + ic_group*KO + ko   (kh*KW+kw == pf_ko)
    wire [SRAM_ADDR_W-1:0] addr_oc_component;
    wire [SRAM_ADDR_W-1:0] addr_ic_component;

    assign addr_oc_component = i_oc_base[SRAM_ADDR_W-1:0] * oc_stride;
    assign addr_ic_component = i_ic_group[SRAM_ADDR_W-1:0]
                             * {{(SRAM_ADDR_W-8){1'b0}}, i_kernel_offsets};

    wire [SRAM_ADDR_W-1:0] sram_rd_addr;
    assign sram_rd_addr = i_wgt_base
                        + addr_oc_component + addr_ic_component
                        + {{(SRAM_ADDR_W-4){1'b0}}, pf_ko}
                        + pf_oc * oc_stride;
```

Wait — `addr_oc_component` was `i_oc_base * i_ic_groups_total * KH * KW` = `i_oc_base * oc_stride` and `pf_oc * oc_stride` is separate; both preserved above. (`oc_stride` per-OC words = icg*KO; `i_oc_base` counts channels and is 0 in current firmware since OC=16 per start — keep the term for hardware completeness.)

- [ ] **Step 1.3: Runtime prefetch loop bound and wgt_vld**

In the `PF_READING` state, change

```verilog
                    if (pf_ko == (KERNEL_OFFSETS - 1) && pf_oc == 5'd15) begin
```
to
```verilog
                    if (pf_ko == (i_kernel_offsets[3:0] - 4'd1) && pf_oc == 5'd15) begin
```

And at the bottom change

```verilog
    assign o_wgt_vld = (pf_state == PF_DONE) && (i_wgt_offset < KERNEL_OFFSETS);
```
to
```verilog
    assign o_wgt_vld = (pf_state == PF_DONE) && (i_wgt_offset < i_kernel_offsets[3:0]);
```

`KERNEL_OFFSETS` (=9) remains as the `wgt_buf` max sizing parameter. Supported runtime offsets: 1..9 (4-bit compare; im2col only supports 3×3 anyway — note in header comment).

- [ ] **Step 1.4: Drive it from npu_top**

In `rtl/npu_top.v` near the wgt_reader instance (~line 510), add above the instance:

```verilog
    // Runtime kernel-offset count for wgt_reader (conv 3x3 -> 9, GEMM 1x1 -> 1)
    wire [7:0] cfg_kernel_offsets = {4'd0, cfg_kh[3:0]} * {4'd0, cfg_kw[3:0]};
```

and inside the `u_wgt_reader` port list add:

```verilog
        .i_kernel_offsets  (cfg_kernel_offsets),
```

- [ ] **Step 1.5: Regression — full sim must stay 10/10**

```bash
cd /e/code/6-10/soc && bash run_all.sh clean >/dev/null 2>&1 && bash run_all.sh sim 2>&1 | grep -E "Result:|TRAP|ALL TESTS|Errors"
```
Expected: `=== Result: 10/10 correct ===`, `ALL TESTS PASSED.`, cycles ≈ 40,873,968 (unchanged ±0). If compile errors or FAIL: fix before proceeding (systematic-debugging skill).

**Checkpoint:** wgt_reader generalized, conv bit-identical. (No commit per project rule.)

---

### Task 2: gemm_en mode — param_regfile + FSM + npu_top

**Files:**
- Modify: `rtl/param_regfile.v`
- Modify: `rtl/top_controller_fsm.v`
- Modify: `rtl/npu_top.v`
- Modify: `firmware/firmware.h` (CTRL bit define — keep both sides in sync)

- [ ] **Step 2.1: param_regfile CTRL[7]**

Header comment line 8: extend to `... [5]relu_en [6]out_ping [7]gemm_en`.
Near `ctrl_out_ping` declaration (~line 161 area) add `reg ctrl_gemm_en;`.
Output ports: after `o_out_ping_sel` add:

```verilog
    output wire                         o_gemm_en,       // CTRL[7]: GEMM/FC mode (bypass im2col)
```

Reset block (after `ctrl_out_ping <= 1'b0;`): `ctrl_gemm_en <= 1'b0;`
Write case `10'h00` (after `ctrl_out_ping   <= s_axi_wdata[6];`):

```verilog
                        ctrl_gemm_en    <= s_axi_wdata[7];
```

Readback case `10'h00`: change to

```verilog
                    10'h00: rdata <= {24'd0, ctrl_gemm_en, ctrl_out_ping, ctrl_relu_en, ctrl_clear_done, ctrl_eltwise_en, ctrl_pool_en, ctrl_ping_pong, ctrl_start};
```

Assign near the other ctrl assigns: `assign o_gemm_en = ctrl_gemm_en;`

- [ ] **Step 2.2: firmware.h define (register-map sync rule)**

After `#define NPU_CTRL_OUT_PING   (1 << 6)`:

```c
#define NPU_CTRL_GEMM_EN    (1 << 7)   // GEMM/FC mode: bypass im2col, vector x matrix
```
Also update the `NPU_CTRL` comment on line 43 to append `[6]out_ping [7]gemm_en`.

- [ ] **Step 2.3: FSM GEMM support (`rtl/top_controller_fsm.v`)**

(a) Port — after `i_eltwise_en` input add:

```verilog
    input  wire                     i_gemm_en,      // GEMM/FC mode: bypass im2col, act = vector
```

(b) Runtime kernel-offset total — near the `ic_groups` derived wire (~line 141):

```verilog
    // Runtime kernel offsets (conv 3x3 -> 9, GEMM 1x1 -> 1); replaces KERNEL_OFFSETS
    wire [7:0] ko_total;
    assign ko_total = {4'd0, i_kernel_kh[3:0]} * {4'd0, i_kernel_kw[3:0]};
```

(c) `S_CALC_KERNEL` (~line 347): replace `if (ko_cnt == (KERNEL_OFFSETS - 1))` with

```verilog
                    if (ko_cnt == (ko_total[3:0] - 4'd1)) begin
```

(d) Act SRAM read mux (~lines 158–165) — GEMM reads the IC-tile word during PREFETCH (1-cycle SRAM latency; `sdp_bram` holds `doa` until next `ena`, so the word is stable through CALC):

```verilog
    wire [SRAM_ADDR_W-1:0] act_rd_addr;
    assign act_rd_addr = act_base_addr
                       + cur_in_row * act_row_stride
                       + cur_in_col * ic_groups
                       + load_tile;   // stream each IC tile of the column

    // GEMM: input vector word index = ic_tile/16 (ceil(IC/16) words at act_base)
    wire [SRAM_ADDR_W-1:0] gemm_act_addr;
    assign gemm_act_addr = act_base_addr + {{(SRAM_ADDR_W-12){1'b0}}, ic_tile[15:4]};

    assign o_act_sram_addr = i_gemm_en ? gemm_act_addr : act_rd_addr;
    assign o_act_sram_en   = i_gemm_en
                           ? ((state == S_PREFETCH_WGT) && (pf_wait_cnt == 16'd0))
                           : ((state == S_LOAD_ROW) && (cur_in_col < i_dim_in_w) && (load_col_cnt > 16'd0));
```

(e) `o_im2col_win_advance` (~line 198): prefix the whole expression with `!i_gemm_en && (` ... `)` so im2col state never advances in GEMM mode (protects the next conv layer's im2col init).

(f) `S_IDLE` transition (~line 278): `state <= S_LOAD_ROW;` →

```verilog
                        state <= i_gemm_en ? S_PREFETCH_WGT : S_LOAD_ROW;
```
(`pf_wait_cnt` is already reset to 0 in the same block.)

(g) `S_NEXT_TILE` OC-advance branch (~line 420–423): replace

```verilog
                            oc_tile <= oc_tile + 16'd16;
                            state <= S_LOAD_ROW;
                            load_col_cnt <= 16'd0;
```
with
```verilog
                            oc_tile <= oc_tile + 16'd16;
                            state <= i_gemm_en ? S_PREFETCH_WGT : S_LOAD_ROW;
                            load_col_cnt <= 16'd0;
                            pf_wait_cnt  <= 16'd0;
```
(Unreachable in current firmware — OC=16 per start — but keeps hardware OC tiling correct in GEMM mode for generality.)

- [ ] **Step 2.4: npu_top wiring + activation mux**

Add wire near other cfg wires: `wire cfg_gemm_en;`
param_regfile instance: add `.o_gemm_en (cfg_gemm_en),`
FSM instance: add `.i_gemm_en (cfg_gemm_en),` after `.i_eltwise_en`.
Systolic activation input (~line 617) — replace `.i_act (im2col_act_window),` with:

```verilog
        // GEMM mode: replicate the current IC-tile activation word to all 16
        // rows (same replication im2col does for conv) — every PE in a column
        // computes the identical dot product, so drain/POST capture is
        // phase-independent. Conv mode: im2col window as before.
        .i_act       (cfg_gemm_en ? {ARRAY_ROWS{act_sram_doa}} : im2col_act_window),
```

- [ ] **Step 2.5: Regression — conv unchanged with gemm_en=0**

```bash
cd /e/code/6-10/soc && bash run_all.sh clean >/dev/null 2>&1 && bash run_all.sh sim 2>&1 | grep -E "Result:|TRAP|ALL TESTS|Errors"
```
Expected: `10/10`, `ALL TESTS PASSED.`, cycles ≈ 40,873,968. Fix any compile error/regression before Task 3.

**Checkpoint:** GEMM datapath in RTL, dormant. Conv regression green.

---

### Task 3: Firmware GEMM helpers + FC1 parity test (red → green)

**Files:**
- Modify: `firmware/deepnet_deploy.c`

- [ ] **Step 3.1: Add FC weight packing + preload (after `preload_conv_weights`)**

```c
// ================================================================
// GEMM/FC 权重打包: KH=KW=1 布局, word = oc_rel*icg + ic_group
// Pack one 16-OC tile of FC weights for the GEMM path. Out-of-range
// oc/ic positions are zero-filled (padded OCs then compute exact 0).
// ================================================================
#define FC1_WGT_BASE 0      // Wgt SRAM PONG bank word offset (FC1: 4 tiles x 1024)
#define FC2_WGT_BASE 4096   // FC2: 1 tile x 64 words
#define FC1_OUT_DDR  NPU_OUT_BUF   // FC1 int8 output staging (4 words used)
#define FC2_OUT_DDR  (NPU_OUT_BUF + 0x100) // FC2 int8 scores (1 word)

static void pack_fc_tile(const int8_t *W, int in_dim, int out_dim,
                         int oc_base, uint32_t stage_ddr)
{
    int icg = (in_dim + 15) / 16;
    volatile int8_t *dst = (volatile int8_t *)stage_ddr;
    for (int o = 0; o < 16; o++) {
        int oc = oc_base + o;
        for (int g = 0; g < icg; g++)
            for (int b = 0; b < 16; b++) {
                int ic = g * 16 + b;
                dst[(o * icg + g) * 16 + b] =
                    (oc < out_dim && ic < in_dim) ? W[oc * in_dim + ic] : 0;
            }
    }
}

// Preload FC weights into the Wgt SRAM PONG bank (conv weights live in PING).
// FC1 packed = 64KB > WGT_BUF staging window, so pack+DMA per 16-OC tile.
static void preload_fc_weights(void)
{
    npu_wr(NPU_DMA_PING_SEL, 0x2);                 // DMA wgt writes -> PONG bank
    for (int t = 0; t < (AFFINE1_OUT + 15) / 16; t++) {
        pack_fc_tile(&affine1_W[0][0], AFFINE1_IN, AFFINE1_OUT, t * 16, WGT_BUF);
        dma_ddr_to_wgt(WGT_BUF, FC1_WGT_BASE + t * 16 * ((AFFINE1_IN + 15) / 16),
                       16 * ((AFFINE1_IN + 15) / 16));
    }
    pack_fc_tile(&affine2_W[0][0], 50, 10, 0, WGT_BUF);
    dma_ddr_to_wgt(WGT_BUF, FC2_WGT_BASE, 16 * ((50 + 15) / 16));
    npu_wr(NPU_DMA_PING_SEL, 0x0);                 // restore: conv DMAs use PING
}
```

(If `deepnet.h` defines AFFINE2 dims, use them instead of literals 50/10.)

- [ ] **Step 3.2: Add general `npu_gemm_pass` (after `npu_conv_pass`)**

```c
// ================================================================
// 通用 GEMM/全连接: out[0..out_dim) = quant(act · W + bias), 可选 ReLU
// General NPU GEMM: arbitrary in_dim (<=1024, HW ic_group width) /
// out_dim (OC tiled by 16 per decision D). Runs with global ping_pong=1:
// input vector in Act PONG, weights resident in Wgt PONG (pack_fc_tile
// layout at wgt_base), output INT8 channel-major at out_ddr.
// ================================================================
static void npu_gemm_pass(int in_dim, int out_dim, int scale_mul_val,
                          const int32_t *biases, int relu_en,
                          uint32_t in_ddr, uint32_t out_ddr, int wgt_base)
{
    int icg        = (in_dim + 15) / 16;
    int oc_passes  = (out_dim + 15) / 16;
    int tile_words = 16 * icg;

    npu_wr(NPU_DMA_PING_SEL, 0x1);                 // input vector -> Act PONG
    dma_ddr_to_act(in_ddr, 0, icg);
    npu_wr(NPU_DMA_PING_SEL, 0x0);

    for (int pass = 0; pass < oc_passes; pass++) {
        int oc_base = pass * 16;
        npu_wr(NPU_IN_W, 1);
        npu_wr(NPU_IN_H, 1);
        npu_wr(NPU_IC, in_dim);
        npu_wr(NPU_OC, 16);                        // decision D: 16 OC per start
        npu_wr(NPU_KERNEL, (1 << 8) | 1);
        npu_wr(NPU_STRIDE, (1 << 8) | 1);
        // ping_pong=1 -> FSM latches the *_B (pong) base registers
        npu_wr(NPU_ACT_ADDR_B, 0);
        npu_wr(NPU_WGT_ADDR_B, (uint32_t)(wgt_base + pass * tile_words));
        npu_wr(NPU_OUT_ADDR_B, (uint32_t)pass);    // word p = OCs p*16..p*16+15

        for (int ch = 0; ch < 16; ch++) {
            int oc = oc_base + ch;
            npu_wr(NPU_BIAS(ch),  (oc < out_dim) ? (uint32_t)biases[oc] : 0u);
            npu_wr(NPU_SCALE(ch), (oc < out_dim) ? (uint32_t)scale_mul_val : 0u);
            npu_wr(NPU_SHIFT(ch), SCALE_SHIFT);
        }

        npu_irq_flag = 0;
        npu_wr(NPU_CTRL, NPU_CTRL_START | NPU_CTRL_PING_PONG | NPU_CTRL_GEMM_EN |
                         (relu_en ? NPU_CTRL_RELU_EN : 0));
        int t = NPU_IRQ_TIMEOUT;
        while (t-- > 0)
            if (npu_irq_flag) break;
        if (t <= 0) print_str("  GEMM IRQ timeout!\n");
    }
    // All passes' words sit at Out SRAM (PING bank, CTRL[6]=0) words 0..passes-1
    dma_out_to_ddr(out_ddr, 0, oc_passes, 0);
}
```

- [ ] **Step 3.3: Wire up call sites + FC1 parity check**

Add near the other switches:

```c
// 1 = run CPU affine AND NPU GEMM, compare results (validation build).
#ifndef NPU_GEMM_PARITY
#define NPU_GEMM_PARITY 1
#endif
```

In `usercode7()` after `preload_conv_weights();` add `preload_fc_weights();`.

In `deepnet_inference`, make reorder write **both** the RAM buffer and DDR:
in the reorder block change the assignment line to:

```c
            {
                int8_t v = pool_out[(pass * n_pos + pos) * 16 + ch_in];
                affine_in[ch * n_pos + pos] = v;
                ((volatile int8_t *)AFFINE_SCR)[ch * n_pos + pos] = v;
            }
```

After the existing CPU affine block (`PROF_ADD(prof_affine);` line), insert:

```c
#if NPU_GEMM_PARITY
    {
        // NPU FC1 vs CPU affine1: must be bit-identical (same quant path)
        npu_gemm_pass(AFFINE1_IN, AFFINE1_OUT, SCALE_AFFINE1, affine1_b, 1,
                      AFFINE_SCR, FC1_OUT_DDR, FC1_WGT_BASE);
        volatile int8_t *nf = (volatile int8_t *)FC1_OUT_DDR;
        int mism = 0;
        for (int i = 0; i < AFFINE1_OUT; i++)
            if (nf[i] != affine_mid[i]) mism++;
        print_str("  FC1 parity: ");
        if (mism == 0) print_str("OK\n");
        else { print_str("FAIL mism="); print_dec((uint32_t)mism); print_chr('\n'); }
    }
#endif
```

- [ ] **Step 3.4: Build + run parity sim**

```bash
cd /e/code/6-10/soc && bash run_all.sh sim 2>&1 | grep -E "parity|Result:|TRAP|ALL TESTS|timeout|Errors"
```
Expected GREEN: ten `FC1 parity: OK` lines, `10/10`, `ALL TESTS PASSED.`
If `FAIL`/`timeout`: use superpowers:systematic-debugging — likely suspects in order: (1) FSM never leaves S_PREFETCH_WGT (wgt prefetch handshake with ko=1), (2) act word not stable at CALC (check `o_act_sram_en` pulse timing), (3) wgt address mismatch vs `pack_fc_tile` layout (dump first weight word via DMA-read of Wgt PONG), (4) wrong bank selects (CTRL[1]=1 + DMA_PING_SEL sequencing). Waveform run: `bash run_all.sh waves`.

**Checkpoint:** NPU FC1 bit-identical to CPU. This is the core acceptance gate.

---

### Task 4: FC2 on NPU + argmax criterion

**Files:**
- Modify: `firmware/deepnet_deploy.c`

- [ ] **Step 4.1: FC2 parity block**

FC2's CPU reference produces **int32** scores; the NPU path quantizes to INT8
(clamped ±127). Decision criterion (from spec): if `argmax(int8 NPU scores)` ==
CPU argmax for **all 10 images**, FC2 ships on NPU; otherwise FC2 stays on CPU
(cost ≈5K cycles — negligible) and that is documented.

Inside the same `#if NPU_GEMM_PARITY` block, after the FC1 compare, add:

```c
        // NPU FC2 (no ReLU): int8-quantized scores; compare argmax vs CPU int32
        npu_gemm_pass(50, 10, SCALE_AFFINE2, affine2_b, 0,
                      FC1_OUT_DDR, FC2_OUT_DDR, FC2_WGT_BASE);
        volatile int8_t *ns = (volatile int8_t *)FC2_OUT_DDR;
        int nbest = 0, cbest = 0;
        for (int i = 1; i < 10; i++) {
            if (ns[i] > ns[nbest]) nbest = i;
            if (scores[i] > scores[cbest]) cbest = i;
        }
        print_str("  FC2 argmax: npu="); print_dec((uint32_t)nbest);
        print_str(" cpu="); print_dec((uint32_t)cbest);
        print_str(nbest == cbest ? " OK\n" : " MISMATCH\n");
```
Note: FC2's input here is the **NPU** FC1 output (`FC1_OUT_DDR`, bytes 50..63
already exact zeros from the padded OC computation) — this exercises the real
deploy chain. `scores` (CPU) is computed from the CPU FC1 output; if FC1 parity
is OK these inputs are identical.

- [ ] **Step 4.2: Run parity sim, apply criterion**

```bash
cd /e/code/6-10/soc && bash run_all.sh sim 2>&1 | grep -E "parity|argmax|Result:|ALL TESTS"
```
Expected: 10× `FC1 parity: OK`, 10× `FC2 argmax: ... OK`, `10/10`.
- All OK → FC2 ships on NPU (Task 5 uses NPU for both).
- Any `MISMATCH` → record which digits, keep FC2 on CPU in Task 5, state the clamping reason in the final report. Do NOT silently rescale.

**Checkpoint:** Parity evidence collected for both FC layers.

---

### Task 5: Switch deploy path to NPU FC, full verify + profile

**Files:**
- Modify: `firmware/deepnet_deploy.c`

- [ ] **Step 5.1: Make NPU the deploy path**

Restructure the tail of `deepnet_inference` (keep the CPU path compilable under
the parity switch — it is the reference oracle):

```c
#if NPU_GEMM_PARITY
    /* parity build: CPU is the deploy result; NPU compared above (Task 3/4 blocks) */
#else
    {
        PROF_T0();
        // ---- FC1 + FC2 on NPU (GEMM mode) ----
        npu_gemm_pass(AFFINE1_IN, AFFINE1_OUT, SCALE_AFFINE1, affine1_b, 1,
                      AFFINE_SCR, FC1_OUT_DDR, FC1_WGT_BASE);
        npu_gemm_pass(50, 10, SCALE_AFFINE2, affine2_b, 0,
                      FC1_OUT_DDR, FC2_OUT_DDR, FC2_WGT_BASE);
        volatile int8_t *ns = (volatile int8_t *)FC2_OUT_DDR;
        for (int i = 0; i < 10; i++)
            scores[i] = (int32_t)ns[i];           // sign-extended int8 scores
        PROF_ADD(prof_affine);
    }
#endif
```

Concretely: wrap the existing CPU reorder-to-RAM line, `cpu_affine_layer` call
and FC2 loop in `#if NPU_GEMM_PARITY ... #endif` (they're the oracle), keep the
DDR write in the reorder loop unconditional, and add the block above. If Task 4
said FC2-on-CPU: in the `#else` branch keep the CPU FC2 loop reading
`(volatile int8_t *)FC1_OUT_DDR` as input instead of the second
`npu_gemm_pass`. Guard unused statics (`affine_in`/`affine_mid`,
`cpu_affine_layer`) with the parity switch or `__attribute__((unused))` to stay
warning-clean.

- [ ] **Step 5.2: Flip switch to deploy mode, full verification**

Set `#define NPU_GEMM_PARITY 0`, then:

```bash
cd /e/code/6-10/soc && bash run_all.sh sim 2>&1 | grep -E "Result:|TRAP|ALL TESTS|timeout|Errors"
```
Expected: `=== Result: 10/10 correct ===`, `ALL TESTS PASSED.`, TRAP cycles ≈ **18–20M** (down from 40,873,968).

- [ ] **Step 5.3: Profile run (evidence for the report)**

Set `#define NPU_PROFILE 1`, run sim, capture the breakdown (expect `affine`
category < 1M, was 22.3M), then set `NPU_PROFILE 0` and rebuild. Record both
numbers.

- [ ] **Step 5.4: Final clean verification**

`NPU_GEMM_PARITY 0`, `NPU_PROFILE 0`: one more `bash run_all.sh sim` → 10/10 +
PASSED + final cycle count. This is the number reported to the user.

**Checkpoint:** Deploy on NPU GEMM verified end-to-end with measured speedup.

---

### Task 6: Documentation + memory

**Files:**
- Modify: `CLAUDE.md`
- Modify: `C:\Users\Lenovo\.claude\projects\E--code-6-10-soc\memory\` (new/updated entries)

- [ ] **Step 6.1: CLAUDE.md updates**
  - Register map table: CTRL row → add `[6]out_ping, [7]gemm_en`.
  - New architecture decision: "F: General GEMM/FC mode — `gemm_en` CTRL[7] bypasses im2col, replicates the IC-tile act word to all 16 rows, streams ceil(IC/16) k-steps; IC ≤ 1024 (wgt ic_group width); OC firmware-tiled per decision D; runs in PONG banks (conv keeps PING); wgt_reader kernel-offset count is runtime `kh*kw`."
  - DeepConvNet table: Affine1/Affine2 rows `[CPU]` → `[NPU]` (or FC2 `[CPU]` if Task 4 criterion failed); update the stale "pooling not used" note while in the file (it is wrong today: Conv2/4/6 fuse pooling).
  - DDR buffer table: note FC1/FC2 output staging inside `NPU_OUT_BUF`.

- [ ] **Step 6.2: Memory updates**
  - Update `soc-npu-pooling.md`-style entry or add `soc-npu-gemm-mode.md`: GEMM mode exists (CTRL[7]), constraints (IC≤1024, PONG-bank convention, pack_fc_tile layout), and the measured cycle results. Link [[soc-npu-general-purpose]].
  - Update `MEMORY.md` index line.

- [ ] **Step 6.3: Offer the user a commit**

Summarize the session's working-tree changes (license fix, overlap switch,
profiling, FAST_MUL, RAM affine cache, GEMM mode, docs) and ask whether to
commit — per project rule, do not commit unprompted.

---

## Self-review notes

- **Spec coverage:** CTRL[7] ✓(T2), act feed mux ✓(T2.4, replication refinement), wgt_reader runtime offsets ✓(T1), FSM branch ✓(T2.3), firmware `npu_gemm_pass` ✓(T3), parity testing ✓(T3/T4), end-to-end + cycles ✓(T5), docs ✓(T6). Edge cases: OC pad ✓(zero weights/bias→exact 0), IC pad ✓(FC2 in=50→4 words, zeros), signedness unchanged, ReLU per-pass ✓.
- **Deviation from spec (documented):** all-rows replication instead of single active row — removes the drain-phase risk the spec itself flagged; behavior matches conv mode exactly.
- **Type consistency:** `npu_gemm_pass(in_dim, out_dim, scale_mul_val, biases, relu_en, in_ddr, out_ddr, wgt_base)` used identically in T3/T4/T5; `pack_fc_tile`/`preload_fc_weights`/`FC*_WGT_BASE`/`FC*_OUT_DDR` defined once in T3 and reused.
- **HW caps documented:** in_dim ≤ 1024; kernel offsets ≤ 9 runtime (4-bit compares).
