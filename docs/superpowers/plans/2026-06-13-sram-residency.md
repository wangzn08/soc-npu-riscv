# On-Chip SRAM Residency (conv→conv) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the DDR round-trip between conv layers by adding an on-chip Out SRAM → Act SRAM copy engine, so each conv layer's output feeds the next layer's input without going through DDR (~30–40K cycles/image saved).

**Architecture:** A small dedicated `sram_copy` FSM time-shares the existing SRAM Port-B paths (Out Port B read → Act Port B write), isolated from the proven `axi_dma` DDR engine. Triggered by a new register, polled for completion. Firmware replaces the `dma_out_to_ddr` + `dma_ddr_to_act` pair at each conv→conv boundary with one on-chip copy; the NPU compute path (im2col/systolic/post/FSM) is unchanged.

**Tech Stack:** Verilog-2001 RTL (ModelSim/Questa), rv32imc firmware (C). No unit-test framework — "the test" is `bash run_all.sh sim`; gate is **byte-identical output, 10/10 MNIST, `ALL TESTS PASSED`**.

---

## CRITICAL build/run note (every task)

`run_all.sh` MUST run with CWD = repo root. The **Bash tool starts in the wrong directory and strips `cd`** — use the **PowerShell tool** instead (its CWD is `E:\code\6-10\soc`): `bash run_all.sh clean`, then `bash run_all.sh sim`. Use PowerShell for git too. Success = `=== Result: 10/10 correct ===`, `ALL TESTS PASSED.`, `Errors: 0`. After editing RTL run `bash run_all.sh clean` first. Branch `feat/sram-residency` already exists — do NOT create a branch.

**Bit-identical gate:** residency only changes *transport* (the data crossing each boundary is byte-identical to what DDR carried), so every layer's output must stay byte-for-byte unchanged. Cycle count WILL drop (fewer DDR loads), so the gate is **predictions 10/10 + per-layer DEBUG_VERBOSE spot-checks matching golden**, not cycle count.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `rtl/sram_copy.v` | The copy FSM: Out Port B read → Act Port B write, `len` words | Create |
| `axi_sys.f` | Add `rtl/sram_copy.v` to the compile list | Modify |
| `rtl/param_regfile.v` | Decode `COPY_TRIG` (0x154) → `o_copy_trig` pulse; expose `i_copy_done` in STATUS[2] | Modify |
| `firmware/firmware.h` | `NPU_DMA_COPY_TRIG` (0x154); STATUS copy-done bit | Modify |
| `rtl/npu_top.v` | Instantiate `sram_copy`; Port-B mux (copy vs dma); wire trigger/done | Modify |
| `firmware/deepnet_deploy.c` | `dma_out_to_act` helper; `npu_conv_pass` destination mode; wire 5 boundaries; skip consumer loads | Modify |

**Unchanged:** im2col, systolic/gp/pe, wgt_reader, post_process, max_pooling, top_controller_fsm, axi_dma (DDR path untouched), SRAM wrappers.

---

## Task 0: Golden baseline

**Files:** none

- [ ] **Step 1: Confirm on the branch and capture the golden predictions + spot-checks**

Run (PowerShell): `bash run_all.sh clean | Out-Null; $env:EXTRA_CFLAGS='-DDEBUG_VERBOSE'; bash run_all.sh sim 2>$null | Select-String -Pattern 'conv1\(7,10\)|conv3\(7,7\)|Pool1\]|Pool2\]|Result:|ALL TESTS' | Select-Object -First 12; Remove-Item Env:\EXTRA_CFLAGS`
Expected: record these golden lines (Conv1(7,10) image-0 bytes, Conv3(7,7) bytes, Pool1/Pool2 nz, 10/10). Save them — every later task diffs against these.

- [ ] **Step 2: Capture the golden cycle count**

Run: `bash run_all.sh sim 2>$null | Select-String 'TRAP after'`
Expected: note the baseline (post-E/quiet ≈ 9,857,218). Residency should reduce it.

---

## Task 1: Add copy trigger register + status bit (dormant)

**Files:**
- Modify: `firmware/firmware.h` (after the `NPU_DMA_PING_SEL` line, ~line 75)
- Modify: `rtl/param_regfile.v` (port, reg, decode at `10'h154`, STATUS readback, output)

- [ ] **Step 1: firmware.h — add the offset + status bit**

After `#define NPU_DMA_PING_SEL     (NPU_BASE + 0x14C)  // ...`:
```c
#define NPU_DMA_COPY_TRIG    (NPU_BASE + 0x154)  // write any value: trigger on-chip Out->Act copy
#define NPU_DMA_STATUS_COPY_DONE (1 << 2)        // NPU_DMA_STATUS bit2: copy complete
```

- [ ] **Step 2: param_regfile.v — add the output port + done input**

In the port list near `o_dma_out_ping_sel` (~line 158) add:
```verilog
    output wire                         o_copy_trig,     // 0x154 write: pulse to start on-chip copy
    input  wire                         i_copy_done      // copy engine: completion (level)
```
(Add a comma to the previous last port as needed.)

- [ ] **Step 3: param_regfile.v — add the trigger pulse reg**

Near `reg dma_wr_req_d;` (~line 228) add:
```verilog
    reg        copy_trig_d;    // 1-cycle delayed pulse for on-chip copy trigger
```
In the reset block near `dma_wr_req_d <= 1'b0;` (~line 314 and ~line 334 — there are two; add to both, mirroring `dma_wr_req_d`):
```verilog
            copy_trig_d      <= 1'b0;
```
(The second one, ~line 334, is the per-cycle pulse self-clear: `copy_trig_d <= 1'b0;` so the pulse is 1 cycle.)

- [ ] **Step 4: param_regfile.v — decode the write at 0x154**

In the write `case` near `10'h150: pad_cfg <= ...` (~line 435) add:
```verilog
                    10'h154: copy_trig_d <= 1'b1;   // trigger on-chip Out->Act copy
```

- [ ] **Step 5: param_regfile.v — expose copy_done in STATUS readback**

Change the DMA status readback (line 506) from:
```verilog
                    10'h140: rdata <= {30'd0, i_dma_wr_done, i_dma_rd_done};
```
to:
```verilog
                    10'h140: rdata <= {29'd0, i_copy_done, i_dma_wr_done, i_dma_rd_done};
```

- [ ] **Step 6: param_regfile.v — drive the output**

Near `assign o_dma_out_ping_sel = dma_out_ping_sel;` (~line 592) add:
```verilog
    assign o_copy_trig = copy_trig_d;
```

- [ ] **Step 7: npu_top.v — connect the new regfile ports (tie done to 0 for now)**

In the `u_param_regfile` instantiation, after `.o_dma_out_ping_sel(cfg_dma_out_ping_sel)` add:
```verilog
        ,.o_copy_trig      (cfg_copy_trig)
        ,.i_copy_done      (copy_done)
```
Declare near the other cfg wires:
```verilog
    wire cfg_copy_trig;
    wire copy_done = 1'b0;   // TEMP: replaced by the copy engine in Task 3
```

- [ ] **Step 8: Compile + run — bit-identical (dormant)**

Run: `bash run_all.sh clean` then `bash run_all.sh sim`. Expected: 10/10, ALL TESTS PASSED, Errors 0, cycle count unchanged (nothing triggers a copy; STATUS[2] reads 0).

- [ ] **Step 9: Commit**

```
git add firmware/firmware.h rtl/param_regfile.v rtl/npu_top.v
git commit -m "feat(npu): add copy-trigger reg (0x154) + STATUS copy_done bit (dormant)"
```

---

## Task 2: Create the `sram_copy` engine module

**Files:**
- Create: `rtl/sram_copy.v`
- Modify: `axi_sys.f` (add the file)

- [ ] **Step 1: Write rtl/sram_copy.v**

```verilog
// Filename: sram_copy.v
// -------------------------------------------------------------------
// On-chip SRAM copy engine (task: SRAM residency).
// Copies `i_len` 128-bit words from Out SRAM (Port B read) to Act SRAM
// (Port B write), src/dst word bases configurable.  One word/cycle with a
// 1-cycle SRAM read latency (read addr issued, data lands next cycle and is
// written).  Banks are selected by the SRAM wrappers' dma_ping_sel (driven by
// cfg_dma_out_ping_sel / cfg_dma_act_ping_sel in npu_top) — not here.
//   i_len is the FULL word count (not count-1).
// -------------------------------------------------------------------
module sram_copy #(
    parameter ADDR_W = 14,
    parameter DATA_W = 128
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                i_trig,        // 1-cycle pulse: start
    input  wire [ADDR_W-1:0]   i_src_base,    // Out SRAM word base
    input  wire [ADDR_W-1:0]   i_dst_base,    // Act SRAM word base
    input  wire [15:0]         i_len,         // word count (full)
    // Out SRAM Port B read
    output wire [ADDR_W-1:0]   o_out_rd_addr,
    output wire                o_out_rd_en,
    input  wire [DATA_W-1:0]   i_out_rd_data,
    // Act SRAM Port B write
    output wire [ADDR_W-1:0]   o_act_wr_addr,
    output wire                o_act_wr_en,
    output wire [DATA_W-1:0]   o_act_wr_data,
    output wire                o_busy,
    output reg                 o_done         // set on completion, cleared on i_trig
);
    reg              busy;
    reg [15:0]       rd_cnt;       // reads issued
    reg [15:0]       wr_cnt;       // writes completed
    reg [15:0]       len_q;
    reg [ADDR_W-1:0] src_q, dst_q;
    reg              rd_vld_d;     // a read was issued last cycle -> data valid now
    reg [ADDR_W-1:0] wr_addr_d;    // matching write address

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; rd_cnt <= 16'd0; wr_cnt <= 16'd0; len_q <= 16'd0;
            src_q <= {ADDR_W{1'b0}}; dst_q <= {ADDR_W{1'b0}};
            rd_vld_d <= 1'b0; wr_addr_d <= {ADDR_W{1'b0}}; o_done <= 1'b0;
        end else begin
            rd_vld_d <= 1'b0;
            if (i_trig) begin
                busy   <= 1'b1;
                rd_cnt <= 16'd0; wr_cnt <= 16'd0;
                len_q  <= i_len; src_q <= i_src_base; dst_q <= i_dst_base;
                o_done <= 1'b0;
            end else if (busy) begin
                // Issue a read each cycle while reads remain
                if (rd_cnt < len_q) begin
                    rd_vld_d  <= 1'b1;
                    wr_addr_d <= dst_q + rd_cnt[ADDR_W-1:0];
                    rd_cnt    <= rd_cnt + 16'd1;
                end
                // A write completes the cycle after its read was issued
                if (rd_vld_d) begin
                    wr_cnt <= wr_cnt + 16'd1;
                    if (wr_cnt + 16'd1 == len_q) begin
                        busy   <= 1'b0;
                        o_done <= 1'b1;
                    end
                end
            end
        end
    end

    assign o_out_rd_addr = src_q + rd_cnt[ADDR_W-1:0];
    assign o_out_rd_en   = busy && (rd_cnt < len_q);
    assign o_act_wr_addr = wr_addr_d;
    assign o_act_wr_en   = rd_vld_d;
    assign o_act_wr_data = i_out_rd_data;   // Out Port B read data -> Act Port B write
    assign o_busy        = busy;
endmodule
```

- [ ] **Step 2: Add to axi_sys.f**

Add the line `rtl/sram_copy.v` to `axi_sys.f` (next to the other `rtl/*.v` entries, e.g. after `rtl/post_process_top.v`).

- [ ] **Step 3: Compile — module compiles, still unused**

Run: `bash run_all.sh clean` then `bash run_all.sh compile`. Expected: `Errors: 0` (the module compiles; it is instantiated in Task 3).

- [ ] **Step 4: Commit**

```
git add rtl/sram_copy.v axi_sys.f
git commit -m "feat(npu): add sram_copy engine (Out PortB -> Act PortB word copy)"
```

---

## Task 3: Instantiate the copy engine + Port-B mux (dormant)

**Files:**
- Modify: `rtl/npu_top.v` (instantiate `sram_copy`; mux Out/Act Port B; replace the temp `copy_done`)

- [ ] **Step 1: Remove the temp copy_done, declare engine wires**

Replace the temp line from Task 1:
```verilog
    wire copy_done = 1'b0;   // TEMP: replaced by the copy engine in Task 3
```
with:
```verilog
    wire                     copy_done;
    wire                     copy_busy;
    wire [SRAM_ADDR_W-1:0]   copy_out_rd_addr;
    wire                     copy_out_rd_en;
    wire [SRAM_ADDR_W-1:0]   copy_act_wr_addr;
    wire                     copy_act_wr_en;
    wire [ACT_DATA_W-1:0]    copy_act_wr_data;
```

- [ ] **Step 2: Instantiate the engine (near the axi_dma instance)**

```verilog
    // On-chip Out->Act copy engine (SRAM residency). src/dst/len reuse the DMA
    // SRAM-base/len registers; banks follow cfg_dma_out_ping_sel / cfg_dma_act_ping_sel
    // via the SRAM wrappers' dma_ping_sel.  i_len = full word count (cfg_dma_rd_len).
    sram_copy #(
        .ADDR_W (SRAM_ADDR_W),
        .DATA_W (ACT_DATA_W)
    ) u_sram_copy (
        .clk           (clk),
        .rst_n         (rst_n),
        .i_trig        (cfg_copy_trig),
        .i_src_base    (cfg_dma_rd_sram_base),
        .i_dst_base    (cfg_dma_wr_sram_base),
        .i_len         (cfg_dma_rd_len),
        .o_out_rd_addr (copy_out_rd_addr),
        .o_out_rd_en   (copy_out_rd_en),
        .i_out_rd_data (out_sram_dob),
        .o_act_wr_addr (copy_act_wr_addr),
        .o_act_wr_en   (copy_act_wr_en),
        .o_act_wr_data (copy_act_wr_data),
        .o_busy        (copy_busy),
        .o_done        (copy_done)
    );
```
(`cfg_dma_rd_sram_base`, `cfg_dma_wr_sram_base`, `cfg_dma_rd_len`, `out_sram_dob` all already exist in npu_top.)

- [ ] **Step 3: Mux the copy onto Act SRAM Port B**

Find the Act Port B assigns (~lines 844-847):
```verilog
    assign act_sram_addrb = act_dma_wr_active ? dma_sram_wr_addr : dma_sram_rd_addr;
    assign act_sram_enb   = act_dma_wr_active | act_dma_rd_active;
    assign act_sram_dib   = dma_sram_wr_data;
    assign act_sram_web   = act_dma_wr_active;
```
Replace with (copy takes priority while busy):
```verilog
    assign act_sram_addrb = copy_busy ? copy_act_wr_addr
                          : act_dma_wr_active ? dma_sram_wr_addr : dma_sram_rd_addr;
    assign act_sram_enb   = copy_busy ? copy_act_wr_en
                          : (act_dma_wr_active | act_dma_rd_active);
    assign act_sram_dib   = copy_busy ? copy_act_wr_data : dma_sram_wr_data;
    assign act_sram_web   = copy_busy ? copy_act_wr_en : act_dma_wr_active;
```

- [ ] **Step 4: Mux the copy onto Out SRAM Port B**

Find the Out Port B final assigns (~lines 870-871):
```verilog
    assign out_sram_addrb = out_sram_addrb_mux;
    assign out_sram_enb   = out_sram_enb_mux;
```
Replace with:
```verilog
    assign out_sram_addrb = copy_busy ? copy_out_rd_addr : out_sram_addrb_mux;
    assign out_sram_enb   = copy_busy ? copy_out_rd_en   : out_sram_enb_mux;
```

- [ ] **Step 5: Compile + run — bit-identical (dormant)**

Run: `bash run_all.sh clean` then `bash run_all.sh sim`. Expected: 10/10, ALL TESTS PASSED, Errors 0, cycle count unchanged. The engine exists and is wired, but firmware never writes `NPU_DMA_COPY_TRIG`, so `copy_busy` stays 0 and the Port-B muxes select the legacy DMA arm — byte-identical.

- [ ] **Step 6: Commit**

```
git add rtl/npu_top.v
git commit -m "feat(npu): instantiate sram_copy + Port-B mux (dormant until firmware triggers)"
```

---

## Task 4: Firmware `dma_out_to_act` helper + `npu_conv_pass` destination mode

**Files:**
- Modify: `firmware/deepnet_deploy.c` (new helper; `npu_conv_pass` gains a copy-dst mode; no boundary enabled yet)

- [ ] **Step 1: Add the copy helper (near `dma_out_to_ddr`, ~line 200)**

```c
// ================================================================
// On-chip copy: Out SRAM -> Act SRAM (no DDR round-trip). Banks: reads Out
// bank `out_bank`, writes Act PING (the conv input bank).  Polls STATUS bit2.
// ================================================================
static void dma_out_to_act(uint32_t act_dst_word, uint32_t out_src_word,
                           int nwords, int out_bank)
{
    PROF_T0();
    // Bank select: Out read bank = out_bank (bit2), Act write bank = PING (bit0=0)
    npu_wr(NPU_DMA_PING_SEL, out_bank ? 0x4 : 0x0);
    npu_wr(NPU_DMA_RD_SRAM_BASE, out_src_word);   // src = Out SRAM word base
    npu_wr(NPU_DMA_WR_SRAM_BASE, act_dst_word);   // dst = Act SRAM word base
    npu_wr(NPU_DMA_RD_LEN, nwords);               // FULL word count (copy convention)
    npu_wr(NPU_DMA_COPY_TRIG, 1);                 // start; clears STATUS copy_done

    int t = DMA_IRQ_TIMEOUT;
    while (t-- > 0)
        if (npu_rd(NPU_DMA_STATUS) & NPU_DMA_STATUS_COPY_DONE) break;
    if (t <= 0) print_str("  COPY timeout!\n");

    npu_wr(NPU_DMA_PING_SEL, 0x0);                // restore default banks
    PROF_ADD(prof_load);
}
```

- [ ] **Step 2: Give `npu_conv_pass` a copy-destination mode**

Change the `npu_conv_pass` signature to add a destination-Act base (`>=0` ⇒ copy to Act SRAM instead of DMA to DDR). Find the signature (`int pad, int row_par)`) and change to:
```c
    int pad,           // hardware-pad amount each side
    int row_par,       // 1 = 16-row spatial parallelism (task E)
    int act_dst)       // >=0: copy output to Act SRAM word base act_dst (resident); <0: DMA to out_ddr_addr
```
In the SERIAL drain branch (the `#else` of `NPU_OC_OVERLAP`, ~line 414), and the OVERLAP drain points (~396, 421), replace each `dma_out_to_ddr(out_ddr_addr + P*out_words*16, 0, out_nbeats, BANK)` with a conditional. For the SERIAL branch:
```c
        if (act_dst >= 0)
            dma_out_to_act((uint32_t)act_dst + pass * out_words, 0, out_words, out_bank);
        else
            dma_out_to_ddr(out_ddr_addr + pass * out_words * 16, 0, out_nbeats, out_bank);
```
For the OVERLAP branches (drain of `prev_pass`/`prev_bank`), mirror with `act_dst + prev_pass*out_words` and `prev_bank`:
```c
        if (prev_pass >= 0) {
            if (act_dst >= 0)
                dma_out_to_act((uint32_t)act_dst + prev_pass * out_words, 0, out_words, prev_bank);
            else
                dma_out_to_ddr(out_ddr_addr + prev_pass * out_words * 16, 0, out_nbeats, prev_bank);
        }
```
and the final-pass drain after the loop likewise.

- [ ] **Step 3: Update ALL existing `npu_conv_pass` call sites to pass `-1` (DDR, unchanged)**

There are 6 conv calls (Conv1..Conv6). Append `, -1` to each (keeping current behavior). Example Conv1:
```c
    npu_conv_pass(30, 30, 16, 16, 3, 3, 1, 1,
                  1, SCALE_CONV1, conv1_b,
                  ACT_BUF_B, 784, CONV1_WGT_BASE, 0, 1, 1, -1);   // act_dst=-1 (DDR)
```
Do the same `, -1` for Conv2..Conv6.

- [ ] **Step 4: Compile + run — bit-identical (all boundaries still DDR)**

Run: `bash run_all.sh clean` then `bash run_all.sh sim`. Expected: 10/10, ALL TESTS PASSED, Errors 0, cycle count unchanged (`act_dst=-1` everywhere ⇒ the DDR path runs exactly as before; the new helper is unused).

- [ ] **Step 5: Commit**

```
git add firmware/deepnet_deploy.c
git commit -m "feat(fw): dma_out_to_act helper + npu_conv_pass copy-dst mode (all boundaries still DDR)"
```

---

## Task 5: Enable the FIRST boundary (Conv1→Conv2) and bring up bit-identical

**Files:**
- Modify: `firmware/deepnet_deploy.c` (Conv1 outputs to Act SRAM; Conv2 skips its DDR load)

Conv1: single OC pass (OC=16), out 28×28×16 = 784 words, non-pool. Conv2 reads 784 words. Conv1's output bank (single pass) = out_bank 0 (Out PING). Copy Out PING → Act PING base 0; Conv2 reads Act PING base 0.

- [ ] **Step 1: Make Conv1 copy its output into Act SRAM (base 0)**

Change Conv1's call `act_dst` from `-1` to `0` (Act SRAM word base 0, where Conv2 reads):
```c
    npu_conv_pass(30, 30, 16, 16, 3, 3, 1, 1,
                  1, SCALE_CONV1, conv1_b,
                  ACT_BUF_B, 784, CONV1_WGT_BASE, 0, 1, 1, 0);   // act_dst=0 (resident)
```

- [ ] **Step 2: Conv2 — skip its DDR load (input already resident in Act SRAM)**

Find the Conv2 input load `dma_ddr_to_act(ACT_BUF_B, 0, 28 * 28 * 1);` (the line right before Conv2's `npu_conv_pass`) and comment it out / remove it:
```c
    // Conv2 input is resident in Act SRAM (copied from Conv1's Out SRAM) — no DDR load.
    // dma_ddr_to_act(ACT_BUF_B, 0, 28 * 28 * 1);
```

- [ ] **Step 3: Compile + run, verify 10/10 + Conv2/Pool1 byte-match golden**

Run: `bash run_all.sh clean | Out-Null; $env:EXTRA_CFLAGS='-DDEBUG_VERBOSE'; bash run_all.sh sim 2>$null | Select-String 'Pool1\]|Result:|ALL TESTS|TIMEOUT|COPY timeout'; Remove-Item Env:\EXTRA_CFLAGS`
Expected: Pool1 nz lines (711/379/784/838 from Task 0 golden) **identical**, 10/10, no COPY/IRQ timeout. (Pool1 is Conv2's pooled output — if Conv2's input came through correctly, it matches.)

- [ ] **Step 4: If NOT identical or a COPY timeout — debug on waveform**

Run `bash run_all.sh waves`. Check `u_sram_copy`: on Conv1's drain, `i_trig` pulses, `o_out_rd_addr` sweeps `src..src+783`, `o_act_wr_en`/`o_act_wr_addr` lag by 1 cycle writing `0..783`, `o_done` pulses after 784 writes, `copy_busy` gates the Port-B muxes. Confirm the Out read bank (`cfg_dma_out_ping_sel`) = Conv1's write bank, and Act write bank (`cfg_dma_act_ping_sel`) = PING. Common bugs: off-by-one on `len` (784 vs 783), bank mismatch, `copy_busy` not muxing Port B. Fix smallest root cause, re-run Step 3. Do not proceed until bit-identical.

- [ ] **Step 5: Commit**

```
git add firmware/deepnet_deploy.c
git commit -m "feat(fw): resident Conv1->Conv2 boundary (on-chip copy), bit-identical 10/10"
```

---

## Task 6: Enable the remaining 4 boundaries (Conv2→3, 3→4, 4→5, 5→6)

**Files:**
- Modify: `firmware/deepnet_deploy.c`

Boundary data sizes (words) and producer OC passes:
- Conv2→3: Conv2 pooled out 14×14×16 = 196 words, 1 pass. Conv3 reads 196.
- Conv3→4: Conv3 out 14×14×32 = 392 words, 2 passes (each 196). Conv4 reads 392.
- Conv4→5: Conv4 pooled out 8×8×32 = 128 words, 2 passes (each 64). Conv5 reads 128.
- Conv5→6: Conv5 out 8×8×64 = 256 words, 4 passes (each 64). Conv6 reads 256.

For multi-pass producers, `npu_conv_pass` already copies per pass to `act_dst + pass*out_words` (Task 4) — pass them `act_dst=0` and they place tiles correctly.

- [ ] **Step 1: Enable Conv2→3** — Conv2 `act_dst=0`; remove Conv3's `dma_ddr_to_act(ACT_BUF_B, 0, 14*14*1);`

- [ ] **Step 2: Run + verify** (PowerShell, DEBUG_VERBOSE): Conv3(7,7) bytes + downstream match golden, 10/10. If fail, waveform-check the pooled-output copy (Conv2 pooled words contiguous in Out SRAM → Act base 0). Fix, re-run.

- [ ] **Step 3: Enable Conv3→4** — Conv3 `act_dst=0`; remove Conv4's `dma_ddr_to_act(ACT_BUF_A, 0, 14*14*2);`. (Conv3 is 2-pass: each pass copies 196 words to Act `0 + pass*196`.)

- [ ] **Step 4: Run + verify** (Pool2 nz 618/458/673 match golden, 10/10). If fail, waveform-check per-pass offset `pass*out_words` for the 2-pass producer.

- [ ] **Step 5: Enable Conv4→5** — Conv4 `act_dst=0`; remove Conv5's `dma_ddr_to_act(ACT_BUF_A, 0, 8*8*2);`. (Conv4 2-pass, pooled, each 64 words → Act `0 + pass*64`.)

- [ ] **Step 6: Run + verify** (10/10, downstream match).

- [ ] **Step 7: Enable Conv5→6** — Conv5 `act_dst=0`; remove Conv6's `dma_ddr_to_act(ACT_BUF_B, 0, 8*8*4);`. (Conv5 4-pass, each 64 → Act `0 + pass*64`.)

- [ ] **Step 8: Run + verify** — full bit-identical 10/10, no timeouts. Conv6 output still DMAs to DDR (`act_dst=-1`, unchanged) for the FC/reorder path.

- [ ] **Step 9: Commit**

```
git add firmware/deepnet_deploy.c
git commit -m "feat(fw): resident all 5 conv->conv boundaries, bit-identical 10/10"
```

---

## Task 7: Profile + confirm the saving

**Files:** none

- [ ] **Step 1:** `$env:EXTRA_CFLAGS='-DNPU_PROFILE=1'; bash run_all.sh clean | Out-Null; bash run_all.sh sim 2>$null | Select-String 'preload|infer_total|infer/image|load |npu |TRAP after|Result:'; Remove-Item Env:\EXTRA_CFLAGS`
- [ ] **Step 2:** Compare `infer/image` and `load` vs the Task-0 golden. Expected: `load` drops substantially (conv→conv DDR reads gone, copy cost folded in), per-image inference ~374K → ~335–345K. Record actuals. (`dma_out_to_act` is timed under `prof_load`, so part of the saving is load-vs-copy net; also check total cycle count dropped.)
- [ ] **Step 3:** If `load` did NOT drop, a boundary is still doing the DDR load (a missed `dma_ddr_to_act` removal) — audit the 5 boundaries.

---

## Task 8: Docs + memory + finalize

**Files:**
- Modify: `CLAUDE.md` (Decision J: SRAM residency)
- Memory: add a residency memory; update the index

- [ ] **Step 1:** Add a "Decision J: On-chip SRAM residency (conv→conv)" section to CLAUDE.md mirroring decisions G/H/I: the `sram_copy` engine, the 0x154 trigger + STATUS[2], reuse of DMA src/dst/len + bank selects, copy Out→Act PING (no Act ping-pong needed in serial flow), scope (5 conv→conv boundaries; Conv6→FC stays DDR), measured saving, generality.
- [ ] **Step 2:** Write a project memory (`soc-npu-sram-residency.md`) with the result + the bank/serial insight; add a one-line index entry to `MEMORY.md`.
- [ ] **Step 3:** Use `superpowers:finishing-a-development-branch` to merge.

---

## Self-review notes

- **Spec coverage:** copy engine (Task 2), register interface reuse + trigger/status (Task 1), Port-B mux/bank handling (Task 3), firmware helper + dest mode (Task 4), serial Out→Act-PING no-ping-pong (Tasks 4–6), 5-boundary scope incl. pooled + multi-pass (Tasks 5–6), bit-identical gate + incremental bring-up (every task), profile (Task 7), generality via layout-agnostic word copy (Task 2/4). Covered.
- **Placeholders:** RTL (sram_copy, muxes, registers) and the firmware helper are given in full. Tasks 5–6 bring-up describes verification gates + waveform debug (real RTL bring-up — exact copy timing confirmed on a waveform), each blocked by a bit-identical gate. The per-pass offset and bank wiring are concrete.
- **Type consistency:** `cfg_copy_trig`/`o_copy_trig`/`i_trig`, `copy_done`/`i_copy_done`/`o_done`, `copy_busy`, `act_dst` param, `dma_out_to_act(act_dst_word,out_src_word,nwords,out_bank)`, `i_len`=full word count — consistent across regfile→npu_top→sram_copy→firmware. `NPU_DMA_COPY_TRIG` (0x154) ↔ regfile `10'h154`; `NPU_DMA_STATUS_COPY_DONE` (1<<2) ↔ STATUS[2].
