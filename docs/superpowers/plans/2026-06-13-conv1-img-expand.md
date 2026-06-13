# Conv1 Image `img_expand` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace Conv1's CPU image formatting (zero-fill + scatter, ~101K cyc/image) with a deleted zero-fill plus a HW `img_expand` engine that turns the packed image into tile-major Act SRAM, cutting pad to ~6K/image.

**Architecture:** Delete the (unnecessary, IC=1 ⇒ ch1..15 don't-care) zero-fill. CPU copies the 784 image bytes contiguously to DDR; existing DMA stages them packed into Act SRAM; a new `img_expand` FSM reads each packed word and writes its 16 bytes as zero-extended 16-ch words to Conv1's input region. Mirrors the `sram_copy` engine/register pattern.

**Tech Stack:** Verilog-2001 (ModelSim), rv32imc firmware. Test = `bash run_all.sh sim`; gate = byte-identical output, 10/10, `ALL TESTS PASSED`.

---

## CRITICAL build note (every task)

Use the **PowerShell tool** for builds/git (Bash tool runs from the wrong dir, strips `cd`): `bash run_all.sh clean`, `bash run_all.sh sim`. Success = `=== Result: 10/10 correct ===`, `ALL TESTS PASSED.`, `Errors: 0`. Branch `feat/conv1-img-expand` exists — do not branch. RTL change → `bash run_all.sh clean` first.

**Golden (post-residency):** Pool1 nz = 711/379/784/838 (DEBUG_VERBOSE); total ≈ 8,829,807. The gate is **Pool1 nz match + 10/10**, not cycle count (which drops).

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `rtl/img_expand.v` | FSM: read packed Act word → write 16 zero-extended words | Create |
| `axi_sys.f` | Add `rtl/img_expand.v` | Modify |
| `rtl/param_regfile.v` | `EXPAND_TRIG` (0x158) → `o_expand_trig`; `i_expand_done` in STATUS[3] | Modify |
| `firmware/firmware.h` | `NPU_EXPAND_TRIG` (0x158), `NPU_DMA_STATUS_EXPAND_DONE` (1<<3) | Modify |
| `rtl/npu_top.v` | Instantiate `img_expand`; Act Port-B mux; status | Modify |
| `firmware/deepnet_deploy.c` | Delete zero-fill; replace scatter with copy+stage+expand | Modify |

**Unchanged:** im2col, systolic/post/FSM, max_pooling, wgt_reader, axi_dma, sram_copy, SRAM wrappers.

---

## Task 1: Delete the zero-fill (verify ch1..15 are don't-cares) — FREE win

**Files:** Modify `firmware/deepnet_deploy.c` (the Conv1 formatting block, ~line 679-688)

- [ ] **Step 1: Remove the zero-fill loop**

Change the block:
```c
        volatile int32_t *abuf = (volatile int32_t *)ACT_BUF_A;
        for (int i = 0; i < 28 * 28 * 16 / 4; i++)
            abuf[i] = 0;
        for (int i = 0; i < 28 * 28; i++)
            ((volatile int8_t *)abuf)[i * 16] = input[i];
```
to (zero-fill deleted; scatter still writes ch0 of each word, ch1..15 left as-is):
```c
        volatile int32_t *abuf = (volatile int32_t *)ACT_BUF_A;
        // No zero-fill: Conv1 IC=1, so packed weights for ch1..15 are zero and
        // those activation lanes are don't-cares. Only ch0 (the pixel) matters.
        for (int i = 0; i < 28 * 28; i++)
            ((volatile int8_t *)abuf)[i * 16] = input[i];
```

- [ ] **Step 2: Run — verify 10/10 + Pool1 byte-match (confirms the don't-care assumption)**

Run (PowerShell): `bash run_all.sh clean | Out-Null; $env:EXTRA_CFLAGS='-DDEBUG_VERBOSE'; bash run_all.sh sim 2>$null | Select-String 'Pool1\]|Result:|ALL TESTS'; Remove-Item Env:\EXTRA_CFLAGS`
Expected: Pool1 nz = 711/379/784/838 (identical), 10/10. **If different, the zero-fill was needed — STOP, revert, and reconsider** (the engine still works but must also zero ch1..15). The expected outcome is identical (ch1..15 don't-cares).

- [ ] **Step 3: Commit**

```
git add firmware/deepnet_deploy.c
git commit -m "perf(fw): delete Conv1 zero-fill (IC=1 -> ch1..15 don't-care), 10/10 (~-75K/img)"
```

---

## Task 2: Add `EXPAND_TRIG` register + STATUS[3] (dormant)

**Files:** Modify `firmware/firmware.h`, `rtl/param_regfile.v`, `rtl/npu_top.v`

- [ ] **Step 1: firmware.h — after `NPU_DMA_COPY_TRIG` line**

```c
#define NPU_DMA_EXPAND_TRIG  (NPU_BASE + 0x158)  // write any value: trigger img_expand
#define NPU_DMA_STATUS_EXPAND_DONE (1 << 3)      // NPU_DMA_STATUS bit3: img_expand complete
```

- [ ] **Step 2: param_regfile.v — port (after `i_copy_done`)**

```verilog
    output wire                         o_expand_trig,   // 0x158 write: pulse to start img_expand
    input  wire                         i_expand_done    // expander completion (level), STATUS[3]
```

- [ ] **Step 3: param_regfile.v — reg `expand_trig_d`** (declare near `copy_trig_d`; reset in BOTH reset/clear blocks like `copy_trig_d`; self-clear pulse in the per-cycle block):
```verilog
    reg        expand_trig_d;
```
add `expand_trig_d <= 1'b0;` in the reset block (near `copy_trig_d <= 1'b0;`) and in the per-cycle auto-clear block (near `copy_trig_d  <= 1'b0;`).

- [ ] **Step 4: param_regfile.v — decode (after `10'h154`)**
```verilog
                    10'h158: expand_trig_d <= 1'b1;   // trigger img_expand
```

- [ ] **Step 5: param_regfile.v — STATUS readback (extend the 0x140 line)**

Change `10'h140: rdata <= {29'd0, i_copy_done, i_dma_wr_done, i_dma_rd_done};` to:
```verilog
                    10'h140: rdata <= {28'd0, i_expand_done, i_copy_done, i_dma_wr_done, i_dma_rd_done};
```

- [ ] **Step 6: param_regfile.v — output (near `o_copy_trig`)**
```verilog
    assign o_expand_trig = expand_trig_d;
```

- [ ] **Step 7: npu_top.v — declare wires + connect ports (temp done)**

Near `wire copy_done;` declarations add:
```verilog
    wire                            cfg_expand_trig;
    wire                            expand_done = 1'b0;   // TEMP: replaced by the engine in Task 4
```
In `u_param_regfile`, after `.i_copy_done(copy_done)`:
```verilog
        ,.o_expand_trig    (cfg_expand_trig)
        ,.i_expand_done    (expand_done)
```

- [ ] **Step 8: Compile + run — bit-identical (dormant)**

`bash run_all.sh clean` then `bash run_all.sh sim`. Expected 10/10, Errors 0, cycle count unchanged from Task 1 (bit decoded, unused).

- [ ] **Step 9: Commit** — `git commit -am "feat(npu): add EXPAND_TRIG reg (0x158) + STATUS[3] (dormant)"`

---

## Task 3: Create the `img_expand` engine module

**Files:** Create `rtl/img_expand.v`; modify `axi_sys.f`

- [ ] **Step 1: Write rtl/img_expand.v**

```verilog
// Filename: img_expand.v
// -------------------------------------------------------------------
// Image expand engine: reads packed bytes from Act SRAM (Port B) and writes
// one zero-extended 128-bit word per byte back to Act SRAM (Port B). Used for
// Conv1 input: 16 packed pixels/word -> 16 tile-major words (pixel in ch0,
// ch1..15 = 0). Act Port B is a single port, so it READS one packed word then
// WRITES its 16 expanded words (1 read + 16 writes per source word).
//   i_n_out = number of OUTPUT words (= pixels); src words = ceil(n_out/16).
// Reads are combinational (act Port B COMB_B=1) -> latch on the read cycle.
// -------------------------------------------------------------------
module img_expand #(
    parameter ADDR_W = 14,
    parameter DATA_W = 128
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                i_trig,       // 1-cycle pulse: start
    input  wire [ADDR_W-1:0]   i_src_base,   // Act scratch word base (packed)
    input  wire [ADDR_W-1:0]   i_dst_base,   // Act output word base (expanded)
    input  wire [15:0]         i_n_out,      // output word count (= pixel count)
    // Act SRAM Port B (shared read/write)
    output reg  [ADDR_W-1:0]   o_addr,
    output reg                 o_en,
    output reg                 o_we,
    output reg  [DATA_W-1:0]   o_wdata,
    input  wire [DATA_W-1:0]   i_rdata,      // act_sram_dob (combinational)
    output wire                o_busy,
    output reg                 o_done
);
    localparam S_IDLE = 2'd0, S_READ = 2'd1, S_WRITE = 2'd2;
    reg [1:0]        state;
    reg [ADDR_W-1:0] src_q, dst_q;
    reg [15:0]       n_q;          // output words remaining target
    reg [15:0]       out_cnt;      // output words written so far
    reg [15:0]       sw;           // current source word index
    reg [3:0]        b;            // byte within the latched word (0..15)
    reg [DATA_W-1:0] word_q;       // latched packed source word

    assign o_busy = (state != S_IDLE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; src_q <= 0; dst_q <= 0; n_q <= 0;
            out_cnt <= 0; sw <= 0; b <= 0; word_q <= 0;
            o_addr <= 0; o_en <= 0; o_we <= 0; o_wdata <= 0; o_done <= 0;
        end else begin
            o_done <= 1'b0;
            case (state)
                S_IDLE: begin
                    o_en <= 1'b0; o_we <= 1'b0;
                    if (i_trig) begin
                        src_q <= i_src_base; dst_q <= i_dst_base; n_q <= i_n_out;
                        out_cnt <= 16'd0; sw <= 16'd0; b <= 4'd0;
                        state <= S_READ;
                    end
                end
                S_READ: begin
                    // Drive a Port-B read of the current packed source word.
                    o_addr <= src_q + sw[ADDR_W-1:0];
                    o_en   <= 1'b1; o_we <= 1'b0;
                    b      <= 4'd0;
                    state  <= S_WRITE;   // i_rdata is valid next cycle (combinational read of the registered addr)
                end
                S_WRITE: begin
                    // On the first write cycle, latch the packed word (read result).
                    // act Port B read is combinational: i_rdata reflects o_addr from
                    // the S_READ cycle. Capture it once.
                    if (b == 4'd0)
                        word_q <= i_rdata;
                    // Write expanded byte b (use i_rdata for b==0, word_q after).
                    o_addr  <= dst_q + (sw[ADDR_W-1:0] << 4) + {{(ADDR_W-4){1'b0}}, b};
                    o_en    <= 1'b1; o_we <= 1'b1;
                    o_wdata <= {{(DATA_W-8){1'b0}},
                                (b == 4'd0 ? i_rdata[7:0] : word_q[{b,3'b0} +: 8])};
                    out_cnt <= out_cnt + 16'd1;
                    if (out_cnt + 16'd1 == n_q) begin
                        state  <= S_IDLE; o_en <= 1'b0; o_we <= 1'b0; o_done <= 1'b1;
                    end else if (b == 4'd15) begin
                        sw    <= sw + 16'd1;
                        state <= S_READ;   // next packed word
                    end else begin
                        b <= b + 4'd1;     // next byte of the same word
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
```

> **Note (bring-up):** the byte-latch timing (`word_q <= i_rdata` on `b==0`, using
> `i_rdata` for byte 0 and `word_q` for 1..15) assumes the combinational read of the
> `o_addr` registered in S_READ is stable through the first S_WRITE cycle. Verify on the
> waveform in Task 5; if the read settles a cycle later, add one settle cycle in S_READ.

- [ ] **Step 2: Add to axi_sys.f** — add `rtl/img_expand.v` after `rtl/sram_copy.v`.

- [ ] **Step 3: Compile** — `bash run_all.sh clean` then `bash run_all.sh compile`. Expected `Errors: 0`.

- [ ] **Step 4: Commit** — `git add rtl/img_expand.v axi_sys.f; git commit -m "feat(npu): add img_expand engine (packed byte -> zero-extended word)"`

---

## Task 4: Instantiate `img_expand` + Act Port-B mux (dormant)

**Files:** Modify `rtl/npu_top.v`

- [ ] **Step 1: Replace temp `expand_done`, declare wires**

Replace `wire expand_done = 1'b0;   // TEMP...` with:
```verilog
    wire                     expand_done;
    wire                     expand_busy;
    wire [SRAM_ADDR_W-1:0]   expand_addr;
    wire                     expand_en;
    wire                     expand_we;
    wire [ACT_DATA_W-1:0]    expand_wdata;
```

- [ ] **Step 2: Instantiate (near `u_sram_copy`)**

```verilog
    img_expand #(
        .ADDR_W (SRAM_ADDR_W),
        .DATA_W (ACT_DATA_W)
    ) u_img_expand (
        .clk        (clk),
        .rst_n      (rst_n),
        .i_trig     (cfg_expand_trig),
        .i_src_base (cfg_dma_rd_sram_base),
        .i_dst_base (cfg_dma_wr_sram_base),
        .i_n_out    (cfg_dma_rd_len),
        .o_addr     (expand_addr),
        .o_en       (expand_en),
        .o_we       (expand_we),
        .o_wdata    (expand_wdata),
        .i_rdata    (act_sram_dob),
        .o_busy     (expand_busy),
        .o_done     (expand_done)
    );
```

- [ ] **Step 3: Add the expander as the top priority on Act Port B**

The Act Port B mux (from the residency work) currently is:
```verilog
    assign act_sram_addrb = copy_busy ? copy_act_wr_addr
                          : act_dma_wr_active ? dma_sram_wr_addr : dma_sram_rd_addr;
    assign act_sram_enb   = copy_busy ? copy_act_wr_en
                          : (act_dma_wr_active | act_dma_rd_active);
    assign act_sram_dib   = copy_busy ? copy_act_wr_data : dma_sram_wr_data;
    assign act_sram_web   = copy_busy ? copy_act_wr_en : act_dma_wr_active;
```
Replace with (expander highest priority):
```verilog
    assign act_sram_addrb = expand_busy ? expand_addr
                          : copy_busy ? copy_act_wr_addr
                          : act_dma_wr_active ? dma_sram_wr_addr : dma_sram_rd_addr;
    assign act_sram_enb   = expand_busy ? expand_en
                          : copy_busy ? copy_act_wr_en
                          : (act_dma_wr_active | act_dma_rd_active);
    assign act_sram_dib   = expand_busy ? expand_wdata
                          : copy_busy ? copy_act_wr_data : dma_sram_wr_data;
    assign act_sram_web   = expand_busy ? expand_we
                          : copy_busy ? copy_act_wr_en : act_dma_wr_active;
```

- [ ] **Step 4: Compile + run — bit-identical (dormant; firmware never triggers expand)**

`bash run_all.sh clean` then `bash run_all.sh sim`. Expected 10/10, Errors 0, cycle count unchanged from Task 1.

- [ ] **Step 5: Commit** — `git commit -am "feat(npu): instantiate img_expand + Act Port-B mux (dormant)"`

---

## Task 5: Wire Conv1 to use `img_expand`; bring up bit-identical

**Files:** Modify `firmware/deepnet_deploy.c`

Scratch region = Act word 2048 (49 packed words; free, doesn't overlap R0 0..783 / R1 1024..1807). `IMG_BUF` = a spare DDR buffer (reuse `AFFINE_SCR` 0x40016000, unused at Conv1 time, or `ACT_BUF_A`).

- [ ] **Step 1: Add the `img_expand` firmware helper (near `dma_out_to_act`)**

```c
// ================================================================
// Trigger the HW img_expand engine: Act scratch (packed bytes) -> Act dst
// (zero-extended 16-ch words). n_out = output word count (= pixels). Poll STATUS[3].
// ================================================================
static void img_expand(uint32_t act_dst_word, uint32_t act_src_word, int n_out)
{
    npu_wr(NPU_DMA_PING_SEL, 0x0);                // Act read+write bank = PING
    npu_wr(NPU_DMA_RD_SRAM_BASE, act_src_word);   // src = Act scratch
    npu_wr(NPU_DMA_WR_SRAM_BASE, act_dst_word);   // dst = Act output region
    npu_wr(NPU_DMA_RD_LEN, n_out);                // output word count
    npu_wr(NPU_DMA_EXPAND_TRIG, 1);

    int t = DMA_IRQ_TIMEOUT;
    while (t-- > 0)
        if (npu_rd(NPU_DMA_STATUS) & NPU_DMA_STATUS_EXPAND_DONE) break;
    if (t <= 0) print_str("  EXPAND timeout!\n");
}
```

- [ ] **Step 2: Replace the Conv1 formatting + load with copy+stage+expand**

Replace the Conv1 formatting block (the `{ ... scatter ... }` from Task 1) AND the
following `dma_ddr_to_act(ACT_BUF_A, 0, 28 * 28 * 1);` with:
```c
    // ---- Conv1 input: HW img_expand (no CPU scatter) ----
    // (1) contiguous copy image bytes -> DDR IMG_BUF (packed words)
    {
        PROF_T0();
        volatile uint32_t *img = (volatile uint32_t *)ACT_BUF_A;   // DDR staging
        const uint32_t *src = (const uint32_t *)input;             // 784 bytes = 196 words
        for (int i = 0; i < 28 * 28 / 4; i++)                      // 196 words
            img[i] = src[i];
        PROF_ADD(prof_pad);
    }
    // (2) stage packed image into Act scratch (49 beats = 784 bytes)
    dma_ddr_to_act(ACT_BUF_A, 2048, 28 * 28 / 16);   // 784/16 = 49 words
    // (3) expand: Act scratch(2048) packed -> Act R0(0), 784 tile-major words
    img_expand(0, 2048, 28 * 28);
```
Note: `input` is `int8_t[784]`; reading it as `uint32_t[196]` packs 4 pixels/word. The
DMA then moves 49×16-byte beats. The expander writes 784 words to Act base 0.

- [ ] **Step 3: Run — verify 10/10 + Pool1 byte-match golden**

`bash run_all.sh clean | Out-Null; $env:EXTRA_CFLAGS='-DDEBUG_VERBOSE'; bash run_all.sh sim 2>$null | Select-String 'Pool1\]|Result:|ALL TESTS|EXPAND timeout|COPY timeout'; Remove-Item Env:\EXTRA_CFLAGS`
Expected: Pool1 nz = 711/379/784/838, 10/10, no timeout.

- [ ] **Step 4: If wrong / timeout — waveform debug**

`bash run_all.sh waves`. In `u_img_expand`: on Conv1, `i_trig` pulses; S_READ drives `o_addr=2048+sw`; S_WRITE writes `o_addr=0+sw*16+b`, `o_wdata={120'b0, byte}`; `o_done` after 784 writes; `expand_busy` gates Act Port B. Check the byte-extract (`word_q[{b,3'b0}+:8]`) picks the right pixel, and the read-latch timing (the S_READ→S_WRITE handoff). Common bugs: byte-order in packing (CPU `uint32` little-endian vs expander byte index), off-by-one in `sw*16`, read not yet settled (add a settle cycle). Fix, re-run Step 3 to bit-identical.

- [ ] **Step 5: Commit** — `git commit -am "feat: Conv1 input via HW img_expand (no CPU scatter), bit-identical 10/10"`

---

## Task 6: Profile + docs + finalize

- [ ] **Step 1:** `-DNPU_PROFILE=1` run; confirm `prof_pad` ~1.01M → ~60K, per-image inference ~289K → ~195K. Record actuals.
- [ ] **Step 2:** Add "Decision K: Conv1 img_expand" to CLAUDE.md (engine, 0x158/STATUS[3], data flow, zero-fill deletion, scope, saving) + register-map rows. Add a memory + index entry.
- [ ] **Step 3:** Use `superpowers:finishing-a-development-branch` to merge.

---

## Self-review notes

- **Spec coverage:** zero-fill deletion (Task 1), register (Task 2), engine (Task 3), mux/instantiate (Task 4), firmware data flow incl. contiguous copy + stage + expand (Task 5), profile/docs (Task 6), generality via param'd byte→word primitive (Task 3). Covered.
- **Placeholders:** engine, registers, mux, firmware helper given in full. Task 5 bring-up has a waveform gate for the read-latch/byte-order timing (real RTL bring-up), blocked by the bit-identical gate.
- **Type consistency:** `cfg_expand_trig`/`o_expand_trig`/`i_trig`, `expand_done`/`i_expand_done`/`o_done`, `expand_busy`, `img_expand(act_dst,act_src,n_out)`, `NPU_DMA_EXPAND_TRIG` (0x158) ↔ regfile `10'h158`, `NPU_DMA_STATUS_EXPAND_DONE` (1<<3) ↔ STATUS[3], scratch=2048, n_out=784. Consistent.
- **Byte-order risk** (flagged Task 5): CPU packs `input` as `uint32` (little-endian: pixel i in byte i%4); the expander indexes `word_q` byte `(i%16)`. The DMA carries DDR words to Act words preserving byte order, so Act-scratch word w byte b = pixel w*16+b. Verify on the waveform.
