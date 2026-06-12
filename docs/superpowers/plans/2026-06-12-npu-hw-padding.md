# NPU Hardware Padding (eliminate CPU pad_activation) — Spec + Plan

> Inline execution (executing-plans). Steps use `- [ ]`. **No commits unless the user asks.**

**Goal:** Eliminate the CPU `pad_activation` (6.4M cycles) by having the conv front-end read the previous layer's output **tile-major directly from Act SRAM** and **inject border zeros in the FSM** (virtual padding). im2col/systolic/post unchanged. General over any pad via a new `NPU_PAD` register; gated by `CTRL[8] hw_pad_en` for incremental bring-up.

**Why this works (verified):** conv & pooled outputs are both regular **tile-major, row-major** in DDR (`word = tile*spatial + pos`) — the existing `pad_activation` already treats them so. So the previous layer's output can be DMA'd contiguously into Act SRAM and read tile-major. The FSM feeds im2col the *same logical padded stream* it gets today (border zeros injected instead of stored), so im2col sees identical windows ⇒ conv output **bit-identical**.

**Acceptance gate:** conv output bit-identical → `10/10` + `ALL TESTS PASSED`, with the per-layer NPU result unchanged. Baseline to beat: 16,136,813 cycles (deploy). Expected after: ~10M.

**Timing model (matches existing 1-cycle SRAM-read delay):** FSM issues act addr in cycle N; `act_sram_doa` + delayed `pixel_vld_d` arrive N+1. New `border_d` (FSM `o_border` registered in npu_top, like `pixel_vld_d`) selects the injected zero at N+1. So `im2col.i_pixel_data = border_d ? 0 : act_sram_doa`.

---

### Task 1: RTL — NPU_PAD register + CTRL[8] hw_pad_en (dormant)

**Files:** `rtl/param_regfile.v`, `firmware/firmware.h`

- [ ] **1.1** param_regfile: add `reg ctrl_hw_pad;` + output `o_hw_pad`; decode `ctrl_hw_pad <= s_axi_wdata[8];` in CTRL write; reset `<=0`; readback bit 8; `assign o_hw_pad = ctrl_hw_pad;`. Header comment append `[8]hw_pad`.
- [ ] **1.2** param_regfile: add `reg [15:0] pad_cfg;` (`{pad_h[15:8], pad_w[7:0]}`); output `o_pad_w[7:0]`, `o_pad_h[7:0]`; decode new case `10'h150: pad_cfg <= s_axi_wdata[15:0];`; reset `<=0`; readback `10'h150: rdata <= {16'd0, pad_cfg};`; `assign o_pad_w = pad_cfg[7:0]; assign o_pad_h = pad_cfg[15:8];`.
- [ ] **1.3** firmware.h: `#define NPU_PAD (NPU_BASE + 0x150)` and `#define NPU_CTRL_HW_PAD (1 << 8)`; update NPU_CTRL comment.
- [ ] **1.4** Build (`run_all.sh clean && sim`) — still 10/10, exactly 16,136,813 (regs unused, dormant).

### Task 2: RTL — FSM tile-major read + border, npu_top zero-inject (dormant)

**Files:** `rtl/top_controller_fsm.v`, `rtl/npu_top.v`

- [ ] **2.1** FSM ports: add `input i_hw_pad`, `input [7:0] i_pad_w`, `input [7:0] i_pad_h`, output `o_border`.
- [ ] **2.2** FSM wires:
```verilog
wire [15:0] unpad_w = i_dim_in_w - {7'd0, i_pad_w, 1'b0};  // in_w - 2*pad_w
wire [15:0] unpad_h = i_dim_in_h - {7'd0, i_pad_h, 1'b0};
wire [SRAM_ADDR_W-1:0] unpad_spatial = unpad_w * unpad_h;
wire at_border = i_hw_pad &&
     ((cur_in_col < {8'd0,i_pad_w}) || (cur_in_col >= i_dim_in_w - {8'd0,i_pad_w}) ||
      (cur_in_row < {8'd0,i_pad_h}) || (cur_in_row >= i_dim_in_h - {8'd0,i_pad_h}));
wire [SRAM_ADDR_W-1:0] tilemaj_addr = act_base_addr
     + load_tile * unpad_spatial
     + (cur_in_row - {8'd0,i_pad_h}) * unpad_w
     + (cur_in_col - {8'd0,i_pad_w});
assign o_border = at_border;
```
- [ ] **2.3** FSM act read mux (replace current `o_act_sram_addr`/`o_act_sram_en` for the conv path; keep GEMM path):
```verilog
assign o_act_sram_addr = i_gemm_en ? gemm_act_addr
                       : i_hw_pad  ? tilemaj_addr
                       :             act_rd_addr;
assign o_act_sram_en   = i_gemm_en ? ((state==S_PREFETCH_WGT)&&(pf_wait_cnt==16'd0))
                       : ((state==S_LOAD_ROW)&&(cur_in_col<i_dim_in_w)&&(load_col_cnt>16'd0)
                          && (!i_hw_pad || !at_border));   // skip SRAM read on border
```
(`o_im2col_pixel_vld` unchanged — still fires for border cols so im2col gets the zero pixel.)
- [ ] **2.4** npu_top: register `o_border` → `border_d` (alongside `fsm_im2col_pixel_vld_d`); change im2col `.i_pixel_data` to `border_d ? {ACT_DATA_W{1'b0}} : fsm_im2col_pixel_data`. Wire FSM `.i_hw_pad(cfg_hw_pad)`, `.i_pad_w(cfg_pad_w)`, `.i_pad_h(cfg_pad_h)`, `.o_border(fsm_border)`; add `cfg_hw_pad/cfg_pad_w/cfg_pad_h` wires + param_regfile connections.
- [ ] **2.5** Build — 10/10, exactly 16,136,813 (hw_pad still 0 everywhere → dormant; the muxes select the old path).

### Task 3: Firmware — convert ONE layer (Conv2) to hw-pad, verify bit-identical

**Files:** `firmware/deepnet_deploy.c`

- [ ] **3.1** Add `pad` parameter to `npu_conv_pass` (or a sibling): it writes `NPU_PAD = (pad<<8)|pad` and sets `NPU_CTRL_HW_PAD` in the CTRL start word. Keep `NPU_IN_W/H = padded` (so out dims unchanged).
- [ ] **3.2** For Conv2 only: remove its `pad_activation` + `dma_ddr_to_act(PAD_BUF,...)`; instead `dma_ddr_to_act(ACT_BUF_B /*prev conv output, unpadded*/, 0, 28*28*tiles)`; call conv with `pad=1, hw_pad`. Leave Conv1/3/4/5/6 on the old CPU-pad path.
- [ ] **3.3** Build+sim → **10/10**, predictions identical. Conv2's path now hardware-padded. If FAIL: waveform-debug the border/tile-major timing (likely `border_d` alignment or `unpad_spatial`). Fix before proceeding.

### Task 4: Firmware — convert ALL conv layers; delete pad_activation

- [ ] **4.1** Convert Conv1 (image input, tile=1), Conv3/4/5/6 to hw-pad like Conv2. Conv1: DMA the raw 28×28 image directly (no inline scatter), pad=1.
- [ ] **4.2** Remove the now-unused `pad_activation` (and Conv1 scatter block); `PAD_BUF` macro may become unused — mark/remove.
- [ ] **4.3** Build+sim → **10/10**, `ALL TESTS PASSED`. Record cycles (expect ~10M).
- [ ] **4.4** Profile (`NPU_PROFILE=1`): confirm `pad` ≈ 0. Revert profile.

### Task 5: Docs + memory + offer commit

- [ ] **5.1** CLAUDE.md: decision **H** (hardware padding: CTRL[8] hw_pad + NPU_PAD, tile-major read + FSM border zero-injection, eliminates pad_activation; general); update cycle figure; note pad_activation removed.
- [ ] **5.2** Memory: `soc-npu-hw-padding.md` + index. Note E will migrate padding fully into im2col when the front-end is rewritten.
- [ ] **5.3** Summarize changes; offer commit.

---

## Self-review
- Coverage: NPU_PAD reg ✓(1.2), CTRL[8] ✓(1.1), FSM tile-major+border ✓(2.2-2.3), zero-inject timing ✓(2.4), incremental gate ✓(per-layer hw_pad, T3 before T4), bit-identical gate ✓(3.3/4.3), delete pad ✓(4.2), general (NPU_PAD any pad) ✓.
- Risk: FSM border/timing — mitigated by dormant checkpoints (1.4/2.5) + one-layer bring-up (T3) before all (T4).
- GEMM unaffected: act mux keeps `i_gemm_en` branch first; hw_pad only affects conv path.
