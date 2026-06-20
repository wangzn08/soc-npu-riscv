# DFL 期望单元 + sigmoid LUT 硬件化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 NPU 侧增加一个流式 DFL 期望硬件单元(`dfl_unit.v`)和一张 sigmoid post-process LUT,使 YOLOv8n 检测头解码的 softmax-期望与 cls-sigmoid 在 SoC 上完成,CPU 只剩整数 argmax/几何/NMS。

**Architecture:** `dfl_unit.v` 仿 `upsample2x.v`，作为 Act-SRAM Port-B 流式引擎：每读入一个 128-bit 字(16×INT8 一坐标的 16 bins)→ max 归约 → 256 项 EXP_LUT 查表 → Σe、Σe·W 累加 → 顺序整数除得 Q8.8 距离；每 4 个输入字(一个 anchor 的 4 坐标)打包成 1 个输出字写回 Act SRAM。sigmoid 复用 `post_process_top.v` 的 LUT 路径加一张表 + 选表 CTRL 位。所有新路径默认 OFF，关时 FSM/post-process 字节级不变，MNIST 10/10 不动。

**Tech Stack:** Verilog (ModelSim/Questa), PicoRV32 firmware (rv32imc, no FPU), Python 生成器, `tests/tb_dfl_unit.v` directed TB, `run_all.sh`/`tests/run_regress.sh`。

---

## 寄存器分配(param_regfile，紧接现有 0x3D0 PAD_VALUE 之后)

```
NPU_DFL_SRC   0x3D4  src_base (Act SRAM word addr)
NPU_DFL_DST   0x3D8  dst_base (Act SRAM word addr)
NPU_DFL_CNT   0x3DC  输入字数 = anchors*4
NPU_DFL_TRIG  0x3E0  写:启动 dfl_unit
NPU_DFL_WLOAD 0x3E4  W_k 装载: wdata[19:16]=idx(0..15), wdata[15:0]=W_k(Q8.8 有符号)
NPU_DFL_ELOAD 0x3E8  EXP_LUT 装载: wdata[23:16]=idx(0..255), wdata[15:0]=e(Q1.15 无符号)
NPU_SIGM_LOAD 0x3EC  sigmoid LUT 装载: wdata[15:8]=idx(0..255), wdata[7:0]=prob(Q0.8)
NPU_DMA_STATUS[5]    dfl_done (沿用 status 聚合; [4]=upsample_done 已用)
NPU_CTRL[19]         sigmoid_en (post-process 选 sigmoid 表替代 SiLU 表)
```

## 文件结构

- Create `rtl/dfl_unit.v` — DFL 期望流式引擎(本计划核心)。
- Create `tools/gen_dfl_vectors.py` — 生成 EXP_LUT/W_k/随机输入字/期望 Q8.8 距离(整数参考模型),写 `tests/dfl_vectors.mem`。
- Create `tests/tb_dfl_unit.v` — 独立自检 TB(喂向量、查结果)。
- Create `tools/gen_sigmoid_lut.py` — 生成 `rtl/sigmoid_lut_q0_8.hex`(256 项 int8→Q0.8)。
- Create `rtl/sigmoid_lut_q0_8.hex` — 由上脚本产出。
- Modify `rtl/post_process_top.v` — 加 sigmoid 表 + CTRL[19] 选表。
- Modify `rtl/param_regfile.v` — 新寄存器 + 输出端口 + W/EXP/SIGM 装载口。
- Modify `rtl/npu_top.v` — 实例化 dfl_unit、Act Port-B mux 加 `dfl_busy`、status[5]、装载口布线。
- Modify `rtl/npu_axi_wrapper.v` — 透传新端口(若该层存在端口列表)。
- Modify `axi_sys.f` — 加 `rtl/dfl_unit.v`、`tests/tb_dfl_unit.v` 不进 f(单独编译)。
- Modify `firmware/firmware.h` — 寄存器宏 + CTRL 位 + status 位。
- Modify `firmware/yolo_ops.c` / `.h` — `yolo_dfl_load_exp_lut`、`yolo_dfl_load_weights`、`yolo_run_dfl`、`yolo_load_sigmoid_lut`。
- Create `firmware/yolo_dfl_smoke.c` — 经真实 MMIO 路跑 dfl_unit + 比对(SoC 端到端证明)。

---

## Task 1: DFL 向量生成器 + 整数参考模型

**Files:**
- Create: `tools/gen_dfl_vectors.py`
- Create (output): `tests/dfl_vectors.mem`

- [ ] **Step 1: 写生成器(整数参考模型,镜像 RTL 定点)**

```python
# tools/gen_dfl_vectors.py — 生成 dfl_unit directed 向量 + 黄金。
# 模型: 每"坐标字" = 16 个 INT8 logit z[0..15].
#   idx_k = int8_max - z_k  (0..255)   e_k = EXP_LUT[idx_k] (Q1.15)
#   Sden = sum e_k ; Snum = sum e_k * W_k (W_k Q8.8 signed)
#   distance_Q8.8 = Snum // Sden  (Q1.15 因子相消)
import random, math
random.seed(1234)

SCALE = 0.05          # 该尺度 bbox 输出 dequant scale (示例)
NWORDS = 64           # 输入字数 (16 anchors * 4 coords)
# W_k = round(k * 1.0 * 256) 标准 DFL 权重 0..15 (Q8.8)，可换成真实 conv63
Wk = [round(k * 256) for k in range(16)]          # Q8.8, 这里 W_k = k
# EXP_LUT[idx] = exp(-idx*SCALE) in Q1.15 (idx=0..255), exp(0)=32768
EXP = [min(32768, round(math.exp(-i*SCALE)*32768)) for i in range(256)]

def s8(v): return v-256 if v>=128 else v

def dfl_word(z):
    mx = max(z)
    Sden=0; Snum=0
    for k in range(16):
        idx = mx - z[k]                # 0..255
        e = EXP[idx]
        Sden += e
        Snum += e * Wk[k]              # Wk signed
    if Sden==0: return 0
    d = Snum // Sden                   # Q8.8
    if d >  32767: d =  32767
    if d < -32768: d = -32768
    return d & 0xFFFF

lines=[]
# header lines: NWORDS, then EXP(256), then Wk(16), then inputs(NWORDS*16 int8 hex), then golden(NWORDS Q8.8)
inputs=[]; golden=[]
for _ in range(NWORDS):
    z=[random.randint(-40,40) for _ in range(16)]
    inputs.append(z); golden.append(dfl_word(z))

with open("tests/dfl_vectors.mem","w") as f:
    f.write(f"{NWORDS}\n")
    for v in EXP: f.write(f"{v:04x}\n")
    for v in Wk:  f.write(f"{v & 0xFFFF:04x}\n")
    for z in inputs:
        # pack 16 int8 little-endian into 128-bit hex (lane0=z[0])
        word=0
        for k in range(16): word |= (z[k]&0xFF) << (8*k)
        f.write(f"{word:032x}\n")
    for g in golden: f.write(f"{g:04x}\n")
print("wrote tests/dfl_vectors.mem", NWORDS, "words")
```

- [ ] **Step 2: 运行生成器**

Run: `python tools/gen_dfl_vectors.py`
Expected: `wrote tests/dfl_vectors.mem 64 words`，文件存在。

- [ ] **Step 3: Commit**

```bash
git add tools/gen_dfl_vectors.py tests/dfl_vectors.mem
git commit -m "feat(yolo): DFL directed-vector generator + integer golden model"
```

---

## Task 2: `dfl_unit.v` + directed TB

**Files:**
- Create: `rtl/dfl_unit.v`
- Create: `tests/tb_dfl_unit.v`

- [ ] **Step 1: 写自检 TB(先失败)**

```verilog
// tests/tb_dfl_unit.v — 读 tests/dfl_vectors.mem，逐字驱动 dfl_unit，比对 Q8.8。
`timescale 1ns/1ps
module tb_dfl_unit;
  localparam ADDR_W=14, DATA_W=128;
  reg clk=0, rst_n=0, trig=0;
  reg  [DATA_W-1:0] mem [0:1023];     // act sram model (port B comb read)
  wire [ADDR_W-1:0] addr; wire en, we; wire [DATA_W-1:0] wdata; reg [DATA_W-1:0] rdata;
  wire busy, done;
  integer nwords, i, k, errors;
  reg [15:0] golden [0:255];

  // load LUT/Wk into DUT via task-style register pokes (hierarchical) handled below
  dfl_unit #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) dut(
    .clk(clk), .rst_n(rst_n), .i_trig(trig),
    .i_src_base(14'd0), .i_dst_base(14'd512), .i_cnt(nwords[15:0]),
    .i_wload_en(1'b0), .i_wload_idx(4'd0), .i_wload_val(16'd0),
    .i_eload_en(1'b0), .i_eload_idx(8'd0), .i_eload_val(16'd0),
    .o_addr(addr), .o_en(en), .o_we(we), .o_wdata(wdata), .i_rdata(rdata),
    .o_busy(busy), .o_done(done));

  always #5 clk=~clk;
  always @(*) rdata = mem[addr];                 // COMB_B=1
  always @(posedge clk) if (en && we) mem[addr] <= wdata;

  // file parse buffers
  reg [15:0] exp_lut [0:255]; reg [15:0] wk [0:15];
  integer fd, code; reg [127:0] tmp;

  initial begin
    // --- parse tests/dfl_vectors.mem manually ---
    $readmemh("tests/dfl_vectors.mem", /*placeholder*/ exp_lut); // replaced by manual parse in impl
    // (实现时改成逐行 $fscanf: nwords, 256 exp, 16 wk, nwords inputs, nwords golden)
    rst_n=0; #20 rst_n=1;
    // poke EXP_LUT and W_k into DUT through load ports
    for (i=0;i<256;i=i+1) begin @(posedge clk); force dut.i_eload_en=1; force dut.i_eload_idx=i[7:0]; force dut.i_eload_val=exp_lut[i]; end
    @(posedge clk); force dut.i_eload_en=0;
    for (i=0;i<16;i=i+1) begin @(posedge clk); force dut.i_wload_en=1; force dut.i_wload_idx=i[3:0]; force dut.i_wload_val=wk[i]; end
    @(posedge clk); force dut.i_wload_en=0; release dut.i_eload_en; release dut.i_wload_en;
    // trigger
    @(posedge clk) trig=1; @(posedge clk) trig=0;
    wait(done);
    errors=0;
    for (i=0;i<nwords;i=i+4) begin
      // output word at dst 512 + (i/4); lanes 0..3 hold 4 Q8.8 distances
      for (k=0;k<4;k=k+1) begin
        if (mem[512+(i/4)][16*k +:16] !== golden[i+k]) begin
          errors=errors+1; $display("MISMATCH word=%0d coord=%0d got=%h exp=%h", i, k, mem[512+(i/4)][16*k +:16], golden[i+k]);
        end
      end
    end
    if (errors==0) $display("TB_DFL_UNIT PASS"); else $display("TB_DFL_UNIT FAIL errors=%0d", errors);
    $finish;
  end
endmodule
```

注:`$readmemh` 占位需在实现时替换为逐行 `$fopen`/`$fscanf` 解析(nwords→256 exp→16 wk→nwords 输入字→nwords golden)。golden 是按"坐标字"的;输出按 4 坐标打包,故 `golden[i+k]` 对应输入字 `i+k`。

- [ ] **Step 2: 运行 TB 确认失败(无 DUT)**

Run: `vlib sim/work_dfl; vlog -sv -work sim/work_dfl rtl/dfl_unit.v tests/tb_dfl_unit.v; vsim -c -lib sim/work_dfl tb_dfl_unit -do "run -all; quit -f"`
Expected: 编译失败(`dfl_unit` 未定义)。

- [ ] **Step 3: 实现 `rtl/dfl_unit.v`**

```verilog
// rtl/dfl_unit.v — 流式 DFL 期望单元(仿 upsample2x.v 的 Port-B 引擎约定)。
// 每输入字=16×INT8(一坐标16 bins)。max→EXP_LUT→Σe,Σe·W→整除→Q8.8。
// 每 4 个输入字(一个 anchor 的 4 坐标)打包成 1 个输出字写回。
module dfl_unit #(parameter ADDR_W=14, DATA_W=128) (
    input  wire               clk, rst_n, i_trig,
    input  wire [ADDR_W-1:0]  i_src_base, i_dst_base,
    input  wire [15:0]        i_cnt,            // 输入字数
    input  wire               i_wload_en, input wire [3:0] i_wload_idx, input wire [15:0] i_wload_val,
    input  wire               i_eload_en, input wire [7:0] i_eload_idx, input wire [15:0] i_eload_val,
    output wire [ADDR_W-1:0]  o_addr, output wire o_en, output wire o_we,
    output wire [DATA_W-1:0]  o_wdata, input wire [DATA_W-1:0] i_rdata,
    output wire o_busy, output reg o_done
);
    reg signed [15:0] wk [0:15];        // Q8.8
    reg        [15:0] elut [0:255];     // Q1.15 unsigned
    integer li;
    // LUT/W 装载(可与空闲并行)
    always @(posedge clk) begin
        if (i_wload_en) wk[i_wload_idx] <= i_wload_val;
        if (i_eload_en) elut[i_eload_idx] <= i_eload_val;
    end

    localparam S_IDLE=3'd0, S_READ=3'd1, S_ACC=3'd2, S_DIV=3'd3, S_WRITE=3'd4;
    reg [2:0]  state;
    reg [15:0] widx, cnt_q;             // 输入字游标 / 总数
    reg [ADDR_W-1:0] src_q, dst_q;
    reg signed [7:0]  z [0:15];
    reg signed [7:0]  zmax;
    reg [4:0]  k;                       // 0..16
    reg [31:0] sden; reg signed [47:0] snum;
    reg signed [15:0] dist;             // Q8.8 商
    reg [DATA_W-1:0]  outw;             // 打包 4 坐标
    reg [1:0]  coord;                   // 0..3 当前 anchor 内坐标

    // 顺序除法器(shift-subtract, snum/sden -> Q8.8 already since 因子相消)
    reg [5:0]  dstep; reg [47:0] rem; reg [47:0] quo; reg signed [47:0] snum_abs; reg neg;

    assign o_busy  = (state != S_IDLE);
    assign o_en    = (state==S_READ) || (state==S_WRITE);
    assign o_we    = (state==S_WRITE);
    assign o_addr  = (state==S_WRITE) ? (dst_q + {2'd0, (widx>>2)}[ADDR_W-1:0])  // 每4字一输出字
                                      : (src_q + widx[ADDR_W-1:0]);
    assign o_wdata = outw;

    integer m;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin state<=S_IDLE; o_done<=1'b0; widx<=0; coord<=0; end
        else begin
            case (state)
            S_IDLE: if (i_trig) begin
                        src_q<=i_src_base; dst_q<=i_dst_base; cnt_q<=i_cnt;
                        widx<=0; coord<=0; o_done<=1'b0; outw<=0; state<=S_READ;
                    end
            S_READ: begin
                        for (m=0;m<16;m=m+1) z[m] <= $signed(i_rdata[8*m +:8]);
                        state<=S_ACC; k<=0; sden<=0; snum<=0;
                        // zmax 在下个态算
                    end
            S_ACC: begin
                        if (k==0) begin
                            // 计算 zmax(组合 max over z[])
                            zmax = z[0];
                            for (m=1;m<16;m=m+1) if (z[m] > zmax) zmax = z[m];
                        end
                        if (k < 16) begin
                            // idx = zmax - z[k] (0..255)
                            sden <= sden + {16'd0, elut[(zmax - z[k]) & 8'hFF]};
                            snum <= snum + $signed(elut[(zmax - z[k]) & 8'hFF]) * wk[k];
                            k <= k + 1;
                        end else begin
                            // 准备除法
                            neg <= (snum < 0); snum_abs <= (snum<0)? -snum : snum;
                            rem<=0; quo<=0; dstep<=0; state<=S_DIV;
                        end
                    end
            S_DIV: begin
                        // 48 步 shift-subtract: quo = snum_abs / sden
                        if (dstep < 48) begin
                            rem = (rem << 1) | ((snum_abs >> (47-dstep)) & 48'd1);
                            quo = quo << 1;
                            if (rem >= {16'd0, sden}) begin rem = rem - {16'd0, sden}; quo = quo | 48'd1; end
                            dstep <= dstep + 1;
                        end else begin
                            dist <= neg ? -quo[15:0] : quo[15:0];
                            state <= S_WRITE_PREP;
                        end
                    end
            // 把 dist 放进 outw 的 coord 槽; 满4或末字则写回
            S_WRITE_PREP: begin
                        outw[16*coord +:16] <= dist;
                        if (coord==2'd3 || (widx+16'd1==cnt_q)) state<=S_WRITE;
                        else begin coord<=coord+2'd1; widx<=widx+16'd1; state<=S_READ; end
                    end
            S_WRITE: begin
                        if (widx+16'd1==cnt_q) begin o_done<=1'b1; state<=S_IDLE; end
                        else begin coord<=2'd0; widx<=widx+16'd1; state<=S_READ; end
                        outw<=0;
                    end
            default: state<=S_IDLE;
            endcase
        end
    end
endmodule
```

注:`S_WRITE_PREP` 也加入 localparam(3'd5)。`o_addr` 在 S_WRITE 用 `widx>>2`(上一字组)；实现时确认写地址用已完成 anchor 的索引(可加 `wr_anchor` 寄存器避免 off-by-one)。本步允许在 TB 下迭代修正时序直到 PASS。

- [ ] **Step 4: 运行 TB 直到 PASS**

Run: `python tools/gen_dfl_vectors.py && vlib sim/work_dfl 2>/dev/null; vlog -sv -work sim/work_dfl rtl/dfl_unit.v tests/tb_dfl_unit.v && vsim -c -lib sim/work_dfl tb_dfl_unit -do "run -all; quit -f" 2>&1 | grep -E "TB_DFL_UNIT|MISMATCH"`
Expected: `TB_DFL_UNIT PASS`（如失败按 MISMATCH 调 FSM/除法器时序）。

- [ ] **Step 5: Commit**

```bash
git add rtl/dfl_unit.v tests/tb_dfl_unit.v
git commit -m "feat(yolo): dfl_unit.v streaming DFL expectation engine + directed TB"
```

---

## Task 3: 接入 npu_top + param_regfile + 寄存器

**Files:**
- Modify: `rtl/param_regfile.v`
- Modify: `rtl/npu_top.v`
- Modify: `rtl/npu_axi_wrapper.v`(若有端口透传层)
- Modify: `axi_sys.f`
- Modify: `firmware/firmware.h`

- [ ] **Step 1: param_regfile 新寄存器 + 输出端口**

在 `param_regfile.v` 写解码段(仿 0x3C8 upsample trig)加：

```verilog
// 输出端口(模块头追加)
output reg  [13:0] o_dfl_src, o_dfl_dst,
output reg  [15:0] o_dfl_cnt,
output wire        o_dfl_trig,
output reg         o_dfl_wload_en, o_dfl_eload_en,
output reg  [3:0]  o_dfl_wload_idx, output reg [15:0] o_dfl_wload_val,
output reg  [7:0]  o_dfl_eload_idx, output reg [15:0] o_dfl_eload_val,
output reg         o_sigm_load_en, output reg [7:0] o_sigm_load_idx, output reg [7:0] o_sigm_load_val,
output reg         o_sigmoid_en,   // CTRL[19]
```

写通道(`s_axi` 写解码 case)：

```verilog
10'h3D4: o_dfl_src <= s_axi_wdata[13:0];
10'h3D8: o_dfl_dst <= s_axi_wdata[13:0];
10'h3DC: o_dfl_cnt <= s_axi_wdata[15:0];
10'h3E0: dfl_trig_pulse <= 1'b1;       // 单拍 pulse, 仿 upsample_trig
10'h3E4: begin o_dfl_wload_en<=1'b1; o_dfl_wload_idx<=s_axi_wdata[19:16]; o_dfl_wload_val<=s_axi_wdata[15:0]; end
10'h3E8: begin o_dfl_eload_en<=1'b1; o_dfl_eload_idx<=s_axi_wdata[23:16]; o_dfl_eload_val<=s_axi_wdata[15:0]; end
10'h3EC: begin o_sigm_load_en<=1'b1; o_sigm_load_idx<=s_axi_wdata[15:8]; o_sigm_load_val<=s_axi_wdata[7:0]; end
```

每个 `*_en` 须在非写周期清 0(仿现有 trig pulse 复位逻辑)。`o_sigmoid_en` 从 CTRL[19] 取(在 CTRL 寄存器赋值处加 `o_sigmoid_en <= ctrl[19];`)。`o_dfl_trig = dfl_trig_pulse`。

- [ ] **Step 2: npu_top 实例化 dfl_unit + Port-B mux + status**

```verilog
wire dfl_busy, dfl_done;
wire [13:0] dfl_addr; wire dfl_en, dfl_we; wire [127:0] dfl_wdata;
dfl_unit #(.ADDR_W(SRAM_ADDR_W), .DATA_W(ACT_DATA_W)) u_dfl (
  .clk(clk), .rst_n(rst_n), .i_trig(cfg_dfl_trig),
  .i_src_base(cfg_dfl_src), .i_dst_base(cfg_dfl_dst), .i_cnt(cfg_dfl_cnt),
  .i_wload_en(cfg_dfl_wload_en), .i_wload_idx(cfg_dfl_wload_idx), .i_wload_val(cfg_dfl_wload_val),
  .i_eload_en(cfg_dfl_eload_en), .i_eload_idx(cfg_dfl_eload_idx), .i_eload_val(cfg_dfl_eload_val),
  .o_addr(dfl_addr), .o_en(dfl_en), .o_we(dfl_we), .o_wdata(dfl_wdata),
  .i_rdata(act_sram_dob), .o_busy(dfl_busy), .o_done(dfl_done));
```

Act Port-B mux 加最高/次高优先级(在 expand/upsample/copy 链上追加 `dfl_busy`)：

```verilog
assign act_sram_addrb = dfl_busy ? dfl_addr : expand_busy ? expand_addr : ... ;
assign act_sram_enb   = dfl_busy ? dfl_en   : ...;
assign act_sram_dib   = dfl_busy ? dfl_wdata: ...;
assign act_sram_web   = dfl_busy ? dfl_we   : ...;
```

status 聚合加 `dfl_done` 到 bit[5]（仿 upsample_done latch 进 NPU_DMA_STATUS）。

- [ ] **Step 3: axi_sys.f 加文件**

在 `axi_sys.f` 加一行 `rtl/dfl_unit.v`。

- [ ] **Step 4: firmware.h 宏**

```c
#define NPU_DFL_SRC   (NPU_BASE + 0x3D4u)
#define NPU_DFL_DST   (NPU_BASE + 0x3D8u)
#define NPU_DFL_CNT   (NPU_BASE + 0x3DCu)
#define NPU_DFL_TRIG  (NPU_BASE + 0x3E0u)
#define NPU_DFL_WLOAD (NPU_BASE + 0x3E4u)
#define NPU_DFL_ELOAD (NPU_BASE + 0x3E8u)
#define NPU_SIGM_LOAD (NPU_BASE + 0x3ECu)
#define NPU_DMA_STATUS_DFL_DONE (1u << 5)
#define NPU_CTRL_SIGMOID_EN     (1u << 19)
```

(`NPU_BASE` 取 firmware.h 现有定义；若现用全地址宏，按现状风格写全 `0x3000_03D4` 等。)

- [ ] **Step 5: RTL 编译通过 + MNIST 不破**

Run: `bash run_all.sh clean && bash run_all.sh sim 2>&1 | grep -iE "correct|DEPLOY|Errors:"`
Expected: `10/10 correct`、`DEPLOY SUCCESS`、`Errors: 0`（dfl 默认空闲，基线字节级不变）。

- [ ] **Step 6: Commit**

```bash
git add rtl/param_regfile.v rtl/npu_top.v rtl/npu_axi_wrapper.v axi_sys.f firmware/firmware.h
git commit -m "feat(yolo): wire dfl_unit into npu_top + registers; MNIST 10/10 intact"
```

---

## Task 4: sigmoid post-process LUT

**Files:**
- Create: `tools/gen_sigmoid_lut.py`, `rtl/sigmoid_lut_q0_8.hex`
- Modify: `rtl/post_process_top.v`
- Create: `tests/tb_sigmoid_lut.v`

- [ ] **Step 1: 生成 sigmoid 表**

```python
# tools/gen_sigmoid_lut.py
import math
SCALE=0.05; ZP=0   # 该尺度 cls 输出 dequant 参数(示例, 部署时按真实值)
with open("rtl/sigmoid_lut_q0_8.hex","w") as f:
    for i in range(256):
        x = i-256 if i>=128 else i          # int8
        v = 1.0/(1.0+math.exp(-((x-ZP)*SCALE)))
        q = min(255, round(v*255))          # Q0.8 unsigned
        f.write(f"{q:02x}\n")
print("wrote rtl/sigmoid_lut_q0_8.hex")
```

Run: `python tools/gen_sigmoid_lut.py`  Expected: 文件 256 行。

- [ ] **Step 2: post_process_top 加表 + 选表(先加 TB 失败)**

`tests/tb_sigmoid_lut.v`: 实例化 post_process_top，置 `i_sigmoid_en=1`，对 256 个 INT8 输入查 `act_val`，与 `rtl/sigmoid_lut_q0_8.hex` 逐项比对；另置 `i_sigmoid_en=0` 时与现有 SiLU 路径字节一致。

```verilog
// 断言: sigmoid_en=1 时 act_val == sigmoid_lut[in&0xFF]; en=0 时不变
```

- [ ] **Step 3: 实现 post_process_top sigmoid 路径**

在 `rtl/post_process_top.v` SiLU LUT 旁加：

```verilog
reg [7:0] sigmoid_lut [0:255];
initial $readmemh("rtl/sigmoid_lut_q0_8.hex", sigmoid_lut);
wire [7:0] sigmoid_val = sigmoid_lut[silu_sat[7:0]];
// act_val 选择追加: i_sigmoid_en ? {sigmoid_val} : (现有 SiLU/requant/linear 链)
```

并在模块头加 `input wire i_sigmoid_en;`，在 npu_top 接 `o_sigmoid_en`。默认 0 时 `act_val` 表达式与现状字节级相同。

- [ ] **Step 4: 运行两个 TB**

Run: sigmoid TB → `TB_SIGMOID PASS`；再 `bash run_all.sh sim` → MNIST 10/10（en=0 不破）。

- [ ] **Step 5: Commit**

```bash
git add tools/gen_sigmoid_lut.py rtl/sigmoid_lut_q0_8.hex rtl/post_process_top.v tests/tb_sigmoid_lut.v rtl/npu_top.v
git commit -m "feat(yolo): sigmoid post-process LUT (CTRL[19]); default-off byte-identical"
```

---

## Task 5: firmware DFL/sigmoid 助手

**Files:**
- Modify: `firmware/yolo_ops.c`, `firmware/yolo_ops.h`

- [ ] **Step 1: 助手声明 + 实现**

```c
// yolo_ops.h
void yolo_dfl_load_weights(const int16_t wk_q8_8[16]);
void yolo_dfl_load_exp_lut(const uint16_t exp_q1_15[256]);
int  yolo_run_dfl(uint32_t src_act_base, uint32_t dst_act_base, uint32_t in_words);
void yolo_load_sigmoid_lut(const uint8_t prob_q0_8[256]);
```

```c
// yolo_ops.c
void yolo_dfl_load_weights(const int16_t wk[16]){
    for (uint32_t i=0;i<16u;i++) npu_wr(NPU_DFL_WLOAD, (i<<16)|((uint32_t)(uint16_t)wk[i]));
}
void yolo_dfl_load_exp_lut(const uint16_t e[256]){
    for (uint32_t i=0;i<256u;i++) npu_wr(NPU_DFL_ELOAD, (i<<16)|e[i]);
}
int yolo_run_dfl(uint32_t src, uint32_t dst, uint32_t n){
    npu_wr(NPU_DFL_SRC, src); npu_wr(NPU_DFL_DST, dst); npu_wr(NPU_DFL_CNT, n);
    npu_wr(NPU_DFL_TRIG, 1u);
    return wait_dma_status(NPU_DMA_STATUS_DFL_DONE);
}
void yolo_load_sigmoid_lut(const uint8_t p[256]){
    for (uint32_t i=0;i<256u;i++) npu_wr(NPU_SIGM_LOAD, (i<<8)|p[i]);
}
```

- [ ] **Step 2: 编译固件无警告**

Run: `bash tests/run_regress.sh compile`（或带任一 smoke 编译）Expected: 无 `-Werror` 失败。

- [ ] **Step 3: Commit**

```bash
git add firmware/yolo_ops.c firmware/yolo_ops.h
git commit -m "feat(yolo): firmware DFL/sigmoid LUT load + run helpers"
```

---

## Task 6: SoC 端到端 DFL smoke(真实 MMIO 路)

**Files:**
- Create: `firmware/yolo_dfl_smoke.c`
- Create: `firmware/yolo_dfl_smoke_data.h`(由 `tools/gen_dfl_vectors.py` 追加导出 C 头)

- [ ] **Step 1: 生成器追加导出 C 头**

在 `tools/gen_dfl_vectors.py` 末尾加：写 `firmware/yolo_dfl_smoke_data.h`，含 `EXP_LUT[256]`(uint16)、`WK[16]`(int16)、输入字 `ACT[NWORDS][4]`(uint32 lanes)、`GOLD[NWORDS]`(uint16)、`NWORDS`。

- [ ] **Step 2: 写 smoke(CPU 装 SRAM + LUT + 触发 + 比对)**

```c
#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_dfl_smoke_data.h"
#define ACT_DDR 0x40090000u
#define SRC_BASE 0u
#define DST_BASE 512u
void usercode7(void){
  uint32_t i, errors=0;
  print_str("YOLO DFL HW SMOKE\n");
  // DMA 输入字到 Act SRAM SRC_BASE
  for (i=0;i<DFL_NWORDS;i++){ volatile uint32_t*p=(volatile uint32_t*)(ACT_DDR+i*16u);
     p[0]=DFL_ACT[i][0];p[1]=DFL_ACT[i][1];p[2]=DFL_ACT[i][2];p[3]=DFL_ACT[i][3]; }
  yolo_dma_ddr_to_act(ACT_DDR, SRC_BASE, DFL_NWORDS);
  yolo_dfl_load_exp_lut(DFL_EXP_LUT);
  yolo_dfl_load_weights(DFL_WK);
  if(!yolo_run_dfl(SRC_BASE, DST_BASE, DFL_NWORDS)){ print_str("dfl run fail\n"); errors++; }
  // 读回 dst(每4输入字=1输出字,4×Q8.8/字),与 GOLD 比对
  for (i=0;i<DFL_NWORDS && errors<=16;i++){
     volatile uint32_t*p=(volatile uint32_t*)(/*Act SRAM 不可 CPU 直读 -> 先 drain 到 DDR*/ 0);
     (void)p; // 实现: yolo_dma_act_to_ddr(OUT_DDR, DST_BASE, DFL_NWORDS/4); 再从 OUT_DDR 读
  }
  if(errors==0){ print_str("YOLO DFL HW SMOKE PASS\n"); return; }
  print_str("YOLO DFL HW SMOKE FAIL\n"); __asm__ volatile("ebreak");
}
```

注:Act SRAM 不能 CPU 直读，须 `yolo_dma_act_to_ddr` 把 DST 区(`DFL_NWORDS/4` 字)搬到 DDR 再逐 lane 读 Q8.8 与 `GOLD` 比对(每输出字 4 个,对应 4 个连续输入字)。实现时补全读回循环。

- [ ] **Step 3: 跑 smoke PASS**

Run: `python tools/gen_dfl_vectors.py && bash tests/run_regress.sh sim yolo_dfl_smoke yolo_ops 2>&1 | grep -E "DFL HW SMOKE|TRAP"`
Expected: `YOLO DFL HW SMOKE PASS`。

- [ ] **Step 4: Commit**

```bash
git add tools/gen_dfl_vectors.py firmware/yolo_dfl_smoke.c firmware/yolo_dfl_smoke_data.h
git commit -m "feat(yolo): on-chip DFL end-to-end smoke through real MMIO path"
```

---

## Task 7: 回归 MNIST 基线

- [ ] **Step 1: 全回归**

Run: `bash run_all.sh clean && bash run_all.sh sim 2>&1 | grep -iE "correct|DEPLOY|TRAP|Errors:"`
Expected: `10/10 correct`、`DEPLOY SUCCESS`、`TRAP after 941155 clock cycles`、`Errors: 0`。

- [ ] **Step 2: Commit(若有 doc 更新)**

```bash
git add -A && git commit -m "test(yolo): DFL/sigmoid HW default-off; MNIST 10/10 byte-identical"
```

---

## Self-Review 结论

- **Spec 覆盖**：DFL 单元(Task 2)、exp LUT 按尺度装载(Task 5 helper + 寄存器 Task 3)、W_k 常数(Task 1/3/5)、sigmoid LUT(Task 4)、寄存器同步(Task 3)、默认 OFF 保 MNIST(Task 3/4/7)、定点格式(Task 1/2 与 spec 表一致)、TB(Task 2/4/6)。argmax/几何/NMS 明确范围外。
- **占位**：`tb_dfl_unit.v` 的 `$readmemh` 占位已标注须改逐行解析；`dfl_unit.v` 的 `S_WRITE_PREP`/写地址 off-by-one 标注为执行期 TB 迭代点——这是 RTL 时序须在仿真器内收敛的固有部分，不是规格缺口。
- **类型一致**：寄存器宏、helper 签名、status/CTRL 位在 Task 3/5 间一致。
- **风险**：写回地址相位与除法器延迟是主要时序风险点，已用独立 TB(Task 2)在接整网前隔离验证。
```
