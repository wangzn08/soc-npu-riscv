# NPU 算子通用性扩展 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 CPU+NPU SoC 上用 RTL 集成 7 个缺失算子(ReLU6/可配 clip、avgpool、global avgpool、通用残差、1×1 pointwise、stride>1、depthwise),把 NPU 从 MNIST 专用扩展为可覆盖 ResNet/MobileNet 主体算子的 INT8 通用核心。

**Architecture:** 每个算子是一个 opt-in CTRL 位或寄存器,默认 OFF/默认值 ⇒ FSM 与 datapath 逐 bit 与当前 10/10 baseline 相同。每个算子配一个定向 testbench(`tests/tb_<op>.v`,内置 golden 自检),先让 TB 失败、实现 RTL、TB 通过、再跑 full-MNIST 回归确认 10/10 未破,然后提交。低风险算子先做。

**Tech Stack:** Verilog/SystemVerilog,ModelSim/Questa(`vlog`/`vsim`),PicoRV32 固件(riscv-none-elf-gcc),Python 生成 golden 向量。

Spec: [docs/superpowers/specs/2026-06-15-npu-operator-generality-design.md](../specs/2026-06-15-npu-operator-generality-design.md)

---

## 公共约定

### 寄存器 / CTRL 位分配(全计划统一)

| 名称 | 地址/位 | 默认 | 用途 |
|---|---|---|---|
| `pw_en` | CTRL[14] | 0 | 1×1 pointwise |
| `dw_en` | CTRL[15] | 0 | depthwise |
| `pool_avg` | CTRL[16] | 0 | avgpool(配合 pool_en) |
| `gpool_en` | CTRL[17] | 0 | global avgpool |
| `NPU_CLIP_MAX` | 0x118 | 127 | post_process 上钳值 |
| `NPU_SKIP_BASE` | 0x11C | 0 | 残差 skip 源 SRAM 基址 |

0x118 / 0x11C 在 param_regfile 当前地址图谱里是空位(0x110/0x114 已用,0x120 起为 DMA),无冲突。

### 运行定向 testbench 的方法(无现成脚本,本计划统一用此 recipe)

在 MSYS/Git Bash 中,从仓库根 `E:\code\6-10\soc` 运行:

```bash
export PATH="/c/msys64/usr/bin:/e/modelsim/win64:$PATH"
export MGC_LICENSE_FILE='E:/modelsim/LICENSE.TXT'
export TMP='E:\code\6-10\soc\.tmp' TEMP='E:\code\6-10\soc\.tmp'; mkdir -p .tmp
# <DUT> = 被测模块及其依赖的 .v 文件列表
vlib sim/work_tb 2>/dev/null; vlog -sv -timescale 1ns/1ps -work sim/work_tb <DUT files> tests/tb_<op>.v 2>&1 | tail -3
vsim -c -lib sim/work_tb tb_<op> -do "run -all; quit -f" 2>&1 | grep -iE "PASS|FAIL|error|Errors:"
```

每个 TB 末尾按现有风格自检:统计 `errors`,`$display("TB_<OP> PASS")` 当 errors==0,否则打印失败并 `$fatal`/`errors` 计数。

### 全量回归(每个算子合入前必跑)

```bash
bash run_all.sh clean && bash run_all.sh sim
# 期望:=== Result: 10/10 correct === 且 DEPLOY SUCCESS.
```

### 提交规范

- 分支:`npu-operator-generality`(已建)。
- commit 尾部:`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。
- 寄存器改动同时改 `rtl/param_regfile.v` 与 `firmware/firmware.h`;新 RTL 文件加入 `axi_sys.f`。

---

## Phase 1 — ReLU6 / 可配 clip(风险:极低)

**Files:**
- Modify: `rtl/param_regfile.v`(新增 `clip_max` 寄存器 + `o_clip_max` 端口)
- Modify: `rtl/post_process_top.v`(新增 `i_clip_max` 输入,stage3 比较改用它)
- Modify: `rtl/npu_top.v`(连线 `o_clip_max` → `i_clip_max`)
- Modify: `firmware/firmware.h`(`#define NPU_CLIP_MAX 0x118`)
- Test: `tests/tb_clip.v`

### Task 1.1: param_regfile 增加 clip_max 寄存器

- [ ] **Step 1: 在 param_regfile.v 声明寄存器并加端口**

在端口列表加 `output wire [7:0] o_clip_max,`。在寄存器声明区加:
```verilog
reg [7:0] clip_max;
```
复位块加 `clip_max <= 8'd127;`。写解码 case 加:
```verilog
10'h118: clip_max <= s_axi_wdata[7:0];
```
读 case 加:
```verilog
10'h118: rdata <= {24'd0, clip_max};
```
输出区加 `assign o_clip_max = clip_max;`。

- [ ] **Step 2: post_process_top 用 i_clip_max 取代常量 127**

端口加 `input wire [7:0] i_clip_max,`。stage3 内:
```verilog
wire is_gt = (s2_val > {24'd0, i_clip_max});      // was: s2_val > 32'd127
...
assign act_val = (i_relu_en && is_neg) ? {ACT_WIDTH{1'b0}} :
                 is_gt                 ? i_clip_max :        // was: 8'd127
                                         s2_val[ACT_WIDTH-1:0];
```

- [ ] **Step 3: npu_top 连线**

在 param_regfile 实例加 `.o_clip_max(cfg_clip_max),`(声明 `wire [7:0] cfg_clip_max;`),在 post_process_top 实例加 `.i_clip_max(cfg_clip_max),`。

- [ ] **Step 4: firmware.h 加宏**

```c
#define NPU_CLIP_MAX   (NPU_BASE + 0x118)   /* 上钳值,默认127;ReLU6=q(6.0) */
```
(沿用 firmware.h 现有 `NPU_BASE + offset` 风格;若现有用别的形式,匹配之。)

### Task 1.2: 定向 TB

- [ ] **Step 1: 写 tests/tb_clip.v(先失败)**

例化 `post_process_top`,`i_relu_en=1`。给一组 `i_psum`(bias=0,scale_mul=1,scale_shift=0,使 s2==psum):
- 用例 A:`i_clip_max=127`,psum=200 → 期望 127;psum=50 → 50;psum=-5 → 0(验证与旧行为一致)。
- 用例 B:`i_clip_max=30`(模拟 ReLU6),psum=200 → 30;psum=20 → 20;psum=-5 → 0。
统计 errors,末尾 `TB_CLIP PASS` / 失败打印期望与实际。

- [ ] **Step 2: 跑 TB,确认按预期(实现前 B 用例失败或编译失败)**

按公共 recipe,`<DUT files>` = `rtl/post_process_top.v rtl/max_pooling_2x2.v`。
期望:实现前 `i_clip_max` 端口不存在 → 编译失败(确认 TB 真的依赖新接口)。

- [ ] **Step 3: 实现 Task 1.1 全部改动**(若尚未完成)

- [ ] **Step 4: 跑 TB,期望 `TB_CLIP PASS`,errors=0**

- [ ] **Step 5: 全量回归**

`bash run_all.sh clean && bash run_all.sh sim` → 10/10。clip_max 默认 127 ⇒ 逐 bit 不变。

- [ ] **Step 6: 提交**
```bash
git add rtl/param_regfile.v rtl/post_process_top.v rtl/npu_top.v firmware/firmware.h tests/tb_clip.v axi_sys.f
git commit -m "feat(npu): configurable activation clip (ReLU6) via NPU_CLIP_MAX, default 127 bit-identical"
```
(`axi_sys.f` 仅当新增 TB 不进主 filelist 时不必改;Phase 1 无新 RTL 文件,可不动 axi_sys.f。)

---

## Phase 2 — avgpool 2×2(风险:低)

**Files:**
- Modify: `rtl/max_pooling_2x2.v`(新增 `i_avg` 输入 + avg 计算 + 输出 mux)
- Modify: `rtl/post_process_top.v`(透传 `i_pool_avg` → pooler `.i_avg`)
- Modify: `rtl/param_regfile.v`(CTRL[16] `pool_avg` + `o_pool_avg`)
- Modify: `rtl/npu_top.v`(连线)
- Modify: `firmware/firmware.h`(`#define NPU_CTRL_POOL_AVG (1<<16)`)
- Test: `tests/tb_avgpool.v`

### Task 2.1: max_pooling_2x2 增加 avg 路

- [ ] **Step 1: 加 i_avg 输入与每通道 avg**

端口加 `input wire i_avg,`。在 `gen_max` 内每通道加:
```verilog
wire [9:0] sum4 = {2'd0, tl} + {2'd0, tr} + {2'd0, bl} + {2'd0, br};
wire [ACT_WIDTH-1:0] avg = sum4[9:2];   // (a+b+c+d)/4,向下取整
assign win_max[gi*ACT_WIDTH +: ACT_WIDTH] =
       i_avg ? avg : ((mtop >= mbot) ? mtop : mbot);
```
(保留 `mtop/mbot` 不变;只在最终赋值处按 `i_avg` 选。)

### Task 2.2: 透传与控制位

- [ ] **Step 1: post_process_top 透传**

端口加 `input wire i_pool_avg,`,在 `u_pool` 实例加 `.i_avg(i_pool_avg),`。

- [ ] **Step 2: param_regfile CTRL[16]**

声明 `reg ctrl_pool_avg;`,复位 `<= 1'b0;`,CTRL 写解码加 `ctrl_pool_avg <= s_axi_wdata[16];`,CTRL 读位拼接加入,输出 `assign o_pool_avg = ctrl_pool_avg;`,端口加 `output wire o_pool_avg,`。

- [ ] **Step 3: npu_top 连线**:param_regfile `.o_pool_avg(cfg_pool_avg)`,post_process `.i_pool_avg(cfg_pool_avg)`。

- [ ] **Step 4: firmware.h**:`#define NPU_CTRL_POOL_AVG (1u<<16)`。

### Task 2.3: 定向 TB

- [ ] **Step 1: 写 tests/tb_avgpool.v(基于 tb_max_pooling_2x2 改)**

复用 MAP A=1..16。`i_avg=1` 时 golden:窗口 {1,2,5,6}→3(=14/4 取整);{3,4,7,8}→5;{9,10,13,14}→11;{11,12,15,16}→13。`i_avg=0` 时与现有 max golden(6,8,14,16)一致。

- [ ] **Step 2: 跑 TB → 实现前失败(无 i_avg)**
`<DUT files>` = `rtl/max_pooling_2x2.v`。

- [ ] **Step 3: 实现 Task 2.1/2.2**

- [ ] **Step 4: TB → `TB_AVGPOOL PASS`**

- [ ] **Step 5: 全量回归 10/10**(pool_avg 默认 0 ⇒ max 路不变)

- [ ] **Step 6: 提交**
```bash
git add rtl/max_pooling_2x2.v rtl/post_process_top.v rtl/param_regfile.v rtl/npu_top.v firmware/firmware.h tests/tb_avgpool.v
git commit -m "feat(npu): 2x2 average pooling via CTRL[16] pool_avg, default max bit-identical"
```

---

## Phase 3 — global avgpool(风险:中,动 FSM 写时序)

**Files:**
- Create: `rtl/global_avg.v`(16 通道 INT32 累加 + 末位 requant 输出)
- Modify: `rtl/top_controller_fsm.v`(gpool_en 时抑制逐位置写,OC tile 末写 1 词)
- Modify: `rtl/npu_top.v`(mux 写地址/数据;例化 global_avg)
- Modify: `rtl/param_regfile.v`(CTRL[17] `gpool_en` + 输出)
- Modify: `firmware/firmware.h`(`#define NPU_CTRL_GPOOL_EN (1<<17)`)
- Modify: `axi_sys.f`(加入 `rtl/global_avg.v`)
- Test: `tests/tb_global_avg.v`

### Task 3.1: global_avg 模块

- [ ] **Step 1: 写 rtl/global_avg.v**

接口(由 TB 与集成共同约束):
```verilog
module global_avg #(parameter NUM_CH=16, ACT_WIDTH=8, PSUM_WIDTH=32, DATA_W=128) (
  input  wire clk, rst_n,
  input  wire i_start,                 // OC tile 起始:清累加器与计数
  input  wire [DATA_W-1:0] i_feat,     // post_process 逐位置 INT8 输出
  input  wire i_feat_vld,
  input  wire i_last,                  // 当前 OC tile 最后一个空间位置
  input  wire [NUM_CH-1:0][PSUM_WIDTH-1:0] i_scale_mul,   // 把 1/(H*W) 折进来
  input  wire [NUM_CH-1:0][5:0]            i_scale_shift,
  output reg  [DATA_W-1:0] o_feat,     // requant 后的均值(每通道 INT8)
  output reg  o_feat_vld               // 在 i_last 那个位置后拉高 1 拍
);
```
逻辑:16 个 INT32 累加器,`i_feat_vld` 时各通道 `acc += i_feat[ch]`;`i_start` 清零。`i_feat_vld && i_last` 时对每通道做 `(acc * scale_mul) >>> scale_shift`,clamp INT8 → `o_feat`,`o_feat_vld<=1`(单拍)。

- [ ] **Step 2: 加入 axi_sys.f**(在 NPU RTL 段加一行 `rtl/global_avg.v`)。

### Task 3.2: FSM 写时序 + 集成

- [ ] **Step 1: param_regfile CTRL[17]**(同 Phase 2 Step 2 模式,`gpool_en`/`o_gpool_en`)。

- [ ] **Step 2: FSM 抑制逐位置写**

在 `top_controller_fsm.v`,`o_out_wr_en` 当 `i_gpool_en` 时改为只在"该 OC tile 最后一个空间位置且 post 完成"拉高;新增 `o_gpool_last`(= 当前是该 OC tile 末位置)给 global_avg。其余位置 `o_out_wr_en=0`。OFF 时维持原 `(state==S_POST)&&i_pp_done`。

- [ ] **Step 3: npu_top 例化 global_avg 并 mux 写路径**

`gpool_en` 时:Out SRAM 写数据 = `global_avg.o_feat`,写使能 = `global_avg.o_feat_vld`,写地址 = OC tile 基址(单地址);OFF 时维持原 post_process → Out 写路径。global_avg 的 `i_feat/i_feat_vld` 接 post_process 输出,`i_scale_mul/shift` 接 param_regfile 的 OC 窗口。

- [ ] **Step 4: firmware.h**:`#define NPU_CTRL_GPOOL_EN (1u<<17)`。

### Task 3.3: 定向 TB

- [ ] **Step 1: 写 tests/tb_global_avg.v**

单独例化 `global_avg`(不拉整条 FSM):喂一张已知 8×8(=64 位置)单通道特征图,scale_mul/shift 设为实现 `÷64`(如 scale_mul=1,scale_shift=6),`i_last` 在第 64 个 `i_feat_vld` 拉高。golden = sum/64。验证 `o_feat` 与 `o_feat_vld` 时序。

- [ ] **Step 2: 跑 TB → 实现前失败**(`<DUT files>`=`rtl/global_avg.v`)

- [ ] **Step 3: 实现 Task 3.1/3.2**

- [ ] **Step 4: TB → `TB_GLOBAL_AVG PASS`**

- [ ] **Step 5: 全量回归 10/10**(gpool_en 默认 0 ⇒ 写序与现状相同)

- [ ] **Step 6: 提交**
```bash
git add rtl/global_avg.v rtl/top_controller_fsm.v rtl/npu_top.v rtl/param_regfile.v firmware/firmware.h tests/tb_global_avg.v axi_sys.f
git commit -m "feat(npu): global average pooling via CTRL[17] gpool_en, default off bit-identical"
```

---

## Phase 4 — 残差通用化 (轻)(风险:中)

**Files:**
- Modify: `rtl/param_regfile.v`(`NPU_SKIP_BASE` 0x11C 寄存器 + 输出)
- Modify: `rtl/npu_top.v`(skip 读地址 = `NPU_SKIP_BASE + 位置偏移`,替换同址)
- Modify: `firmware/firmware.h`(`#define NPU_SKIP_BASE 0x11C`)
- Test: `tests/tb_residual.v`

### Task 4.1: 可配 skip 基址

- [ ] **Step 1: param_regfile 加 skip_base 寄存器**

`reg [SRAM_ADDR_W-1:0] skip_base;` 复位 `<= 0;`;写 `10'h11C: skip_base <= s_axi_wdata[SRAM_ADDR_W-1:0];`;读 `10'h11C: rdata <= {... , skip_base};`;端口 `output wire [SRAM_ADDR_W-1:0] o_skip_base;` + `assign o_skip_base = skip_base;`。

- [ ] **Step 2: npu_top skip 读地址改用基址**

现状 [npu_top.v:1159-1160](../../../rtl/npu_top.v):`skip_rd_addr = fsm_out_wr_addr`。改为:
```verilog
wire [OUT_SRAM_ADDR_W-1:0] skip_rd_addr =
     cfg_skip_base[OUT_SRAM_ADDR_W-1:0] + fsm_out_wr_addr[OUT_SRAM_ADDR_W-1:0];
```
(声明 `wire [SRAM_ADDR_W-1:0] cfg_skip_base;`,接 param_regfile `.o_skip_base`。)若 `NPU_SKIP_BASE=0` ⇒ 与现状同址行为一致(向后兼容)。

- [ ] **Step 3: firmware.h**:`#define NPU_SKIP_BASE (NPU_BASE + 0x11C)`。

### Task 4.2: 定向 TB

- [ ] **Step 1: 写 tests/tb_residual.v**

例化 `vector_alu`,`i_eltwise_en=1`:给一组 `i_conv_res` 与 `i_skip_res`,验证逐通道 `min(conv+skip,127)`,负溢出按现有逻辑。(地址基址逻辑在 npu_top 层,vector_alu 层只验加法/饱和;基址用例可选在集成层验证或用一个迷你 FSM-less harness 驱动 npu_top 的 skip_rd_addr 计算——优先 vector_alu 单元验证 + 代码走读确认基址加法。)

- [ ] **Step 2: 跑 TB → 确认加法/饱和**(`<DUT files>`=`rtl/vector_alu.v`)

- [ ] **Step 3: 实现 Task 4.1**

- [ ] **Step 4: TB → `TB_RESIDUAL PASS`**

- [ ] **Step 5: 全量回归 10/10**(eltwise_en 默认 0 ⇒ 不读 skip,不变)

- [ ] **Step 6: 提交**
```bash
git add rtl/param_regfile.v rtl/npu_top.v firmware/firmware.h tests/tb_residual.v
git commit -m "feat(npu): configurable residual skip base (NPU_SKIP_BASE), INT8 add, default same-addr"
```

---

## Phase 5 — 1×1 pointwise conv(风险:中)

**Files:**
- Modify: `rtl/top_controller_fsm.v`(`pw_en`:ko_total=1 + 直喂像素 IC 向量 + im2col 旁路)
- Modify: `rtl/npu_top.v` 或 `rtl/im2col_line_buffer.v`(激活喂入 mux)
- Modify: `rtl/param_regfile.v`(CTRL[14] `pw_en`)
- Modify: `firmware/firmware.h`(`#define NPU_CTRL_PW_EN (1<<14)`)
- Test: `tests/tb_pointwise.v`(集成级:小输入跑通 npu_top)

### Task 5.1: pw_en 数据通路

- [ ] **Step 1: param_regfile CTRL[14] `pw_en` + 输出**(同位拼接模式)。

- [ ] **Step 2: FSM pointwise 路径**

`i_pw_en` 时:
- `ko_total` 强制为 1(`assign ko_total = i_pw_en ? 8'd1 : (i_gemm_en?...) : kh*kw;`),CALC 单周期;
- 激活地址走 tile-major 直读(复用 `tilemaj_addr` 但 `pad=0`,即直接按 `(row*W+col)*ic_groups + tile`),逐 IC tile 在 CALC 累加;
- 空间扫描、row_par(16 位置/sweep)、oc_single 复用现状;
- im2col 窗口逻辑旁路:`o_act_window` 的喂入在 `pw_en` 时取"当前像素的 IC tile 向量"而非 line-buffer 窗口。
> 本路径内部细节在执行时 test-first 对照 `tb_pointwise` 调试;先把接口与扫描搭起来,再逐项对齐 golden。

- [ ] **Step 3: 激活喂入 mux**

在 im2col 输出或 npu_top 阵列输入处:`pw_en ? direct_pixel_vec : o_act_window`。direct_pixel_vec = 当前 IC tile 的 128-bit 激活字复制到 16 行(row_par 时 16 个位置各自的字)。

- [ ] **Step 4: firmware.h**:`#define NPU_CTRL_PW_EN (1u<<14)`。

### Task 5.2: 定向 TB(集成级)

- [ ] **Step 1: 写 tests/tb_pointwise.v**

最小 harness 驱动 `npu_top`(或一个裹住 FSM+array+post 的子系统):预置 Act SRAM 一张 4×4×32 输入、Wgt SRAM 1×1×32×16 权重,`pw_en=1` 启动,读 Out SRAM,对比 Python/内联 golden 点积(逐位置 32→16 的矩阵乘)。

- [ ] **Step 2: 跑 TB → 实现前失败**

- [ ] **Step 3: 实现 Task 5.1,迭代至 golden 对齐**

- [ ] **Step 4: TB → `TB_POINTWISE PASS`**

- [ ] **Step 5: 全量回归 10/10**(pw_en 默认 0 ⇒ conv 窗口路径不变)

- [ ] **Step 6: 提交**
```bash
git add rtl/top_controller_fsm.v rtl/npu_top.v rtl/im2col_line_buffer.v rtl/param_regfile.v firmware/firmware.h tests/tb_pointwise.v
git commit -m "feat(npu): 1x1 pointwise conv via CTRL[14] pw_en (reuses array IC-reduction), default off"
```

---

## Phase 6 — stride>1 conv(风险:中高,动 im2col 最脆处)

**Files:**
- Modify: `rtl/im2col_line_buffer.v`(列寻址按 stride)
- Modify: `rtl/top_controller_fsm.v`(把 stride 传入 im2col 列步进)
- Test: `tests/tb_stride2.v`

### Task 6.1: im2col 列步进按 stride

- [ ] **Step 1: im2col 加 i_stride 输入**

端口加 `input wire [7:0] i_stride,`。row_par 列计算:
```verilog
// was: rp_col = i_group_base + off_col + gi
wire [15:0] rp_col = i_group_base*i_stride + {14'd0, off_col_dec} + gi*i_stride;
```
legacy(非 row_par)路径:窗口推进步距由 1 改为 `i_stride`(对应 `win_advance` 时 rd_ptr/cur_x 增量,或在 FSM 侧通过多次 advance 实现——执行时按 TB 对齐二选一,优先在 im2col 内乘 stride)。

- [ ] **Step 2: FSM 传 stride**:`.i_stride(i_stride_sx)` 接到 im2col 实例。

### Task 6.2: 定向 TB

- [ ] **Step 1: 写 tests/tb_stride2.v**

最小 harness:一张 6×6×16 输入,3×3 stride-2 conv(out 2×2),对比 golden。另加 stride-1 用例确认未回退。

- [ ] **Step 2: 跑 TB → 实现前 stride-2 结果错**

- [ ] **Step 3: 实现 Task 6.1,迭代对齐**

- [ ] **Step 4: TB → `TB_STRIDE2 PASS`(含 stride-1 回归用例)**

- [ ] **Step 5: 全量回归 10/10**(MNIST 全是 stride-1 ⇒ 必须逐 bit 不变)

- [ ] **Step 6: 提交**
```bash
git add rtl/im2col_line_buffer.v rtl/top_controller_fsm.v tests/tb_stride2.v
git commit -m "feat(npu): stride>1 conv (im2col column stride), stride-1 bit-identical"
```

---

## Phase 7 — depthwise conv(风险:中高,新通路旁路阵列)

**Files:**
- Create: `rtl/depthwise_mac.v`(16 通道并行 MAC)
- Modify: `rtl/top_controller_fsm.v`(`dw_en`:CALC 路由到 depthwise_mac;drain 时序)
- Modify: `rtl/npu_top.v`(post_process 输入在 阵列/depthwise_mac 间 mux;例化)
- Modify: `rtl/wgt_reader.v`(depthwise 权重供给:每通道 9 权重)
- Modify: `rtl/param_regfile.v`(CTRL[15] `dw_en`)
- Modify: `firmware/firmware.h`(`#define NPU_CTRL_DW_EN (1<<15)`)
- Modify: `axi_sys.f`(加入 `rtl/depthwise_mac.v`)
- Test: `tests/tb_depthwise.v`

### Task 7.1: depthwise_mac 模块

- [ ] **Step 1: 写 rtl/depthwise_mac.v**

接口:
```verilog
module depthwise_mac #(parameter NUM_CH=16, ACT_WIDTH=8, KOFF=9, PSUM_WIDTH=32, DATA_W=128) (
  input  wire clk, rst_n,
  input  wire i_clear,                       // kernel 累加起始清零
  input  wire i_vld,                         // 1 个 kernel offset 有效
  input  wire [DATA_W-1:0] i_act,            // 该 offset 的 16 通道激活(im2col win[tile][r][c])
  input  wire [DATA_W-1:0] i_wgt,            // 该 offset 的 16 通道权重(每通道 1 字节)
  input  wire i_latch,                       // 9 个 offset 完 → 锁存
  output wire [NUM_CH-1:0][PSUM_WIDTH-1:0] o_psum,  // 16 通道结果,接 post_process
  output wire o_vld
);
```
逻辑:16 个独立 INT32 累加器,`i_clear` 清零;`i_vld` 时各通道 `acc[ch] += signed(i_act[ch]) * signed(i_wgt[ch])`(**通道间不求和**);`i_latch` 输出 `o_psum`,`o_vld` 拉高。

- [ ] **Step 2: 加入 axi_sys.f**。

### Task 7.2: FSM 路由 + 权重供给 + 集成

- [ ] **Step 1: param_regfile CTRL[15] `dw_en` + 输出**。

- [ ] **Step 2: FSM 在 dw_en 时驱动 depthwise_mac**

复用现有空间扫描 + im2col(`win[tile][r][c]` 即每通道 3×3 窗口)。CALC 9 个周期把 9 个 offset 的 `i_act`(im2col 选 offset)与 `i_wgt`(wgt_reader 供该 offset 的 16 通道权重)送入 depthwise_mac;第 9 拍 `i_latch`。结果走现有 drain/post。

- [ ] **Step 3: wgt_reader depthwise 布局**

depthwise 每通道 9 个权重(无 OC×IC 矩阵)。wgt_reader 在 dw_en 时按"offset 维 × 16 通道"供给(每周期一个 offset 的 16 字节权重)。具体布局与 gen_weights 对齐在执行时定;TB 用直接预置的 Wgt SRAM 内容验证。

- [ ] **Step 4: npu_top mux**:post_process 的 `i_psum` 在 `dw_en` 时取 `depthwise_mac.o_psum`,否则取阵列列和;`i_psum_vld` 同理。

- [ ] **Step 5: firmware.h**:`#define NPU_CTRL_DW_EN (1u<<15)`。

### Task 7.3: 定向 TB

- [ ] **Step 1: 写 tests/tb_depthwise.v**

单元级先验 depthwise_mac:给 16 通道、9 offset 的 act/wgt,验证每通道 = Σ act×wgt(无跨通道)。再集成级(可选):8×8×16 输入 + 3×3 depthwise,对比 golden。

- [ ] **Step 2: 跑 TB → 实现前失败**(`<DUT files>`=`rtl/depthwise_mac.v`)

- [ ] **Step 3: 实现 Task 7.1/7.2**

- [ ] **Step 4: TB → `TB_DEPTHWISE PASS`**

- [ ] **Step 5: 全量回归 10/10**(dw_en 默认 0 ⇒ 走阵列,不变)

- [ ] **Step 6: 提交**
```bash
git add rtl/depthwise_mac.v rtl/top_controller_fsm.v rtl/npu_top.v rtl/wgt_reader.v rtl/param_regfile.v firmware/firmware.h tests/tb_depthwise.v axi_sys.f
git commit -m "feat(npu): depthwise conv via CTRL[15] dw_en (dedicated channel-parallel MAC), default off"
```

---

## 收尾

- [ ] 7 个算子全部合入后,跑一次完整回归确认 10/10 + DEPLOY SUCCESS。
- [ ] 更新 `CLAUDE.md` 的 "Current Optimization State" 与 CTRL 位表(CTRL[14..17] + 两个新寄存器)。
- [ ] 更新 memory:在 MEMORY.md 加一条算子通用性扩展的指针。

## Self-Review 备注(计划作者已核对)

- **Spec 覆盖**:spec 第 3 节 7 个算子 ↔ Phase 1–7 一一对应;统领架构(CTRL 位/寄存器/铁律/顺序)↔ 公共约定;验证策略 ↔ 每 Phase 的 TB + 全量回归。
- **类型/命名一致**:CTRL 位 pw_en[14]/dw_en[15]/pool_avg[16]/gpool_en[17]、寄存器 NPU_CLIP_MAX(0x118)/NPU_SKIP_BASE(0x11C) 全计划统一。
- **已知风险点(执行时重点对齐 golden)**:Phase 5/6/7 内部 datapath 在 spec/plan 中给到接口+算法+集成点,具体寄存器级实现 test-first 对照各自 TB 调试——这是 RTL 必要的迭代,不是 placeholder。
