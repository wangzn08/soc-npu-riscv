# NPU 算子通用性扩展 — 设计文档

- 日期: 2026-06-15
- 状态: 已批准设计,待 writing-plans
- 范围: 在现有 CPU+NPU SoC 上,用 RTL 集成一组缺失算子,使 NPU 从"MNIST 专用"
  扩展为可覆盖主流 INT8 CNN(ResNet / MobileNet 类)的通用核心。
- 数据类型: 固定 INT8(不引入 FP16/INT4)。

## 1. 背景与动机

当前 NPU 已能跑 6 层 3×3 conv + 2×2 maxpool + FC1/FC2 的 MNIST 网络(10/10,
941,155 cycles)。但从算子覆盖看,它只支持:3×3 conv(stride 1)、2×2 max pool、
ReLU(钳到 127)、GEMM/FC。要跑更复杂的网络,缺以下算子。

本设计在**不破坏现有 10/10 baseline** 的前提下,以"opt-in 模式位 / 寄存器,默认
OFF ⇒ 逐 bit 与现状相同"的方式逐个加入,沿用现有 D–Q 决策的工程模式。

**明确推迟(本设计不含)**: 可配 kernel(5×5/7×7)、特征图宽度 >256、空洞卷积
(dilation)、多分支 concat。这些是结构性重写或固件可绕,CIFAR(32×32)与目标
CNN 家族暂不需要,留待真实需求再立项。

## 2. 统领架构(所有算子共享)

### 2.1 控制位 / 寄存器分配

现有 CTRL 已用到 [13](int32_out)。新增:

| 算子 | 控制 | 默认 | 含义 |
|---|---|---|---|
| 1×1 pointwise | `CTRL[14] pw_en` | 0 | im2col 旁路,直喂像素 IC 向量 |
| depthwise | `CTRL[15] dw_en` | 0 | CALC 路由到 depthwise_mac,旁路阵列 |
| avgpool 2×2 | `CTRL[16] pool_avg` | 0 | 配合 pool_en;0=max,1=avg |
| global avgpool | `CTRL[17] gpool_en` | 0 | 全图通道均值,末尾写 1 词/OC tile |
| ReLU6/可配 clip | 寄存器 `NPU_CLIP_MAX`(默认 127) | 127 | post_process 上钳值;127=现状 |
| 残差通用化 | 复用 `CTRL[3] eltwise_en` + 寄存器 `NPU_SKIP_BASE` | OFF | 可配 skip 源基址 |
| stride>1 | 复用现有 stride 寄存器(无新位) | sx=sy=1 | im2col 列步进按 stride |

寄存器地址在 param_regfile 现有图谱里挑空位(0x150 之后、避开 0x160..0x39C 的
resident 参数块与 0x3A0+ 性能计数器);`NPU_CLIP_MAX`、`NPU_SKIP_BASE` 各占一个
32-bit 字。最终地址在实现计划里定。

### 2.2 三条铁律

1. **baseline 不破**: 每个新位/寄存器默认 OFF/现状值时,FSM 与 datapath 逐 bit
   与当前一致。每步合入前必须跑一次 `bash run_all.sh sim` 确认 10/10。
2. **定向验证**: 因为暂无新模型,full-MNIST 无法验证新算子。每个算子配一个定向
   testbench `tests/tb_<op>.v`,TB 内置 golden 参考向量自检(沿用 tb_max_pooling_2x2
   / tb_post_process_pool / tb_axi_upsizer 的风格)。
3. **同步约定**: 寄存器改动同时改 `rtl/param_regfile.v` 与 `firmware/firmware.h`;
   新 RTL 文件加入 `axi_sys.f`;固件 C 在严格 CFLAGS 下零警告。

### 2.3 实现顺序(低→高风险,各自独立合入)

`① ReLU6/clip → ② avgpool → ③ global avgpool → ④ 残差通用化 → ⑤ 1×1 pointwise
→ ⑥ stride>1 → ⑦ depthwise`

每个增量是一次独立可验证、可合入的改动。

## 3. 各算子设计

### 3.1 ReLU6 / 可配 clip(风险:极低)

- **现状**: `post_process_top.v` stage3 硬编码 `is_gt127 = (s2_val > 32'd127)`,
  超过钳到 127([post_process_top.v:131-138](../../../rtl/post_process_top.v))。
- **改动**: 新增寄存器 `NPU_CLIP_MAX`(默认 127),param_regfile 输出 `o_clip_max`,
  post_process 比较用 `i_clip_max` 取代常量 127。ReLU6 = 把 6.0 的量化整数写入。
- **触点**: param_regfile(寄存器 + 端口)、post_process_top(`i_clip_max` 输入 +
  比较逻辑)、npu_top(连线)、firmware.h。
- **不变性**: clip_max=127 时逐 bit 与现状相同。
- **TB**: `tests/tb_clip.v` — 给定一组 psum,验证在可配上钳值处饱和;clip_max=127
  时与旧行为一致。

### 3.2 avgpool 2×2(风险:低)

- **现状**: `max_pooling_2x2.v` 每通道取 4 邻域 max([max_pooling_2x2.v:73-86](../../../rtl/max_pooling_2x2.v))。
- **改动**: 在同一组合块里加 `win_avg = (tl+tr+bl+br) >> 2`(每通道),按新输入
  `i_avg` 在 max/avg 间选。窗口/相位/line buffer 逻辑完全复用。
- **触点**: max_pooling_2x2(avg 路 + `i_avg`)、post_process_top(透传 `pool_avg`)、
  param_regfile CTRL[16]、npu_top、firmware.h。
- **不变性**: pool_avg=0 时走 max,逐 bit 不变。
- **TB**: 扩展 `tests/tb_post_process_pool.v` 或新增 `tests/tb_avgpool.v`,对同一输入
  验证 avg 结果 = 4 邻域算术平均(向下取整),max 路不受影响。

### 3.3 global avgpool(风险:中)

- **语义**: 对整张特征图,每通道求空间均值,输出 1×1×C。
- **改动**: 新块 `rtl/global_avg.v` — 抽 post_process 的逐位置输出(INT8 stage3 或
  INT32 stage2),16 通道各一个 INT32 累加器 + 空间位置计数;扫描到当前 OC tile 的
  最后一个位置时,emit 一个词:`sum` 经 scale/shift requant(把 1/(H·W) 折进
  scale_mul,与 BN-fold 同理),写入 Out SRAM 该 OC tile 的单地址。
- **FSM**: `gpool_en` 时抑制逐位置 Out 写,改为每个 OC tile 末尾写 1 词。
- **触点**: 新文件、top_controller_fsm(写时序分支)、param_regfile CTRL[17]、
  npu_top(mux 写地址/数据)、firmware.h。
- **不变性**: gpool_en=0 时新块旁路、FSM 写序与现状相同。
- **TB**: `tests/tb_global_avg.v` — 喂一张已知特征图,验证每通道输出 = 空间均值
  requant 后的值。

### 3.4 残差通用化 —— (轻)方案(风险:中)

- **现状**: vector_alu 做 INT8 饱和加 [0,127],skip 源 = `out_sram_dob` 且地址 =
  当前写地址([npu_top.v:927,1159-1160](../../../rtl/npu_top.v))——只能"同址 in-place 累加"。
- **改动((轻))**: 保持 INT8 饱和加,新增寄存器 `NPU_SKIP_BASE`,让 skip 读地址 =
  `NPU_SKIP_BASE + 当前位置偏移`,从而能加"任意早层暂存在某 Out/Act 区域的张量"。
  **量化时保证两路同 scale**(标准做法),硬件不做 INT32 域 requant。
- **触点**: param_regfile(`NPU_SKIP_BASE` 寄存器 + 端口)、npu_top(skip 读地址计算
  改用基址 + 偏移,替换 `skip_rd_addr = fsm_out_wr_addr`)、firmware.h。
- **不变性**: eltwise_en=0 时不读 skip;若 `NPU_SKIP_BASE` 配成与写地址一致,则与
  现状同址行为相同。
- **TB**: `tests/tb_residual.v` — 预置 skip 张量于某基址,conv 结果 + skip,验证逐通道
  饱和加正确,且地址来自可配基址。

### 3.5 1×1 pointwise conv(风险:中)

- **关键洞察**: 16×16 阵列的 IC 求和本身就是 1×1 点积,故阵列、post_process、空间
  扫描、row_par、oc_single **全部复用**。
- **改动**: `pw_en` 时 im2col 不做 3×3 窗口,FSM 直接把每个空间位置的 IC-tile 向量
  喂给阵列(kh=kw=1、ko_total=1、无邻域、无 padding);权重驻留复用。激活地址生成
  类似 hw_pad 的 tile-major 直读但 pad=0。row_par 下 16 个空间位置映射到 16 阵列行。
- **触点**: top_controller_fsm(pw 激活地址 + im2col 旁路 + ko_total=1 路径)、
  im2col_line_buffer 或 npu_top 的激活喂入 mux、param_regfile CTRL[14]、firmware.h。
- **不变性**: pw_en=0 时走原 conv 窗口路径,逐 bit 不变。
- **TB**: `tests/tb_pointwise.v` — 小输入(如 4×4×32 → 4×4×16),对比 CPU golden 点积。

### 3.6 stride>1 conv(风险:中高)

- **现状**: out 维度算式已含 stride([fsm:391-394](../../../rtl/top_controller_fsm.v)),
  空间步进 `cur_in_col += group_size*stride` 已在;但 im2col row_par 读连续列
  `group_base + off_col + gi`([im2col:283](../../../rtl/im2col_line_buffer.v)),未按
  stride 取样。
- **改动**: im2col 列寻址参数化:row_par 读 `group_base*stride + off_col + gi*stride`;
  legacy 路径窗口推进按 stride。FSM 把 stride 传入 im2col。
- **触点**: im2col_line_buffer(列寻址按 stride)、top_controller_fsm(传 stride 给
  im2col)。**这是最脆的模块,改动需特别小心。**
- **不变性**: stride=1 时逐 bit 与现状相同。
- **TB**: `tests/tb_stride2.v` — 一层 3×3 stride-2 conv,对比 CPU golden;并跑一次
  stride-1 回归确认未回退。

### 3.7 depthwise conv —— 独立 MAC 旁路阵列(风险:中高)

- **冲突**: 脉动阵列天生对 16 输入通道求和,depthwise 要求通道间不求和。
- **改动**: 新块 `rtl/depthwise_mac.v` — 16 条并行通道 MAC,吃 im2col 输出的
  `win[tile][r][c]`(每 IC tile 的 16 通道 3×3 窗口,**im2col 不改**),每通道
  × 自己的 9 个权重并累加,得 16 通道输出 → 接现有 post_process(bias/quant/relu/
  pool 全复用)。`dw_en` 时 FSM 在 CALC 把数据路由到 depthwise_mac 而非阵列,9 个
  kernel 周期累加后正常 drain/post。
- **权重布局**: depthwise 权重每通道 9 个,需 gen_weights / wgt_reader 以 depthwise
  布局供给(实现计划里细化权重读路径)。
- **触点**: 新文件、top_controller_fsm(dw 路由 + drain 时序)、npu_top(post_process
  输入在 阵列 / depthwise_mac 间 mux)、wgt_reader(depthwise 权重供给)、
  param_regfile CTRL[15]、firmware.h。
- **不变性**: dw_en=0 时 depthwise_mac 旁路,数据走阵列,逐 bit 不变。
- **TB**: `tests/tb_depthwise.v` — 小输入(如 8×8×16,3×3 depthwise),对比 CPU golden。

## 4. 验证策略

- 每个算子一个定向 TB,golden 参考在 TB 内或用 Python 生成(沿用 extract_images.py
  风格)。
- 每步合入前: 该算子 TB 通过 + `bash run_all.sh sim` 仍 10/10(回归 baseline)。
- 风险较高的 ⑤⑥⑦(动 im2col / FSM / 新通路)先有 TB 通过再谈集成,遵循
  CLAUDE.md "risky RTL timing changes 先加定向 TB" 的工作规则。

## 5. 非目标 / YAGNI

- 不做可配 kernel(5×5/7×7)、>256 宽、dilation、concat。
- 不做 INT32 域残差 requant(选(轻))。
- 不做激活 LUT(sigmoid/GELU 等);只做 ReLU + 可配上钳(覆盖 ReLU/ReLU6)。
- 不在本设计内做描述符 DMA / MMIO 削减(已另议,暂缓)。

## 6. 完成后的能力

做完 ①–⑦,NPU 的算子覆盖将达到:
3×3 conv(stride 1/2) + 1×1 pointwise + depthwise 3×3 + 2×2 max/avg pool +
global avgpool + 通用残差 add + ReLU/ReLU6 + GEMM/FC。这覆盖 ResNet 与 MobileNet
两大家族的主体算子(在 ≤256 宽、3×3/1×1 kernel 约束内)。
