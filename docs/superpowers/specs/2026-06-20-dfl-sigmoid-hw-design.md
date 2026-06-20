# YOLOv8n 检测头后处理硬件化：DFL 期望单元 + sigmoid LUT — 设计

> 目标：让 YOLOv8n 的检测头解码在 **SoC 上完整完成**(无 host/Python),把
> 后处理里唯一需要超越-INT8-MAC 的两类运算——DFL 的 softmax 期望、cls 的
> sigmoid——做成 NPU 侧硬件查表/归约,使 PicoRV32 只剩整数 argmax/几何/NMS。
> 这样整网端到端可在 RTL 仿真里跑完,TRAP 周期数 ÷200MHz = 真实推理时间。

## 背景与黄金参考

纯 C 引擎 `yolov8n_int8/yolov8n_infer.c`(mAP50-95=0.352)是逐元素黄金。其
decode(`yolov8n_infer.c:633-728`)：

- **DFL**：bbox 头每尺度输出 64 通道 = 4 坐标 × 16 bins(INT8)。对每 (anchor,
  coord)：对 16 个 bin 做 softmax,再与 conv63 反量化权重 `W_k = wscale[0]·w[k]`
  (16 个常数)加权求和 → 1 个距离值(grid 单位)。
- **cls**：80 类 logit 过 `sigmoid` 得分数。
- 其余：argmax(80 类)、box 几何(anchor 中心 ± 距离 ×stride)、NMS。

**关键数值事实(决定硬件可行性)**：

1. NPU tile-major 布局下(16ch/word,通道主序),**一个 128-bit 字恰好 = 某坐标
   在某 anchor 的 16 个 bin**(ch_group = coord)。故 DFL 单元是干净的 16→1/字,
   bbox 头 = 4 字/anchor。
2. softmax 里 bbox 的零点 **自动相消**：`z_k = (int8_k − zp)·scale`,
   `z_k − max = (int8_k − int8_max)·scale`,zp 消失。故 `exp` 仅依赖该尺度的
   `scale`,可用 **256 项 LUT 按整数差 `(int8_k − int8_max) ∈ [−255,0]` 索引**。
3. cls 的 `sigmoid(dequant(int8))` 是单个 INT8 的固定函数 → 256 项 LUT,与现有
   SiLU LUT 同构。

## 架构

后处理沿用本 SoC 既有"片上数据搬运引擎"范式(`img_expand`/`sram_copy`/
`upsample2x`：Port-B 流式、`NPU_DMA_*_TRIG` 触发、`NPU_DMA_STATUS` 出完成位、
`npu_top` 内 mux)。新增一个 DFL 引擎;sigmoid 复用 post-process LUT。

```
检测头 bbox conv(NPU,LINEAR INT8) ──drain──▶ DDR/Act SRAM (64ch=4字/anchor)
                                                    │ CPU DMA 进 Act SRAM(逐尺度)
                                                    ▼
                                         ┌─────────────────────┐
                                         │  dfl_unit.v (新)     │  每字16×INT8
                                         │  max→EXP_LUT→Σe,Σe·W │  → 1×Q8.8 距离
                                         │  →整数除→Q8.8        │
                                         └─────────────────────┘
                                                    │ 距离写回 SRAM/DDR
检测头 cls conv(NPU)──post_process LUT(sigmoid 表)──▶ INT8/Q0.8 概率──▶ DDR
                                                    │
                              CPU(PicoRV32,纯整数/定点):
                              argmax(80类,单调,可对原始或概率)+ box几何 + NMS
```

## 组件

### 1. `rtl/dfl_unit.v`(新)

- **输入**：每拍/每事务从 Act SRAM Port-B 读一个 128-bit 字 = 16 个 INT8 logit。
- **数据通路**：
  1. max 归约 → `int8_max`(有符号 8-bit)。
  2. 每 k：`diff = z_k − int8_max ∈ [−255,0]`;`e_k = EXP_LUT[diff]`(256×16-bit
     无符号 Q1.15,`exp(0)=32768`)。
  3. `Sden = Σ_16 e_k`(32-bit 无符号);`Snum = Σ_16 e_k·W_k`,`W_k` 为 16 项
     有符号 Q8.8 常数(48-bit 有符号累加)。
  4. `distance = Snum / Sden`(迭代有符号除法器)。因 Q1.15 因子相消,商直接为
     **Q8.8**(`Snum/Sden = 2^8·Σe_kW_k/Σe_k`),无需预移位。输出 16-bit 有符号。
- **输出**：每字 → 1 个 Q8.8 距离,写回 SRAM(4 个/anchor → 连续)。
- **配置**：字数(= anchor 数 ×4)、源/目的 SRAM 基址。
- **装载口**：`EXP_LUT`(256 项,按尺度重载,共 3 次)、`W_k`(16 项,全程驻留)。

### 2. sigmoid:复用 `post_process_top.v` LUT

- 新增一张 256 项 sigmoid 表(INT8→Q0.8 无符号概率),与 `silu_lut` 并列。
- 新 CTRL 位选表(SiLU / sigmoid)。head 的 cls conv 内联过表,直接输出概率字节。
- 离线用 C 同公式生成表(`sigmoid(dequant(int8))` 量化到 Q0.8)。

### 3. CPU 侧(firmware,纯整数/定点)

读距离(Q8.8)与 cls 概率,做 argmax、box 几何(定点)、NMS(定点 IoU)。argmax
利用 sigmoid 单调性,可直接对原始/概率比较;最终框 conf 用概率。

## 寄存器(两边同步 `param_regfile.v` + `firmware.h`)

```
NPU_DFL_TRIG     写:启动 dfl_unit
NPU_DFL_CFG      字数(anchor×4) + 源/目的 SRAM 基址
NPU_DFL_EXP_LUT  EXP_LUT 装载口(256 项流式写)
NPU_DFL_W        W_k 16 项常数 regfile
NPU_SIGM_LUT     sigmoid LUT 装载口(256 项)
NPU_CTRL[..]     sigmoid 选表位
NPU_DMA_STATUS[..] dfl_done 完成位
```

新文件进 `axi_sys.f`。算子默认 OFF,FSM/post-process 在关时字节级不变。

## 数据流(逐尺度)

1. NPU 跑 head bbox/cls conv,INT8 输出 drain 到 DDR。
2. cls conv 经 sigmoid LUT → 概率(可在 conv 时内联完成)。
3. CPU 把该尺度 bbox 的 64ch logits DMA 进 Act SRAM;装 `EXP_LUT`(该尺度 scale)。
4. 触发 `dfl_unit` → 距离写回 SRAM → drain DDR。
5. 三尺度循环后,CPU argmax+几何+NMS → 最终框。

## 错误处理 / 边界

- `Sden==0` 不会发生(`exp(0)=32768` 至少一项;max 那项 diff=0)。除法器对 0
  仍给安全默认(0)。
- diff 钳到 [−255,0];`int8_max` 由硬件 max 保证 diff≤0。
- exp 下溢(大负 diff)→ LUT 项为 0,正常。

## 验证(TDD,逐算子 + 集成,默认 OFF 保 MNIST 10/10)

- `tests/tb_dfl_unit.v`：喂 C 参考 dump 的某尺度 bbox logits(INT8)+ 对应
  `W_k`/`EXP_LUT`,逐 anchor 比对 distance(Q8.8)与 C 的 `bbox_dfl`,≤ 容差。
- sigmoid LUT TB：全 256 输入 == C `sigmoid(dequant(int8))` 量化值。
- `post_process_top` 关 sigmoid 位时与现有 SiLU/LINEAR 路径字节级一致。
- 每阶段后 `run_all.sh sim` 保 MNIST 10/10。

## 定点格式汇总

| 量 | 格式 | 位宽 |
|----|------|------|
| `e_k`(exp) | 无符号 Q1.15(exp0=32768) | 16 |
| `W_k` | 有符号 Q8.8 | 16 |
| `Sden` | 无符号整数累加 | 32 |
| `Snum` | 有符号整数累加 | 48 |
| `distance` | 有符号 Q8.8 | 16 |
| cls 概率 | 无符号 Q0.8 | 8 |

## 范围外(YAGNI)

- argmax / box 几何 / NMS 留 CPU 定点(便宜、易对齐 C)。
- 不做全 decode 硬件化(几何需 anchor 网格 + stride,NMS 需排序/IoU,性价比低)。
- 这是检测头后处理;整网前向卷积组装(`yolo_full.c`)是另一条并行工作线。
```
