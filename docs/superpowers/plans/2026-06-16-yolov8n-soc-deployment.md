# YOLOv8n INT8 → SoC NPU 部署计划（RTL 仿真报告 · 640×640 · 放大 SRAM 整网驻留 · SiLU LUT）

> **新窗口执行说明**：这是一份**部署可行性计划**(目标=RTL 仿真报告,不上 FPGA),不是逐行
> TDD 脚本。按阶段推进,每阶段有**门控**和**C 黄金参考逐层对齐**验收。新窗口先读本文件 +
> `CLAUDE.md` + `yolov8n_int8/REPORT.md`。

**目标**：在现有 PicoRV32 + 16×16 NPU SoC 的 **RTL 仿真**上跑通 YOLOv8n INT8(640×640),
每层输出与已验证的纯 C 参考引擎逐层数值对齐,产出仿真报告(功能 + 数值对齐 + 性能计数器)。

**已锁定决策**：
1. **交付 = RTL 仿真报告**(不做 FPGA 验证)。→ 因此**直接在 `sram_models.v` 把片上 SRAM 放大到
   能整网驻留,砍掉空间分块**(空间分块是 FPGA 资源现实路线,本次不做)。
2. **保持 640×640** 输入。
3. **SiLU 用 NPU post-process LUT**(256 项 int8→int8),原模型不重训。
4. **可行性优先**:先放大 SRAM + 补算子,把代表性子集跑通对齐,再向整网扩。

> **诚实声明(必须写进报告)**:放大 SRAM 到数 MB + 维持 `COMB_B=1`(Port-B 组合读)是**仿真
> 约定下的功能/数值验证简化**,**不代表 FPGA 资源/时序现实**(典型 FPGA BRAM 只 0.5–4MB,且大
> SRAM 在 FPGA 上要 +1 拍寄存器输出)。本工作证明的是"YOLOv8n INT8 能在该 NPU 数据通路上数值
> 正确地跑通",FPGA 落地需另做空间分块/资源优化。

---

## 0. 现状与差距(执行者必读)

**黄金参考(本仓库 `yolov8n_int8/`)**：
- `yolov8n_infer.c`：纯 C 全 INT8 引擎,数据路径与 NPU 一致(int8×int8→int32 + 输入 zero-point
  校正 + 定点 epilogue),已与 onnxruntime 对齐(conv0 ~1 LSB),INT8 mAP50-95=0.352。**逐层 bit 级
  黄金输出来源**(`YOLO_DEBUG_DUMP` 钩子;按需给更多层加 dump)。
- `yolov8n_int8.onnx`(已正确量化)、`weights/*.bin`(int8 权重)、`yolov8n_layers.h`
  (`yolo_conv[64]` 维度 + `yolo_act_quant[64]` 激活 scale/zp + `yolo_glue_quant`)。

**现有 NPU 已支持(见 CLAUDE.md,多在 `npu-operator-generality` 分支)**：3×3 conv、`pw_en`
(1×1,YOLOv8n 大量用)、maxpool、`NPU_SKIP_BASE`(残差 add)、`NPU_CLIP_MAX`(ReLU6)、16×16
array、`oc_single`/`row_par`/`gemm_reduce`、AXI DMA、`img_expand`/`sram_copy`、perf 计数器。
**大通道(256+ OC/IC)由现有 ic/oc 分块天然支持,不是新能力。**

**本计划要补的差距**(空间分块已砍,所以只剩算子 + 容量):
1. **片上 SRAM 容量**:放大 `sram_models.v` 的 Act/Wgt/Out 到整网驻留(Phase 1)。这是"用仿真
   自由度换掉最难的工程问题"。
2. **SiLU**:post_process 加 256 项 LUT。
3. **最近邻 2× 上采样**(neck):新引擎。
4. **Concat dataflow**(C2f/SPPF/neck):多源 int8 requant 到统一 scale 聚连续区。
5. **检测头 decode(DFL/box/sigmoid/NMS)保持 CPU 浮点**,照搬 C 参考,不上 NPU。

**瓶颈预期(供报告性能叙事)**:放大 SRAM + 整网片上驻留后,DDR 只搬"输入图 + 权重 + 最终
logits",**带宽不再是瓶颈**;阵列对 YOLOv8n(4.35 GMAC,峰值 0.82 TMAC/s,理想 ~5.3ms/图)
**过配**。剩余 cyc 主导预计是**一次性权重预载 + CPU 逐层 MMIO 调度**(与 MNIST 同构:`npu_busy`
占比低、`array_util` 低)。→ 后续优化杠杆是 descriptor 驱动 DMA,不是加算力。

---

## 验证方法论(贯穿全程)

**C 引擎逐层 int8 输出 = 黄金**。每个 NPU 层/算子上线后 dump 其 Out-SRAM int8 结果,与 C 引擎
对应层**逐元素比对,bit-exact 或 ≤±1 LSB**。**不允许只看"有框",必须逐层对数。**

---

## Phase 0：尺寸与仿真预算分析(纯分析,门控)

**目的**:算清 SRAM 要放多大、整网 cyc 量级,确认仿真可承受。

- [ ] **0.1 逐层尺寸表**:脚本读 `yolov8n_layers.h`,对全 64 层列 IC/OC/H/W/KH/KW/stride/pad、
  激活字节(H×W×ceil(C/16)×16)、权重字节、im2col 字节。
- [ ] **0.2 峰值工作集**:求"前一层激活(ping)+ 当前层激活(pong)+ Out + im2col + 当前层权重"的
  峰值 → 定 **Act/Out/Wgt SRAM 各放多大**(预计 Act 需 ~3–4MB,Out ~2MB,Wgt 看是否全网驻留 3.2MB
  还是按层换入)。
- [ ] **0.3 仿真 cyc 预算**:理想计算 4.35GMAC/4096 ≈ 1.06M cyc;按 MNIST 利用率经验放大估总 cyc
  (可能 5–30M)。评估 ModelSim 跑得动否;若太慢,先用**子集**(Phase 3)而非整网做主要验证。

**门控**:产出 `docs/notes/yolov8n-deploy-sizing.md`(尺寸表 + SRAM 目标大小 + cyc 预算)。

---

## Phase 1：放大片上 SRAM(仿真),保住 MNIST 基线

- [ ] **1.1 放大 `rtl/sram_models.v`** 的 Act/Wgt/Out 深度到 Phase 0.2 的目标(整网驻留)。
- [ ] **1.2 保持 `COMB_B=1`**(组合读)——仿真约定,**报告里标注为非 FPGA 现实**(见顶部声明)。
- [ ] **1.3 回归保护**:`bash run_all.sh sim` 确认 **MNIST 仍 10/10**(地址/位宽/驻留逻辑没被放大
  破坏)。SRAM 基址常量同步检查 `firmware.h` / `param_regfile.v`。

**门控**:SRAM 放大后 MNIST 基线不回归。

---

## Phase 2：NPU 算子扩展(RTL,每个独立 TB + C 黄金对齐)

每个默认 OFF(CTRL bit 关时 FSM 字节级不变,沿用 opt-in 惯例)。寄存器/CTRL 改动同步
`param_regfile.v` + `firmware.h` + `axi_sys.f`;新 RTL 文件进 `axi_sys.f`;TB 进 `tests/`。

- [ ] **2.1 SiLU LUT**(`post_process_top.v`/`vector_alu.v`):256 项 int8→int8 表 + `silu_en` CTRL +
  装表寄存器。LUT 初值离线用 C 同公式(dequant→SiLU→requant)生成。TB:全 256 输入 == C SiLU epilogue。
- [ ] **2.2 最近邻 2× 上采样**(`rtl/upsample2x.v`,新):仿 `img_expand`/`sram_copy`,每像素复制
  2×2,scale/zp 不变,`NPU_DMA_*_TRIG` 触发 + `NPU_DMA_STATUS` 完成位。TB == C `upsample2x`。
- [ ] **2.3 Concat 聚合**(扩展 `sram_copy.v` 或新):N 源按**下游 conv 的 in_scale**(C 参考就是这么
  做,不查 glue 表)做 requant 后聚到连续 Act 区。TB == C `concat2_to_conv` / C2f 内 concat。
- [ ] **2.4 残差 Add 复核**:C2f 残差。确认 `NPU_SKIP_BASE` 能"两路各 dequant→相加→requant 到 Add
  自己 calibrated scale"(C 用 `yolo_glue_quant[add_idx]`);不够则扩展。TB == C backbone C2f Add。

---

## Phase 3：代表性子集 bring-up(整网驻留,无分块)

- [ ] **3.1 选子集**:stem `conv0,conv1` + 一个 C2f(`model.2`=conv2..5,覆盖 1×1+3×3+残差+concat+
  SiLU)+ SPPF 一段 maxpool + **一个检测尺度**(如 P5)head 卷积 + DFL conv63。覆盖全部新算子。
- [ ] **3.2 固件调度**(新 `firmware/yolov8n_deploy.c`,参考 `deepnet_deploy.c`):CPU 逐层 program
  NPU/DMA,层间 `sram_copy` 片上驻留(SRAM 已够大,无 DDR 往返)。检测头后 CPU 跑 DFL softmax +
  box 解码 + sigmoid + NMS(浮点,照搬 C 逻辑)。
- [ ] **3.3 逐层对齐**:每层 Out-SRAM == C 引擎对应层,≤±1 LSB。
- [ ] **3.4 子集端到端**:RTL 仿真前向 + CPU decode 跑通,检测框与 C 引擎在同一真图上类别/位置
  一致;读 perf 计数器(cyc_total / npu_busy / array_util / rd_beats / wr_beats)。

**门控**:子集逐层对齐 + 跑通,产出阶段报告。

---

## Phase 4：扩到整网端到端(本路径下可达)

没有空间分块阻碍后,逐步加 backbone 其余层、SPPF 全段、neck/FPN(用 2.2 上采样 + 2.3 concat)、
三个检测尺度,直到整网 640×640 在 RTL 仿真端到端跑通,**整网逐层对齐 C 黄金**,出最终仿真报告:
功能正确 + 数值对齐 + 性能计数器 + 顶部那段 FPGA 现实性声明。

---

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 640 cycle-accurate 仿真慢 | Phase 0.3 估 cyc;主要验证用子集;`run.do` 避免 `log -r /*` |
| 放大 SRAM 破坏 MNIST 基线(地址/位宽) | Phase 1.3 强制 MNIST 10/10 回归门控 |
| SiLU LUT 精度 | LUT 离线 C 同公式生成,TB 全 256 值对齐 |
| Concat/残差 scale 处理错 | 严格按 C 参考:concat 用下游 conv in_scale;Add 用 `yolo_glue_quant` |
| 报告把仿真简化当 FPGA 结论 | 顶部"诚实声明"必须进报告:SRAM/COMB_B 是仿真约定,非 FPGA 现实 |

## 关键文件

```
黄金参考: yolov8n_int8/yolov8n_infer.c (+ YOLO_DEBUG_DUMP), yolov8n_layers.h, weights/*.bin
RTL:      rtl/sram_models.v (放大), rtl/post_process_top.v (SiLU LUT), rtl/upsample2x.v (新),
          rtl/sram_copy.v (concat扩展), rtl/param_regfile.v (寄存器)
固件:     firmware/yolov8n_deploy.c (新, 参考 deepnet_deploy.c), firmware/firmware.h
构建:     axi_sys.f (新文件), run_all.sh (FW_C_SRCS 切换), tests/ (每算子 directed TB)
约定:     寄存器两边同步; 算子默认 OFF 保持 FSM 字节级不变; 改 RTL 后先 run_all.sh clean
```

## 与现有工作的关系

- 这是 `yolov8n_int8/` INT8 验证(已完成,mAP 0.352)的**下游硬件仿真部署**。
- **新分支**(如 `yolov8n-npu-deploy`)从 master 做,保持 MNIST 10/10 绿。
- 复用 `npu-operator-generality` 的 pw/pool/residual/ReLU6 成果(若未并入 master,先评估合并)。
- **空间分块 / FPGA 资源优化是本次明确的非目标**,留作后续真正上板的工作。
