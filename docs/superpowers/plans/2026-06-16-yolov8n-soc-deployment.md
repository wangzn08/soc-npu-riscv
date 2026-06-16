# YOLOv8n INT8 → SoC NPU 部署计划（可行性验证优先 · 640×640 · SiLU LUT）

> **新窗口执行说明**：这是一份**部署可行性计划**,不是逐行 TDD 脚本——后期阶段的具体
> 代码依赖 Phase 0 的尺寸分析结果。请按阶段推进,每个阶段都有明确的**门控(gate)**和
> **黄金参考对齐**验收。建议在新窗口里先读本文件 + `CLAUDE.md` + `yolov8n_int8/REPORT.md`。

**目标**：在现有 PicoRV32 + 16×16 NPU SoC 的 RTL 仿真上,跑通 YOLOv8n INT8 的一个**代表性
子集**(640×640 输入),每层输出与已验证的纯 C 参考引擎逐层数值对齐,作为完整部署前的
de-risk。

**已锁定决策**：
1. **可行性验证优先**——先扩展算子 + 解决大特征图数据通路,跑通代表性子集,不强求一上来
   就整网端到端。
2. **保持 640×640**——真实部署分辨率;因此**片上存储/空间分块是第一难题**,必须 Phase 0/2
   优先解决。
3. **SiLU 用 NPU post-process LUT**——256 项 int8→int8 查表,与现有 bias/scale/shift epilogue
   同级,原模型不需重训/重量化。

---

## 0. 现状与差距(执行者必读)

**已有的黄金参考(本仓库 `yolov8n_int8/`)**：
- `yolov8n_infer.c`：纯 C 全 INT8 引擎,数据路径与 NPU 一致(int8×int8→int32 累加 + 输入
  zero-point 校正 + 定点 epilogue),已与 onnxruntime 对齐(conv0 ~1 LSB),COCO val2017
  INT8 mAP50-95=0.352。**这是每一层的 bit 级黄金输出来源**(用 `YOLO_DEBUG_DUMP` 钩子)。
- `yolov8n_int8.onnx`(已正确量化,排除 decode 尾)、`weights/*.bin`(int8 权重)、
  `yolov8n_layers.h`(`yolo_conv[64]` 维度 + `yolo_act_quant[64]` 激活 scale/zp + `yolo_glue_quant`)。

**现有 NPU 能力(见 CLAUDE.md)**——大部分 YOLOv8n 需要的算子其实已经在 `npu-operator-generality`
分支里:
- 3×3 conv、`pw_en`(1×1 pointwise,YOLOv8n 大量用)、`dw_en`(depthwise,YOLOv8n 不用)、
  maxpool(SPPF 用)、`NPU_SKIP_BASE`(残差 add,C2f 用)、`NPU_CLIP_MAX`(ReLU6 钳位)、
  16×16 array、`oc_single`/`row_par`/`gemm_reduce`、AXI DMA、`img_expand`/`sram_copy`、perf 计数器。
- 大通道(YOLOv8n 最多 256+ OC)：现有 `ic_groups`/`oc_single` 分块逻辑天然支持,只是迭代更多次,**不是新能力**。

**YOLOv8n 相对当前 NPU 的真实差距(本计划要解决的)**：
1. **大特征图空间分块**(最难)：640×640 → conv0 输出 320×320×16 ≈ 1.6 MB,远超 Act SRAM
   256 KB。当前 NPU 只在 MNIST(28×28)验证过,整图驻留片上。**必须引入空间 tiling + DDR
   streaming + 3×3 卷积 tile 边界 halo/padding**。
2. **SiLU**：post_process 无此算子 → 加 256 项 LUT。
3. **最近邻 2× 上采样**(neck FPN/PAN)：无此引擎 → 仿 `img_expand`/`sram_copy` 写一个。
4. **Concat dataflow**(C2f/SPPF/neck)：需把多源 int8 块 requant 到统一 scale 后聚到连续
   Act 区域;可基于 `sram_copy` 扩展。
5. **检测头 decode(DFL softmax/box 解码/sigmoid/NMS)保持 CPU 浮点**,与 C 参考一致,不上 NPU。

---

## 验证方法论(贯穿所有阶段)

**C 引擎逐层 int8 输出 = 黄金**。每个 NPU 层/算子上线后,dump 其 Out-SRAM int8 结果,与 C 引擎
对应层输出**逐元素比对,要求 bit-exact 或 ≤±1 LSB**(定点 shift 舍入)。复用 `yolov8n_infer.c`
的 `YOLO_DEBUG_DUMP`,必要时给 C 引擎加更多层的 dump 钩子。**不允许只看"能跑/有框",必须逐层对数。**

---

## Phase 0：可行性与尺寸分析(纯分析,不动 RTL)——门控

**目的**：用数字确认 640×640 的存储/计算需求,定下空间分块方案,再投入 RTL。

- [ ] **0.1 逐层尺寸表**：对子集涉及的每层(见 Phase 3 选层),列 IC/OC/H/W/KH/KW/stride/pad、
  激活字节数(H×W×ceil(C/16)×16)、权重字节数、im2col 中间字节数;对比 Act/Wgt/Out SRAM 容量
  (256KB/?/128KB,以 `rtl/sram_models.v` 实际为准)。脚本可直接读 `yolov8n_layers.h`。
- [ ] **0.2 标记超容层**:哪些层单层激活就放不下 SRAM → 必须空间分块。预计 conv0..浅层全部超容。
- [ ] **0.3 定空间分块方案**:行条带(row-strip)还是 tile;条带高度/重叠 halo 行数(3×3 conv 需
  上下各 1 行 halo,stride/pad 相关);DDR↔Act SRAM 的 streaming 顺序。给出每方案的 SRAM 占用 +
  DDR 往返估算 + 仿真 cycle 量级估算。
- [ ] **0.4 仿真时间评估**:640×640 在 cycle-accurate ModelSim 上单层就可能极慢。估算子集总
  cycle,决定是否需要先用更小的代表性输入(如 160×160 的真图)做功能对齐、再单独验证一层 640 分块。

**门控**:产出 `docs/notes/yolov8n-deploy-sizing.md`(尺寸表 + 分块决策 + 仿真预算)。确认可行、
方案选定后才进入 Phase 1。**若 0.3 表明 640 分块在仿真上不可承受**,回到用户讨论是否退守
320×320 跑通流程、只在一层上验证 640 分块。

---

## Phase 1：NPU 算子扩展(RTL,每个独立 TB + C 黄金对齐)

每个算子默认 OFF(CTRL bit 关时 FSM 字节级不变,沿用 CLAUDE.md 的 opt-in 惯例),独立 directed TB。

- [ ] **1.1 SiLU LUT**(`rtl/post_process_top.v` / `vector_alu.v`)：256 项 int8→int8 查表,新增
  CTRL bit(如 `silu_en`)+ 寄存器装表。表内容 = 对每个 int8 输入 q,算 dequant→SiLU→requant→int8
  (离线用 C 同公式生成 LUT 初值)。TB:对全 256 个输入值,RTL 输出 == C 的 SiLU epilogue。
  寄存器/CTRL 改动同步 `rtl/param_regfile.v` + `firmware/firmware.h` + `axi_sys.f`。
- [ ] **1.2 最近邻 2× 上采样引擎**(`rtl/upsample2x.v`,新文件)：仿 `img_expand`/`sram_copy`,
  Act/Out SRAM Port B → Act SRAM,每像素复制成 2×2,scale/zp 不变。`NPU_DMA_*_TRIG` 风格触发 +
  `NPU_DMA_STATUS` 完成位。TB:小图上采样结果 == C `upsample2x`。新文件加 `axi_sys.f`。
- [ ] **1.3 Concat 聚合**(扩展 `rtl/sram_copy.v` 或新增):把 N 个源块按目标 conv 的 in_scale
  做 requant(dequant→requant)再聚到连续 Act 区域。TB:两源 concat 结果 == C `concat2_to_conv`/
  C2f 内 concat。**注意 C 参考里 concat 目标 scale = 下游 conv 的 in_scale**(不查 glue 表)。
- [ ] **1.4 残差 Add 复核**:C2f 残差用现有 `NPU_SKIP_BASE`。确认它能做"两路 int8 各 dequant→相加
  →requant 到 Add 自己 calibrated scale"(C 参考用 `yolo_glue_quant[add_idx]`)。若现有 skip 只支持
  同 scale 相加,需扩展。TB 对齐 C 的 backbone C2f Add。

---

## Phase 2：大特征图空间分块数据通路(核心新能力)——门控

- [ ] **2.1 单卷积 spike**:先只做"一个 3×3 conv(stride1,pad1)对 640×640 输入按行条带分块",
  从 DDR 流式读条带(含 halo 行)→ Act SRAM → im2col → array → Out SRAM → 写回 DDR。**先在一层上
  把分块/halo/边界 padding 跑对**,与 C 该层输出逐元素对齐。
- [ ] **2.2 扩展 FSM/DMA**(`rtl/top_controller_fsm.v`,`rtl/axi_dma.v`):把分块循环固化进调度,
  支持任意 H/W 的条带遍历 + 跨条带 halo 复用 + tile 边界 padding(复用 `hw_pad`)。
- [ ] **2.3 stride-2 conv 分块**(conv0 等下采样层)的边界对齐。

**门控**:单卷积 640 分块与 C bit 对齐通过,才扩展到多层。这是整份计划风险最高的一步,**务必先
spike 再泛化**(见 CLAUDE.md "为有风险的 RTL 时序改动先写 directed TB")。

---

## Phase 3：代表性子集 bring-up on RTL 仿真

- [ ] **3.1 选子集**(建议):stem `conv0,conv1` + 一个 C2f(`model.2`=conv2..5,含 1×1+3×3+残差+
  concat)+ SPPF 的一段 maxpool + **一个检测尺度**(如 P5)的 head 特征卷积(cv2/cv3 各几层)+
  DFL conv63。覆盖:3×3、1×1(pw)、SiLU、残差、concat、maxpool、空间分块、检测头卷积。
- [ ] **3.2 固件调度**(新 `firmware/yolov8n_deploy.c`,参考 `deepnet_deploy.c` 的逐层 NPU/DMA
  寄存器编程 + IRQ 协作):CPU 按层 program NPU,层间用 `sram_copy`/DDR streaming。检测头之后 CPU
  跑 DFL softmax + box 解码 + sigmoid + NMS(浮点,照搬 C 参考逻辑)。
- [ ] **3.3 逐层对齐**:每层 Out-SRAM 结果与 C 引擎对应层逐元素比对,≤±1 LSB。
- [ ] **3.4 端到端子集结果**:子集前向 + CPU decode 在 RTL 仿真跑通,检测框与 C 引擎在同一张真图上
  基本一致(类别/位置合理,不强求 NMS 逐框相同)。

**门控**:子集逐层对齐 + 跑通,产出阶段报告(cycle 数、SRAM 占用、对齐结果、瓶颈)。

---

## Phase 4(stretch):扩展覆盖

跑通子集后,逐步加:neck/FPN(用 1.2 上采样 + 1.3 concat)、其余两个检测尺度、完整 backbone,
直到整网端到端。每加一块仍按"逐层对齐 C 黄金"推进。

---

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| **640 片上存储不够**(最大) | Phase 0 先定空间分块;Phase 2 先单卷积 spike |
| **640 cycle-accurate 仿真极慢** | Phase 0.4 评估;必要时小分辨率验功能 + 单层验 640 分块;`run.do` 避免 `log -r /*` |
| tile 边界 halo/padding 错 | Phase 2.1 单卷积 spike 逐元素对齐后再泛化 |
| SiLU LUT 精度 | LUT 离线用 C 同公式生成,TB 全 256 值对齐 |
| FPGA BRAM 映射(COMB_B) | 沿用 CLAUDE.md 警告:Port-B 组合读是 sim 约定,上 FPGA 需整条读路径同步加一拍 |
| 权重/激活 SRAM 容量(256+ 通道) | 现有 ic/oc 分块支持;Wgt SRAM 容量在 Phase 0 核对 |

## 关键文件

```
黄金参考: yolov8n_int8/yolov8n_infer.c (+ YOLO_DEBUG_DUMP), yolov8n_layers.h, weights/*.bin
RTL 改动: rtl/post_process_top.v (SiLU LUT), rtl/upsample2x.v (新), rtl/sram_copy.v (concat扩展),
          rtl/top_controller_fsm.v + rtl/axi_dma.v (空间分块), rtl/param_regfile.v (寄存器)
固件:     firmware/yolov8n_deploy.c (新, 参考 deepnet_deploy.c), firmware/firmware.h
构建:     axi_sys.f (新 RTL 文件), run_all.sh (FW_C_SRCS 切换), 每个算子的 directed TB 进 tests/
约定:     寄存器改动两边同步(param_regfile.v + firmware.h); 算子默认 OFF 保持 FSM 字节级不变
```

## 与现有 SoC 工作的关系

- 这是 `yolov8n_int8/` INT8 验证(已完成,mAP 0.352)的**下游硬件部署**,不影响现有 MNIST 10/10 基线。
- 建议在**新分支**上做(如 `yolov8n-npu-deploy`),从 master,保持 MNIST 回归绿。
- 复用现有 `npu-operator-generality` 分支的 pw/dw/pool/residual/ReLU6 成果(若尚未并入 master,先评估合并)。
