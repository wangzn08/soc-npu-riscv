# YOLOv8n INT8 部署验证 — 设计文档

日期: 2026-06-16
状态: 已批准

## 背景

`yolov8n_int8/` 目录已存在一份此前未提交的工作：YOLOv8n 权重导出
（`.pt` / FP32 ONNX / INT8 ONNX）、按 conv 拆分的二进制权重文件
（`weights/`, `weights_fp32/`）、纯 C 推理引擎（`yolov8n_infer.c/.h`,
`yolov8n_layers.h`）、以及未跑出结果的验证脚本（`validate_coco.py`,
`test_infer.c`）。这些文件未被 git 跟踪。

代码检查发现两个关键事实：

1. **已有 C 引擎不是真·INT8**：`yolov8n_infer.c` 中权重是 int8，但激活值、
   bias、累加全程 float32（`extract_weights.py` 把 ultralytics 导出的 INT8
   ONNX 反量化回 float、融合 BN 后，只对权重重新做 per-tensor INT8 量化）。
   这种方案精度接近 FP32，但不代表 SoC NPU 的真实数据路径（NPU 是 INT8
   MAC + 定点 bias/scale/shift）。
2. **`yolov8n_int8.onnx` 是真正的全 INT8 QDQ 模型**：图中有 242 个激活
   `QuantizeLinear` 节点，每个都带 per-tensor 校准 scale（例如输入层
   scale=1/255，中间层 0.22~1.04 不等），64 个 Conv，SiLU 表示为
   Sigmoid+Mul，3 个 MaxPool，2 个 Resize（上采样）。这些激活 scale
   可以直接复用，无需重新校准。

机器环境：RTX 4060 Laptop GPU (8GB)，当前装的是 CPU 版 torch
（`2.8.0+cpu`）和不带 CUDA 的 onnxruntime；gcc 在 `C:\msys64\mingw64\bin`
可用。

## 目标

1. 产出 SoC NPU 真实定点路径对应的 INT8 精度数字（COCO val2017 全量
   mAP50 / mAP50-95），而不是“权重 INT8、激活 FP32”的近似数字。
2. 产出一份纯 C 的全 INT8 YOLOv8n 推理实现，数据路径（INT8×INT8→INT32
   累加 → 定点 scale/shift 反量化 → INT8）与 SoC NPU 一致，可作为后续
   往 NPU 移植的软件参考模型。
3. 验证该 C 引擎在真实图片上的输出与 `yolov8n_int8.onnx` 数值对齐。

## 非目标

- 不在本轮把 YOLOv8n 接到 RTL/NPU 仿真里跑（那是后续单独任务）。
- 不解决 SiLU 在 SoC NPU 上怎么映射的问题——C 参考实现忠实保留 SiLU，
  并在文档中记录这是后续 NPU 适配需要解决的缺口（当前 NPU 仅有
  ReLU/ReLU6/maxpool/avgpool 这类算子，参见 CLAUDE.md 的算子列表）。
- 不追求纯 C 引擎跑满 COCO val2017 5000 张的性能（无 SIMD 的逐元素卷积
  在 5000 张 640x640 图上太慢）；精度数字由 onnxruntime 跑出，C 引擎只做
  小规模数值对齐验证。

## 架构

### 1. 环境准备

- 重装 CUDA 版 PyTorch（`cu124` 系列wheel）和 `onnxruntime-gpu`，验证
  `torch.cuda.is_available()` 和 onnxruntime CUDA provider 都能用。
- 失败兜底：若 CUDA 安装因网络/驱动问题失败，退回 CPU 版重新跑（仅影响
  耗时，不影响正确性），并明确告知用户改用了 CPU。

### 2. 真·INT8 精度基线（权威数字）

- 直接用现有 `yolov8n_int8.onnx`（已验证是真全 INT8 QDQ 图，不重新导出）
  跑 `ultralytics` 的 `model.val(data='coco.yaml', imgsz=640)`，数据集用
  COCO val2017 全量（5000 张，自动下载或使用已有 ultralytics 缓存）。
- 同时跑 FP32 `yolov8n.pt` 作为对照基线。
- 输出 mAP50、mAP50-95、Precision、Recall 的 FP32 vs INT8 对比表。
- 验收：跑出实际数字即可，不预设任何“合格阈值”——数字本身就是交付物，
  报告里如实记录，不因为掉点多就判定失败。

### 3. 纯 C 全 INT8 推理引擎

**权重/激活抽取（改 `extract_weights.py`）**

- 解析 `yolov8n_int8.onnx` 图时，除了现有的 Conv+BN 融合权重抽取，
  额外抽取每个 QuantizeLinear/DequantizeLinear 节点的 **激活 scale**
  （per-tensor，float32），按执行顺序关联到对应的 conv/激活节点。
- 每个 conv 节点计算定点 requant 参数：
  `output_int8 = clamp(round(acc_int32 * (in_scale * w_scale / out_scale)), -128, 127)`，
  把 `in_scale * w_scale / out_scale` 转换成 `(multiplier, shift)` 定点对
  （沿用 SoC NPU 现有的 bias/scale/shift 寄存器语义，参考
  `rtl/param_regfile.v` 和 `NPU_BIAS/SCALE/SHIFT` 寄存器格式）。
- bias 同样按 `bias_int32 = round(bias_float / (in_scale * w_scale))`
  量化为 int32，与累加器同尺度相加，再一起 requant。
- 导出文件新增/调整：
  - `weights/convN_w.bin`：int8 权重（不变）
  - `weights/convN_b.bin`：改为 int32 定点 bias（原来是 float32 bias）
  - `weights/convN_s.bin`：改为 `{multiplier:int32, shift:int8}` 定点
    requant 参数（原来是单个 float32 scale）
  - `yolov8n_layers.h`：新增每个激活张量的 int8 zero-point（若校准用了
    非零 zp，需要保留；若都是 symmetric/zp=0 则可以省略字段但要在头文件
    注释里写清楚）

**C 引擎改写（改 `yolov8n_infer.c/.h`）**

- Tensor 结构从 `float *data` 改为 `int8_t *data`（保留一个内部 int32
  累加 buffer，每个 conv 输出位置算完整 IC*KH*KW 的 INT32 累加和再
  requant 成 int8）。
- Conv kernel：`acc_int32 += (int32_t)in_int8 * (int32_t)w_int8`，完整
  累加完后加 int32 bias，按 `(acc * multiplier) >> shift` 定点 requant，
  clamp 到 int8。
- SiLU（Sigmoid+Mul）：保留为查表 LUT（int8 输入 -> int8/int16 输出的
  256 项表，预计算），避免在纯整数路径里跑浮点 exp。
- MaxPool/Resize/Concat/Add：在 int8 域直接做（maxpool 是 element-wise
  max，不需要重新量化；concat 需要check相邻分支的 scale 是否一致，如不
  一致需 requant 到统一 scale 再 concat——按 ONNX 图里 concat 输入的
  实际 scale 处理，不假设)。
- 检测头 decode（bbox 回归、objectness、NMS）：维持现有 float 实现，
  只在网络最后输出处把 int8 dequant 回 float 即可，不强求这部分变成
  整数（SoC NPU 之外的 CPU 端后处理本来就是 float，不在“INT8 数据路径”
  讨论范围内）。

**对齐验证**

- 写一个小脚本/复用 `test_infer.c`：用 onnxruntime 跑
  `yolov8n_int8.onnx` 在几张真实 COCO 图片上，导出中间若干层（至少
  第一个conv输出、一个 C2f block 输出、最终检测头输出）的 int8 张量，
  和 C 引擎同样位置的输出做逐元素比较，允许 ±1 LSB 误差（定点 shift
  舍入差异），不允许系统性偏差或形状不对。
- 在真实图片上跑 C 引擎得到检测框，肉眼/打印核对框位置和类别合理
  （不要求精确复现 onnxruntime 的 NMS 结果，因为 NMS 阈值/实现细节
  可能有差异，但目标框应该基本一致）。

### 4. 报告

新增 `yolov8n_int8/REPORT.md`，内容：

- FP32 vs INT8(QDQ onnxruntime) 在 COCO val2017 全量上的 mAP50/mAP50-95/P/R
- C 引擎与 ONNX 的数值对齐结果（哪几层、误差范围）
- 已知缺口：SiLU 在当前 SoC NPU 上没有硬件算子（NPU 目前是
  ReLU/ReLU6/maxpool/avgpool/depthwise/pointwise，参见 CLAUDE.md），
  这份 C 参考实现忠实保留 SiLU，移植到 NPU 时需要额外方案（查表或近似）
- 已知缺口：纯 C 引擎性能未优化，不代表 NPU 移植后的速度

## 测试策略

- 对齐验证用真实 COCO 图片（不用 coco8 的 8 张，那个数据集太小不能代表
  分布；用 COCO val2017 里随机抽 5~10 张）。
- mAP 走 COCO val2017 全量标准评测协议（ultralytics 自带，复用现有
  `validate_coco.py` 的下载/评测逻辑，按需补全）。
- 不写单元测试框架；这是一次性验证任务，验收标准是“报告里的数字和对齐
  结果真实可信”，不是自动化测试套件。

## 风险/已知限制

- CUDA 环境重装可能因网络/版本不兼容失败 → 兜底用 CPU 跑，只影响耗时。
- COCO val2017 标注+图片下载约 1GB+，需要网络（国内可能需代理，参考
  memory 里的 `reference_github_repo.md` 提到的代理经验）。
- `axi_upsizer_32_128`/SoC 当前 NPU 一次只验证过 MNIST 规模的 conv（最大
  64 OC, 28x28 输入附近）；YOLOv8n 640x640、80 OC+ 的规模远超当前 NPU
  实际跑过的测试范围，这份工作只产出 C 软件参考和精度数字，**不代表
  YOLOv8n 已经能在当前 RTL 上跑通**——那是真正部署到 SoC 的下一步工作，
  不在本次范围内。
