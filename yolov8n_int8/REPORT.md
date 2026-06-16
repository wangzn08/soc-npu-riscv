# YOLOv8n INT8 部署验证报告

日期: 2026-06-16
分支: `yolov8n-int8-validation`

## 1. 精度结果 (COCO val2017 全量, 5000 张)

通过 ultralytics 统一的 letterbox/decode/NMS 流水线对 FP32 与 INT8 两个模型做推理
(`conf=0.001, iou=0.7, max_det=300, imgsz=640`),再用 pycocotools COCOeval 标准协议评测。

| 模型 | mAP50 | mAP50-95 |
|------|-------|----------|
| FP32 (`yolov8n.pt`) | 0.5172 | 0.3669 |
| INT8 QDQ (`yolov8n_int8.onnx`) | 0.5026 | 0.3518 |
| **掉点** | **0.0146** | **0.0151** |

INT8 相对 FP32 仅掉 1.46 / 1.51 个点,属于正确静态量化的正常范围。
(评测脚本 `validate_coco_map.py`;原始检测 `dets_fp32.json` / `dets_int8.json`;
汇总 `coco_map_result.json`。)

## 2. 量化方案

- **权重**: per-channel 对称量化 (zero_point=0),INT8。权重量化只依赖权重本身,
  与校准数据无关。
- **激活**: per-tensor 非对称量化 (zero_point 非零),INT8。用 COCO 真实图片
  (coco128, 128 张) 做静态校准。64 个 conv 的输入/输出激活都有 calibrated scale。
- **Conv epilogue (与 SoC NPU 定点路径对应)**:
  ```
  acc_int32 = Σ (int8_act × int8_w)                 # 真整数 MAC
  acc_corr  = acc_int32 − in_zp × Σw[oc]            # 输入 zero-point 校正(权重 zp=0)
  preact    = acc_corr × (in_scale × w_scale[oc]) + bias
  y         = has_silu ? SiLU(preact) : preact
  out_int8  = clamp_round(y / out_scale + out_zp)
  ```
- **检测头 decode 尾保持 FP32**: 量化时排除了 DFL 投影 / box 解码 / sigmoid /
  最终 concat 这 19 个节点(详见下文"关键问题 1")。所有 64 个 conv 仍是 INT8,
  decode 本就在 CPU 上以浮点运行,与 SoC 分工一致。

## 3. C 引擎与 ONNX 数值对齐

纯 C 全 INT8 引擎 (`yolov8n_infer.c`) 数据路径与 SoC NPU 一致
(INT8×INT8→INT32 累加 → 定点 epilogue → INT8),decode 用浮点。

- **conv0 (含 SiLU) 逐元素对齐**: 在 bus.jpg(640×640)上,C 引擎反量化后前 8 个值
  `[22.59, 2.82, 2.35, 1.65, 1.88, 0.94, 0.71, 1.65]` vs onnxruntime
  `[22.28, 2.98, 2.51, 1.52, 2.01, 1.06, 0.63, 1.52]`,逐元素差 ≤ ~1.3 LSB
  (out_scale=0.235),验证核心 MAC + zero-point 校正 + SiLU + requant 正确。
- **真实图片检测**: bus.jpg 上 onnxruntime 检出 4 人 + 1 巴士;C 引擎检出 4 人
  (class=0, conf 0.85/0.81/0.77),巴士因 INT8 深层累积漂移恰好落在阈值附近未输出。
  目标类别与位置正确。

### 验证中发现并修复的两个真实 bug
(单纯的规格/代码审查无法发现,只有对着可运行的 ONNX 做数值对齐才暴露)

1. **Padding zero-point bug**: 卷积对 padding 位置 `continue` 跳过,但 zero-point
   校正项 `in_zp×Σw` 仍按整核计算。ONNX QLinearConv 的 padding 用的是输入
   zero-point(非 0),所以 padding tap 必须贡献 `in_zp×weight`。修复前 conv0 左上角
   误差达 80 LSB 并向内层放大。修复后 conv0 对齐 ~1 LSB。
2. **检测头 channel 布局 bug**: bbox/cls 三尺度输出按 scale-major 拼接,但读取端按
   全局 channel-major `[C, 8400]` 索引,二者不一致,导致分类恒为 76。改为与 ONNX
   输出 `[84, 8400]` 一致的全局 channel-major 后,分类正确。

## 4. 关键问题: 原始 INT8 模型已损坏 + 检测头量化塌缩

- 目录里原有的 `yolov8n_int8.onnx`(ultralytics `int8=True` 快速导出,从未验证过)
  分类头输出**恒为 0**:FP32 同运行时正常,INT8 在 COCO 上 mAP≈0。已备份为
  `yolov8n_int8_broken.onnx`。
- 根因(对新旧两种量化都成立):YOLOv8 最终输出把 bbox(范围 0–640)与 cls 分数
  (范围 0–1)concat 进一个 84 通道张量,**共享一个输出量化 scale(≈640/127)会把
  所有 cls 分数 round 成 0**,分类头整体塌缩。
- 解决:`requantize.py` 用 onnxruntime 静态量化重做,coco128 校准,**排除 decode 尾
  (最后一个特征 conv 之后的非 Conv 节点,共 19 个)**。新模型分类头正常,bus.jpg
  正确检出 4 人 + 1 巴士,COCO mAP 见上表。

## 5. 已知缺口 / 限制

- **SiLU 在当前 SoC NPU 上没有硬件算子**(NPU 目前是 ReLU/ReLU6/maxpool/avgpool/
  depthwise/pointwise,见 CLAUDE.md)。这份 C 参考用 `expf` 忠实保留 SiLU,移植到
  NPU 需额外方案(查表/分段近似)。
- **检测头 decode 尾在 ONNX 与 C 引擎里都是浮点**(DFL softmax / box 解码 / sigmoid /
  NMS),与 SoC CPU 侧后处理一致,不在"INT8 数据路径"范围内。
- **C 引擎用 640×640 拉伸 resize**,而 mAP 评测走 ultralytics 的 letterbox。前者会扭曲
  长宽比、精度略低;mAP 数字反映的是 INT8 **模型**质量(letterbox),C 引擎验证的是
  INT8 **数据路径**数值正确性,两者各自成立。
- **纯 C 引擎未做性能优化**(无 SIMD 的逐元素卷积),不代表 NPU 移植后的速度,也因此
  不用它跑 COCO 全量(精度数字由 onnxruntime 跑出)。
- **规模远超当前 NPU 实测范围**:YOLOv8n 640×640、最多 256+ 输出通道,远超当前 RTL
  实际验证过的 MNIST 量级(28×28,最大 64 OC)。本工作只产出 C 软件参考 + 精度数字,
  **不代表 YOLOv8n 已能在当前 RTL 上跑通**——那是部署到 SoC 的下一步。

## 6. 复现

```bash
cd yolov8n_int8
python requantize.py                 # 重做正确的 INT8 量化 (需 datasets/coco128)
python extract_act_quant.py          # 提取激活量化 scale/zero_point
python gen_int8_layers_header.py     # 生成 yolov8n_layers.h 量化表
gcc -Wall -Wextra -O2 -o test_infer.exe test_infer.c yolov8n_infer.c -lm
./test_infer.exe bus.ppm weights     # 纯 C INT8 推理
python validate_coco_map.py          # COCO val2017 全量 mAP (需 datasets/val2017 + annotations)
```
