# NPU 片上 SRAM 容量分析 (general-purpose, structural)

通用 NPU 的片上 SRAM 该多大——结合计算单元 + AXI 性能的结构分析，既不浪费面积又保最大性能。
**不按 YOLO 等具体模型设计**（NPU 保持通用，尺寸靠寄存器/tiling 配置）。

## 1. 架构参数

| 项 | 值 @200MHz |
|---|---|
| 阵列 | 16 行 × 16 列 PE，每 PE 16 INT8 lane = **4096 MAC/cyc = 1.64 TOPS** |
| 阵列操作数需求 | 16 pos×16 IC (act) + 16 OC×16 IC (wgt) = **512 B/cyc** |
| 复用缓冲 (im2col line buffer / wgt_buf, 2048-bit) | 256+256 B/cyc — **真正喂阵列的高带宽层** |
| 片上 SRAM (128-bit, DATA_W=128) | **16 B/cyc** — 与 AXI 等宽 |
| AXI / DDR (128-bit) | 16 B/cyc = 3.2 GB/s |
| 当前容量 | Act 256KB / Wgt 256KB / Out 128KB (全 ping-pong) = 640KB |

## 2. 两个决定性结构事实

**(A) SRAM 不放大带宽。** SRAM 端口 128-bit = 16 B/cyc，和 AXI 一样。把 16 B/cyc 放大到阵列要的
512 B/cyc 的是两个 **2048-bit 复用缓冲**（line buffer + wgt_buf），不是 SRAM 容量。
→ **加大 SRAM 容量不会提升喂阵列速度。** SRAM 容量只解决：①权重 load-once 复用 ②层间激活驻留
③双缓冲隐藏 DDR 延迟。

**(B) Roofline 平衡点 = 4096/16 = 256 MAC/字节。** 要 compute-bound，每个 DDR 字节须复用 ≥256 次。
这是**工作负载属性**：卷积天然 AI 几百~几千（权重在 H×W 复用、激活在 OC 复用）→ 轻松 compute-bound；
GEMM/1×1 小空间最坏，方阵 tile 边长 T 时 AI≈T/2，要 T≥512 才平衡；低复用工作负载（单 token FC）无论
SRAM 多大都 DDR-bound。→ **SRAM 按"最坏 GEMM/1×1 的 load-once tile + 双缓冲"定，不按峰值带宽定。**

## 3. 三个容量驱动因素（floor 依据）

- **(A) 权重 load-once** → Wgt SRAM：双缓冲"当前 OC-chunk 权重" = MAX_OC_RESIDENT × IC × K×K × 2。
  单 array tile (16 OC×256 IC×9)=36KB；64-OC chunk×双缓冲≈256–294KB。
- **(B) 层间激活驻留** → Act SRAM：双缓冲输入条带 (row_par 行 + K-1 halo) × 宽 × IC 组。宽层靠列分块。
- **(C) output-stationary psum** → Out SRAM：输出条带 INT8 + 深 K 归约的 INT32 部分和，留片上至 K 扫完。

## 4. 性能最优配置（不浪费面积）

| Buffer | 当前 | 性能最优 | 依据 |
|---|---|---|---|
| Act | 256KB | 256KB | 双缓冲输入条带 + 层间驻留；再大边际收益低 |
| Wgt | 256KB | 256KB | 双缓冲 OC-chunk（最坏 147KB×2） |
| Out/Psum | 128KB | 128KB（可拆专用 INT32 psum 区） | output-stationary 深 K 归约 |

**当前 256/256/128 本就接近结构最优**——恰好覆盖最坏 GEMM/1×1 双缓冲 tile + 大 OC-chunk 权重双缓冲 +
充裕 psum。再加是浪费面积（DDR 流量非瓶颈、复用是工作负载属性）。

## 5. 面积优化配置（SRAM 能省则省）

**⚠️ 实测约束(2026-06-24)**：直接把 Act/Wgt 减半(64KB/半,4096字)+ Out 减到 32KB(1024字/半)、跑回归，
**连 MNIST 都立刻崩**(Digit0 pred 错 + TIMEOUT,地址溢出)。原因:**当前两套固件的 ping/pong 逻辑地址已用到
接近满**——MNIST residency R1=word1024 + FC pong;YOLO conv0 输入条带 ~5760 字、conv26(512→256 1×1)权重
8192 字都接近/超过单半区 4096 字。**结论:缩 SRAM 不是免费的**——仿真里 SRAM 大小不改周期,真正约束是
**逻辑地址 fit**;要真缩必须先**重新 tiling**(更小条带 / 分块权重 / 缩 residency 区)让地址落进小 buffer。
下面的"floor"是重新 tiling 后的理论下限,不能直接改 DEPTH 套用。

前提：SRAM 非带宽放大器 + DDR 非瓶颈 → 重新 tiling 后缩小只多几次 DMA 重载（CPU 编排/延迟），**不掉吞吐**。
三块都能朝"双缓冲一个工作 tile"的下限收（需配合 tiling 改动）：

| Buffer | 当前 | **面积优化** | 省 | floor / 旋钮 |
|---|---|---|---|---|
| Act | 256KB | **128KB** | 2× | 双缓冲输入列块条带；旋钮=列分块粒度。保守(兼顾层间驻留) |
| Wgt | 256KB | **128KB** | 2× | 双缓冲 OC-chunk；旋钮=`MAX_OC_RESIDENT` 64→16~32 |
| **Out/Psum** | 128KB | **32KB** | **4×** | 双缓冲输出条带 + INT32 psum；旋钮=输出条带粒度 |
| 合计 | **640KB** | **~288KB** | **−55%** | — |

**再小就崩的边界**：
- Out/Psum <8KB：每 start 算的活太少，启动开销占比上升。
- Wgt <双缓冲单 array tile (72KB)：weight-stationary 复用断裂。
- Act 太小：层间驻留余量没了，宽/大层被迫更多列块 + DMA。

## 6. 唯一不能省的 + 真正的性能杠杆

**复用缓冲（im2col line buffer + wgt_buf, 2048-bit）不要省，甚至该加**——它们是喂阵列的高带宽层，
面积小价值高（`ICG_MAX`/`ICG_BUF` 4→8 实测省 ~100M 周期）。省面积省在大块 SRAM 容量，不省在这。

容量够之后，把当前 ~10% 阵列占用推高的真正杠杆（按收益）：
1. **SRAM 读端口分 bank 加宽**（16×128-bit = 2048-bit）：128-bit SRAM=16B/cyc，1×1/低-OC 层里复用缓冲
   耗尽时阵列等 SRAM 刷新（需 256 B/cyc）。分 bank 让复用缓冲一拍填满 → 提升 1×1/GEMM 利用率，比加容量值钱。
2. **复用缓冲深度** `ICG_MAX`/`ICG_BUF`（已 4→8，通用可参数化到 16）。
3. **专用 INT32 psum SRAM + HW 跨块累加**（ACC_FIRST/ADD/FINAL 通路已搭好）。

## 一句话

容量上：**性能最优 = 当前 640KB**。面积优化目标 ~288KB(Out 32KB + Act/Wgt 各 128KB)理论上对吞吐几乎无损,
**但实测直接减半会地址溢出崩(连 MNIST)——必须先重新 tiling 让逻辑地址 fit,缩 SRAM 才安全**,不能只改 DEPTH。
要榨性能不靠堆 SRAM 面积,靠 **SRAM 分 bank 加宽喂阵列 + 复用缓冲深度 + 片上 psum 累加**。
