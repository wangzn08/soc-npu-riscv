# 非池化待解决问题清单

## B. DMA 完成中断 / IO 异步

### 问题
- 当前 NPU done IRQ 已有独立路径，但 DMA 完成仍主要依赖 `NPU_DMA_STATUS` 轮询。
- `dma_ddr_to_sram`、`dma_sram_to_ddr` 等搬运流程如果一直轮询，会让 CPU 在 DMA 等待期间不能做其它准备工作。
- 后续做 ping-pong overlap 时，DMA 完成必须能用中断或明确的异步完成机制通知 CPU，否则 CPU 仍会卡在等待 DMA 上。

### 需要检查 / 修改
- 检查 `rtl/axi_dma.v` 是否有 DMA done / error 信号。
- 检查 `rtl/axi_sys.v` 是否把 DMA done / error 接到 CPU IRQ 线。
- 如果没有 DMA IRQ，补充 DMA completion IRQ，并在 `firmware/irq.c` 中增加对应 bit 的处理。
- 将 firmware 中 DMA 等待逻辑从纯轮询改为“启动 DMA -> 等 DMA IRQ flag”，或明确保留轮询并文档化原因。
- DMA read done、DMA write done、DMA error 最好区分清楚，避免只知道“有中断”但不知道是哪一路完成。

### 验证
- 仿真中执行 DDR -> SRAM 和 SRAM -> DDR 两类 DMA。
- 确认 DMA 完成后 CPU 能通过 IRQ flag 继续运行。
- 确认 DMA error 不会被误判为正常完成。

### 风险
- 中低。功能独立，但会影响后续 ping-pong overlap。

## C. Ping-pong 真正同步 overlap

### 问题
- 当前 ping-pong 更像是 bank 选择，不是真正的计算和搬运重叠。
- 现在流程基本仍是：DMA 搬完 -> NPU 算 -> DMA 搬回/搬下一层，CPU 等待较多。
- 如果 Out SRAM 的 DMA 读 bank 和 NPU 写 bank 使用同一个选择信号，会导致 DMA 和 NPU 不能安全地同时访问不同 bank。

### 需要检查 / 修改
- 检查 `rtl/npu_top.v` 中 Out SRAM 的 DMA bank 选择和 NPU 写 bank 选择是否被同一个 `cfg_ping_pong_sel` 绑定。
- 如果目前 DMA 读和 NPU 写总是同 bank，需要拆成两个选择：
  - NPU 写当前层输出 bank。
  - DMA 读上一层或已完成 bank。
- firmware 侧需要有 DMA ping/pong 选择寄存器，例如已有或计划中的 `0x14C`。
- 每层循环需要重构为：
  - NPU 计算第 N 层时，DMA/CPU 准备第 N+1 层输入或搬走第 N-1 层输出。
  - NPU done IRQ 和 DMA done IRQ 都到达后，再切换 bank。
- 必须避免 NPU 写 bank 和 DMA 读/写 bank 冲突。

### 依赖
- 依赖 B：DMA 完成中断或等价异步完成机制。
- 如果后续启用 NPU 池化，也依赖池化输出写回地址和长度正确。

### 验证
- 增加 bank trace，打印或波形观察每一层：
  - NPU 当前写哪个 bank。
  - DMA 当前读/写哪个 bank。
  - bank 切换是否发生在 done 之后。
- 对比 10 张 MNIST 结果，确认 overlap 后精度不变。

### 风险
- 高。容易引入 bank 冲突、旧数据覆盖、层间数据错位。

## D. 多 OC tile 硬件完整性

### 问题
- 当前 bias / scale / shift 寄存器路径更像只覆盖 16 路输出通道。
- 当 `NPU_OC = 32 / 64` 时，需要确认每个 OC tile 是否能拿到正确的 bias / scale / shift。
- 如果所有 OC tile 共用同一组 16 路参数，而 firmware 没有在每个 tile 前及时重写参数，就会导致后续通道量化错误。

### 需要检查 / 修改
- 检查 `rtl/param_regfile.v` 中 bias / scale / shift 的寄存器数量和索引方式。
- 检查 `rtl/npu_top.v` / `rtl/top_controller_fsm.v` 是否根据 `oc_tile` 选择对应参数。
- 二选一明确架构：
  - 支持一次 NPU start 覆盖 OC=32/64，并为每个 OC tile 准备独立 bias / scale / shift。
  - 或文档化并强制“一次 start 只处理 16 个 OC”，由 firmware 分多次启动并更新参数。

### 验证
- 构造 OC=32 或 OC=64 的小测试。
- 每个 OC tile 使用不同 bias / scale / shift。
- 检查第 0、16、32、48 通道的输出是否分别匹配 golden。

### 风险
- 中。当前模型若靠 firmware 分 tile 已能通过，可以先文档化；若目标是硬件完整支持多 OC tile，需要修改。

## E. 层间驻留和 CPU 卸载

### 问题
- 当前部署中，很多层间数据会走 Out SRAM -> DDR -> Act SRAM，增加大量往返搬运。
- padding、flatten / reorder、affine 等操作仍在 CPU 上完成。
- 如果目标是提高端到端性能，需要决定哪些操作继续由 CPU 做，哪些搬到硬件或 NPU 数据通路中。

### 待决策项
- #13：卷积/池化输出是否保留在 SRAM，作为下一层输入，减少 DDR 往返。
- #14：padding 是否继续 CPU 预处理，还是由 im2col / NPU 输入侧支持边界补零。
- #15：flatten / reorder 是否继续 CPU 完成，还是在 DMA 或硬件地址映射中完成。
- #16：affine / 全连接是否继续 CPU 完成，还是映射到 NPU 或增加专用硬件路径。

### 需要检查 / 修改
- 明确每层输出的位置：
  - Out SRAM。
  - Act SRAM。
  - DDR activation buffer。
- 明确每层输入是否必须回 DDR。
- 如果做 SRAM 驻留，需要补充层间地址规划，避免覆盖还未消费的数据。
- 如果保留 CPU padding / flatten / affine，需要在文档中明确这是架构选择，不要误认为 NPU 已完成全流程。

### 验证
- 统计每张图片 DDR 搬运次数和搬运字节数。
- 对比优化前后总 cycle。
- 每减少一次 DDR 往返，都要确认 10 张 MNIST 结果不变。

### 风险
- 中高。属于架构改动，容易和 ping-pong、DMA 长度、地址布局互相影响。

## F. P2 清理 / 微优化

### #18 wgt_reader 跨空间位置缓存权重 tile

#### 问题
- 权重 tile 在相邻空间位置之间可能被重复读取。
- 如果每个输出点都重新走完整权重读取，会浪费带宽和 cycle。

#### 需要检查 / 修改
- 检查 `wgt_reader` 是否能在同一 OC/IC/kernel 配置下复用已经加载的权重 tile。
- 如果不能复用，评估增加权重 tile cache 或延长当前 Wgt SRAM 驻留策略。

### #19 conv1 lane 利用率

#### 问题
- conv1 输入通道数很小，16 lane 阵列利用率天然偏低。
- 这可能是结构性低利用率，不一定需要硬改。

#### 需要处理
- 文档化 conv1 的 lane 利用率和原因。
- 如果要优化，评估是否值得增加特殊输入打包逻辑。

### #20 debug 打印和误导性 probe

#### 问题
- `firmware/deepnet_deploy.c` 中存在大量 layer dump / debug 打印。
- Wgt SRAM probe，例如 `WgtBuf lane0`，在权重常驻后可能看到的是后续层残留权重，容易误导 debug。

#### 需要处理
- 精简默认 debug 打印。
- 将大量 dump 放到宏开关后面。
- 修改或删除容易误导的 Wgt SRAM probe。
- 如果保留 probe，必须打印当前 layer / oc_tile / ic_group / weight base。

### #21 寄存器 / 宏一致性

#### 问题
- RTL 寄存器定义、firmware 宏、文档说明可能存在不同步风险。
- 一旦寄存器 bit 或地址改动，firmware 可能仍使用旧宏。

#### 需要处理
- 对齐 `rtl/param_regfile.v`、`firmware/firmware.h`、`CLAUDE.md` 或其它寄存器文档。
- 确认 `CTRL`、`STATUS`、DMA 相关寄存器、ping-pong 选择寄存器地址一致。
- 最好新增一个单一寄存器表，后续 RTL 和 firmware 都按这个表维护。

### 风险
- 低到中。大多是清理和微优化，但能减少后续 debug 干扰。

## 建议处理顺序

1. 先做 B：DMA 完成中断。
2. 再做 C：ping-pong overlap。
3. 决定 D：多 OC tile 是硬件支持还是 firmware 分次启动。
4. 决定 E：哪些层间操作继续 CPU，哪些进入硬件。
5. 最后做 F：debug、文档、寄存器一致性清理。
