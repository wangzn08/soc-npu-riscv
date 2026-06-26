# Hardware Descriptor Queue V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic hardware NPU descriptor queue and use it to run one MNIST image inference per queue start while preserving the existing direct MMIO fallback.

**Architecture:** Add descriptor-control MMIO registers, a read-only AXI descriptor engine, and a descriptor-mode config mux inside `npu_top`. V1 executes a linear 16-word command stream, first proving NOP/STOP, DMA/copy/expand, qparam load, Conv/GEMM, then MNIST descriptor-mode inference. The descriptor format reserves YOLO fields/opcodes but v1 only enables the MNIST-required op subset.

**Tech Stack:** Verilog RTL, bare-metal C firmware, existing ModelSim/Questa flow via `bash run_all.sh`, directed testbenches under `tests/`, full SoC simulation with `rtl/axi_sys_tb.v`.

## Global Constraints

- Do not hardcode MNIST layer numbers, MNIST addresses, or YOLO layer numbers in RTL.
- Keep the existing direct MMIO path and default MNIST deployment working.
- Descriptor format is fixed at 16 x 32-bit words, four 128-bit AXI read beats per descriptor.
- Descriptor v1 is linear: no branches, loops, dynamic graph import, or automatic memory planning.
- Descriptor opcodes and fields must reserve YOLO phase-2 needs: `flags`, `scratch0`, `scratch1`, `wgt_words_per_oc`, `strip_out_rows`, `pad_value`, and reserved YOLO opcodes.
- Register-map changes must update both `rtl/param_regfile.v` and `firmware/firmware.h`.
- Descriptor registers live at offsets `0x400..0x418`; widen `REG_ADDR_W`/`ADDR_W` from 10 to 12 bits along the NPU register path so the existing 4KB NPU MMIO window (`0x3000_0000..0x3000_0FFF`) is fully reachable.
- New RTL files must be added to `axi_sys.f`.
- Firmware C must remain warning-clean under existing strict CFLAGS.
- Verification uses ModelSim/Questa through `bash run_all.sh`, not VCS.

---

## File Structure

- Create `rtl/descriptor_engine.v`: hardware queue FSM, descriptor AXI read master, qparam loader, op sequencer, descriptor-mode config outputs.
- Modify `rtl/npu_top.v`: instantiate descriptor engine, mux descriptor AXI reads with existing DMA AXI master, mux descriptor config/start signals with CPU MMIO config signals, route descriptor status/IRQ.
- Modify `rtl/param_regfile.v`: add CPU-visible descriptor registers and status inputs.
- Modify `firmware/firmware.h`: add descriptor MMIO definitions, opcodes, status bits, and C descriptor word indexes.
- Modify `firmware/npu_desc.h`: add hardware descriptor constants and packed software representation.
- Modify `firmware/deepnet_deploy.c`: add `NPU_HW_DESC=1` path that emits one descriptor list per image and submits the queue.
- Modify `axi_sys.f`: add `rtl/descriptor_engine.v`.
- Create `tests/tb_descriptor_engine.v`: standalone descriptor-engine directed tests with a simple 128-bit read-memory model.
- Modify or add `tests/tb_npu_desc_queue.v`: integration test for `npu_top` descriptor NOP/STOP, qparam, and tiny GEMM/CONV paths.
- Optionally modify `run_all.sh` only if directed test invocation needs a helper target; otherwise use direct `vlog/vsim` commands inside the plan steps.

---

### Task 1: Add Descriptor Register Map Constants

**Files:**
- Modify: `firmware/firmware.h`
- Modify: `firmware/npu_desc.h`

**Interfaces:**
- Produces:
  - Descriptor MMIO constants: `NPU_DESC_BASE_LO`, `NPU_DESC_COUNT`, `NPU_DESC_CTRL`, `NPU_DESC_STATUS`, `NPU_DESC_PC`, `NPU_DESC_ERR`
  - Opcode constants: `NPU_HW_DESC_OP_*`
  - Software layout type: `typedef struct npu_hw_desc_t`
- Consumes: existing `NPU_BASE`, existing `NPU_CTRL_*` flags.

- [ ] **Step 1: Add descriptor MMIO constants to `firmware/firmware.h`**

Add these definitions after the current extended NPU register definitions, using offsets that do not overlap current `0x3A0..0x3FC` registers:

```c
// Hardware descriptor queue registers (RTL descriptor_engine.v)
#define NPU_DESC_BASE_LO   (NPU_BASE + 0x400)
#define NPU_DESC_BASE_HI   (NPU_BASE + 0x404)  // reserved, write 0 in v1
#define NPU_DESC_COUNT     (NPU_BASE + 0x408)
#define NPU_DESC_CTRL      (NPU_BASE + 0x40C)
#define NPU_DESC_STATUS    (NPU_BASE + 0x410)
#define NPU_DESC_PC        (NPU_BASE + 0x414)
#define NPU_DESC_ERR       (NPU_BASE + 0x418)

#define NPU_DESC_CTRL_START      (1u << 0)
#define NPU_DESC_CTRL_ABORT      (1u << 1)
#define NPU_DESC_CTRL_IRQ_EN     (1u << 2)
#define NPU_DESC_CTRL_CLEAR_DONE (1u << 3)

#define NPU_DESC_STATUS_BUSY    (1u << 0)
#define NPU_DESC_STATUS_DONE    (1u << 1)
#define NPU_DESC_STATUS_ERR     (1u << 2)
#define NPU_DESC_STATUS_ABORTED (1u << 3)
```

- [ ] **Step 2: Add hardware descriptor constants to `firmware/npu_desc.h`**

Append the hardware descriptor layout below the existing firmware runtime API:

```c
#define NPU_HW_DESC_VERSION 1u
#define NPU_HW_DESC_WORDS   16u

#define NPU_HW_DESC_OP_NOP                 0x00u
#define NPU_HW_DESC_OP_DMA_DDR_TO_ACT      0x01u
#define NPU_HW_DESC_OP_DMA_ACT_TO_DDR      0x02u
#define NPU_HW_DESC_OP_DMA_OUT_TO_DDR      0x03u
#define NPU_HW_DESC_OP_IMG_EXPAND          0x04u
#define NPU_HW_DESC_OP_SRAM_COPY_OUT_TO_ACT 0x05u
#define NPU_HW_DESC_OP_CONV2D              0x06u
#define NPU_HW_DESC_OP_GEMM                0x07u
#define NPU_HW_DESC_OP_STOP_IRQ            0x08u

#define NPU_HW_DESC_OP_UPSAMPLE2X          0x20u
#define NPU_HW_DESC_OP_MAXPOOL5X5          0x21u
#define NPU_HW_DESC_OP_ELTWISE_ADD         0x22u
#define NPU_HW_DESC_OP_DFL                 0x23u
#define NPU_HW_DESC_OP_LUT_LOAD            0x24u
#define NPU_HW_DESC_OP_ACTIVATION_CFG      0x25u

#define NPU_HW_DESC_ERR_NONE               0u
#define NPU_HW_DESC_ERR_BAD_VERSION        1u
#define NPU_HW_DESC_ERR_BAD_OPCODE         2u
#define NPU_HW_DESC_ERR_UNSUPPORTED_OP     3u
#define NPU_HW_DESC_ERR_BAD_COUNT          4u
#define NPU_HW_DESC_ERR_BAD_ALIGNMENT      5u
#define NPU_HW_DESC_ERR_BAD_SHAPE          6u
#define NPU_HW_DESC_ERR_BUSY_AT_START      7u
#define NPU_HW_DESC_ERR_AXI_DESC_READ      8u
#define NPU_HW_DESC_ERR_AXI_QPARAM_READ    9u
#define NPU_HW_DESC_ERR_ENGINE_TIMEOUT     10u

typedef struct {
    uint32_t w[NPU_HW_DESC_WORDS];
} npu_hw_desc_t;

static inline void npu_hw_desc_clear(npu_hw_desc_t *d)
{
    uint32_t i;
    for (i = 0u; i < NPU_HW_DESC_WORDS; i++)
        d->w[i] = 0u;
}

static inline void npu_hw_desc_set_op(npu_hw_desc_t *d, uint32_t op, uint32_t flags)
{
    d->w[0] = (op & 0xFFu) | ((NPU_HW_DESC_VERSION & 0xFFu) << 8) |
              ((flags & 0xFFFFu) << 16);
    d->w[1] = flags >> 16;
}
```

- [ ] **Step 3: Compile default firmware**

Run:

```bash
bash run_all.sh fw
```

Expected:

```text
[OK] 固件编译完成
```

- [ ] **Step 4: Commit**

```bash
git add firmware/firmware.h firmware/npu_desc.h
git commit -m "Define hardware descriptor queue ABI"
```

---

### Task 2: Add Descriptor MMIO Registers In `param_regfile`

**Files:**
- Modify: `rtl/param_regfile.v`
- Test: create or extend a minimal register test if convenient; otherwise compile RTL.

**Interfaces:**
- Consumes: offsets from Task 1.
- Produces outputs to `npu_top`:
  - `o_desc_base_lo[31:0]`
  - `o_desc_base_hi[31:0]`
  - `o_desc_count[15:0]`
  - `o_desc_start`
  - `o_desc_abort`
  - `o_desc_irq_en`
  - `o_desc_clear_done`
- Consumes descriptor status inputs:
  - `i_desc_busy`
  - `i_desc_done`
  - `i_desc_err`
  - `i_desc_aborted`
  - `i_desc_pc[15:0]`
  - `i_desc_err_code[7:0]`

- [ ] **Step 1: Add ports to `param_regfile.v`**

Add descriptor outputs near the DMA/register outputs:

```verilog
    output wire [31:0]                  o_desc_base_lo,
    output wire [31:0]                  o_desc_base_hi,
    output wire [15:0]                  o_desc_count,
    output wire                         o_desc_start,
    output wire                         o_desc_abort,
    output wire                         o_desc_irq_en,
    output wire                         o_desc_clear_done,
    input  wire                         i_desc_busy,
    input  wire                         i_desc_done,
    input  wire                         i_desc_err,
    input  wire                         i_desc_aborted,
    input  wire [15:0]                  i_desc_pc,
    input  wire [7:0]                   i_desc_err_code,
```

- [ ] **Step 2: Add descriptor registers**

Add internal regs:

```verilog
    reg [31:0] desc_base_lo;
    reg [31:0] desc_base_hi;
    reg [15:0] desc_count;
    reg        desc_irq_en;
    reg        desc_start_pulse;
    reg        desc_abort_pulse;
    reg        desc_clear_done_pulse;
```

Clear pulse regs to zero every clock, as existing trigger pulses do:

```verilog
        desc_start_pulse      <= 1'b0;
        desc_abort_pulse      <= 1'b0;
        desc_clear_done_pulse <= 1'b0;
```

- [ ] **Step 3: Decode descriptor writes**

In the AXI write case statement, add:

```verilog
                    10'h400: desc_base_lo <= s_axi_wdata;
                    10'h404: desc_base_hi <= s_axi_wdata;
                    10'h408: desc_count   <= s_axi_wdata[15:0];
                    10'h40C: begin
                        desc_start_pulse      <= s_axi_wdata[0];
                        desc_abort_pulse      <= s_axi_wdata[1];
                        desc_irq_en           <= s_axi_wdata[2];
                        desc_clear_done_pulse <= s_axi_wdata[3];
                    end
```

- [ ] **Step 4: Decode descriptor reads**

In the read mux, add:

```verilog
            10'h400: rdata_next = desc_base_lo;
            10'h404: rdata_next = desc_base_hi;
            10'h408: rdata_next = {16'b0, desc_count};
            10'h40C: rdata_next = {29'b0, desc_irq_en, 2'b0};
            10'h410: rdata_next = {28'b0, i_desc_aborted, i_desc_err, i_desc_done, i_desc_busy};
            10'h414: rdata_next = {16'b0, i_desc_pc};
            10'h418: rdata_next = {24'b0, i_desc_err_code};
```

- [ ] **Step 5: Assign outputs**

Add continuous assignments:

```verilog
    assign o_desc_base_lo    = desc_base_lo;
    assign o_desc_base_hi    = desc_base_hi;
    assign o_desc_count      = desc_count;
    assign o_desc_start      = desc_start_pulse;
    assign o_desc_abort      = desc_abort_pulse;
    assign o_desc_irq_en     = desc_irq_en;
    assign o_desc_clear_done = desc_clear_done_pulse;
```

- [ ] **Step 6: Compile RTL**

Run:

```bash
bash run_all.sh compile
```

Expected:

```text
Errors: 0
```

- [ ] **Step 7: Commit**

```bash
git add rtl/param_regfile.v
git commit -m "Add descriptor queue control registers"
```

---

### Task 3: Create Descriptor Engine Skeleton With NOP/STOP

**Files:**
- Create: `rtl/descriptor_engine.v`
- Create: `tests/tb_descriptor_engine.v`
- Modify: `axi_sys.f`

**Interfaces:**
- Consumes descriptor register signals from Task 2.
- Produces:
  - AXI read-only descriptor master signals
  - status outputs `o_busy`, `o_done`, `o_err`, `o_aborted`, `o_pc`, `o_err_code`
  - no NPU/DMA config outputs yet except reserved zero outputs.

- [ ] **Step 1: Add `rtl/descriptor_engine.v` module skeleton**

Create the module with these ports:

```verilog
module descriptor_engine #(
    parameter ADDR_W = 32,
    parameter SRAM_ADDR_W = 14
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [31:0] i_desc_base_lo,
    input  wire [31:0] i_desc_base_hi,
    input  wire [15:0] i_desc_count,
    input  wire        i_desc_start,
    input  wire        i_desc_abort,
    input  wire        i_desc_irq_en,
    input  wire        i_desc_clear_done,
    input  wire        i_global_idle,

    output reg         o_busy,
    output reg         o_done,
    output reg         o_err,
    output reg         o_aborted,
    output reg [15:0]  o_pc,
    output reg [7:0]   o_err_code,
    output wire        o_irq,

    output reg         m_axi_arvalid,
    input  wire        m_axi_arready,
    output reg [ADDR_W-1:0] m_axi_araddr,
    output reg [7:0]   m_axi_arlen,
    output wire [2:0]  m_axi_arsize,
    output wire [1:0]  m_axi_arburst,
    input  wire        m_axi_rvalid,
    output reg         m_axi_rready,
    input  wire [127:0] m_axi_rdata,
    input  wire [1:0]  m_axi_rresp,
    input  wire        m_axi_rlast
);
```

Use constants:

```verilog
localparam OP_NOP      = 8'h00;
localparam OP_STOP_IRQ = 8'h08;
localparam VERSION     = 8'h01;
localparam ERR_NONE    = 8'd0;
localparam ERR_BAD_VERSION = 8'd1;
localparam ERR_BAD_OPCODE  = 8'd2;
localparam ERR_BAD_COUNT   = 8'd4;
localparam ERR_BUSY_AT_START = 8'd7;
localparam ERR_AXI_DESC_READ = 8'd8;
```

- [ ] **Step 2: Implement fixed descriptor fetch**

Fetch four 128-bit beats from:

```verilog
desc_addr = i_desc_base_lo + {o_pc, 6'b0};
```

Store unpacked words in:

```verilog
reg [31:0] desc_w [0:15];
```

Beat unpacking:

```verilog
desc_w[{beat_idx, 2'b00} + 0] <= m_axi_rdata[31:0];
desc_w[{beat_idx, 2'b00} + 1] <= m_axi_rdata[63:32];
desc_w[{beat_idx, 2'b00} + 2] <= m_axi_rdata[95:64];
desc_w[{beat_idx, 2'b00} + 3] <= m_axi_rdata[127:96];
```

- [ ] **Step 3: Implement NOP/STOP execution**

Decode:

```verilog
wire [7:0] op      = desc_w[0][7:0];
wire [7:0] version = desc_w[0][15:8];
```

Rules:

- count zero at start -> `ERR_BAD_COUNT`
- start while `i_global_idle==0` -> `ERR_BUSY_AT_START`
- bad version -> `ERR_BAD_VERSION`
- `OP_NOP` -> increment PC
- `OP_STOP_IRQ` -> done
- unknown op -> `ERR_BAD_OPCODE`

- [ ] **Step 4: Write standalone testbench for NOP + STOP**

In `tests/tb_descriptor_engine.v`, instantiate `descriptor_engine`, provide a simple AXI read memory:

```verilog
reg [127:0] mem [0:15];
always @(posedge clk) begin
    if (!rst_n) begin
        arready <= 1'b1;
        rvalid <= 1'b0;
        rlast  <= 1'b0;
    end else begin
        if (arvalid && arready) begin
            rd_base <= araddr[9:4];
            rd_cnt  <= 0;
            rvalid  <= 1'b1;
        end else if (rvalid && rready) begin
            rd_cnt <= rd_cnt + 1;
            if (rd_cnt == 3) begin
                rlast  <= 1'b1;
            end
            if (rlast) begin
                rvalid <= 1'b0;
                rlast  <= 1'b0;
            end
        end
    end
end
always @(*) rdata = mem[rd_base + rd_cnt];
```

Initialize two descriptors:

```verilog
// desc0 NOP
mem[0] = {32'h0, 32'h0, 32'h0, 32'h0000_0100};
mem[1] = 128'h0;
mem[2] = 128'h0;
mem[3] = 128'h0;
// desc1 STOP_IRQ
mem[4] = {32'h0, 32'h0, 32'h0, 32'h0000_0108};
mem[5] = 128'h0;
mem[6] = 128'h0;
mem[7] = 128'h0;
```

Expected checks:

```verilog
if (!done || err || pc != 16'd1) begin
    $display("FAIL nop_stop done=%0b err=%0b pc=%0d", done, err, pc);
    errors = errors + 1;
end
```

- [ ] **Step 5: Add file to `axi_sys.f`**

Add:

```text
rtl/descriptor_engine.v
```

- [ ] **Step 6: Run standalone test**

Run:

```bash
mkdir -p sim/desc && cd sim/desc
vlib work
vlog -sv -timescale 1ns/1ps ../../rtl/descriptor_engine.v ../../tests/tb_descriptor_engine.v
vsim -c tb_descriptor_engine -do "run -all; quit -f"
```

Expected:

```text
DESC_ENGINE TEST PASS
Errors: 0
```

- [ ] **Step 7: Commit**

```bash
git add rtl/descriptor_engine.v tests/tb_descriptor_engine.v axi_sys.f
git commit -m "Add descriptor engine NOP queue"
```

---

### Task 4: Integrate Descriptor Engine Into `npu_top` For NOP/STOP

**Files:**
- Modify: `rtl/npu_top.v`
- Modify: `rtl/param_regfile.v` connection list if needed
- Test: create `tests/tb_npu_desc_queue.v`

**Interfaces:**
- Consumes Task 2 descriptor register outputs and Task 3 engine.
- Produces descriptor done IRQ through `irq_done` when STOP completes.

- [ ] **Step 1: Wire descriptor register ports in `npu_top.v`**

Declare wires:

```verilog
wire [31:0] desc_base_lo, desc_base_hi;
wire [15:0] desc_count;
wire desc_start, desc_abort, desc_irq_en, desc_clear_done;
wire desc_busy, desc_done, desc_err, desc_aborted;
wire [15:0] desc_pc;
wire [7:0] desc_err_code;
wire desc_irq;
```

Connect them to `u_param_regfile`.

- [ ] **Step 2: Instantiate `descriptor_engine`**

Instantiate in `npu_top.v`:

```verilog
descriptor_engine #(
    .ADDR_W(32),
    .SRAM_ADDR_W(SRAM_ADDR_W)
) u_descriptor_engine (
    .clk(clk),
    .rst_n(rst_n),
    .i_desc_base_lo(desc_base_lo),
    .i_desc_base_hi(desc_base_hi),
    .i_desc_count(desc_count),
    .i_desc_start(desc_start),
    .i_desc_abort(desc_abort),
    .i_desc_irq_en(desc_irq_en),
    .i_desc_clear_done(desc_clear_done),
    .i_global_idle(~npu_busy_visible & ~dma_rd_busy & ~dma_wr_busy),
    .o_busy(desc_busy),
    .o_done(desc_done),
    .o_err(desc_err),
    .o_aborted(desc_aborted),
    .o_pc(desc_pc),
    .o_err_code(desc_err_code),
    .o_irq(desc_irq),
    ...
);
```

Use existing or newly declared DMA busy wires. If `axi_dma` does not expose busy,
derive conservative busy as pending valid/done state in `npu_top` or use the
descriptor engine only when no descriptor op launches DMA in this task.

- [ ] **Step 3: Add AXI read mux between descriptor engine and existing DMA**

For Task 4, descriptor engine only reads descriptors before any DMA operation.
Mux read address/data channels:

```verilog
wire desc_axi_sel = desc_busy && desc_rd_active;
assign m_axi_arvalid = desc_axi_sel ? desc_arvalid : dma_arvalid;
assign m_axi_araddr  = desc_axi_sel ? desc_araddr  : dma_araddr;
assign m_axi_arlen   = desc_axi_sel ? desc_arlen   : dma_arlen;
assign m_axi_arsize  = desc_axi_sel ? desc_arsize  : dma_arsize;
assign m_axi_arburst = desc_axi_sel ? desc_arburst : dma_arburst;
assign desc_arready  = desc_axi_sel ? m_axi_arready : 1'b0;
assign dma_arready_i = desc_axi_sel ? 1'b0 : m_axi_arready;
assign desc_rvalid   = desc_axi_sel ? m_axi_rvalid : 1'b0;
assign dma_rvalid_i  = desc_axi_sel ? 1'b0 : m_axi_rvalid;
assign m_axi_rready  = desc_axi_sel ? desc_rready : dma_rready;
```

If `npu_top` currently connects `axi_dma` directly to top-level AXI signals, first
rename those connections to internal `dma_*` wires, then drive top-level signals
from the mux.

- [ ] **Step 4: Route descriptor IRQ**

For v1 integration:

```verilog
assign irq_done = desc_irq | npu_done_irq;
```

Keep `status_done_irq` feeding old NPU done status as `npu_done_irq`. Descriptor
status is read from `NPU_DESC_STATUS`, not `NPU_STATUS`.

- [ ] **Step 5: Write `tests/tb_npu_desc_queue.v` NOP/STOP integration**

Copy the AXI-lite tasks from `tests/tb_npu_integ.v`. Add a simple DDR read model
that returns two descriptors at DDR address `0x40000000`.

Test sequence:

```verilog
reg_write(10'h400, 32'h4000_0000);
reg_write(10'h404, 32'h0);
reg_write(10'h408, 32'd2);
reg_write(10'h40C, 32'h5); // start + irq_en
wait_done(1000);
reg_read(10'h410, status);
reg_read(10'h414, pc);
if ((status & 32'h2) == 0 || pc != 32'd1) errors = errors + 1;
```

- [ ] **Step 6: Run integration test**

Run:

```bash
mkdir -p sim/desc_npu && cd sim/desc_npu
vlib work
vlog -sv -timescale 1ns/1ps -f ../../axi_sys.f ../../tests/tb_npu_desc_queue.v
vsim -c tb_npu_desc_queue -do "run -all; quit -f"
```

Expected:

```text
NPU DESC QUEUE TEST PASS
Errors: 0
```

- [ ] **Step 7: Commit**

```bash
git add rtl/npu_top.v rtl/param_regfile.v tests/tb_npu_desc_queue.v
git commit -m "Integrate descriptor queue status path"
```

---

### Task 5: Add Descriptor DMA, Copy, And Expand Ops

**Files:**
- Modify: `rtl/descriptor_engine.v`
- Modify: `rtl/npu_top.v`
- Test: extend `tests/tb_npu_desc_queue.v`

**Interfaces:**
- Produces descriptor-mode config outputs:
  - `o_desc_mode`
  - `o_dma_rd_trig`, `o_dma_wr_trig`
  - `o_dma_rd_ddr_addr`, `o_dma_wr_ddr_addr`
  - `o_dma_rd_len`, `o_dma_wr_len`
  - `o_dma_rd_sram_base`, `o_dma_wr_sram_base`
  - `o_dma_sram_sel`, `o_dma_path_ctl`, `o_dma_ping_sel`
  - `o_copy_trig`, `o_expand_trig`
- Consumes done inputs:
  - `i_dma_rd_done`, `i_dma_wr_done`, `i_copy_done`, `i_expand_done`

- [ ] **Step 1: Add descriptor engine op constants**

Add:

```verilog
localparam OP_DMA_DDR_TO_ACT       = 8'h01;
localparam OP_DMA_ACT_TO_DDR       = 8'h02;
localparam OP_DMA_OUT_TO_DDR       = 8'h03;
localparam OP_IMG_EXPAND           = 8'h04;
localparam OP_SRAM_COPY_OUT_TO_ACT = 8'h05;
```

- [ ] **Step 2: Add config output ports**

Add all outputs listed in the Interfaces block. Pulse trigger outputs for one
cycle in `START_OP`, then wait for the matching done.

- [ ] **Step 3: Implement op programming rules**

Use descriptor fields:

```text
src0 = desc_w[2]
dst  = desc_w[4]
words = desc_w[7]
```

Rules:

```verilog
OP_DMA_DDR_TO_ACT:
  dma_sram_sel     = 0
  dma_path_ctl     = 0x2
  dma_rd_ddr_addr  = src0
  dma_rd_len       = words - 1
  dma_rd_sram_base = dst[SRAM_ADDR_W-1:0]
  wait rd_done

OP_DMA_ACT_TO_DDR:
  dma_path_ctl     = 0x2
  dma_wr_ddr_addr  = dst
  dma_wr_len       = words - 1
  dma_wr_sram_base = src0[SRAM_ADDR_W-1:0]
  wait wr_done

OP_DMA_OUT_TO_DDR:
  dma_path_ctl     = 0x1
  dma_wr_ddr_addr  = dst
  dma_wr_len       = words - 1
  dma_wr_sram_base = src0[SRAM_ADDR_W-1:0]
  wait wr_done

OP_IMG_EXPAND:
  dma_ping_sel     = desc_w[1][2:0]
  dma_rd_sram_base = src0[SRAM_ADDR_W-1:0]
  dma_wr_sram_base = dst[SRAM_ADDR_W-1:0]
  dma_rd_len       = words
  wait expand_done

OP_SRAM_COPY_OUT_TO_ACT:
  dma_ping_sel     = desc_w[1][2:0]
  dma_rd_sram_base = src0[SRAM_ADDR_W-1:0]
  dma_wr_sram_base = dst[SRAM_ADDR_W-1:0]
  dma_rd_len       = words
  wait copy_done
```

- [ ] **Step 4: Mux descriptor DMA config in `npu_top.v`**

Create muxed config wires before `axi_dma`, `sram_copy`, and `img_expand`:

```verilog
wire [31:0] mux_dma_rd_ddr_addr = desc_busy ? desc_dma_rd_ddr_addr : cfg_dma_rd_ddr_addr;
wire        mux_dma_rd_trig     = desc_busy ? desc_dma_rd_trig     : cfg_dma_rd_trig;
```

Apply the same pattern for all affected DMA/copy/expand config signals.

- [ ] **Step 5: Extend integration test for a DMA_DDR_TO_ACT descriptor**

DDR model should return descriptor beats for descriptor fetch and return data
beats for DMA read. Use a descriptor:

```text
op=DMA_DDR_TO_ACT
src0=0x40000100
dst=4
words=2
```

After queue completes, check:

```verilog
if (dut.u_act_sram.u_bram.mem[4] != 128'h00112233445566778899aabbccddeeff) errors++;
if (dut.u_act_sram.u_bram.mem[5] != 128'hffeeddccbbaa99887766554433221100) errors++;
```

- [ ] **Step 6: Extend integration test for IMG_EXPAND and SRAM_COPY**

Use small word counts:

```text
IMG_EXPAND src0=0 dst=16 words=4
SRAM_COPY_OUT_TO_ACT src0=0 dst=32 words=4
```

Check done status and a representative destination word.

- [ ] **Step 7: Run tests**

Run:

```bash
cd sim/desc_npu
vlog -sv -timescale 1ns/1ps -f ../../axi_sys.f ../../tests/tb_npu_desc_queue.v
vsim -c tb_npu_desc_queue -do "run -all; quit -f"
```

Expected:

```text
NPU DESC QUEUE TEST PASS
Errors: 0
```

- [ ] **Step 8: Commit**

```bash
git add rtl/descriptor_engine.v rtl/npu_top.v tests/tb_npu_desc_queue.v
git commit -m "Execute descriptor data movement ops"
```

---

### Task 6: Add QParam Loader

**Files:**
- Modify: `rtl/descriptor_engine.v`
- Modify: `rtl/npu_top.v`
- Test: extend `tests/tb_descriptor_engine.v` or `tests/tb_npu_desc_queue.v`

**Interfaces:**
- Produces descriptor qparam arrays:
  - `o_qparam_we`
  - `o_qparam_idx[5:0]`
  - `o_qparam_bias[31:0]`
  - `o_qparam_scale[31:0]`
  - `o_qparam_shift[5:0]`
- Consumes `qparam_base = desc_w[11]`, `qparam_count = desc_w[12][5:0]`.

- [ ] **Step 1: Add qparam read states**

Add states:

```verilog
S_QPARAM_AR
S_QPARAM_R
```

For each channel, read one 128-bit beat from:

```verilog
qparam_addr = desc_w[11] + {qparam_idx, 4'b0};
```

Unpack:

```verilog
o_qparam_bias  <= m_axi_rdata[31:0];
o_qparam_scale <= m_axi_rdata[63:32];
o_qparam_shift <= m_axi_rdata[101:96];
```

Assert `o_qparam_we` for one cycle.

- [ ] **Step 2: Add qparam storage in `npu_top.v` descriptor path**

Add descriptor resident arrays:

```verilog
reg [63:0][31:0] desc_bias_val;
reg [63:0][31:0] desc_scale_mul;
reg [63:0][5:0]  desc_scale_shift;
```

On `desc_qparam_we`:

```verilog
desc_bias_val[desc_qparam_idx]   <= desc_qparam_bias;
desc_scale_mul[desc_qparam_idx]  <= desc_qparam_scale;
desc_scale_shift[desc_qparam_idx]<= desc_qparam_shift;
```

Mux existing NPU qparam inputs:

```verilog
assign mux_bias_val[ch]    = desc_busy ? desc_bias_val[ch]    : cfg_bias_val[ch];
assign mux_scale_mul[ch]   = desc_busy ? desc_scale_mul[ch]   : cfg_scale_mul[ch];
assign mux_scale_shift[ch] = desc_busy ? desc_scale_shift[ch] : cfg_scale_shift[ch];
```

- [ ] **Step 3: Add qparam loader test**

Create descriptor:

```text
op=NOP
qparam_base=0x40000200
qparam_count=3
```

Allow NOP to load qparams for test purposes, or use a CONV2D descriptor that
stops before compute in the standalone engine test. Feed qparam beats:

```text
ch0 bias=1 scale=11 shift=2
ch1 bias=2 scale=12 shift=3
ch2 bias=3 scale=13 shift=4
```

Check qparam write pulses and values in the testbench.

- [ ] **Step 4: Run standalone and integration tests**

Run:

```bash
cd sim/desc
vlog -sv -timescale 1ns/1ps ../../rtl/descriptor_engine.v ../../tests/tb_descriptor_engine.v
vsim -c tb_descriptor_engine -do "run -all; quit -f"
cd ../desc_npu
vlog -sv -timescale 1ns/1ps -f ../../axi_sys.f ../../tests/tb_npu_desc_queue.v
vsim -c tb_npu_desc_queue -do "run -all; quit -f"
```

Expected:

```text
DESC_ENGINE TEST PASS
NPU DESC QUEUE TEST PASS
```

- [ ] **Step 5: Commit**

```bash
git add rtl/descriptor_engine.v rtl/npu_top.v tests/tb_descriptor_engine.v tests/tb_npu_desc_queue.v
git commit -m "Load descriptor qparams from DDR"
```

---

### Task 7: Add CONV2D And GEMM Descriptor Starts

**Files:**
- Modify: `rtl/descriptor_engine.v`
- Modify: `rtl/npu_top.v`
- Test: extend `tests/tb_npu_desc_queue.v`

**Interfaces:**
- Descriptor engine outputs compute config:
  - shape: `in_w`, `in_h`, `in_c`, `out_c`, `kernel`, `stride`, `pad`
  - bases: `act_addr`, `wgt_addr`, `out_addr`
  - control bits: `ctrl_flags`, `start`
- Consumes `i_npu_done`.

- [ ] **Step 1: Add compute output ports**

Add:

```verilog
output reg desc_npu_start,
output reg [31:0] desc_ctrl_flags,
output reg [15:0] desc_in_w,
output reg [15:0] desc_in_h,
output reg [15:0] desc_in_c,
output reg [15:0] desc_out_c,
output reg [15:0] desc_kernel,
output reg [15:0] desc_stride,
output reg [15:0] desc_pad,
output reg [SRAM_ADDR_W-1:0] desc_act_addr,
output reg [SRAM_ADDR_W-1:0] desc_wgt_addr,
output reg [SRAM_ADDR_W-1:0] desc_out_addr,
input wire i_npu_done
```

- [ ] **Step 2: Decode CONV2D and GEMM**

For both ops:

```verilog
desc_in_w     <= desc_w[8][15:0];
desc_in_h     <= desc_w[8][31:16];
desc_in_c     <= desc_w[9][15:0];
desc_out_c    <= desc_w[9][31:16];
desc_kernel   <= {desc_w[10][7:0], desc_w[10][15:8]};
desc_stride   <= {desc_w[10][23:16], desc_w[10][23:16]};
desc_pad      <= {desc_w[10][31:24], desc_w[10][31:24]};
desc_act_addr <= desc_w[2][SRAM_ADDR_W-1:0];
desc_wgt_addr <= desc_w[3][SRAM_ADDR_W-1:0];
desc_out_addr <= desc_w[4][SRAM_ADDR_W-1:0];
desc_ctrl_flags <= {desc_w[1], desc_w[0][31:16]};
```

For `OP_GEMM`, force `NPU_CTRL_GEMM_EN` in `desc_ctrl_flags`.

- [ ] **Step 3: Mux compute config in `npu_top.v`**

Before `top_controller_fsm`, mux:

```verilog
wire mux_start = desc_busy ? desc_npu_start : cfg_start;
wire [15:0] mux_in_w = desc_busy ? desc_in_w : cfg_in_w;
```

Apply to all shape/base/control wires consumed by compute, post-process, and
special engines. Keep direct-mode names untouched where possible by introducing
`run_*` wires:

```verilog
wire run_gemm_en = desc_busy ? desc_ctrl_flags[7] : cfg_gemm_en;
wire run_row_par = desc_busy ? desc_ctrl_flags[9] : cfg_row_par_en;
```

- [ ] **Step 4: Add tiny GEMM descriptor integration test**

Use the existing `tb_npu_integ.v` GEMM pattern:

```text
IC=16, OC=16, act[ic]=1, wgt[oc][ic]=(ic<=oc)
qparams bias=0 scale=1 shift=0
```

Descriptor list:

```text
GEMM qparam_base=0x40001000 act=0 wgt=0 out=0 in_w=1 in_h=1 in_c=16 out_c=16 flags=RELU|GEMM
STOP_IRQ
```

Expected output:

```text
out[oc] = oc + 1
```

- [ ] **Step 5: Add tiny CONV2D descriptor integration test**

Use a 1x1 pointwise-style CONV2D descriptor:

```text
in_w=2 in_h=1 in_c=16 out_c=16 kh=1 kw=1 stride=1 pad=0 flags=RELU|OC_SINGLE|PW_EN
```

Preload Act/Wgt SRAM hierarchically and check two output words.

- [ ] **Step 6: Run integration test**

Run:

```bash
cd sim/desc_npu
vlog -sv -timescale 1ns/1ps -f ../../axi_sys.f ../../tests/tb_npu_desc_queue.v
vsim -c tb_npu_desc_queue -do "run -all; quit -f"
```

Expected:

```text
NPU DESC QUEUE TEST PASS
Errors: 0
```

- [ ] **Step 7: Commit**

```bash
git add rtl/descriptor_engine.v rtl/npu_top.v tests/tb_npu_desc_queue.v
git commit -m "Start NPU compute from descriptors"
```

---

### Task 8: Add MNIST Hardware Descriptor Firmware Path

**Files:**
- Modify: `firmware/deepnet_deploy.c`
- Modify: `firmware/npu_desc.h`

**Interfaces:**
- Consumes hardware descriptor ABI from Task 1 and RTL from Tasks 2-7.
- Produces:
  - `NPU_HW_DESC=1` build path
  - qparam tables in packed descriptor format
  - per-image descriptor list builder
  - queue submit/wait helper

- [ ] **Step 1: Add build switch**

Near the top of `deepnet_deploy.c`:

```c
#ifndef NPU_HW_DESC
#define NPU_HW_DESC 0
#endif
```

- [ ] **Step 2: Add packed qparam type**

Add:

```c
typedef struct {
    int32_t bias;
    uint32_t scale_mul;
    uint32_t scale_shift;
    uint32_t reserved;
} hw_qparam_t;
```

- [ ] **Step 3: Add descriptor submit helper**

Add:

```c
static int hw_desc_submit(uint32_t desc_ddr, uint32_t count)
{
    desc_irq_flag = 0;
    npu_wr(NPU_DESC_BASE_LO, desc_ddr);
    npu_wr(NPU_DESC_BASE_HI, 0);
    npu_wr(NPU_DESC_COUNT, count);
    npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_START | NPU_DESC_CTRL_IRQ_EN);
    int t = NPU_IRQ_TIMEOUT;
    while (t-- > 0) {
        if (desc_irq_flag)
            break;
    }
    if (t <= 0) {
        print_str("  DESC timeout pc=");
        print_dec(npu_rd(NPU_DESC_PC));
        print_str(" err=");
        print_dec(npu_rd(NPU_DESC_ERR));
        print_chr('\n');
        return 0;
    }
    if (npu_rd(NPU_DESC_STATUS) & NPU_DESC_STATUS_ERR) {
        print_str("  DESC err pc=");
        print_dec(npu_rd(NPU_DESC_PC));
        print_str(" err=");
        print_dec(npu_rd(NPU_DESC_ERR));
        print_chr('\n');
        return 0;
    }
    npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_CLEAR_DONE | NPU_DESC_CTRL_IRQ_EN);
    return 1;
}
```

This step also requires `irq.c` to set `desc_irq_flag` from the same NPU IRQ bit
or requires `hw_desc_submit()` to use `npu_irq_flag` if descriptor completion
shares bit 3. Pick one implementation and keep the direct NPU wait path working.

- [ ] **Step 4: Add descriptor builders**

Add helpers:

```c
static void hw_desc_dma_ddr_to_act(npu_hw_desc_t *d, uint32_t src, uint32_t act_base, uint32_t words);
static void hw_desc_img_expand(npu_hw_desc_t *d, uint32_t src_act, uint32_t dst_act, uint32_t words);
static void hw_desc_sram_copy(npu_hw_desc_t *d, uint32_t out_base, uint32_t act_base, uint32_t words, uint32_t ping_sel);
static void hw_desc_conv(npu_hw_desc_t *d, uint32_t act, uint32_t wgt, uint32_t out,
                         uint32_t in_w, uint32_t in_h, uint32_t ic, uint32_t oc,
                         uint32_t kh, uint32_t kw, uint32_t stride, uint32_t pad,
                         uint32_t flags, uint32_t qparam_ddr, uint32_t qparam_count);
static void hw_desc_gemm(npu_hw_desc_t *d, uint32_t act, uint32_t wgt, uint32_t out,
                         uint32_t ic, uint32_t oc, uint32_t flags,
                         uint32_t qparam_ddr, uint32_t qparam_count);
static void hw_desc_dma_out_to_ddr(npu_hw_desc_t *d, uint32_t out_base, uint32_t dst, uint32_t words, uint32_t out_ping);
static void hw_desc_stop(npu_hw_desc_t *d);
```

Each helper must call `npu_hw_desc_clear()` and `npu_hw_desc_set_op()` first.

- [ ] **Step 5: Build MNIST descriptor list**

Under `#if NPU_HW_DESC`, add:

```c
#define HW_DESC_DDR_BASE 0x4003C000u
#define HW_QPARAM_DDR_BASE 0x4003E000u
#define HW_DESC_MAX 32u

static npu_hw_desc_t hw_desc_list[HW_DESC_MAX];
static hw_qparam_t hw_qparams[256];
```

Build the current direct path sequence for one image:

```text
DMA raw image -> Act scratch
IMG_EXPAND
CONV1
SRAM_COPY
CONV2 pool
SRAM_COPY
CONV3
SRAM_COPY
CONV4 pool
SRAM_COPY
CONV5
SRAM_COPY
CONV6 pool
SRAM_COPY to FC1 input
GEMM FC1
DMA_OUT_TO_DDR FC1 staging if needed by current verified path
DMA_DDR_TO_ACT FC1 staging if needed by current verified path
GEMM FC2 INT32_OUT
DMA_OUT_TO_DDR FC2 logits
STOP_IRQ
```

Use the existing addresses and flags from `deepnet_inference()` and
`npu_conv_pass()`; do not invent new activation layout in this task.

- [ ] **Step 6: Add descriptor mode call site**

In `deepnet_inference()` or `usercode7()`, branch:

```c
#if NPU_HW_DESC
    if (!deepnet_inference_hw_desc(d)) {
        print_str("DESC INFER FAIL\n");
        return;
    }
#else
    deepnet_inference(mnist_images[d], (int32_t *)SCORES, d);
#endif
```

Preserve the old direct call under `#else`.

- [ ] **Step 7: Compile firmware direct and descriptor modes**

Run:

```bash
bash run_all.sh fw
EXTRA_CFLAGS=-DNPU_HW_DESC=1 bash run_all.sh fw
```

Expected both:

```text
[OK] 固件编译完成
```

- [ ] **Step 8: Commit**

```bash
git add firmware/deepnet_deploy.c firmware/npu_desc.h firmware/irq.c firmware/firmware.h
git commit -m "Build MNIST hardware descriptor stream"
```

---

### Task 9: Run MNIST Descriptor Integration And Preserve Direct Regression

**Files:**
- Modify only if verification exposes bugs:
  - `rtl/descriptor_engine.v`
  - `rtl/npu_top.v`
  - `firmware/deepnet_deploy.c`

**Interfaces:**
- Consumes all previous tasks.
- Produces verified descriptor-mode MNIST and direct-mode MNIST.

- [ ] **Step 1: Run direct MNIST regression**

Run:

```bash
rm -f .yolo_ddr
bash run_all.sh sim
```

Expected:

```text
=== Result: 10/10 correct ===
DEPLOY SUCCESS.
ALL TESTS PASSED.
```

- [ ] **Step 2: Run descriptor MNIST regression**

Run:

```bash
rm -f .yolo_ddr
EXTRA_CFLAGS=-DNPU_HW_DESC=1 bash run_all.sh sim
```

Expected:

```text
=== Result: 10/10 correct ===
DEPLOY SUCCESS.
ALL TESTS PASSED.
```

- [ ] **Step 3: Print descriptor counters/status**

Add temporary or permanent print lines:

```c
print_str("  desc_pc="); print_dec(npu_rd(NPU_DESC_PC));
print_str(" desc_status="); print_hex(npu_rd(NPU_DESC_STATUS));
print_str(" desc_err="); print_dec(npu_rd(NPU_DESC_ERR));
print_chr('\n');
```

Expected for successful final descriptor:

```text
desc_status has DONE set and ERR clear
desc_err=0
```

- [ ] **Step 4: Compare cycle headline**

Record:

```text
direct TRAP cycles
descriptor TRAP cycles
cyc_total/npu_busy/arr_active
```

Descriptor mode does not need to beat direct mode on the first RTL cut, but it
must reduce CPU per-layer MMIO scheduling in the code path and expose descriptor
setup time as hardware queue work.

- [ ] **Step 5: Commit any final fixes**

If code changed during this task:

```bash
git add rtl/descriptor_engine.v rtl/npu_top.v firmware/deepnet_deploy.c firmware/irq.c
git commit -m "Verify MNIST hardware descriptor mode"
```

If no code changed, do not create an empty commit.

---

### Task 10: Document Results And YOLO Phase-2 Handoff

**Files:**
- Create: `docs/notes/hw-descriptor-queue-v1.md`
- Optionally modify: `docs/notes/descriptor-runtime-v1.md`

**Interfaces:**
- Consumes final verified command outputs.
- Produces project-facing status and next-step handoff.

- [ ] **Step 1: Create results note**

Create `docs/notes/hw-descriptor-queue-v1.md`:

```markdown
# Hardware Descriptor Queue V1

Hardware descriptor queue v1 adds an RTL command-stream executor for the NPU.
The CPU writes descriptor base/count/start once per MNIST image, then waits for
descriptor completion instead of programming every layer through direct MMIO.

Verified:

- Direct MNIST path still reaches 10/10 correctness.
- Descriptor MNIST path reaches 10/10 correctness.
- Descriptor status reports PC, DONE, and ERR state.

V1 scope:

- Linear command stream.
- DMA, image expand, SRAM copy, Conv2D, GEMM, and STOP/IRQ.
- Qparam loading from DDR into the resident param regfile.

YOLO phase-2 handoff:

- Descriptor format already reserves YOLO fields for scratch buffers, tiled conv,
  pad value, weight words per OC, and strip rows.
- Reserved opcodes cover upsample, maxpool5x5, eltwise add, DFL, and LUT/config
  loads.
- Start phase 2 with the SPPF + neck movement region that is already represented
  by firmware descriptors.
```

- [ ] **Step 2: Commit docs**

```bash
git add docs/notes/hw-descriptor-queue-v1.md docs/notes/descriptor-runtime-v1.md
git commit -m "Document hardware descriptor queue v1"
```

---

## Final Verification

- [ ] **Run directed descriptor tests**

```bash
cd sim/desc
vlog -sv -timescale 1ns/1ps ../../rtl/descriptor_engine.v ../../tests/tb_descriptor_engine.v
vsim -c tb_descriptor_engine -do "run -all; quit -f"
cd ../desc_npu
vlog -sv -timescale 1ns/1ps -f ../../axi_sys.f ../../tests/tb_npu_desc_queue.v
vsim -c tb_npu_desc_queue -do "run -all; quit -f"
```

Expected:

```text
DESC_ENGINE TEST PASS
NPU DESC QUEUE TEST PASS
```

- [ ] **Run direct MNIST**

```bash
rm -f .yolo_ddr
bash run_all.sh sim
```

Expected:

```text
=== Result: 10/10 correct ===
DEPLOY SUCCESS.
ALL TESTS PASSED.
```

- [ ] **Run descriptor MNIST**

```bash
rm -f .yolo_ddr
EXTRA_CFLAGS=-DNPU_HW_DESC=1 bash run_all.sh sim
```

Expected:

```text
=== Result: 10/10 correct ===
DEPLOY SUCCESS.
ALL TESTS PASSED.
```

- [ ] **Check git status**

```bash
git status --short
```

Expected: only pre-existing unrelated dirty files remain.

---

## Self-Review

- Spec coverage: register ABI, 16-word format, descriptor engine FSM, qparam loading, MNIST-first path, YOLO reserved fields, error reporting, and verification are mapped to tasks.
- Placeholder scan: no TODO/TBD placeholders are intentionally left; every task has concrete files, interfaces, commands, and expected outputs.
- Type consistency: firmware constants use `NPU_HW_DESC_*`; RTL opcodes mirror the same numeric values; descriptor register offsets match between `firmware.h` and `param_regfile.v`.
- Scope control: v1 implements MNIST-required ops only; YOLO full migration is explicitly a later phase with preserved descriptor fields/opcodes.
