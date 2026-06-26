// Filename: npu_axi_wrapper.v
// -------------------------------------------------------------------
// Wrapper that integrates the 16x16 systolic NPU into the 飞腾杯 SoC.
//
// Provides:
//   - Register interface for CPU MMIO (config registers via param_regfile)
//   - AXI4-Full master with 128-bit → 32-bit width conversion (DMA)
//   - Interrupt output (irq_done)
//
// The SoC (axi_sys.v) drives reg_wr_en/reg_rd_en for register access,
// and connects the 32-bit AXI-Full master through the arbiter to shared
// memory.
// -------------------------------------------------------------------

module npu_axi_wrapper #(
    parameter NPU_AXI_DATA_W  = 128,   // NPU native DMA width
    parameter SOC_AXI_DATA_W  = 128,    // SoC bus width
    parameter AXI_ADDR_W      = 32,
    parameter AXI_ID_W        = 4,
    parameter AXI_LEN_W       = 8,
    parameter REG_ADDR_W      = 12
) (
    input  wire                         clk,
    input  wire                         rst_n,

    // === Register interface (from axi_sys MMIO logic) ===
    input  wire                         reg_wr_en,
    input  wire [REG_ADDR_W-1:0]        reg_wr_addr,
    input  wire [31:0]                  reg_wr_data,

    input  wire                         reg_rd_en,
    input  wire [REG_ADDR_W-1:0]        reg_rd_addr,
    output wire [31:0]                  reg_rd_data,
    output wire                         reg_rd_data_valid,
    output wire                         reg_wr_done,

    // === AXI4-Full Master 32-bit (to arbiter → shared memory) ===
    // Write Address
    output wire                         m_axi_awvalid,
    input  wire                         m_axi_awready,
    output wire [AXI_ADDR_W-1:0]        m_axi_awaddr,
    output wire [7:0]                   m_axi_awlen,
    output wire [2:0]                   m_axi_awsize,
    output wire [1:0]                   m_axi_awburst,
    // Write Data
    output wire                         m_axi_wvalid,
    input  wire                         m_axi_wready,
    output wire [SOC_AXI_DATA_W-1:0]    m_axi_wdata,
    output wire                         m_axi_wlast,
    // Write Response
    input  wire                         m_axi_bvalid,
    output wire                         m_axi_bready,
    input  wire [1:0]                   m_axi_bresp,
    // Read Address
    output wire                         m_axi_arvalid,
    input  wire                         m_axi_arready,
    output wire [AXI_ADDR_W-1:0]        m_axi_araddr,
    output wire [7:0]                   m_axi_arlen,
    output wire [2:0]                   m_axi_arsize,
    output wire [1:0]                   m_axi_arburst,
    // Read Data
    input  wire                         m_axi_rvalid,
    output wire                         m_axi_rready,
    input  wire [SOC_AXI_DATA_W-1:0]    m_axi_rdata,
    input  wire                         m_axi_rlast,
    input  wire [1:0]                   m_axi_rresp,

    // === Interrupt ===
    output wire                         irq_done,

    // === DMA completion signals ===
    output wire                         dma_rd_done,
    output wire                         dma_wr_done
);

    // ===================================================================
    // Internal AXI-Lite wires (wrapper ↔ NPU param_regfile)
    // ===================================================================
    wire        npu_awvalid, npu_awready;
    wire [REG_ADDR_W-1:0] npu_awaddr;
    wire        npu_wvalid,  npu_wready;
    wire [31:0] npu_wdata;
    wire [3:0]  npu_wstrb;
    wire        npu_bvalid,  npu_bready;
    wire [1:0]  npu_bresp;
    wire        npu_arvalid, npu_arready;
    wire [REG_ADDR_W-1:0] npu_araddr;
    wire        npu_rvalid,  npu_rready;
    wire [31:0] npu_rdata;
    wire [1:0]  npu_rresp;

    // ===================================================================
    // Internal 128-bit AXI wires (NPU DMA ↔ width converter)
    // ===================================================================
    wire                    dma_awvalid, dma_awready;
    wire [AXI_ADDR_W-1:0]   dma_awaddr;
    wire [AXI_LEN_W-1:0]    dma_awlen;
    wire [2:0]              dma_awsize;
    wire [1:0]              dma_awburst;
    wire                    dma_wvalid,  dma_wready;
    wire [NPU_AXI_DATA_W-1:0] dma_wdata;
    wire [NPU_AXI_DATA_W/8-1:0] dma_wstrb;
    wire                    dma_wlast;
    wire                    dma_bvalid,  dma_bready;
    wire [1:0]              dma_bresp;
    wire                    dma_arvalid, dma_arready;
    wire [AXI_ADDR_W-1:0]   dma_araddr;
    wire [AXI_LEN_W-1:0]    dma_arlen;
    wire [2:0]              dma_arsize;
    wire [1:0]              dma_arburst;
    wire                    dma_rvalid,  dma_rready;
    wire [NPU_AXI_DATA_W-1:0] dma_rdata;
    wire [1:0]              dma_rresp;
    wire                    dma_rlast;

    // ===================================================================
    // Register interface → AXI-Lite conversion
    // ===================================================================
    // Write: assert awvalid+wvalid for one cycle, wait for response
    reg wr_pending;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            wr_pending <= 1'b0;
        else if (reg_wr_en && !wr_pending)
            wr_pending <= 1'b1;
        else if (npu_bvalid && npu_bready)
            wr_pending <= 1'b0;
    end

    assign npu_awvalid = reg_wr_en && !wr_pending;
    assign npu_awaddr  = {reg_wr_addr, 2'b00};  // word addr → byte addr
    assign npu_wvalid  = reg_wr_en && !wr_pending;
    assign npu_wdata   = reg_wr_data;
    assign npu_wstrb   = 4'hF;
    assign npu_bready  = 1'b1;

    // Write done: pulse when NPU register write completes
    assign reg_wr_done = npu_bvalid && npu_bready;

    // synthesis translate_off
    `ifdef DEBUG
    always @(posedge clk) begin
        if (reg_wr_en)
            $display("WRAP_WR: reg_wr_en addr=0x%03h data=0x%08h wr_pending=%0b", reg_wr_addr, reg_wr_data, wr_pending);
        if (npu_awvalid && npu_awready)
            $display("WRAP_AW: awvalid accepted addr=0x%08h", npu_awaddr);
        if (npu_wvalid && npu_wready)
            $display("WRAP_W: wvalid accepted data=0x%08h", npu_wdata);
        if (npu_bvalid && npu_bready)
            $display("WRAP_B: bvalid resp=%0d", npu_bresp);
        if (reg_wr_done)
            $display("WRAP_DONE: write complete");
    end
    `endif
    // synthesis translate_on

    // Read: assert arvalid for one cycle, capture rdata
    // Hold rd_pending until read data is delivered to upstream
    reg rd_pending;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rd_pending <= 1'b0;
        else if (reg_rd_en && !rd_pending)
            rd_pending <= 1'b1;
        else if (reg_rd_data_valid)
            rd_pending <= 1'b0;
    end

    assign npu_arvalid = reg_rd_en && !rd_pending;
    assign npu_araddr  = {reg_rd_addr, 2'b00};  // word addr → byte addr
    assign npu_rready  = 1'b1;

    assign reg_rd_data       = npu_rdata;
    assign reg_rd_data_valid = npu_rvalid;

    // ===================================================================
    // Instantiate the NPU
    // ===================================================================
    npu_top #(
        .AXI_ADDR_W     (AXI_ADDR_W),
        .AXI_DATA_W     (NPU_AXI_DATA_W),
        .AXI_ID_W       (AXI_ID_W),
        .AXI_LEN_W      (AXI_LEN_W),
        .REG_ADDR_W      (REG_ADDR_W)
    ) u_npu_top (
        .clk            (clk),
        .rst_n          (rst_n),

        // AXI-Lite slave (config)
        .s_axi_awvalid  (npu_awvalid),
        .s_axi_awready  (npu_awready),
        .s_axi_awaddr   (npu_awaddr),
        .s_axi_wvalid   (npu_wvalid),
        .s_axi_wready   (npu_wready),
        .s_axi_wdata    (npu_wdata),
        .s_axi_wstrb    (npu_wstrb),
        .s_axi_bvalid   (npu_bvalid),
        .s_axi_bready   (npu_bready),
        .s_axi_bresp    (npu_bresp),
        .s_axi_arvalid  (npu_arvalid),
        .s_axi_arready  (npu_arready),
        .s_axi_araddr   (npu_araddr),
        .s_axi_rvalid   (npu_rvalid),
        .s_axi_rready   (npu_rready),
        .s_axi_rdata    (npu_rdata),
        .s_axi_rresp    (npu_rresp),

        // AXI-Full master (DMA, 128-bit)
        .m_axi_arvalid  (dma_arvalid),
        .m_axi_arready  (dma_arready),
        .m_axi_arid     (),
        .m_axi_araddr   (dma_araddr),
        .m_axi_arlen    (dma_arlen),
        .m_axi_arsize   (dma_arsize),
        .m_axi_arburst  (dma_arburst),
        .m_axi_rvalid   (dma_rvalid),
        .m_axi_rready   (dma_rready),
        .m_axi_rid      ({AXI_ID_W{1'b0}}),
        .m_axi_rdata    (dma_rdata),
        .m_axi_rresp    (dma_rresp),
        .m_axi_rlast    (dma_rlast),
        .m_axi_awvalid  (dma_awvalid),
        .m_axi_awready  (dma_awready),
        .m_axi_awid     (),
        .m_axi_awaddr   (dma_awaddr),
        .m_axi_awlen    (dma_awlen),
        .m_axi_awsize   (dma_awsize),
        .m_axi_awburst  (dma_awburst),
        .m_axi_wvalid   (dma_wvalid),
        .m_axi_wready   (dma_wready),
        .m_axi_wdata    (dma_wdata),
        .m_axi_wstrb    (dma_wstrb),
        .m_axi_wlast    (dma_wlast),
        .m_axi_bvalid   (dma_bvalid),
        .m_axi_bready   (dma_bready),
        .m_axi_bid      ({AXI_ID_W{1'b0}}),
        .m_axi_bresp    (dma_bresp),

        // Interrupt
        .irq_done       (irq_done),

        // DMA completion
        .dma_rd_done    (dma_rd_done),
        .dma_wr_done    (dma_wr_done)
    );

    // ===================================================================
    // DMA AXI pass-through (Stage 2): npu_top's native 128-bit DMA master
    // drives the SoC bus directly. The old 128->32 width converter is removed;
    // the SoC data path / arbiter / shared memory are now 128-bit, so there is
    // no 4x down-conversion. NPU writes full beats, so axi_sys ties the write
    // strobe to all-ones and this wrapper carries no WSTRB port. dma_wstrb from
    // npu_top is intentionally left unused (full-beat writes).
    // ===================================================================
    // Write address
    assign m_axi_awvalid = dma_awvalid;
    assign dma_awready   = m_axi_awready;
    assign m_axi_awaddr  = dma_awaddr;
    assign m_axi_awlen   = dma_awlen;
    assign m_axi_awsize  = dma_awsize;
    assign m_axi_awburst = dma_awburst;
    // Write data
    assign m_axi_wvalid  = dma_wvalid;
    assign dma_wready    = m_axi_wready;
    assign m_axi_wdata   = dma_wdata;
    assign m_axi_wlast   = dma_wlast;
    // Write response
    assign dma_bvalid    = m_axi_bvalid;
    assign m_axi_bready  = dma_bready;
    assign dma_bresp     = m_axi_bresp;
    // Read address
    assign m_axi_arvalid = dma_arvalid;
    assign dma_arready   = m_axi_arready;
    assign m_axi_araddr  = dma_araddr;
    assign m_axi_arlen   = dma_arlen;
    assign m_axi_arsize  = dma_arsize;
    assign m_axi_arburst = dma_arburst;
    // Read data
    assign dma_rvalid    = m_axi_rvalid;
    assign m_axi_rready  = dma_rready;
    assign dma_rdata     = m_axi_rdata;
    assign dma_rresp     = m_axi_rresp;
    assign dma_rlast     = m_axi_rlast;

endmodule
