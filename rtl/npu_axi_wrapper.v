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
    parameter SOC_AXI_DATA_W  = 32,    // SoC bus width
    parameter AXI_ADDR_W      = 32,
    parameter AXI_ID_W        = 4,
    parameter AXI_LEN_W       = 8,
    parameter REG_ADDR_W      = 10
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
    output wire                         irq_done
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
        .irq_done       (irq_done)
    );

    // ===================================================================
    // AXI Width Converter: 128-bit (NPU) ↔ 32-bit (SoC)
    // ===================================================================
    // Each 128-bit beat → 4 × 32-bit beats.
    // Write: latch 128-bit, serialize 4 × 32-bit
    // Read:  collect 4 × 32-bit, assemble 128-bit

    // ---- Write path state machine ----
    // DMA sends burst: one address (WR_ADDR) then N data beats (WR_DATA).
    // Each 128-bit DMA beat → 4 × 32-bit slave beats.
    // Total slave burst = 4*(dma_awlen+1) beats, single address, wlast on last.
    //
    // States: WR_IDLE → WR_WAIT_DATA → WR_SEND → (WR_WAIT_DATA or WR_WAIT_RESP)

    localparam WR_IDLE      = 2'd0;
    localparam WR_WAIT_DATA = 2'd1;
    localparam WR_SEND      = 2'd2;
    localparam WR_WAIT_RESP = 2'd3;

    reg [1:0]                wr_state;
    reg [AXI_ADDR_W-1:0]    wr_base_addr;
    reg [7:0]               wr_awlen;        // DMA burst length (N-1)
    reg [7:0]               wr_dma_beat_cnt;  // current DMA beat index
    reg [NPU_AXI_DATA_W-1:0] wr_data_buf;
    reg [1:0]               wr_sub_cnt;       // 32-bit sub-beat index (0..3)
    reg                     wr_aw_sent;       // awvalid accepted by slave

    wire wr_xfer = m_axi_wvalid && m_axi_wready;

    // DMA handshake: accept address in IDLE, accept data in WAIT_DATA
    assign dma_awready = (wr_state == WR_IDLE);
    assign dma_wready  = (wr_state == WR_WAIT_DATA);

    // Write state machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_state        <= WR_IDLE;
            wr_base_addr    <= {AXI_ADDR_W{1'b0}};
            wr_awlen        <= 8'd0;
            wr_dma_beat_cnt <= 8'd0;
            wr_data_buf     <= {NPU_AXI_DATA_W{1'b0}};
            wr_sub_cnt      <= 2'd0;
            wr_aw_sent      <= 1'b0;
        end else begin
            case (wr_state)
                WR_IDLE: begin
                    if (dma_awvalid) begin
                        wr_base_addr    <= dma_awaddr;
                        wr_awlen        <= dma_awlen;
                        wr_dma_beat_cnt <= 8'd0;
                        wr_state        <= WR_WAIT_DATA;
                    end
                end

                WR_WAIT_DATA: begin
                    if (dma_wvalid) begin
                        wr_data_buf <= dma_wdata;
                        wr_sub_cnt  <= 2'd0;
                        wr_aw_sent  <= 1'b0;
                        wr_state    <= WR_SEND;
                    end
                end

                WR_SEND: begin
                    // Track awvalid/awready handshake
                    if (m_axi_awvalid && m_axi_awready)
                        wr_aw_sent <= 1'b1;
                    // Track wvalid/wready data beats
                    if (wr_xfer) begin
                        if (wr_sub_cnt == 2'd3) begin
                            wr_dma_beat_cnt <= wr_dma_beat_cnt + 8'd1;
                            if (wr_dma_beat_cnt == wr_awlen)
                                wr_state <= WR_WAIT_RESP;  // last beat done
                            else
                                wr_state <= WR_WAIT_DATA;   // more beats
                        end else begin
                            wr_sub_cnt <= wr_sub_cnt + 2'd1;
                        end
                    end
                end

                WR_WAIT_RESP: begin
                    if (m_axi_bvalid)
                        wr_state <= WR_IDLE;
                end
            endcase
        end
    end

    // ---- Slave write address channel ----
    // awvalid sent once at start of burst, cleared when slave accepts
    assign m_axi_awvalid = (wr_state == WR_SEND) && !wr_aw_sent;
    assign m_axi_awaddr  = wr_base_addr;
    assign m_axi_awlen   = {wr_awlen, 2'b11};   // ×4: NPU awlen=N-1 → SoC awlen=4N-1
    assign m_axi_awsize  = 3'd2;                  // 4 bytes
    assign m_axi_awburst = 2'b01;                 // INCR

    // ---- Slave write data channel ----
    reg [SOC_AXI_DATA_W-1:0] wdata_mux;
    always @(*) begin
        case (wr_sub_cnt)
            2'd0:    wdata_mux = wr_data_buf[31:0];
            2'd1:    wdata_mux = wr_data_buf[63:32];
            2'd2:    wdata_mux = wr_data_buf[95:64];
            default: wdata_mux = wr_data_buf[127:96];
        endcase
    end

    assign m_axi_wvalid = (wr_state == WR_SEND);
    assign m_axi_wdata  = wdata_mux;
    assign m_axi_wlast  = (wr_state == WR_SEND) && (wr_sub_cnt == 2'd3) &&
                          (wr_dma_beat_cnt == wr_awlen);

    // ---- Write response ----
    assign m_axi_bready  = 1'b1;
    assign dma_bvalid    = m_axi_bvalid && (wr_state == WR_WAIT_RESP);
    assign dma_bresp     = m_axi_bresp;

    // ---- Read path: AXI width conversion with burst splitting ----
    // Each 128-bit DMA beat → 4 × 32-bit SoC beats.
    // Maximum AXI burst length is 256 (arlen=255), supporting 64 DMA beats
    // per SoC burst. Larger DMA transfers are split into multiple AXI bursts.

    localparam RD_IDLE = 2'd0;
    localparam RD_ADDR = 2'd1;
    localparam RD_DATA = 2'd2;

    reg [1:0]               rd_state;
    reg [AXI_ADDR_W-1:0]    rd_addr;
    reg [7:0]               rd_dma_total;   // DMA burst length (N-1)
    reg [7:0]               rd_dma_done;    // DMA beats completed
    reg [7:0]               rd_soc_arlen;   // current SoC burst arlen
    reg [1:0]               rd_sub_cnt;     // 32-bit sub-beat index
    reg [NPU_AXI_DATA_W-1:0] rd_data_buf;
    reg                     rd_word_done;

    // DMA address handshake
    assign dma_arready = (rd_state == RD_IDLE);

    // SoC address channel
    assign m_axi_arvalid = (rd_state == RD_ADDR);
    assign m_axi_araddr  = rd_addr;
    assign m_axi_arlen   = rd_soc_arlen;
    assign m_axi_arsize  = 3'd2;
    assign m_axi_arburst = 2'b01;

    // SoC data channel
    assign m_axi_rready = 1'b1;
    wire rd_xfer = m_axi_rvalid && m_axi_rready;

    // DMA data channel
    assign dma_rvalid = rd_word_done;
    assign dma_rdata  = rd_data_buf;
    assign dma_rresp  = m_axi_rresp;

    // dma_rlast: registered, sampled in the same cycle as rd_word_done assertion.
    // Using the registered rd_dma_done ONE CYCLE LATER would cause rlast to fire
    // one beat early (rd_dma_done was already incremented past the current beat).
    reg dma_rlast_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            dma_rlast_r <= 1'b0;
        else if (rd_state == RD_DATA && rd_xfer &&
                 (m_axi_rlast || rd_sub_cnt == 2'd3))
            dma_rlast_r <= (rd_dma_done == rd_dma_total);
        else
            dma_rlast_r <= 1'b0;
    end
    assign dma_rlast = dma_rlast_r;

    // Combinational: remaining DMA beats after current
    wire [7:0] rd_dma_remaining = rd_dma_total - rd_dma_done;
    // Next SoC burst chunk size (DMA beats, max 64)
    wire [5:0] rd_next_chunk = (rd_dma_remaining > 8'd64) ? 6'd64 : rd_dma_remaining[5:0];
    // Next SoC arlen = (chunk-1)*4 + 3 = {chunk-1, 2'b11}
    wire [7:0] rd_next_arlen = {rd_next_chunk - 6'd1, 2'b11};
    // Next SoC address = current + (arlen+1)*4
    wire [AXI_ADDR_W-1:0] rd_next_addr = rd_addr +
        {{(AXI_ADDR_W-10){1'b0}}, rd_soc_arlen, 2'b00} + 32'd4;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_state     <= RD_IDLE;
            rd_addr      <= {AXI_ADDR_W{1'b0}};
            rd_dma_total <= 8'd0;
            rd_dma_done  <= 8'd0;
            rd_soc_arlen <= 8'd0;
            rd_sub_cnt   <= 2'd0;
            rd_data_buf  <= {NPU_AXI_DATA_W{1'b0}};
            rd_word_done <= 1'b0;
        end else begin
            rd_word_done <= 1'b0;

            case (rd_state)
                RD_IDLE: begin
                    if (dma_arvalid) begin
                        rd_addr      <= dma_araddr;
                        rd_dma_total <= dma_arlen;
                        rd_dma_done  <= 8'd0;
                        rd_sub_cnt   <= 2'd0;
                        // First burst: min(dma_arlen+1, 64) DMA beats
                        rd_soc_arlen <= (dma_arlen > 8'd63) ?
                            8'd255 :
                            {dma_arlen[5:0], 2'b00} + 8'd3;
                        rd_state     <= RD_ADDR;
                    end
                end

                RD_ADDR: begin
                    if (m_axi_arready) begin
                        rd_state <= RD_DATA;
                    end
                end

                RD_DATA: begin
                    if (rd_xfer) begin
                        case (rd_sub_cnt)
                            2'd0: rd_data_buf[31:0]   <= m_axi_rdata;
                            2'd1: rd_data_buf[63:32]  <= m_axi_rdata;
                            2'd2: rd_data_buf[95:64]  <= m_axi_rdata;
                            2'd3: rd_data_buf[127:96] <= m_axi_rdata;
                        endcase

                        if (m_axi_rlast || rd_sub_cnt == 2'd3) begin
                            rd_sub_cnt   <= 2'd0;
                            rd_word_done <= 1'b1;

                            if (rd_dma_done == rd_dma_total) begin
                                rd_state <= RD_IDLE;
                            end else begin
                                rd_dma_done  <= rd_dma_done + 8'd1;
                                if (m_axi_rlast) begin
                                    rd_addr      <= rd_next_addr;
                                    rd_soc_arlen <= rd_next_arlen;
                                    rd_state     <= RD_ADDR;
                                end
                            end
                        end else begin
                            rd_sub_cnt <= rd_sub_cnt + 2'd1;
                        end
                    end
                end
            endcase
        end
    end

    // synthesis translate_off
    // always @(posedge clk) begin
    //     if (dma_arvalid && dma_arready)
    //         $display("WRAP_RD_ACCEPT: addr=%0h len=%0d", dma_araddr, dma_arlen);
    //     if (m_axi_arvalid && m_axi_arready)
    //         $display("WRAP_RD_SEND: addr=%0h len=%0d", m_axi_araddr, m_axi_arlen);
    //     if (rd_xfer)
    //         $display("WRAP_RD_DATA: rdata=%0h sub=%0d rlast=%0b", m_axi_rdata, rd_sub_cnt, m_axi_rlast);
    //     if (rd_word_done)
    //         $display("WRAP_RD_WORD: data=%0h done_cnt=%0d total=%0d", rd_data_buf, rd_dma_done, rd_dma_total);
    //     if (dma_rvalid)
    //         $display("WRAP_RD_DMA: rdata=%0h rlast=%0b", dma_rdata, dma_rlast);
    // end
    // synthesis translate_on

endmodule
