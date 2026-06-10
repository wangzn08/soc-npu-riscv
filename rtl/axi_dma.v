// Filename: axi_dma.v
// -------------------------------------------------------------------
// AXI DMA — simplified AXI4-Full master for burst read/write to DDR.
// Moves data between off-chip DDR and on-chip SRAM buffers.
//
// Read path:  DMA reads from DDR → writes to SRAM (Fill)
// Write path: DMA reads from SRAM → writes to DDR (Drain)
//
// Supports configurable burst length and 128-bit data width.
// -------------------------------------------------------------------

module axi_dma #(
    parameter AXI_ADDR_W    = 32,
    parameter AXI_DATA_W    = 128,
    parameter AXI_ID_W      = 4,
    parameter AXI_LEN_W     = 8,
    parameter BURST_MAX     = 16,
    parameter SRAM_ADDR_W   = 14,
    parameter SRAM_DATA_W   = 128
) (
    input  wire                         clk,
    input  wire                         rst_n,

    // === Control interface (from FSM) ===
    input  wire                         i_dma_rd_req,
    input  wire [AXI_ADDR_W-1:0]        i_dma_rd_ddr_addr,
    input  wire [AXI_LEN_W-1:0]         i_dma_rd_len,
    input  wire [SRAM_ADDR_W-1:0]       i_dma_rd_sram_base,
    output wire                         o_dma_rd_done,
    output wire                         o_dma_rd_err,

    input  wire                         i_dma_wr_req,
    input  wire [AXI_ADDR_W-1:0]        i_dma_wr_ddr_addr,
    input  wire [AXI_LEN_W-1:0]         i_dma_wr_len,
    input  wire [SRAM_ADDR_W-1:0]       i_dma_wr_sram_base,
    output wire                         o_dma_wr_done,
    output wire                         o_dma_wr_err,

    // === SRAM write port (for read-from-DDR data) ===
    output wire [SRAM_ADDR_W-1:0]       o_sram_wr_addr,
    output wire                         o_sram_wr_en,
    output wire [SRAM_DATA_W-1:0]       o_sram_wr_data,

    // === SRAM read port (for write-to-DDR data) ===
    output wire [SRAM_ADDR_W-1:0]       o_sram_rd_addr,
    output wire                         o_sram_rd_en,
    input  wire [SRAM_DATA_W-1:0]       i_sram_rd_data,

    // === AXI4-Full Read Address Channel ===
    output wire                         m_axi_arvalid,
    input  wire                         m_axi_arready,
    output wire [AXI_ID_W-1:0]          m_axi_arid,
    output wire [AXI_ADDR_W-1:0]        m_axi_araddr,
    output wire [AXI_LEN_W-1:0]         m_axi_arlen,
    output wire [2:0]                   m_axi_arsize,
    output wire [1:0]                   m_axi_arburst,

    // === AXI4-Full Read Data Channel ===
    input  wire                         m_axi_rvalid,
    output wire                         m_axi_rready,
    input  wire [AXI_ID_W-1:0]          m_axi_rid,
    input  wire [AXI_DATA_W-1:0]        m_axi_rdata,
    input  wire [1:0]                   m_axi_rresp,
    input  wire                         m_axi_rlast,

    // === AXI4-Full Write Address Channel ===
    output wire                         m_axi_awvalid,
    input  wire                         m_axi_awready,
    output wire [AXI_ID_W-1:0]          m_axi_awid,
    output wire [AXI_ADDR_W-1:0]        m_axi_awaddr,
    output wire [AXI_LEN_W-1:0]         m_axi_awlen,
    output wire [2:0]                   m_axi_awsize,
    output wire [1:0]                   m_axi_awburst,

    // === AXI4-Full Write Data Channel ===
    output wire                         m_axi_wvalid,
    input  wire                         m_axi_wready,
    output wire [AXI_DATA_W-1:0]        m_axi_wdata,
    output wire [AXI_DATA_W/8-1:0]      m_axi_wstrb,
    output wire                         m_axi_wlast,

    // === AXI4-Full Write Response Channel ===
    input  wire                         m_axi_bvalid,
    output wire                         m_axi_bready,
    input  wire [AXI_ID_W-1:0]          m_axi_bid,
    input  wire [1:0]                   m_axi_bresp
);

    // -------------------------------------------------------------------
    // Read DMA state machine
    // -------------------------------------------------------------------
    localparam RD_IDLE   = 2'd0;
    localparam RD_ADDR   = 2'd1;
    localparam RD_DATA   = 2'd2;

    reg [1:0] rd_state;
    reg [AXI_ADDR_W-1:0]  rd_ddr_addr;
    reg [AXI_LEN_W-1:0]   rd_len;
    reg [SRAM_ADDR_W-1:0] rd_sram_addr;
    reg [AXI_LEN_W-1:0]   rd_beat_cnt;
    reg                   rd_done_r;
    reg                   rd_err_r;

    assign m_axi_arvalid = (rd_state == RD_ADDR);
    assign m_axi_arid    = {AXI_ID_W{1'b0}};
    assign m_axi_araddr  = rd_ddr_addr;
    assign m_axi_arlen   = rd_len;
    assign m_axi_arsize  = 3'd4;   // 2^4 = 16 bytes = 128 bits
    assign m_axi_arburst = 2'd1;   // INCR burst

    assign m_axi_rready  = (rd_state == RD_DATA);

    assign o_sram_wr_addr = rd_sram_addr;
    assign o_sram_wr_en   = m_axi_rvalid && m_axi_rready;
    assign o_sram_wr_data = m_axi_rdata;

    // synthesis translate_off
    `ifdef DEBUG
    always @(posedge clk) begin
        if (m_axi_rvalid && m_axi_rready)
            $display("DMA_RD: beat=%0d sram_addr=%0h data=%0h last=%0b",
                     rd_beat_cnt, rd_sram_addr, m_axi_rdata, m_axi_rlast);
        if (o_sram_wr_en)
            $display("DMA_SRAM_WR: addr=%0h data=%0h", o_sram_wr_addr, o_sram_wr_data);
    end
    `endif
    // synthesis translate_on

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_state     <= RD_IDLE;
            rd_ddr_addr  <= {AXI_ADDR_W{1'b0}};
            rd_len       <= {AXI_LEN_W{1'b0}};
            rd_sram_addr <= {SRAM_ADDR_W{1'b0}};
            rd_beat_cnt  <= {AXI_LEN_W{1'b0}};
            rd_done_r    <= 1'b0;
            rd_err_r     <= 1'b0;
        end else begin
            // Clear done when a new request starts
            if (i_dma_rd_req && rd_state == RD_IDLE) begin
                rd_done_r <= 1'b0;
                rd_err_r  <= 1'b0;
            end

            case (rd_state)
                RD_IDLE: begin
                    if (i_dma_rd_req) begin
                        // synthesis translate_off
                        `ifdef DEBUG
                        $display("DMA_RD_START: ddr=%0h len=%0d sram_base=%0h",
                                 i_dma_rd_ddr_addr, i_dma_rd_len, i_dma_rd_sram_base);
                        `endif
                        // synthesis translate_on
                        rd_ddr_addr  <= i_dma_rd_ddr_addr;
                        rd_len       <= i_dma_rd_len;
                        rd_sram_addr <= i_dma_rd_sram_base;
                        rd_beat_cnt  <= {AXI_LEN_W{1'b0}};
                        rd_state     <= RD_ADDR;
                    end
                end

                RD_ADDR: begin
                    if (m_axi_arvalid && m_axi_arready) begin
                        rd_state <= RD_DATA;
                    end
                end

                RD_DATA: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        if (m_axi_rresp != 2'b00)
                            rd_err_r <= 1'b1;
                        // Always increment address — SRAM write (combinational)
                        // uses current rd_sram_addr before the increment takes
                        // effect on the next clock edge.  On the rlast beat the
                        // write hits the correct address; the incremented value
                        // is unused until the next transfer.
                        rd_sram_addr <= rd_sram_addr + {{SRAM_ADDR_W-1{1'b0}}, 1'b1};
                        rd_beat_cnt  <= rd_beat_cnt + {{AXI_LEN_W-1{1'b0}}, 1'b1};
                        if (m_axi_rlast) begin
                            rd_state  <= RD_IDLE;
                            rd_done_r <= 1'b1;
                        end
                    end
                end

                default: rd_state <= RD_IDLE;
            endcase
        end
    end

    assign o_dma_rd_done = rd_done_r;
    assign o_dma_rd_err  = rd_err_r;

    // -------------------------------------------------------------------
    // Write DMA state machine
    // -------------------------------------------------------------------
    localparam WR_IDLE   = 2'd0;
    localparam WR_ADDR   = 2'd1;
    localparam WR_DATA   = 2'd2;
    localparam WR_RESP   = 2'd3;

    reg [1:0] wr_state;
    reg [AXI_ADDR_W-1:0]  wr_ddr_addr;
    reg [AXI_LEN_W-1:0]   wr_len;
    reg [SRAM_ADDR_W-1:0] wr_sram_addr;
    reg [AXI_LEN_W-1:0]   wr_beat_cnt;
    reg                   wr_done_r;
    reg                   wr_err_r;

    // SRAM read for write data
    assign o_sram_rd_addr = wr_sram_addr;
    assign o_sram_rd_en   = (wr_state == WR_ADDR) || (wr_state == WR_DATA);

    // Write address channel
    assign m_axi_awvalid = (wr_state == WR_ADDR);
    assign m_axi_awid    = {AXI_ID_W{1'b0}};
    assign m_axi_awaddr  = wr_ddr_addr;
    assign m_axi_awlen   = wr_len;
    assign m_axi_awsize  = 3'd4;
    assign m_axi_awburst = 2'd1;

    // Write data channel: with combinational SRAM read (COMB_B=1),
    // i_sram_rd_data reflects current wr_sram_addr in the same cycle.
    // Feed directly to wdata — no pipeline register needed.
    assign m_axi_wvalid = (wr_state == WR_DATA);
    assign m_axi_wdata  = i_sram_rd_data;
    assign m_axi_wstrb  = {(AXI_DATA_W/8){1'b1}};
    assign m_axi_wlast  = (wr_beat_cnt == wr_len);

    // Write response
    assign m_axi_bready = (wr_state == WR_RESP);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_state     <= WR_IDLE;
            wr_ddr_addr  <= {AXI_ADDR_W{1'b0}};
            wr_len       <= {AXI_LEN_W{1'b0}};
            wr_sram_addr <= {SRAM_ADDR_W{1'b0}};
            wr_beat_cnt  <= {AXI_LEN_W{1'b0}};
            wr_done_r    <= 1'b0;
            wr_err_r     <= 1'b0;
        end else begin
            // Clear done when a new request starts
            if (i_dma_wr_req && wr_state == WR_IDLE) begin
                wr_done_r <= 1'b0;
                wr_err_r  <= 1'b0;
            end

            case (wr_state)
                WR_IDLE: begin
                    if (i_dma_wr_req) begin
                        // synthesis translate_off
                        `ifdef DEBUG
                        $display("DMA_WR_START: ddr=%0h len=%0d sram_base=%0h",
                                 i_dma_wr_ddr_addr, i_dma_wr_len, i_dma_wr_sram_base);
                        `endif
                        // synthesis translate_on
                        wr_ddr_addr  <= i_dma_wr_ddr_addr;
                        wr_len       <= i_dma_wr_len;
                        wr_sram_addr <= i_dma_wr_sram_base;
                        wr_beat_cnt  <= {AXI_LEN_W{1'b0}};
                        wr_state     <= WR_ADDR;
                    end
                end

                WR_ADDR: begin
                    if (m_axi_awvalid && m_axi_awready) begin
                        wr_state <= WR_DATA;
                    end
                end

                WR_DATA: begin
                    // synthesis translate_off
                    `ifdef DEBUG
                    $display("DMA_SRAM_RD: sram_addr=%0h rd_data=%0h",
                             wr_sram_addr, i_sram_rd_data);
                    `endif
                    // synthesis translate_on
                    if (m_axi_wvalid && m_axi_wready) begin
                        // synthesis translate_off
                        `ifdef DEBUG
                        $display("DMA_WR: beat=%0d/%0d sram_addr=%0h data=%0h last=%0b",
                                 wr_beat_cnt, wr_len, wr_sram_addr, i_sram_rd_data, m_axi_wlast);
                        `endif
                        // synthesis translate_on
                        wr_sram_addr <= wr_sram_addr + {{SRAM_ADDR_W-1{1'b0}}, 1'b1};
                        wr_beat_cnt  <= wr_beat_cnt + {{AXI_LEN_W-1{1'b0}}, 1'b1};
                        if (wr_beat_cnt == wr_len) begin
                            wr_state <= WR_RESP;
                        end
                    end
                end

                WR_RESP: begin
                    if (m_axi_bvalid && m_axi_bready) begin
                        if (m_axi_bresp != 2'b00)
                            wr_err_r <= 1'b1;
                        wr_state  <= WR_IDLE;
                        wr_done_r <= 1'b1;
                    end
                end

                default: wr_state <= WR_IDLE;
            endcase
        end
    end

    assign o_dma_wr_done = wr_done_r;
    assign o_dma_wr_err  = wr_err_r;

endmodule
