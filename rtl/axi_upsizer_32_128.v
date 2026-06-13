`timescale 1ns/1ps
// ---------------------------------------------------------------------------
// axi_upsizer_32_128
//
// Width adapter for a NARROW (32-bit) single-beat AXI master talking to a WIDE
// (128-bit) AXI slave / interconnect. Used on the CPU path so the CPU can share
// the 128-bit memory with the NPU without a 4x down-conversion on the NPU side.
//
//   * The CPU bridge (axi_lite_to_axi_full) issues single-beat AXI4 (awlen=0).
//   * A 32-bit WRITE -> one 128-bit beat. The 32-bit data is steered to the
//     lane selected by addr[3:2]; only those 4 byte-strobes are set, so the
//     other 12 bytes are left untouched (true partial write via WSTRB).
//   * A 32-bit READ  -> one 128-bit beat; the 32-bit word at addr[3:2] is
//     muxed back out.
//
// Single outstanding transaction (the CPU is low-traffic and its bridge is
// single-outstanding), so a small FSM is plenty and avoids reordering.
// ---------------------------------------------------------------------------
module axi_upsizer_32_128 (
    input  wire         clk,
    input  wire         resetn,

    // -------- 32-bit slave (from CPU bridge) --------
    input  wire         s_awvalid,
    output wire         s_awready,
    input  wire [31:0]  s_awaddr,
    input  wire [2:0]   s_awsize,
    input  wire [1:0]   s_awburst,
    input  wire         s_wvalid,
    output wire         s_wready,
    input  wire [31:0]  s_wdata,
    input  wire [3:0]   s_wstrb,
    input  wire         s_wlast,
    output wire         s_bvalid,
    input  wire         s_bready,
    output wire [1:0]   s_bresp,
    input  wire         s_arvalid,
    output wire         s_arready,
    input  wire [31:0]  s_araddr,
    input  wire [2:0]   s_arsize,
    input  wire [1:0]   s_arburst,
    output wire         s_rvalid,
    input  wire         s_rready,
    output wire [31:0]  s_rdata,
    output wire [1:0]   s_rresp,

    // -------- 128-bit master (to arbiter / memory) --------
    output wire         m_awvalid,
    input  wire         m_awready,
    output wire [31:0]  m_awaddr,
    output wire [7:0]   m_awlen,
    output wire [2:0]   m_awsize,
    output wire [1:0]   m_awburst,
    output wire         m_wvalid,
    input  wire         m_wready,
    output wire [127:0] m_wdata,
    output wire [15:0]  m_wstrb,
    output wire         m_wlast,
    input  wire         m_bvalid,
    output wire         m_bready,
    input  wire [1:0]   m_bresp,
    output wire         m_arvalid,
    input  wire         m_arready,
    output wire [31:0]  m_araddr,
    output wire [7:0]   m_arlen,
    output wire [2:0]   m_arsize,
    output wire [1:0]   m_arburst,
    input  wire         m_rvalid,
    output wire         m_rready,
    input  wire [127:0] m_rdata,
    input  wire [1:0]   m_rresp,
    input  wire         m_rlast
);

    localparam [2:0] IDLE   = 3'd0,
                     WCAP   = 3'd1,   // capture 32-bit write data
                     WISSUE = 3'd2,   // drive 128-bit AW + W
                     WRESP  = 3'd3,   // forward B
                     RD     = 3'd4,   // drive 128-bit AR
                     RDATA  = 3'd5;   // mux R back to 32-bit

    reg  [2:0]  state;
    reg  [31:0] addr_q;
    reg  [1:0]  sel_q;       // addr[3:2]: which 32-bit lane of the 128-bit word
    reg  [31:0] wdata_q;
    reg  [3:0]  wstrb_q;
    reg         aw_done, w_done;

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            state   <= IDLE;
            addr_q  <= 32'd0; sel_q   <= 2'd0;
            wdata_q <= 32'd0; wstrb_q <= 4'd0;
            aw_done <= 1'b0;  w_done  <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    aw_done <= 1'b0; w_done <= 1'b0;
                    if (s_awvalid && s_wvalid) begin
                        // fast path: AXI-Lite presents AW+W together -> capture
                        // both in one cycle and skip WCAP (saves a cycle/write).
                        addr_q  <= s_awaddr; sel_q <= s_awaddr[3:2];
                        wdata_q <= s_wdata;  wstrb_q <= s_wstrb;
                        state   <= WISSUE;
                    end else if (s_awvalid) begin
                        addr_q <= s_awaddr; sel_q <= s_awaddr[3:2];
                        state  <= WCAP;
                    end else if (s_arvalid) begin
                        addr_q <= s_araddr; sel_q <= s_araddr[3:2];
                        state  <= RD;
                    end
                end
                WCAP: if (s_wvalid) begin
                    wdata_q <= s_wdata; wstrb_q <= s_wstrb;
                    state   <= WISSUE;
                end
                WISSUE: begin
                    if (m_awvalid && m_awready) aw_done <= 1'b1;
                    if (m_wvalid  && m_wready ) w_done  <= 1'b1;
                    if ((aw_done || (m_awvalid && m_awready)) &&
                        (w_done  || (m_wvalid  && m_wready )))
                        state <= WRESP;
                end
                WRESP:  if (m_bvalid && s_bready) state <= IDLE;
                RD:     if (m_arvalid && m_arready) state <= RDATA;
                RDATA:  if (m_rvalid  && s_rready)  state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end

    // ---- slave-side handshakes ----
    assign s_awready = (state == IDLE);
    assign s_arready = (state == IDLE) && !s_awvalid;   // AW wins a tie
    assign s_wready  = (state == WCAP) || ((state == IDLE) && s_awvalid);
    assign s_bvalid  = (state == WRESP) && m_bvalid;
    assign s_bresp   = m_bresp;
    assign s_rvalid  = (state == RDATA) && m_rvalid;
    assign s_rdata   = m_rdata[{sel_q, 5'b0} +: 32];    // mux selected 32-bit word (sel*32)
    assign s_rresp   = m_rresp;

    // ---- master-side write ----
    assign m_awvalid = (state == WISSUE) && !aw_done;
    assign m_awaddr  = {addr_q[31:4], 4'b0000};         // 16-byte aligned
    assign m_awlen   = 8'd0;
    assign m_awsize  = 3'd4;                             // 16 bytes / beat
    assign m_awburst = 2'b01;
    assign m_wvalid  = (state == WISSUE) && !w_done;
    assign m_wdata   = {4{wdata_q}};                    // replicate; WSTRB selects lane
    assign m_wstrb   = {12'b0, wstrb_q} << {sel_q, 2'b00};  // sel*4, no width-trunc
    assign m_wlast   = 1'b1;
    assign m_bready  = (state == WRESP) && s_bready;

    // ---- master-side read ----
    assign m_arvalid = (state == RD);
    assign m_araddr  = {addr_q[31:4], 4'b0000};
    assign m_arlen   = 8'd0;
    assign m_arsize  = 3'd4;
    assign m_arburst = 2'b01;
    assign m_rready  = (state == RDATA) && s_rready;

endmodule
