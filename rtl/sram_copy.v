// Filename: sram_copy.v
// -------------------------------------------------------------------
// On-chip SRAM copy engine (SRAM residency).
// Copies `i_len` 128-bit words from Out SRAM (Port B read) to Act SRAM
// (Port B write), src/dst word bases configurable.  One word/cycle with a
// 1-cycle SRAM read latency (read addr issued at cycle T, data lands at T+1 and
// is written).  Banks are selected by the SRAM wrappers' dma_ping_sel (driven by
// cfg_dma_out_ping_sel / cfg_dma_act_ping_sel in npu_top) — not here.
//   i_len is the FULL word count (not count-1).
//
// 片上 SRAM 拷贝引擎（SRAM 驻留）：Out SRAM(Port B 读) → Act SRAM(Port B 写)，
// 1 字/周期，1 拍 SRAM 读延迟。bank 由 wrapper 的 dma_ping_sel 决定。
// -------------------------------------------------------------------
module sram_copy #(
    parameter ADDR_W = 14,
    parameter DATA_W = 128
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                i_trig,        // 1-cycle pulse: start
    input  wire [ADDR_W-1:0]   i_src_base,    // Out SRAM word base
    input  wire [ADDR_W-1:0]   i_dst_base,    // Act SRAM word base
    input  wire [15:0]         i_len,         // word count (full)
    // Out SRAM Port B read
    output wire [ADDR_W-1:0]   o_out_rd_addr,
    output wire                o_out_rd_en,
    input  wire [DATA_W-1:0]   i_out_rd_data,
    // Act SRAM Port B write
    output wire [ADDR_W-1:0]   o_act_wr_addr,
    output wire                o_act_wr_en,
    output wire [DATA_W-1:0]   o_act_wr_data,
    output wire                o_busy,
    output reg                 o_done         // set on completion, cleared on i_trig
);
    reg              busy;
    reg [15:0]       rd_cnt;       // reads issued
    reg [15:0]       wr_cnt;       // writes completed
    reg [15:0]       len_q;
    reg [ADDR_W-1:0] src_q, dst_q;
    reg              rd_vld_d;     // a read was issued last cycle -> data valid now
    reg [ADDR_W-1:0] wr_addr_d;    // matching write address

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; rd_cnt <= 16'd0; wr_cnt <= 16'd0; len_q <= 16'd0;
            src_q <= {ADDR_W{1'b0}}; dst_q <= {ADDR_W{1'b0}};
            rd_vld_d <= 1'b0; wr_addr_d <= {ADDR_W{1'b0}}; o_done <= 1'b0;
        end else begin
            rd_vld_d <= 1'b0;
            if (i_trig) begin
                busy   <= 1'b1;
                rd_cnt <= 16'd0; wr_cnt <= 16'd0;
                len_q  <= i_len; src_q <= i_src_base; dst_q <= i_dst_base;
                o_done <= 1'b0;
            end else if (busy) begin
                // Issue a read each cycle while reads remain
                if (rd_cnt < len_q) begin
                    rd_vld_d  <= 1'b1;
                    wr_addr_d <= dst_q + rd_cnt[ADDR_W-1:0];
                    rd_cnt    <= rd_cnt + 16'd1;
                end
                // A write completes the cycle after its read was issued
                if (rd_vld_d) begin
                    wr_cnt <= wr_cnt + 16'd1;
                    if (wr_cnt + 16'd1 == len_q) begin
                        busy   <= 1'b0;
                        o_done <= 1'b1;
                    end
                end
            end
        end
    end

    assign o_out_rd_addr = src_q + rd_cnt[ADDR_W-1:0];
    assign o_out_rd_en   = busy && (rd_cnt < len_q);
    assign o_act_wr_addr = wr_addr_d;
    assign o_act_wr_en   = rd_vld_d;
    assign o_act_wr_data = i_out_rd_data;   // Out Port B read data -> Act Port B write
    assign o_busy        = busy;
endmodule
