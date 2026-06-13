// Filename: sram_copy.v
// -------------------------------------------------------------------
// On-chip SRAM copy engine (SRAM residency).
// Copies `i_len` 128-bit words from Out SRAM (Port B read) to Act SRAM
// (Port B write), src/dst word bases configurable.  Out Port B read is
// COMBINATIONAL (out_sram_wrapper COMB_B=1): the word at addr src+cnt is
// available the SAME cycle, so it is written to Act addr dst+cnt that cycle
// (Act write is synchronous, latched on the edge).  One word/cycle, no pipeline.
// Banks are selected by the SRAM wrappers' dma_ping_sel (driven by
// cfg_dma_out_ping_sel / cfg_dma_act_ping_sel in npu_top) — not here.
//   i_len is the FULL word count (not count-1).
//
// 片上 SRAM 拷贝引擎（SRAM 驻留）：Out SRAM(Port B 组合读) → Act SRAM(Port B 写)，
// 1 字/周期，读写同拍（Out 读是组合的）。bank 由 wrapper 的 dma_ping_sel 决定。
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
    reg [15:0]       cnt;          // words processed (read+write same cycle)
    reg [15:0]       len_q;
    reg [ADDR_W-1:0] src_q, dst_q;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; cnt <= 16'd0; len_q <= 16'd0;
            src_q <= {ADDR_W{1'b0}}; dst_q <= {ADDR_W{1'b0}}; o_done <= 1'b0;
        end else begin
            if (i_trig) begin
                busy  <= 1'b1;
                cnt   <= 16'd0;
                len_q <= i_len; src_q <= i_src_base; dst_q <= i_dst_base;
                o_done <= 1'b0;
            end else if (busy) begin
                if (cnt < len_q) begin
                    // read Out[src+cnt] (combinational) and write Act[dst+cnt] this cycle
                    cnt <= cnt + 16'd1;
                    if (cnt + 16'd1 == len_q) begin
                        busy   <= 1'b0;
                        o_done <= 1'b1;
                    end
                end
            end
        end
    end

    wire active = busy && (cnt < len_q);
    assign o_out_rd_addr = src_q + cnt[ADDR_W-1:0];
    assign o_out_rd_en   = active;
    assign o_act_wr_addr = dst_q + cnt[ADDR_W-1:0];
    assign o_act_wr_en   = active;
    assign o_act_wr_data = i_out_rd_data;   // combinational Out read -> Act write (same cycle)
    assign o_busy        = busy;
endmodule
