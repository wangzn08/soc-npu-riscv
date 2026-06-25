// Filename: eltwise_add_engine.v
// -------------------------------------------------------------------
// Generic Act-SRAM signed INT8 eltwise add engine.
//
// Layout is a flat sequence of 128-bit words. For each word:
//   out = sat_s8( s8(src1) + term(src0,zp,ratio) )
//   term = ratio_en ? round((s8(src0)-s8(zp))*mul >> shift)
//                   : s8(src0)-s8(zp)
//
// This is a standalone SRAM post-process/data-movement engine, not tied to
// any network topology. C2f uses src0=skip/prev and src1=conv result.
// -------------------------------------------------------------------
module eltwise_add_engine #(
    parameter ADDR_W = 14,
    parameter DATA_W = 128
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                i_trig,
    input  wire [ADDR_W-1:0]   i_src0_base,
    input  wire [ADDR_W-1:0]   i_src1_base,
    input  wire [ADDR_W-1:0]   i_dst_base,
    input  wire [15:0]         i_len,
    input  wire [7:0]          i_zp,
    input  wire                i_ratio_en,
    input  wire [16:0]         i_ratio_mul,
    input  wire [5:0]          i_ratio_shift,
    output wire [ADDR_W-1:0]   o_addr,
    output wire                o_en,
    output wire                o_we,
    output wire [DATA_W-1:0]   o_wdata,
    input  wire [DATA_W-1:0]   i_rdata,
    output wire                o_busy,
    output reg                 o_done
);
    localparam S_IDLE  = 2'd0;
    localparam S_READ0 = 2'd1;
    localparam S_READ1 = 2'd2;
    localparam S_WRITE = 2'd3;

    reg [1:0] state;
    reg [ADDR_W-1:0] src0_q, src1_q, dst_q;
    reg [15:0] len_q, idx;
    reg [7:0] zp_q;
    reg ratio_en_q;
    reg [16:0] ratio_mul_q;
    reg [5:0] ratio_shift_q;
    reg [DATA_W-1:0] src0_word, out_word;

    assign o_busy  = (state != S_IDLE);
    assign o_en    = (state != S_IDLE);
    assign o_we    = (state == S_WRITE);
    assign o_addr  = (state == S_READ0) ? src0_q + idx[ADDR_W-1:0] :
                     (state == S_READ1) ? src1_q + idx[ADDR_W-1:0] :
                                          dst_q  + idx[ADDR_W-1:0];
    assign o_wdata = out_word;

    integer lane;
    reg signed [9:0] skip_minus_zp;
    reg signed [27:0] prod;
    reg signed [27:0] rounded;
    reg signed [11:0] term;
    reg signed [11:0] sum;
    reg signed [7:0] src0_s;
    reg signed [7:0] src1_s;
    reg signed [7:0] zp_s;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            src0_q <= {ADDR_W{1'b0}};
            src1_q <= {ADDR_W{1'b0}};
            dst_q <= {ADDR_W{1'b0}};
            len_q <= 16'd0;
            idx <= 16'd0;
            zp_q <= 8'd0;
            ratio_en_q <= 1'b0;
            ratio_mul_q <= 17'd1;
            ratio_shift_q <= 6'd0;
            src0_word <= {DATA_W{1'b0}};
            out_word <= {DATA_W{1'b0}};
            o_done <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (i_trig) begin
                        src0_q <= i_src0_base;
                        src1_q <= i_src1_base;
                        dst_q <= i_dst_base;
                        len_q <= i_len;
                        idx <= 16'd0;
                        zp_q <= i_zp;
                        ratio_en_q <= i_ratio_en;
                        ratio_mul_q <= i_ratio_mul;
                        ratio_shift_q <= i_ratio_shift;
                        o_done <= 1'b0;
                        state <= (i_len == 16'd0) ? S_IDLE : S_READ0;
                        if (i_len == 16'd0) o_done <= 1'b1;
                    end
                end

                S_READ0: begin
                    src0_word <= i_rdata;
                    state <= S_READ1;
                end

                S_READ1: begin
                    for (lane = 0; lane < 16; lane = lane + 1) begin
                        src0_s = src0_word[lane*8 +: 8];
                        src1_s = i_rdata[lane*8 +: 8];
                        zp_s   = zp_q;
                        skip_minus_zp = $signed({{2{src0_s[7]}}, src0_s})
                                      - $signed({{2{zp_s[7]}},   zp_s});
                        prod = skip_minus_zp * $signed({1'b0, ratio_mul_q});
                        if (ratio_en_q) begin
                            rounded = (ratio_shift_q == 6'd0)
                                    ? prod
                                    : prod + (28'sd1 <<< (ratio_shift_q - 6'd1));
                            term = rounded >>> ratio_shift_q;
                        end else begin
                            term = {{2{skip_minus_zp[9]}}, skip_minus_zp};
                        end
                        sum = $signed({{4{src1_s[7]}}, src1_s}) + term;
                        if (sum > 12'sd127)
                            out_word[lane*8 +: 8] <= 8'h7F;
                        else if (sum < -12'sd128)
                            out_word[lane*8 +: 8] <= 8'h80;
                        else
                            out_word[lane*8 +: 8] <= sum[7:0];
                    end
                    state <= S_WRITE;
                end

                S_WRITE: begin
                    if (idx + 16'd1 == len_q) begin
                        o_done <= 1'b1;
                        state <= S_IDLE;
                    end else begin
                        idx <= idx + 16'd1;
                        state <= S_READ0;
                    end
                end
            endcase
        end
    end
endmodule
