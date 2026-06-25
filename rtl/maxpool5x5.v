// Filename: maxpool5x5.v
// -------------------------------------------------------------------
// Generic Act-SRAM 5x5 stride-1 signed INT8 maxpool engine.
//
// Layout matches the NPU activation tile-major layout:
//   addr = base + ic_group * H*W + y*W + x
//
// Each 128-bit word contains 16 signed INT8 lanes. Border locations outside
// the input tensor are treated as -128, so the output has the same H/W.
// This is a generic SRAM data-movement/post-process engine, not tied to YOLO.
// -------------------------------------------------------------------
module maxpool5x5 #(
    parameter ADDR_W = 14,
    parameter DATA_W = 128
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                i_trig,
    input  wire [ADDR_W-1:0]   i_src_base,
    input  wire [ADDR_W-1:0]   i_dst_base,
    input  wire [15:0]         i_w,
    input  wire [15:0]         i_h,
    input  wire [15:0]         i_ic_groups,
    output wire [ADDR_W-1:0]   o_addr,
    output wire                o_en,
    output wire                o_we,
    output wire [DATA_W-1:0]   o_wdata,
    input  wire [DATA_W-1:0]   i_rdata,
    output wire                o_busy,
    output reg                 o_done
);
    localparam S_IDLE  = 2'd0;
    localparam S_SCAN  = 2'd1;
    localparam S_WRITE = 2'd2;

    reg [1:0]        state;
    reg [ADDR_W-1:0] src_q, dst_q;
    reg [15:0]       w_q, h_q, groups_q;
    reg [15:0]       g, y, x;
    reg [2:0]        ky, kx;
    reg [DATA_W-1:0] max_word;

    wire signed [17:0] win_y_s = $signed({1'b0, y}) + $signed({15'd0, ky}) - 18'sd2;
    wire signed [17:0] win_x_s = $signed({1'b0, x}) + $signed({15'd0, kx}) - 18'sd2;
    wire               in_bounds = (win_y_s >= 0) && (win_y_s < $signed({1'b0, h_q})) &&
                                   (win_x_s >= 0) && (win_x_s < $signed({1'b0, w_q}));
    wire [31:0]        spatial = w_q * h_q;
    wire [31:0]        rd_offset = g * spatial + win_y_s[15:0] * w_q + win_x_s[15:0];
    wire [31:0]        wr_offset = g * spatial + y * w_q + x;
    wire               last_tap = (ky == 3'd4) && (kx == 3'd4);
    wire               last_out = (g == groups_q - 16'd1) &&
                                  (y == h_q - 16'd1) &&
                                  (x == w_q - 16'd1);

    assign o_busy  = (state != S_IDLE);
    assign o_en    = (state == S_SCAN && in_bounds) || (state == S_WRITE);
    assign o_we    = (state == S_WRITE);
    assign o_addr  = (state == S_WRITE)
                   ? dst_q + wr_offset[ADDR_W-1:0]
                   : src_q + rd_offset[ADDR_W-1:0];
    assign o_wdata = max_word;

    integer lane;
    reg signed [7:0] cur_lane;
    reg signed [7:0] max_lane;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            src_q <= {ADDR_W{1'b0}};
            dst_q <= {ADDR_W{1'b0}};
            w_q <= 16'd0;
            h_q <= 16'd0;
            groups_q <= 16'd0;
            g <= 16'd0;
            y <= 16'd0;
            x <= 16'd0;
            ky <= 3'd0;
            kx <= 3'd0;
            max_word <= {16{8'h80}};
            o_done <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (i_trig) begin
                        src_q <= i_src_base;
                        dst_q <= i_dst_base;
                        w_q <= i_w;
                        h_q <= i_h;
                        groups_q <= i_ic_groups;
                        g <= 16'd0;
                        y <= 16'd0;
                        x <= 16'd0;
                        ky <= 3'd0;
                        kx <= 3'd0;
                        max_word <= {16{8'h80}};
                        o_done <= 1'b0;
                        state <= S_SCAN;
                    end
                end

                S_SCAN: begin
                    if (in_bounds) begin
                        for (lane = 0; lane < 16; lane = lane + 1) begin
                            cur_lane = i_rdata[lane*8 +: 8];
                            max_lane = max_word[lane*8 +: 8];
                            if (cur_lane > max_lane)
                                max_word[lane*8 +: 8] <= cur_lane[7:0];
                        end
                    end

                    if (last_tap) begin
                        state <= S_WRITE;
                    end else if (kx == 3'd4) begin
                        kx <= 3'd0;
                        ky <= ky + 3'd1;
                    end else begin
                        kx <= kx + 3'd1;
                    end
                end

                S_WRITE: begin
                    if (last_out) begin
                        state <= S_IDLE;
                        o_done <= 1'b1;
                    end else begin
                        state <= S_SCAN;
                        ky <= 3'd0;
                        kx <= 3'd0;
                        max_word <= {16{8'h80}};
                        if (x + 16'd1 < w_q) begin
                            x <= x + 16'd1;
                        end else begin
                            x <= 16'd0;
                            if (y + 16'd1 < h_q) begin
                                y <= y + 16'd1;
                            end else begin
                                y <= 16'd0;
                                g <= g + 16'd1;
                            end
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
