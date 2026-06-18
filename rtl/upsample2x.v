// Filename: upsample2x.v
// -------------------------------------------------------------------
// Shared Act-SRAM nearest-neighbor 2x upsample engine.
//
// Layout matches the NPU activation tile-major layout:
//   addr = base + ic_group * H*W + y*W + x
//
// Each 128-bit word is one spatial position for one 16-channel IC group.
// The engine reads a source word from Act SRAM Port B, then writes it to the
// four corresponding destination positions in a 2x larger tensor. Act Port B
// is shared and assumed combinational-read (COMB_B=1), same as img_expand.
// -------------------------------------------------------------------
module upsample2x #(
    parameter ADDR_W = 14,
    parameter DATA_W = 128
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                i_trig,
    input  wire [ADDR_W-1:0]   i_src_base,
    input  wire [ADDR_W-1:0]   i_dst_base,
    input  wire [15:0]         i_in_w,
    input  wire [15:0]         i_in_h,
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
    localparam S_READ  = 2'd1;
    localparam S_WRITE = 2'd2;

    reg [1:0]        state;
    reg [ADDR_W-1:0] src_q, dst_q;
    reg [15:0]       in_w_q, in_h_q, ic_groups_q;
    reg [15:0]       g, y, x;
    reg [1:0]        wr_phase;
    reg [DATA_W-1:0] word_q;

    wire [31:0] spatial      = in_w_q * in_h_q;
    wire [31:0] out_w        = {15'd0, in_w_q, 1'b0};
    wire [31:0] out_spatial  = spatial << 2;
    wire [31:0] src_offset   = g * spatial + y * in_w_q + x;
    wire [31:0] dst_y        = {15'd0, y, 1'b0} + wr_phase[1];
    wire [31:0] dst_x        = {15'd0, x, 1'b0} + wr_phase[0];
    wire [31:0] dst_offset   = g * out_spatial + dst_y * out_w + dst_x;
    wire        last_src     = (g == ic_groups_q - 16'd1) &&
                               (y == in_h_q - 16'd1) &&
                               (x == in_w_q - 16'd1);

    assign o_busy  = (state != S_IDLE);
    assign o_en    = (state == S_READ) || (state == S_WRITE);
    assign o_we    = (state == S_WRITE);
    assign o_addr  = (state == S_WRITE)
                   ? dst_q + dst_offset[ADDR_W-1:0]
                   : src_q + src_offset[ADDR_W-1:0];
    assign o_wdata = word_q;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            src_q <= {ADDR_W{1'b0}};
            dst_q <= {ADDR_W{1'b0}};
            in_w_q <= 16'd0;
            in_h_q <= 16'd0;
            ic_groups_q <= 16'd0;
            g <= 16'd0;
            y <= 16'd0;
            x <= 16'd0;
            wr_phase <= 2'd0;
            word_q <= {DATA_W{1'b0}};
            o_done <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (i_trig) begin
                        src_q <= i_src_base;
                        dst_q <= i_dst_base;
                        in_w_q <= i_in_w;
                        in_h_q <= i_in_h;
                        ic_groups_q <= i_ic_groups;
                        g <= 16'd0;
                        y <= 16'd0;
                        x <= 16'd0;
                        wr_phase <= 2'd0;
                        o_done <= 1'b0;
                        state <= S_READ;
                    end
                end

                S_READ: begin
                    word_q <= i_rdata;
                    wr_phase <= 2'd0;
                    state <= S_WRITE;
                end

                S_WRITE: begin
                    if (wr_phase == 2'd3) begin
                        if (last_src) begin
                            state <= S_IDLE;
                            o_done <= 1'b1;
                        end else begin
                            state <= S_READ;
                            if (x + 16'd1 < in_w_q) begin
                                x <= x + 16'd1;
                            end else begin
                                x <= 16'd0;
                                if (y + 16'd1 < in_h_q) begin
                                    y <= y + 16'd1;
                                end else begin
                                    y <= 16'd0;
                                    g <= g + 16'd1;
                                end
                            end
                        end
                    end
                    wr_phase <= wr_phase + 2'd1;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
