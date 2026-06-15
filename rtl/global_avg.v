// Filename: global_avg.v
// -------------------------------------------------------------------
// Global Average Pooling accumulator (CTRL[17] gpool_en).
//
// Accumulates the per-position INT8 activations (NUM_CH channels) of one
// OC tile into NUM_CH INT32 channel sums, then on i_last emits a single
// requantized word holding each channel's mean.  The mean divide by the
// spatial count N (= out_w*out_h) is done as a reciprocal multiply+shift
// (i_avg_mul / i_avg_shift, set by firmware = round(2^shift / N)), so any
// N is supported without a hardware divider.
//
// gpool_en off ⇒ this block is bypassed in npu_top ⇒ byte-identical.
//
// 全局平均池化累加器：把一个 OC tile 内所有空间位置的 16 通道 INT8 激活
// 累加为 16 个 INT32 通道和；i_last 时用倒数乘移位 (mul>>shift) 实现 /N,
// 输出一个词(16 通道均值,INT8)。
// -------------------------------------------------------------------
module global_avg #(
    parameter NUM_CH     = 16,
    parameter ACT_WIDTH  = 8,
    parameter PSUM_WIDTH = 32,
    parameter DATA_W     = NUM_CH * ACT_WIDTH   // 128
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  i_start,      // clear accumulators & count (OC-tile start)
    input  wire [DATA_W-1:0]     i_feat,       // per-position INT8 activations (NUM_CH ch)
    input  wire                  i_feat_vld,
    input  wire                  i_last,       // last spatial position of this OC tile
    input  wire [PSUM_WIDTH-1:0] i_avg_mul,    // reciprocal multiplier  = round(2^shift / N)
    input  wire [5:0]            i_avg_shift,  // reciprocal shift
    output reg  [DATA_W-1:0]     o_feat,       // requantized mean (NUM_CH ch INT8)
    output reg                   o_feat_vld    // single-cycle pulse after i_last
);

    reg [PSUM_WIDTH-1:0] acc [0:NUM_CH-1];

    integer c;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (c = 0; c < NUM_CH; c = c + 1) acc[c] <= {PSUM_WIDTH{1'b0}};
            o_feat     <= {DATA_W{1'b0}};
            o_feat_vld <= 1'b0;
        end else begin
            o_feat_vld <= 1'b0;   // default: single-cycle pulse

            if (i_start) begin
                for (c = 0; c < NUM_CH; c = c + 1) acc[c] <= {PSUM_WIDTH{1'b0}};
            end else if (i_feat_vld) begin
                for (c = 0; c < NUM_CH; c = c + 1) begin
                    // activations are post-ReLU unsigned [0,127]; zero-extend
                    acc[c] <= acc[c] + {{(PSUM_WIDTH-ACT_WIDTH){1'b0}},
                                        i_feat[c*ACT_WIDTH +: ACT_WIDTH]};
                end
            end

            // Emit on the last position (combine this position into the sum first).
            if (i_feat_vld && i_last) begin
                for (c = 0; c < NUM_CH; c = c + 1) begin : gen_emit
                    reg [PSUM_WIDTH-1:0] sum_c;
                    reg [63:0]           prod_c;
                    reg [PSUM_WIDTH-1:0] mean_c;
                    sum_c  = acc[c] + {{(PSUM_WIDTH-ACT_WIDTH){1'b0}},
                                       i_feat[c*ACT_WIDTH +: ACT_WIDTH]};
                    prod_c = sum_c * i_avg_mul;
                    mean_c = prod_c >> i_avg_shift;
                    // clamp to INT8 [0,127] (mean of [0,127] inputs is in range;
                    // cap guards reciprocal rounding).
                    o_feat[c*ACT_WIDTH +: ACT_WIDTH] <= (mean_c > 32'd127)
                                                        ? 8'd127 : mean_c[ACT_WIDTH-1:0];
                end
                o_feat_vld <= 1'b1;
            end
        end
    end

endmodule
