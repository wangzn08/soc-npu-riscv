// Filename: post_process_top.v
// -------------------------------------------------------------------
// Post-Processing Top — four-stage pipeline:
//   S1: PSUM + Bias  → 32-bit sum
//   S2: Quantization → (sum * scale_mul) >>> scale_shift, clamp INT8
//   S3: ReLU         → max(0, val), cap at 127
//   S4: MaxPool 2×2  → optional spatial down-sample
//
// 16 parallel output channels, each independently parametrized.
// -------------------------------------------------------------------

module post_process_top #(
    parameter NUM_OC        = 16,
    parameter PSUM_WIDTH    = 32,
    parameter SCALE_WIDTH   = 32,
    parameter SHIFT_WIDTH   = 6,
    parameter ACT_WIDTH     = 8,
    parameter DATA_W        = NUM_OC * ACT_WIDTH,    // 128
    parameter MAX_WIDTH     = 256
) (
    input  wire                             clk,
    input  wire                             rst_n,

    // Partial sums from systolic array columns
    input  wire [NUM_OC-1:0][PSUM_WIDTH-1:0] i_psum,
    input  wire                             i_psum_vld,

    // Per-output-channel parameters (from param_regfile)
    input  wire [NUM_OC-1:0][PSUM_WIDTH-1:0] i_bias,
    input  wire [NUM_OC-1:0][SCALE_WIDTH-1:0] i_scale_mul,
    input  wire [NUM_OC-1:0][SHIFT_WIDTH-1:0] i_scale_shift,

    // Feature map geometry (for pooling)
    input  wire [15:0]                      i_width,
    input  wire                             i_pool_en,
    input  wire                             i_relu_en,

    // Pooling state control (from FSM)
    input  wire                             i_in_drain,
    input  wire                             i_in_post,

    // Output feature data
    output wire [DATA_W-1:0]                o_feat,
    output wire                             o_feat_vld
);

    // -------------------------------------------------------------------
    // Stage 1: PSUM + Bias → s1_sum
    // -------------------------------------------------------------------
    reg [NUM_OC-1:0][PSUM_WIDTH-1:0] s1_sum;
    reg                               s1_vld;

    genvar gi;
    generate
        for (gi = 0; gi < NUM_OC; gi = gi + 1) begin : gen_s1
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    s1_sum[gi] <= {PSUM_WIDTH{1'b0}};
                else if (i_psum_vld) begin
                    s1_sum[gi] <= i_psum[gi] + i_bias[gi];
                end
            end
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            s1_vld <= 1'b0;
        else
            s1_vld <= i_psum_vld;
    end

    // -------------------------------------------------------------------
    // Stage 2: Quantization — (s1_sum * scale_mul) >>> scale_shift
    //   Product is 64-bit (32b × 32b); arithmetic right-shift then clamp
    // -------------------------------------------------------------------
    reg [NUM_OC-1:0][PSUM_WIDTH-1:0] s2_quant;
    reg                               s2_vld;

    generate
        for (gi = 0; gi < NUM_OC; gi = gi + 1) begin : gen_s2
            // 32×32 → 64-bit product; keep relevant bits for shift
            wire signed [63:0] product;
            assign product = $signed(s1_sum[gi]) * $signed(i_scale_mul[gi]);

            // Arithmetic right-shift by scale_shift
            wire signed [31:0] shifted;
            assign shifted = product >>> i_scale_shift[gi];

            always @(posedge clk) begin
                if (s1_vld) begin
                    s2_quant[gi] <= shifted[PSUM_WIDTH-1:0];
                end
            end
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            s2_vld <= 1'b0;
        else
            s2_vld <= s1_vld;
    end

    // -------------------------------------------------------------------
    // Stage 3: ReLU + clamp to [0, 127]
    // -------------------------------------------------------------------
    reg [DATA_W-1:0] s3_act;
    reg              s3_vld;

    generate
        for (gi = 0; gi < NUM_OC; gi = gi + 1) begin : gen_s3
            wire [PSUM_WIDTH-1:0] s2_val = s2_quant[gi];
            wire is_neg  = s2_val[PSUM_WIDTH-1];  // sign bit
            wire is_gt127 = (s2_val > 32'd127);

            wire [ACT_WIDTH-1:0] act_val;
            // When relu_en: negative→0, >127→127, else pass through
            // When !relu_en: only clamp >127→127, keep negative values
            assign act_val = (i_relu_en && is_neg) ? {ACT_WIDTH{1'b0}} :
                             is_gt127              ? 8'd127 :
                                                     s2_val[ACT_WIDTH-1:0];

            always @(posedge clk) begin
                if (s2_vld)
                    s3_act[gi*ACT_WIDTH +: ACT_WIDTH] <= act_val;
            end
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            s3_vld <= 1'b0;
        else
            s3_vld <= s2_vld;
    end

    // -------------------------------------------------------------------
    // Pooling input gating: feed pooler only at POST time (one valid per
    // spatial position), not during DRAIN (16 valids per position).
    // -------------------------------------------------------------------
    reg [DATA_W-1:0] pool_gated_data;
    reg              pool_gated_vld;
    reg              pool_data_latched;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pool_gated_data  <= {DATA_W{1'b0}};
            pool_gated_vld   <= 1'b0;
            pool_data_latched <= 1'b0;
        end else begin
            // Latch accumulated data on FIRST drain cycle only (PE row 15 = final result)
            if (i_in_drain && s3_vld && !pool_data_latched) begin
                pool_gated_data  <= s3_act;
                pool_data_latched <= 1'b1;
            end
            // Reset latch flag when drain ends
            if (!i_in_drain)
                pool_data_latched <= 1'b0;
            // Generate single valid pulse at POST entry
            pool_gated_vld <= s3_vld && i_in_post;
        end
    end

    // -------------------------------------------------------------------
    // Stage 4: Optional 2×2 Max Pooling
    // -------------------------------------------------------------------
    wire [DATA_W-1:0] pool_out;
    wire              pool_vld;

    max_pooling_2x2 #(
        .MAX_WIDTH (MAX_WIDTH),
        .ACT_WIDTH (ACT_WIDTH),
        .NUM_CH    (NUM_OC),
        .DATA_W    (DATA_W)
    ) u_pool (
        .clk         (clk),
        .rst_n       (rst_n),
        .i_feat      (pool_gated_data),
        .i_feat_vld  (pool_gated_vld),
        .i_width     (i_width),
        .o_pool      (pool_out),
        .o_pool_vld  (pool_vld)
    );

    // Bypass mux: when pooling disabled, match latency of max_pooling_2x2
    // The pooling module adds ~3 cycles of pipeline delay.
    // We delay the direct path by the same amount.
    localparam POOL_LATENCY = 3;

    reg [DATA_W-1:0] bypass_dly [0:POOL_LATENCY-1];
    reg              bypass_vld_dly [0:POOL_LATENCY-1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bypass_dly[0]     <= {DATA_W{1'b0}};
            bypass_dly[1]     <= {DATA_W{1'b0}};
            bypass_dly[2]     <= {DATA_W{1'b0}};
            bypass_vld_dly[0] <= 1'b0;
            bypass_vld_dly[1] <= 1'b0;
            bypass_vld_dly[2] <= 1'b0;
        end else begin
            bypass_dly[0]     <= s3_act;
            bypass_dly[1]     <= bypass_dly[0];
            bypass_dly[2]     <= bypass_dly[1];
            bypass_vld_dly[0] <= s3_vld;
            bypass_vld_dly[1] <= bypass_vld_dly[0];
            bypass_vld_dly[2] <= bypass_vld_dly[1];
        end
    end

    assign o_feat     = i_pool_en ? pool_out      : bypass_dly[POOL_LATENCY-1];
    assign o_feat_vld = i_pool_en ? pool_vld      : bypass_vld_dly[POOL_LATENCY-1];

endmodule
