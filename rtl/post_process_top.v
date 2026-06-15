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
    input  wire [7:0]                       i_clip_max,   // upper clamp (default 127 = ReLU; ReLU6 = q(6.0))

    // Pooling state control (from FSM)
    input  wire                             i_start,      // op-start pulse: reset pool phase
    input  wire                             i_in_drain,
    input  wire                             i_in_post,

    // Row-parallel (task E): the drain delivers 16 distinct pixels (one per array
    // row), reverse column order (row 15 first). For pooling we reorder them to
    // row-major and replay group_size pixels into the pooler.
    input  wire                             i_row_par_en,
    input  wire [15:0]                      i_group_size,
    input  wire [3:0]                       i_rows_per_grp,   // #4: R rows packed (1 = byte-identical)
    input  wire [1:0]                       i_oc_tile,        // decision P: active OC-tile (0 ⇒ legacy)
    output wire [1:0]                       o_pool_tile,      // decision P: tile of o_feat (pool), aligned w/ o_feat_vld
    output wire                             o_rp_pool_done,   // replay complete (FSM advance)

    // Output feature data
    output wire [DATA_W-1:0]                o_feat,
    output wire                             o_feat_vld,

    // Decision Q: raw INT32 output — the scaled, un-clamped stage-2 result
    // (16 channels × 32 bit), delay-matched to o_feat/o_feat_vld. Used by the
    // int32 write sequencer in npu_top for final-classifier FC (FC2).
    output wire [NUM_OC*PSUM_WIDTH-1:0]     o_feat32
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
            wire is_gt   = (s2_val > {24'd0, i_clip_max});  // exceeds configurable upper clamp

            wire [ACT_WIDTH-1:0] act_val;
            // When relu_en: negative→0, >clip_max→clip_max, else pass through
            // When !relu_en: only clamp >clip_max→clip_max, keep negative values
            // clip_max defaults to 127 (legacy ReLU); set lower for ReLU6.
            assign act_val = (i_relu_en && is_neg) ? {ACT_WIDTH{1'b0}} :
                             is_gt                 ? i_clip_max :
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
    reg              in_post_d;        // delayed i_in_post, for S_POST-entry edge

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pool_gated_data   <= {DATA_W{1'b0}};
            pool_gated_vld    <= 1'b0;
            pool_data_latched <= 1'b0;
            in_post_d         <= 1'b0;
        end else begin
            // Latch the conv result once, on the first post-processed value
            // of the drain phase; hold it stable through S_POST.
            if (i_in_drain && s3_vld && !pool_data_latched) begin
                pool_gated_data   <= s3_act;
                pool_data_latched <= 1'b1;
            end
            if (!i_in_drain)
                pool_data_latched <= 1'b0;
            // Feed the pooler EXACTLY ONCE per conv point: a single-cycle
            // pulse on the rising edge of i_in_post (S_POST entry).  The old
            // "s3_vld && i_in_post" stayed high for ~4 cycles and fed the
            // pooler ~4x per point, corrupting the 2x2 windowing.
            in_post_d      <= i_in_post;
            pool_gated_vld <= i_in_post && !in_post_d;
        end
    end

    // -------------------------------------------------------------------
    // Row-parallel pool feed (task E): the drain delivers 16 distinct pixels
    // in REVERSE column order (drain cycle k -> array row 15-k -> output column
    // group_base + 15 - k).  Capture all 16 into rp_buf indexed by column-within-
    // group (15-k), then REPLAY group_size of them row-major (col 0,1,..) into the
    // pooler so its row-major 2x2 windowing works unchanged.  Capture is armed at
    // DRAIN start (like the non-pool sequencer) because the post-process pipeline
    // emits most valids during S_DRAIN, before S_POST.
    //
    // POOLER BOUNDARY ASSUMPTION: 2x2 pooling pairs even/odd columns, so a group
    // boundary must land on an EVEN column or a pair would split across groups.
    // Non-final groups are always 16 (even); the final group = out_w mod 16. All
    // pooled layers here have even out_w (28=16+12, 16, 8) so every boundary is even
    // -> safe. A pooled layer with ODD out_w would need the split aligned to an even
    // column (not just 16) — out of scope for the current model.
    // -------------------------------------------------------------------
    reg [DATA_W-1:0] rp_buf [0:15];
    reg [4:0]        rp_cap_cnt;
    reg              rp_cap_active;
    reg              rp_in_drain_d;
    reg [4:0]        rp_play_cnt;
    reg              rp_play_active;
    wire [19:0]      rp_play_total = i_group_size * {16'd0, i_rows_per_grp}; // R*group_size pixels
    reg [DATA_W-1:0] rp_play_data;
    reg              rp_play_vld;
    reg              rp_play_done_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rp_cap_cnt     <= 5'd0;
            rp_cap_active  <= 1'b0;
            rp_in_drain_d  <= 1'b0;
            rp_play_cnt    <= 5'd0;
            rp_play_active <= 1'b0;
            rp_play_data   <= {DATA_W{1'b0}};
            rp_play_vld    <= 1'b0;
            rp_play_done_r <= 1'b0;
        end else begin
            rp_in_drain_d  <= i_in_drain;
            rp_play_done_r <= 1'b0;     // default: 1-cycle pulse

            // ---- Capture: arm at drain start, store 16 drained pixels ----
            if (i_in_drain && !rp_in_drain_d) begin
                rp_cap_cnt    <= 5'd0;
                rp_cap_active <= i_row_par_en;
            end else if (rp_cap_active && s3_vld) begin
                rp_buf[5'd15 - rp_cap_cnt] <= s3_act;   // column-within-group = 15-k
                if (rp_cap_cnt == 5'd15)
                    rp_cap_active <= 1'b0;
                rp_cap_cnt <= rp_cap_cnt + 5'd1;
            end

            // ---- Replay: after the 16th capture, stream group_size row-major ----
            if (rp_cap_active && s3_vld && (rp_cap_cnt == 5'd15)) begin
                rp_play_active <= 1'b1;
                rp_play_cnt    <= 5'd0;
                rp_play_vld    <= 1'b0;
            end else if (rp_play_active) begin
                rp_play_data <= rp_buf[rp_play_cnt];
                rp_play_vld  <= 1'b1;
                // #4 row-block: replay R*group_size pixels (R rows row-major:
                // rp_buf[r] index = b*group_size+c = r). R=1 → group_size (legacy).
                if (rp_play_cnt == rp_play_total[4:0] - 5'd1) begin
                    rp_play_active <= 1'b0;
                    rp_play_done_r <= 1'b1;
                end
                rp_play_cnt <= rp_play_cnt + 5'd1;
            end else begin
                rp_play_vld <= 1'b0;
            end
        end
    end

    assign o_rp_pool_done = rp_play_done_r;

    // Pooler feed: row-par replay vs legacy single-latch gating.
    wire [DATA_W-1:0] pool_feat_in     = i_row_par_en ? rp_play_data : pool_gated_data;
    wire              pool_feat_vld_in = i_row_par_en ? rp_play_vld  : pool_gated_vld;

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
        .i_start     (i_start),
        .i_feat      (pool_feat_in),
        .i_feat_vld  (pool_feat_vld_in),
        .i_width     (i_width),
        .i_tile      (i_oc_tile),   // decision P: active OC-tile (oc_single OC-inner loop)
        .o_pool      (pool_out),
        .o_pool_vld  (pool_vld),
        .o_pool_tile (o_pool_tile)
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

    // -------------------------------------------------------------------
    // Decision Q: raw INT32 output. s2_quant is the scaled, un-clamped 32-bit
    // result (= CPU's (psum+bias)*scale>>shift). Mirror the INT8 bypass delay
    // (s3 register + POOL_LATENCY) so o_feat32 aligns with o_feat_vld. Used only
    // in int32_out mode (GEMM/final-FC, non-pool); off ⇒ unused ⇒ byte-identical.
    // -------------------------------------------------------------------
    wire [NUM_OC*PSUM_WIDTH-1:0] s2_flat;
    generate
        for (gi = 0; gi < NUM_OC; gi = gi + 1) begin : gen_s2flat
            assign s2_flat[gi*PSUM_WIDTH +: PSUM_WIDTH] = s2_quant[gi];
        end
    endgenerate

    reg [NUM_OC*PSUM_WIDTH-1:0] s3_i32;                       // mirrors s3_act timing
    reg [NUM_OC*PSUM_WIDTH-1:0] i32_dly [0:POOL_LATENCY-1];   // mirrors bypass_dly
    integer di;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3_i32 <= {NUM_OC*PSUM_WIDTH{1'b0}};
            for (di = 0; di < POOL_LATENCY; di = di + 1)
                i32_dly[di] <= {NUM_OC*PSUM_WIDTH{1'b0}};
        end else begin
            if (s2_vld) s3_i32 <= s2_flat;        // == s3_act register stage
            i32_dly[0] <= s3_i32;
            i32_dly[1] <= i32_dly[0];
            i32_dly[2] <= i32_dly[1];
        end
    end
    assign o_feat32 = i32_dly[POOL_LATENCY-1];

endmodule
