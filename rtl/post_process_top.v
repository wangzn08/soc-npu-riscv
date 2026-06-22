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
    input  wire                             i_pool_avg,   // 0=max pool (legacy), 1=2x2 average pool
    input  wire                             i_relu_en,
    input  wire                             i_silu_en,    // YOLO SiLU LUT path (default-off via CTRL[18])
    input  wire                             i_silu_requant_en, // optional SiLU Q4.4 -> output INT8 requant
    input  wire [15:0]                      i_silu_requant_mul,
    input  wire [5:0]                       i_silu_requant_shift,
    input  wire [7:0]                       i_silu_requant_zp,
    input  wire [7:0]                       i_clip_max,   // upper clamp (default 127 = ReLU; ReLU6 = q(6.0))
    input  wire                             i_sigmoid_en, // CTRL[21]: sigmoid LUT activation (detect-head cls)
    input  wire                             i_sigm_load_en, // runtime sigmoid LUT load (per scale)
    input  wire [7:0]                       i_sigm_load_idx,
    input  wire [7:0]                       i_sigm_load_val,
    input  wire                             i_silu_exact_en, // CTRL[22]: per-layer exact SiLU LUT (out-grid indexed)
    input  wire                             i_silu_load_en,  // runtime exact-SiLU LUT load (per layer)
    input  wire [7:0]                       i_silu_load_idx,
    input  wire [7:0]                       i_silu_load_val,

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

    // INT32 psum accumulate modes (IC-chunk streaming): a conv's IC dimension
    // can be split across multiple NPU passes and accumulated in INT32.
    //   0 = NONE  : legacy single-pass (bit-identical when tied to 0)
    //   1 = FIRST : first IC chunk -> o_feat32 = raw i_psum (no bias, no readback)
    //   2 = ADD   : middle IC chunk -> o_feat32 = i_psum + i_psum_readback (no bias)
    //   3 = FINAL : last IC chunk  -> normal requant+SiLU on (i_psum +
    //               i_psum_readback + bias) -> o_feat (INT8)
    // i_psum_readback carries the prior accumulated INT32 (one per OC lane),
    // read back from Out SRAM by the FSM (wired in a later task).
    input  wire [1:0]                       i_acc_mode,
    input  wire [NUM_OC*PSUM_WIDTH-1:0]     i_psum_readback,

    // Decision Q: raw INT32 output — the scaled, un-clamped stage-2 result
    // (16 channels × 32 bit), delay-matched to o_feat/o_feat_vld. Used by the
    // int32 write sequencer in npu_top for final-classifier FC (FC2).
    output wire [NUM_OC*PSUM_WIDTH-1:0]     o_feat32
);

    localparam [1:0] ACC_NONE  = 2'd0;
    localparam [1:0] ACC_FIRST = 2'd1;
    localparam [1:0] ACC_ADD   = 2'd2;
    localparam [1:0] ACC_FINAL = 2'd3;

    // acc_in[oc] = i_psum[oc] + (ADD/FINAL ? readback : 0). NONE selects the
    // plain i_psum so the rest of the pipeline is bit-identical to legacy.
    wire add_readback = (i_acc_mode == ACC_ADD) || (i_acc_mode == ACC_FINAL);
    wire [NUM_OC-1:0][PSUM_WIDTH-1:0] acc_in;
    genvar ga;
    generate
        for (ga = 0; ga < NUM_OC; ga = ga + 1) begin : gen_acc_in
            assign acc_in[ga] = add_readback
                ? (i_psum[ga] + i_psum_readback[ga*PSUM_WIDTH +: PSUM_WIDTH])
                : i_psum[ga];
        end
    endgenerate

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
                    // acc_in == i_psum in ACC_NONE/FIRST; == i_psum+readback in
                    // ACC_ADD/FINAL. ACC_FINAL adds bias for the final requant.
                    s1_sum[gi] <= acc_in[gi] + i_bias[gi];
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

    // ACC FIRST/ADD raw INT32 sum (no bias, no requant), registered at the
    // s1 stage so it aligns with s2_quant timing. Captured every i_psum_vld
    // like s1_sum; only consumed when i_acc_mode is FIRST/ADD.
    reg [NUM_OC-1:0][PSUM_WIDTH-1:0] s1_acc_raw;
    reg                               s1_acc_int32;   // FIRST/ADD: route raw to o_feat32
    generate
        for (gi = 0; gi < NUM_OC; gi = gi + 1) begin : gen_s1acc
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    s1_acc_raw[gi] <= {PSUM_WIDTH{1'b0}};
                else if (i_psum_vld)
                    s1_acc_raw[gi] <= acc_in[gi];
            end
        end
    endgenerate
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            s1_acc_int32 <= 1'b0;
        else if (i_psum_vld)
            s1_acc_int32 <= (i_acc_mode == ACC_FIRST) || (i_acc_mode == ACC_ADD);
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
    // Stage 3: activation (legacy ReLU/clip or YOLO SiLU LUT)
    // -------------------------------------------------------------------
    localparam signed [PSUM_WIDTH-1:0] SILU_IN_MIN = -128;
    localparam signed [PSUM_WIDTH-1:0] SILU_IN_MAX =  127;

    reg [ACT_WIDTH-1:0] silu_lut [0:255];
    initial begin
        $readmemh("rtl/silu_lut_q4_4.hex", silu_lut);
    end

    // Sigmoid LUT (INT8 logit -> Q0.8 prob): boot default + runtime-loadable per
    // detect scale via NPU_SIGM_LOAD. Used by the detect-head cls path (CTRL[21]).
    reg [ACT_WIDTH-1:0] sigmoid_lut [0:255];
    initial begin
        $readmemh("rtl/sigmoid_lut_q0_8.hex", sigmoid_lut);
    end
    always @(posedge clk) begin
        if (i_sigm_load_en) sigmoid_lut[i_sigm_load_idx] <= i_sigm_load_val;
    end

    // Exact per-layer SiLU LUT (out-grid INT8 -> out-grid INT8 SiLU): runtime-
    // loaded per layer via NPU_SILU_LOAD. Indexed by the LINEAR output-quantized
    // preact (the same value the linear-requant path produces, lin_requant_val),
    // so it never saturates the fixed Q4.4 (+/-8) range of the legacy silu_lut.
    // No boot default: must be loaded before i_silu_exact_en is used.
    reg [ACT_WIDTH-1:0] silu_exact_lut [0:255];
    always @(posedge clk) begin
        if (i_silu_load_en) silu_exact_lut[i_silu_load_idx] <= i_silu_load_val;
    end

    reg [DATA_W-1:0] s3_act;
    reg              s3_vld;

    generate
        for (gi = 0; gi < NUM_OC; gi = gi + 1) begin : gen_s3
            wire [PSUM_WIDTH-1:0] s2_val = s2_quant[gi];
            wire is_neg  = s2_val[PSUM_WIDTH-1];  // sign bit
            wire is_gt   = (s2_val > {24'd0, i_clip_max});  // exceeds configurable upper clamp
            wire signed [PSUM_WIDTH-1:0] s2_signed = $signed(s2_quant[gi]);
            wire signed [7:0] silu_sat = (s2_signed < SILU_IN_MIN) ? 8'sh80 :
                                         (s2_signed > SILU_IN_MAX) ? 8'sh7f :
                                                                     s2_signed[7:0];
            wire [ACT_WIDTH-1:0] silu_val = silu_lut[silu_sat[7:0]];
            wire [ACT_WIDTH-1:0] sigmoid_val = sigmoid_lut[silu_sat[7:0]];
            wire signed [7:0] silu_signed = $signed(silu_val);
            wire signed [31:0] silu_rq_prod = $signed({{24{silu_signed[7]}}, silu_signed})
                                            * $signed({16'd0, i_silu_requant_mul});
            wire signed [31:0] silu_rq_shifted = silu_rq_prod >>> i_silu_requant_shift;
            wire signed [31:0] silu_rq_biased = silu_rq_shifted + $signed({{24{i_silu_requant_zp[7]}}, i_silu_requant_zp});
            wire [ACT_WIDTH-1:0] silu_requant_val =
                (silu_rq_biased < -32'sd128) ? 8'h80 :
                (silu_rq_biased >  32'sd127) ? 8'h7f :
                                                silu_rq_biased[7:0];

            // Linear-output requant (silu_requant_en && !silu_en): for LINEAR
            // quantized convs (detect-head outputs, has_silu=0). Stage-2 scale_mul/
            // shift are set so s2_quant == round(real_preact / out_scale); here we
            // just add the output zero-point (reusing i_silu_requant_zp) and clamp
            // to signed INT8. Bypasses the SiLU LUT and its Q4.4 (+/-8) clamp.
            wire signed [PSUM_WIDTH-1:0] lin_biased =
                s2_signed + $signed({{(PSUM_WIDTH-8){i_silu_requant_zp[7]}}, i_silu_requant_zp});
            wire [ACT_WIDTH-1:0] lin_requant_val =
                (lin_biased < -32'sd128) ? 8'h80 :
                (lin_biased >  32'sd127) ? 8'h7f :
                                            lin_biased[ACT_WIDTH-1:0];

            // Exact SiLU: index the per-layer loadable LUT by the linear output-
            // quantized preact (lin_requant_val). The LUT maps that out-grid INT8
            // to round(SiLU(dequant)/out_scale + out_zp). Exact to INT8, no +/-8 clamp.
            wire [ACT_WIDTH-1:0] silu_exact_out = silu_exact_lut[lin_requant_val];

            wire [ACT_WIDTH-1:0] act_val;
            // When relu_en: negative→0, >clip_max→clip_max, else pass through
            // When !relu_en: only clamp >clip_max→clip_max, keep negative values
            // clip_max defaults to 127 (legacy ReLU); set lower for ReLU6.
            assign act_val = i_silu_exact_en       ? silu_exact_out :
                             i_sigmoid_en          ? sigmoid_val :
                             (i_silu_en && i_silu_requant_en) ? silu_requant_val :
                             i_silu_en             ? silu_val :
                             i_silu_requant_en     ? lin_requant_val :
                             (i_relu_en && is_neg) ? {ACT_WIDTH{1'b0}} :
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
        .i_avg       (i_pool_avg),  // CTRL[16]: 2x2 average vs max
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
    // Align the raw-acc INT32 value (FIRST/ADD) to the s2 stage: s2_quant is
    // captured from s1 on s1_vld, so register s1_acc_raw/flag forward one
    // stage with the same enable to land in the same cycle as s2_quant.
    reg [NUM_OC-1:0][PSUM_WIDTH-1:0] s2_acc_raw;
    reg                               s2_acc_int32;
    generate
        for (gi = 0; gi < NUM_OC; gi = gi + 1) begin : gen_s2acc
            always @(posedge clk) begin
                if (s1_vld) s2_acc_raw[gi] <= s1_acc_raw[gi];
            end
        end
    endgenerate
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)       s2_acc_int32 <= 1'b0;
        else if (s1_vld)  s2_acc_int32 <= s1_acc_int32;
    end

    // o_feat32 source: requant (s2_quant) by default; raw acc sum in FIRST/ADD.
    wire [NUM_OC*PSUM_WIDTH-1:0] s2_flat;
    generate
        for (gi = 0; gi < NUM_OC; gi = gi + 1) begin : gen_s2flat
            assign s2_flat[gi*PSUM_WIDTH +: PSUM_WIDTH] =
                s2_acc_int32 ? s2_acc_raw[gi] : s2_quant[gi];
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
