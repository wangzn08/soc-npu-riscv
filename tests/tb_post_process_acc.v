`timescale 1ns/1ps
// ===================================================================
// Directed test for post_process_top's INT32 psum ACCUMULATE modes.
//
// Adds i_acc_mode[1:0] (0=NONE,1=FIRST,2=ADD,3=FINAL) and
// i_psum_readback (prior accumulated INT32, one per OC lane).
//
// Drives a simple non-pool, plain-requant config (bias/scale/shift,
// relu on) and checks three accumulate cases against TB golden:
//   ACC_FIRST: o_feat32 == i_psum            (raw, no bias, no readback)
//   ACC_ADD  : o_feat32 == R + i_psum        (running INT32 sum, no bias)
//   ACC_FINAL: o_feat   == single-pass post-process of (R + i_psum)
//
// o_feat32/o_feat carry the same delay as the legacy paths, so we
// just capture the first valid INT32/INT8 output after each feed.
// ===================================================================
module tb_post_process_acc;
    localparam NUM_OC = 16;

    reg clk = 1, rst_n = 0;
    reg [NUM_OC-1:0][31:0] i_psum;
    reg                    i_psum_vld;
    reg [NUM_OC-1:0][31:0] i_bias;
    reg [NUM_OC-1:0][31:0] i_scale_mul;
    reg [NUM_OC-1:0][5:0]  i_scale_shift;
    reg [15:0]             i_width;
    reg                    i_pool_en, i_pool_avg, i_relu_en, i_silu_en, i_start, i_in_drain, i_in_post;
    reg                    i_row_par_en;
    reg [15:0]             i_group_size;
    reg [3:0]              i_rows_per_grp;
    reg [1:0]              i_oc_tile;
    reg [7:0]              i_clip_max;
    reg [1:0]              i_acc_mode;
    reg [NUM_OC-1:0][31:0] i_psum_readback;
    wire [127:0]           o_feat;
    wire                   o_feat_vld;
    wire [NUM_OC*32-1:0]   o_feat32;
    wire [1:0]             o_pool_tile;
    wire                   o_rp_pool_done;

    integer k, errors = 0;

    always #5 clk = ~clk;

    initial begin
        #100000;
        $display("  TB WATCHDOG TIMEOUT");
        $display("TB_FAIL");
        $finish;
    end

    post_process_top #(.NUM_OC(NUM_OC), .MAX_WIDTH(256)) dut (
        .clk(clk), .rst_n(rst_n),
        .i_psum(i_psum), .i_psum_vld(i_psum_vld),
        .i_bias(i_bias), .i_scale_mul(i_scale_mul), .i_scale_shift(i_scale_shift),
        .i_width(i_width), .i_pool_en(i_pool_en), .i_pool_avg(i_pool_avg),
        .i_relu_en(i_relu_en), .i_silu_en(i_silu_en),
        .i_silu_requant_en(1'b0), .i_silu_requant_mul(16'd0),
        .i_silu_requant_shift(6'd0), .i_silu_requant_zp(8'd0),
        .i_clip_max(i_clip_max),
        .i_sigmoid_en(1'b0), .i_sigm_load_en(1'b0), .i_sigm_load_idx(8'd0), .i_sigm_load_val(8'd0),
        .i_silu_exact_en(1'b0), .i_silu_load_en(1'b0), .i_silu_load_idx(8'd0), .i_silu_load_val(8'd0),
        .i_start(i_start), .i_in_drain(i_in_drain), .i_in_post(i_in_post),
        .i_row_par_en(i_row_par_en), .i_group_size(i_group_size),
        .i_rows_per_grp(i_rows_per_grp), .i_oc_tile(i_oc_tile),
        .i_acc_mode(i_acc_mode), .i_psum_readback(i_psum_readback),
        .o_pool_tile(o_pool_tile), .o_rp_pool_done(o_rp_pool_done),
        .o_feat(o_feat), .o_feat_vld(o_feat_vld), .o_feat32(o_feat32)
    );

    // ---- capture: latch first valid output after a feed ----
    reg cap_arm;
    reg [127:0] cap_feat;
    reg [NUM_OC*32-1:0] cap_feat32;
    reg cap_got;
    always @(posedge clk) if (rst_n && cap_arm && o_feat_vld && !cap_got) begin
        cap_feat   <= o_feat;
        cap_feat32 <= o_feat32;
        cap_got    <= 1'b1;
    end

    // Single non-pool conv point. Present one i_psum_vld pulse; the
    // pipeline produces one o_feat_vld POOL_LATENCY+stages later.
    task feed_point;
        begin
            @(negedge clk); i_psum_vld = 1;
            @(negedge clk); i_psum_vld = 0;
            // wait for the captured output
            repeat (60) begin
                @(negedge clk);
                if (cap_got) disable feed_point;
            end
        end
    endtask

    // golden single-pass post-process (plain requant, relu on, no pool)
    function [7:0] golden_pp;
        input signed [31:0] sum;       // acc_in + bias
        input signed [31:0] mul;
        input [5:0]         shift;
        input [7:0]         clip;
        reg signed [63:0]   prod;
        reg signed [31:0]   sh;
        begin
            prod = sum * mul;
            sh   = prod >>> shift;
            // relu_en path: neg->0, >clip->clip, else low byte
            if (sh[31])
                golden_pp = 8'd0;
            else if (sh > {24'd0, clip})
                golden_pp = clip;
            else
                golden_pp = sh[7:0];
        end
    endfunction

    integer R0;
    reg [31:0] psum_v [0:NUM_OC-1];
    reg [31:0] rb_v   [0:NUM_OC-1];

    initial begin
        i_psum_vld = 0; i_in_drain = 0; i_in_post = 0; i_start = 0;
        i_pool_en = 0; i_pool_avg = 0; i_relu_en = 1; i_silu_en = 0; i_width = 16'd4;
        i_row_par_en = 0; i_group_size = 1; i_rows_per_grp = 1; i_oc_tile = 0; i_clip_max = 8'd127;
        i_acc_mode = 2'd0; cap_arm = 0; cap_got = 0;
        for (k = 0; k < NUM_OC; k = k + 1) begin
            i_bias[k] = 32'd0; i_scale_mul[k] = 32'd1048576; i_scale_shift[k] = 6'd20;
            i_psum[k] = 0; i_psum_readback[k] = 0;
        end
        repeat (4) @(negedge clk); rst_n = 1; repeat (2) @(negedge clk);

        // ============ CASE 1: ACC_FIRST -> o_feat32 == i_psum ============
        cap_got = 0; cap_arm = 1;
        for (k = 0; k < NUM_OC; k = k + 1) begin
            psum_v[k] = 100 + k*3;            // arbitrary INT32
            i_psum[k] = psum_v[k];
            i_psum_readback[k] = 32'hDEAD0000 + k;   // must be IGNORED
        end
        i_acc_mode = 2'd1;   // FIRST
        feed_point;
        if (!cap_got) begin $display("CASE1: no output captured"); errors = errors + 1; end
        for (k = 0; k < NUM_OC; k = k + 1)
            if (cap_feat32[k*32 +: 32] !== psum_v[k]) begin
                $display("CASE1 FIRST oc%0d: got %0d exp %0d", k, $signed(cap_feat32[k*32 +: 32]), $signed(psum_v[k]));
                errors = errors + 1;
            end
        cap_arm = 0; repeat (8) @(negedge clk);

        // ============ CASE 2: ACC_ADD -> o_feat32 == R + i_psum ============
        cap_got = 0; cap_arm = 1;
        for (k = 0; k < NUM_OC; k = k + 1) begin
            psum_v[k] = 50 - k*2;
            rb_v[k]   = 1000 + k*7;
            i_psum[k] = psum_v[k];
            i_psum_readback[k] = rb_v[k];
        end
        i_acc_mode = 2'd2;   // ADD
        feed_point;
        if (!cap_got) begin $display("CASE2: no output captured"); errors = errors + 1; end
        for (k = 0; k < NUM_OC; k = k + 1)
            if (cap_feat32[k*32 +: 32] !== (rb_v[k] + psum_v[k])) begin
                $display("CASE2 ADD oc%0d: got %0d exp %0d", k, $signed(cap_feat32[k*32 +: 32]), $signed(rb_v[k]+psum_v[k]));
                errors = errors + 1;
            end
        cap_arm = 0; repeat (8) @(negedge clk);

        // ============ CASE 3: ACC_FINAL -> o_feat == pp(R + i_psum + bias) ===
        cap_got = 0; cap_arm = 1;
        for (k = 0; k < NUM_OC; k = k + 1) begin
            psum_v[k] = 20 + k;          // small so post-process gives in-range INT8
            rb_v[k]   = 30 + k*2;
            i_psum[k] = psum_v[k];
            i_psum_readback[k] = rb_v[k];
            i_bias[k] = 5;               // bias added in FINAL
        end
        i_acc_mode = 2'd3;   // FINAL
        feed_point;
        if (!cap_got) begin $display("CASE3: no output captured"); errors = errors + 1; end
        for (k = 0; k < NUM_OC; k = k + 1) begin
            reg [7:0] g;
            g = golden_pp(rb_v[k] + psum_v[k] + 5, 32'd1048576, 6'd20, 8'd127);
            if (cap_feat[k*8 +: 8] !== g) begin
                $display("CASE3 FINAL oc%0d: got %0d exp %0d", k, $signed(cap_feat[k*8 +: 8]), $signed(g));
                errors = errors + 1;
            end
        end
        cap_arm = 0; repeat (8) @(negedge clk);

        if (errors == 0) $display("TB_PASS");
        else $display("TB_FAIL (%0d diffs)", errors);
        $finish;
    end
endmodule
