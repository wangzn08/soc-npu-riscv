`timescale 1ns/1ps
// ===================================================================
// Directed test for post_process_top's POOLING path.
//
// Replicates the FSM drain->post protocol for a 4x4 feature map (16
// conv points) and checks the pooled o_feat / o_feat_vld stream.
//
// Per conv point the FSM presents:
//   S_DRAIN : i_in_drain=1, i_psum_vld=1 for 16 cycles
//   S_POST  : i_in_post=1 ; i_psum_vld=1 on the first post cycle (pp_start)
//   gap     : a few idle cycles (next-tile / prefetch / calc / k_end)
//
// Quant is identity (bias=0, scale_mul=2^20, shift=20, relu on) so
// o_feat channel == clamp(i_psum,0,127).  All 16 channels carry the
// same value, so o_feat[7:0] == pooled value.
//
// Feature map A = 1..16 row-major -> golden pooled = 6 8 14 16.
// Probes dut.pool_gated_vld to count how many valids actually reach
// the pooler per point (exposes the multi-pulse feed bug).
// ===================================================================
module tb_post_process_pool;
    localparam NUM_OC = 16;

    reg clk = 1, rst_n = 0;
    reg [NUM_OC-1:0][31:0] i_psum;
    reg                    i_psum_vld;
    reg [NUM_OC-1:0][31:0] i_bias;
    reg [NUM_OC-1:0][31:0] i_scale_mul;
    reg [NUM_OC-1:0][5:0]  i_scale_shift;
    reg [15:0]             i_width;
    reg                    i_pool_en, i_relu_en, i_start, i_in_drain, i_in_post;
    wire [127:0]           o_feat;
    wire                   o_feat_vld;

    integer p, j, k, cap_idx = 0, feed_pulses = 0;
    reg [7:0] cap [0:63];
    reg [7:0] mapA [0:15];

    always #5 clk = ~clk;

    // Safety watchdog: stop no matter what.
    initial begin
        #100000;
        $display("  TB WATCHDOG TIMEOUT");
        $finish;
    end

    post_process_top #(.NUM_OC(NUM_OC), .MAX_WIDTH(256)) dut (
        .clk(clk), .rst_n(rst_n),
        .i_psum(i_psum), .i_psum_vld(i_psum_vld),
        .i_bias(i_bias), .i_scale_mul(i_scale_mul), .i_scale_shift(i_scale_shift),
        .i_width(i_width), .i_pool_en(i_pool_en), .i_relu_en(i_relu_en),
        .i_start(i_start), .i_in_drain(i_in_drain), .i_in_post(i_in_post),
        .o_feat(o_feat), .o_feat_vld(o_feat_vld)
    );

    always @(posedge clk) if (rst_n && o_feat_vld && cap_idx < 64) begin
        cap[cap_idx] = o_feat[7:0];
        $display("    o_feat_vld: OUT[%0d] = %0d", cap_idx, o_feat[7:0]);
        cap_idx = cap_idx + 1;
    end
    always @(posedge clk) if (rst_n && dut.pool_gated_vld) feed_pulses = feed_pulses + 1;

    task set_psum(input [7:0] v);
        begin for (k = 0; k < NUM_OC; k = k + 1) i_psum[k] = {24'b0, v}; end
    endtask

    task conv_point(input [7:0] v);
        begin
            for (j = 0; j < 16; j = j + 1) begin   // S_DRAIN
                @(negedge clk);
                i_in_drain = 1; i_in_post = 0; i_psum_vld = 1; set_psum(v);
            end
            @(negedge clk);                         // S_POST entry (pp_start)
            i_in_drain = 0; i_in_post = 1; i_psum_vld = 1; set_psum(v);
            for (j = 0; j < 5; j = j + 1) begin     // S_POST hold
                @(negedge clk); i_in_post = 1; i_psum_vld = 0;
            end
            for (j = 0; j < 4; j = j + 1) begin     // gap
                @(negedge clk); i_in_drain = 0; i_in_post = 0; i_psum_vld = 0;
            end
        end
    endtask

    initial begin
        for (p = 0; p < 16; p = p + 1) mapA[p] = p + 1;
        i_psum_vld = 0; i_in_drain = 0; i_in_post = 0; i_start = 0;
        i_pool_en = 1; i_relu_en = 1; i_width = 16'd4;
        for (k = 0; k < NUM_OC; k = k + 1) begin
            i_bias[k] = 0; i_scale_mul[k] = 32'd1048576; i_scale_shift[k] = 6'd20; i_psum[k] = 0;
        end
        repeat (4) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);

        @(negedge clk); i_start = 1;
        @(negedge clk); i_start = 0;

        $display("=== POOL path: feed 4x4 (1..16), golden = 6 8 14 16 ===");
        for (p = 0; p < 16; p = p + 1) conv_point(mapA[p]);
        repeat (10) @(negedge clk);

        $display("=== RESULT ===");
        $display("  pooler i_feat_vld pulses = %0d  (ideal 16, one per conv point)", feed_pulses);
        $display("  pooled outputs captured  = %0d  (ideal 4)", cap_idx);
        if (cap_idx >= 4)
            $display("  first 4 outputs = %0d %0d %0d %0d  (golden 6 8 14 16)",
                     cap[0], cap[1], cap[2], cap[3]);
        $finish;
    end
endmodule
