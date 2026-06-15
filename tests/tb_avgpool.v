`timescale 1ns/1ps
// ===================================================================
// Directed test for 2x2 average pooling (CTRL[16] pool_avg) in
// max_pooling_2x2. Feeds MAP A = 1..16 row-major (4x4):
//    1  2  3  4
//    5  6  7  8
//    9 10 11 12
//   13 14 15 16
//   avg golden -> {1,2,5,6}/4=3  {3,4,7,8}/4=5  {9,10,13,14}/4=11  {11,12,15,16}/4=13
//   max golden -> 6 8 14 16  (i_avg=0 must stay byte-identical to legacy)
// ===================================================================
module tb_avgpool;
    reg clk = 1, rst_n = 0;
    reg          i_start, i_avg;
    reg  [127:0] i_feat;
    reg          i_feat_vld;
    reg  [15:0]  i_width;
    wire [127:0] o_pool;
    wire         o_pool_vld;
    wire [1:0]   o_pool_tile;

    integer i, errors = 0, cap_idx = 0;
    reg [7:0] cap   [0:31];
    reg [7:0] mapA  [0:15];
    reg [7:0] goldAvg [0:3];
    reg [7:0] goldMax [0:3];

    always #5 clk = ~clk;

    max_pooling_2x2 #(.MAX_WIDTH(256), .ACT_WIDTH(8), .NUM_CH(16), .DATA_W(128)) dut (
        .clk(clk), .rst_n(rst_n), .i_start(i_start),
        .i_feat(i_feat), .i_feat_vld(i_feat_vld), .i_width(i_width),
        .i_tile(2'd0), .i_avg(i_avg),
        .o_pool(o_pool), .o_pool_vld(o_pool_vld), .o_pool_tile(o_pool_tile)
    );

    always @(posedge clk) if (rst_n && o_pool_vld) begin
        cap[cap_idx] = o_pool[7:0];
        cap_idx = cap_idx + 1;
    end

    task pulse_start; begin
        @(negedge clk); i_start = 1'b1; i_feat_vld = 1'b0;
        @(negedge clk); i_start = 1'b0;
    end endtask

    task feed(input [7:0] v); begin
        @(negedge clk); i_feat = {16{v}}; i_feat_vld = 1'b1;
    end endtask

    task idle(input integer n); begin
        @(negedge clk); i_feat_vld = 1'b0; repeat (n) @(negedge clk);
    end endtask

    task check(input [8*5-1:0] name, input integer base, input [7:0] g0, g1, g2, g3); begin
        if (cap[base]!==g0 || cap[base+1]!==g1 || cap[base+2]!==g2 || cap[base+3]!==g3) begin
            $display("  [FAIL] %0s: got %0d %0d %0d %0d  expected %0d %0d %0d %0d",
                     name, cap[base], cap[base+1], cap[base+2], cap[base+3], g0, g1, g2, g3);
            errors = errors + 1;
        end else
            $display("  [PASS] %0s: %0d %0d %0d %0d", name, g0, g1, g2, g3);
    end endtask

    initial begin
        for (i = 0; i < 16; i = i + 1) mapA[i] = i + 1;
        goldAvg[0]=3; goldAvg[1]=5; goldAvg[2]=11; goldAvg[3]=13;
        goldMax[0]=6; goldMax[1]=8; goldMax[2]=14; goldMax[3]=16;

        i_feat = 0; i_feat_vld = 0; i_start = 0; i_avg = 0; i_width = 16'd4;
        repeat (4) @(negedge clk); rst_n = 1; repeat (2) @(negedge clk);

        // avg path
        i_avg = 1; pulse_start;
        for (i = 0; i < 16; i = i + 1) feed(mapA[i]);
        idle(8);

        // max path (must equal legacy)
        i_avg = 0; pulse_start;
        for (i = 0; i < 16; i = i + 1) feed(mapA[i]);
        idle(8);

        check("AVG", 0, goldAvg[0], goldAvg[1], goldAvg[2], goldAvg[3]);
        check("MAX", 4, goldMax[0], goldMax[1], goldMax[2], goldMax[3]);
        if (errors == 0 && cap_idx == 8) $display("TB_AVGPOOL PASS");
        else $display("TB_AVGPOOL FAIL errors=%0d cap_idx=%0d", errors, cap_idx);
        $finish;
    end
endmodule
