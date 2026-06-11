`timescale 1ns/1ps
// ===================================================================
// Standalone directed test for max_pooling_2x2 (rewritten version).
// Feeds known 4x4 feature maps row-major, pulses i_start before each
// map, and self-checks the 2x2 stride-2 maxpool result.
//
// MAP A = 1..16 row-major:        MAP B = 16..1 row-major:
//    1  2  3  4                      16 15 14 13
//    5  6  7  8                      12 11 10  9
//    9 10 11 12                       8  7  6  5
//   13 14 15 16                       4  3  2  1
//   golden A -> 6 8 14 16            golden B -> 16 14 8 6
//
// All 16 channels carry the same byte, so o_pool[7:0] == pooled value.
// ===================================================================
module tb_max_pooling_2x2;
    reg clk = 1, rst_n = 0;
    reg          i_start;
    reg  [127:0] i_feat;
    reg          i_feat_vld;
    reg  [15:0]  i_width;
    wire [127:0] o_pool;
    wire         o_pool_vld;

    integer i, errors = 0;
    integer cap_idx = 0;
    reg [7:0] cap [0:31];
    reg [7:0] mapA [0:15];
    reg [7:0] mapB [0:15];
    reg [7:0] goldA [0:3];
    reg [7:0] goldB [0:3];

    always #5 clk = ~clk;

    max_pooling_2x2 #(.MAX_WIDTH(256), .ACT_WIDTH(8), .NUM_CH(16), .DATA_W(128)) dut (
        .clk(clk), .rst_n(rst_n), .i_start(i_start),
        .i_feat(i_feat), .i_feat_vld(i_feat_vld), .i_width(i_width),
        .o_pool(o_pool), .o_pool_vld(o_pool_vld)
    );

    always @(posedge clk) if (rst_n && o_pool_vld) begin
        cap[cap_idx] = o_pool[7:0];
        $display("    OUT[%0d] = %0d", cap_idx, o_pool[7:0]);
        cap_idx = cap_idx + 1;
    end

    task pulse_start;
        begin
            @(negedge clk); i_start = 1'b1; i_feat_vld = 1'b0;
            @(negedge clk); i_start = 1'b0;
        end
    endtask

    task feed(input [7:0] v);
        begin
            @(negedge clk);
            i_feat     = {16{v}};
            i_feat_vld = 1'b1;
        end
    endtask

    task idle(input integer n);
        begin
            @(negedge clk); i_feat_vld = 1'b0;
            repeat (n) @(negedge clk);
        end
    endtask

    task check(input [8*5-1:0] name, input integer base, input [7:0] g0, g1, g2, g3);
        begin
            if (cap[base]!==g0 || cap[base+1]!==g1 || cap[base+2]!==g2 || cap[base+3]!==g3) begin
                $display("  [FAIL] %0s: got %0d %0d %0d %0d  expected %0d %0d %0d %0d",
                         name, cap[base], cap[base+1], cap[base+2], cap[base+3], g0, g1, g2, g3);
                errors = errors + 1;
            end else begin
                $display("  [PASS] %0s: %0d %0d %0d %0d", name, g0, g1, g2, g3);
            end
        end
    endtask

    initial begin
        for (i = 0; i < 16; i = i + 1) mapA[i] = i + 1;
        for (i = 0; i < 16; i = i + 1) mapB[i] = 8'd16 - i;
        goldA[0]=6; goldA[1]=8; goldA[2]=14; goldA[3]=16;
        goldB[0]=16; goldB[1]=14; goldB[2]=8; goldB[3]=6;

        i_feat = 0; i_feat_vld = 0; i_start = 0; i_width = 16'd4;
        repeat (4) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);

        $display("=== MAP A (start, feed 1..16) ===");
        pulse_start;
        for (i = 0; i < 16; i = i + 1) feed(mapA[i]);
        idle(8);

        $display("=== MAP B (start, feed 16..1) ===");
        pulse_start;
        for (i = 0; i < 16; i = i + 1) feed(mapB[i]);
        idle(8);

        $display("=== RESULT ===");
        if (cap_idx !== 8)
            $display("  [FAIL] output count = %0d, expected 8", cap_idx);
        check("MAP_A", 0, goldA[0], goldA[1], goldA[2], goldA[3]);
        check("MAP_B", 4, goldB[0], goldB[1], goldB[2], goldB[3]);
        if (errors == 0 && cap_idx == 8)
            $display("  ALL POOL TESTS PASSED.");
        else
            $display("  POOL TESTS FAILED (errors=%0d).", errors);
        $finish;
    end
endmodule
