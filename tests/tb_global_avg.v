`timescale 1ns/1ps
// ===================================================================
// Directed test for global_avg. Feeds N=64 positions of a single channel
// value, with reciprocal mul=1, shift=6 (=> /64), and checks the mean.
//   Test A: all positions = 10  -> mean 10
//   Test B: positions = 0..63   -> sum 2016, /64 = 31 (floor)
// All 16 channels carry the same byte; checks channel 0.
// ===================================================================
module tb_global_avg;
    localparam NUM_CH = 16, PSUM_WIDTH = 32;
    reg clk = 1, rst_n = 0;
    reg          i_start, i_feat_vld, i_last;
    reg  [127:0] i_feat;
    reg  [PSUM_WIDTH-1:0] i_avg_mul;
    reg  [5:0]   i_avg_shift;
    wire [127:0] o_feat;
    wire         o_feat_vld;

    integer i, errors = 0;
    reg [7:0] capA, capB;
    reg gotA, gotB;

    always #5 clk = ~clk;

    global_avg #(.NUM_CH(NUM_CH)) dut (
        .clk(clk), .rst_n(rst_n), .i_start(i_start),
        .i_feat(i_feat), .i_feat_vld(i_feat_vld), .i_last(i_last),
        .i_avg_mul(i_avg_mul), .i_avg_shift(i_avg_shift),
        .o_feat(o_feat), .o_feat_vld(o_feat_vld)
    );

    task feed(input [7:0] v, input integer last);
        begin
            @(negedge clk);
            i_feat = {16{v}}; i_feat_vld = 1'b1; i_last = last[0];
        end
    endtask

    task run_pool;
        begin
            @(negedge clk); i_start = 1'b1; i_feat_vld = 1'b0; i_last = 1'b0;
            @(negedge clk); i_start = 1'b0;
        end
    endtask

    initial begin
        i_feat = 0; i_feat_vld = 0; i_last = 0; i_start = 0;
        i_avg_mul = 32'd1; i_avg_shift = 6'd6;   // /64
        gotA = 0; gotB = 0;
        repeat (4) @(negedge clk); rst_n = 1; repeat (2) @(negedge clk);

        // Test A: 64x value 10 -> mean 10
        run_pool;
        for (i = 0; i < 64; i = i + 1) feed(8'd10, (i == 63) ? 1 : 0);
        @(negedge clk); i_feat_vld = 0; i_last = 0;
        repeat (3) @(negedge clk);

        // Test B: values 0..63 -> sum 2016 /64 = 31
        run_pool;
        for (i = 0; i < 64; i = i + 1) feed(i[7:0], (i == 63) ? 1 : 0);
        @(negedge clk); i_feat_vld = 0; i_last = 0;
        repeat (3) @(negedge clk);

        if (!gotA || capA !== 8'd10) begin
            errors = errors + 1; $display("  [FAIL] A got=%0d exp=10 (got_vld=%0b)", capA, gotA);
        end else $display("  [PASS] A mean=10");
        if (!gotB || capB !== 8'd31) begin
            errors = errors + 1; $display("  [FAIL] B got=%0d exp=31 (got_vld=%0b)", capB, gotB);
        end else $display("  [PASS] B mean=31");

        if (errors == 0) $display("TB_GLOBAL_AVG PASS");
        else $display("TB_GLOBAL_AVG FAIL errors=%0d", errors);
        $finish;
    end

    // capture: first pulse -> A, second -> B
    always @(posedge clk) if (rst_n && o_feat_vld) begin
        if (!gotA) begin capA = o_feat[7:0]; gotA = 1; end
        else if (!gotB) begin capB = o_feat[7:0]; gotB = 1; end
    end
endmodule
