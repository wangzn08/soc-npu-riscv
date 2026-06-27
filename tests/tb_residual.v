`timescale 1ns/1ps
// ===================================================================
// Directed test for vector_alu (residual eltwise add). o_res is
// combinational: per channel sat(conv+skip) to [0,127] when eltwise_en,
// else pass-through conv. (The configurable skip base address lives in
// npu_top; here we verify the add/saturate datapath residual relies on.)
//   conv=100 skip=50  -> 127 (saturated)
//   conv=30  skip=20  -> 50
//   conv=127 skip=127 -> 127
//   eltwise_en=0      -> conv pass-through (100)
// ===================================================================
module tb_residual;
    reg          clk = 1, rst_n = 0;
    reg  [127:0] i_conv_res, i_skip_res;
    reg          i_eltwise_en, i_vld;
    wire [127:0] o_res;
    wire         o_vld;

    integer errors = 0;

    always #5 clk = ~clk;

    vector_alu dut (
        .clk(clk), .rst_n(rst_n),
        .i_conv_res(i_conv_res), .i_skip_res(i_skip_res),
        .i_eltwise_en(i_eltwise_en),
        .i_signed_mode(1'b0),        // legacy unsigned [0,127] residual (this TB's scope)
        .i_elt_zp(8'd0),
        .i_elt_ratio_en(1'b0), .i_elt_ratio_mul(17'd1), .i_elt_ratio_shift(6'd0),
        .i_vld(i_vld),
        .o_res(o_res), .o_vld(o_vld)
    );

    task chk(input [8*8-1:0] name, input [7:0] got, input [7:0] exp);
        begin
            if (got !== exp) begin
                $display("  [FAIL] %0s: got=%0d exp=%0d", name, got, exp);
                errors = errors + 1;
            end else
                $display("  [PASS] %0s: %0d", name, got);
        end
    endtask

    task drive(input [7:0] c, input [7:0] s, input en);
        begin
            i_conv_res = {16{c}}; i_skip_res = {16{s}}; i_eltwise_en = en; i_vld = 1;
            #1;  // settle combinational o_res
        end
    endtask

    initial begin
        i_conv_res = 0; i_skip_res = 0; i_eltwise_en = 0; i_vld = 0;
        repeat (2) @(negedge clk); rst_n = 1; @(negedge clk);

        drive(8'd100, 8'd50,  1'b1); chk("SAT_150",  o_res[7:0], 8'd127);
        drive(8'd30,  8'd20,  1'b1); chk("ADD_50",   o_res[7:0], 8'd50);
        drive(8'd127, 8'd127, 1'b1); chk("SAT_254",  o_res[7:0], 8'd127);
        drive(8'd0,   8'd0,   1'b1); chk("ZERO",     o_res[7:0], 8'd0);
        drive(8'd100, 8'd50,  1'b0); chk("BYPASS",   o_res[7:0], 8'd100);

        if (errors == 0) $display("TB_RESIDUAL PASS");
        else $display("TB_RESIDUAL FAIL errors=%0d", errors);
        $finish;
    end
endmodule
