`timescale 1ns/1ps
// ===================================================================
// Directed test for vector_alu signed+zero-point eltwise mode.
//
// Default mode (i_signed_mode=0) must stay byte-identical to the
// legacy MNIST residual: unsigned per-lane add saturated to [0,127].
//
// Signed mode (i_signed_mode=1) computes the YOLO C2f residual add on
// two operands already requantized to the glue scale/zero-point:
//   out = sat_s8( s8(conv) + s8(skip) - s8(zp) )   range [-128,127]
// All 16 lanes carry the same byte, so o_res[7:0] is the result.
// ===================================================================
module tb_vector_alu_signed;
    reg          clk = 1, rst_n = 0;
    reg  [127:0] i_conv_res;
    reg  [127:0] i_skip_res;
    reg          i_eltwise_en;
    reg          i_signed_mode;
    reg  [7:0]   i_elt_zp;
    reg          i_vld;
    wire [127:0] o_res;
    wire         o_vld;

    integer errors = 0;

    always #5 clk = ~clk;

    vector_alu #(.NUM_CH(16), .DATA_W(128)) dut (
        .clk(clk), .rst_n(rst_n),
        .i_conv_res(i_conv_res),
        .i_skip_res(i_skip_res),
        .i_eltwise_en(i_eltwise_en),
        .i_signed_mode(i_signed_mode),
        .i_elt_zp(i_elt_zp),
        .i_elt_ratio_en(1'b0),       // ratio disabled -> legacy signed (skip - zp)
        .i_elt_ratio_mul(17'd1),
        .i_elt_ratio_shift(6'd0),
        .i_vld(i_vld),
        .o_res(o_res),
        .o_vld(o_vld)
    );

    // check the low lane (all lanes identical) combinationally
    task chk(input [8*8-1:0] name, input signed [8:0] expect_s);
        reg [7:0] got;
        reg [7:0] exp;
        begin
            #1; // let combinational settle
            got = o_res[7:0];
            exp = expect_s[7:0];
            if (got !== exp) begin
                $display("  [FAIL] %0s: got %0d (0x%02x) expected %0d (0x%02x)",
                         name, $signed(got), got, $signed(exp), exp);
                errors = errors + 1;
            end else begin
                $display("  [PASS] %0s: %0d", name, $signed(got));
            end
        end
    endtask

    task drive(input [7:0] cv, input [7:0] sk, input mode,
               input [7:0] zp, input en);
        begin
            i_conv_res   = {16{cv}};
            i_skip_res   = {16{sk}};
            i_signed_mode= mode;
            i_elt_zp     = zp;
            i_eltwise_en = en;
            i_vld        = 1'b1;
        end
    endtask

    initial begin
        i_conv_res=0; i_skip_res=0; i_eltwise_en=0; i_signed_mode=0;
        i_elt_zp=0; i_vld=0;
        repeat (4) @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        $display("=== default unsigned mode (legacy) ===");
        drive(8'd10, 8'd20, 1'b0, 8'd0, 1'b1); chk("u_add",    9'sd30);
        drive(8'd100,8'd100,1'b0, 8'd0, 1'b1); chk("u_satpos", 9'sd127);
        drive(8'd10, 8'd20, 1'b0, 8'd0, 1'b0); chk("u_bypass", 9'sd10);

        $display("=== signed + zp mode (YOLO C2f) ===");
        // zp = -126 = 8'h82
        drive(8'h82, 8'h82, 1'b1, 8'h82, 1'b1); chk("s_both_zp", -9'sd126);
        drive(8'd0,  8'd0,  1'b1, 8'h82, 1'b1); chk("s_zero",     9'sd126);
        drive(8'd127,8'd127,1'b1, 8'h82, 1'b1); chk("s_satpos",   9'sd127);
        drive(8'h80, 8'h80, 1'b1, 8'd10, 1'b1); chk("s_satneg",  -9'sd128);
        drive(8'd10, 8'h82, 1'b1, 8'd0,  1'b0); chk("s_bypass",   9'sd10);

        $display("=== RESULT ===");
        if (errors == 0)
            $display("  ALL VECTOR_ALU SIGNED TESTS PASSED.");
        else
            $display("  VECTOR_ALU SIGNED TESTS FAILED (errors=%0d).", errors);
        $finish;
    end
endmodule
