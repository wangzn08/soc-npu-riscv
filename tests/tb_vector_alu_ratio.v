`timescale 1ns/1ps
// ===================================================================
// Directed test for vector_alu signed eltwise with the ratio extension.
// Checks, per random/directed vector, that the signed-mode output equals
// the firmware add_word() residual:
//   ratio_en=0 (legacy signed): out = sat_s8( conv + skip - zp )
//   ratio_en=1 (C2f residual):  out = sat_s8( conv + ((skip-zp)*mul + round)>>sh )
//                               round = (sh==0)?0:(1<<(sh-1))
// Also checks unsigned-mode (MNIST legacy) is untouched, and the
// real YOLO ratios (c2f2/c2f4/c2f8) reproduce the CPU math exactly.
// All 16 lanes carry independent bytes.
// ===================================================================
module tb_vector_alu_ratio;
    localparam RW = 17;
    reg          clk = 1, rst_n = 0;
    reg  [127:0] i_conv_res, i_skip_res;
    reg          i_eltwise_en, i_signed_mode, i_elt_ratio_en;
    reg  [7:0]   i_elt_zp;
    reg  [RW-1:0] i_elt_ratio_mul;
    reg  [5:0]   i_elt_ratio_shift;
    reg          i_vld;
    wire [127:0] o_res;
    wire         o_vld;

    integer errors = 0, t, lane;

    always #5 clk = ~clk;

    vector_alu #(.NUM_CH(16), .DATA_W(128), .RATIO_MUL_W(RW)) dut (
        .clk(clk), .rst_n(rst_n),
        .i_conv_res(i_conv_res), .i_skip_res(i_skip_res),
        .i_eltwise_en(i_eltwise_en), .i_signed_mode(i_signed_mode),
        .i_elt_zp(i_elt_zp),
        .i_elt_ratio_en(i_elt_ratio_en), .i_elt_ratio_mul(i_elt_ratio_mul),
        .i_elt_ratio_shift(i_elt_ratio_shift),
        .i_vld(i_vld), .o_res(o_res), .o_vld(o_vld)
    );

    // signed 8-bit value of a byte
    function integer s8; input [7:0] b; begin s8 = b[7] ? (b - 256) : b; end endfunction
    function [7:0] sat8; input integer v; begin
        sat8 = (v > 127) ? 8'h7F : (v < -128) ? 8'h80 : v[7:0];
    end endfunction
    // arithmetic right shift with round-half (matches CPU add_word)
    function integer ars_round; input integer x; input integer sh; integer rnd; begin
        if (sh == 0) ars_round = x;
        else begin rnd = (1 << (sh-1)); ars_round = (x + rnd) >>> sh; end
    end endfunction

    // expected signed-mode output for one lane
    function [7:0] exp_signed;
        input [7:0] cb; input [7:0] sb; input [7:0] zb;
        input ren; input integer mul; input integer sh;
        integer skip_term;
        begin
            if (ren) skip_term = ars_round( (s8(sb) - s8(zb)) * mul, sh );
            else     skip_term = s8(sb) - s8(zb);
            exp_signed = sat8( s8(cb) + skip_term );
        end
    endfunction
    function [7:0] exp_unsigned; input [7:0] cb; input [7:0] sb; integer u; begin
        u = cb + sb; exp_unsigned = (u > 127) ? 8'd127 : u[7:0];
    end endfunction

    // drive one cycle and check after the registered valid
    task run_vec;
        input signed_mode; input ren; input [7:0] zb;
        input integer mul; input integer sh;
        integer L; reg [7:0] cb, sb, eb, gb;
        begin
            @(negedge clk);
            i_eltwise_en = 1; i_signed_mode = signed_mode; i_elt_ratio_en = ren;
            i_elt_zp = zb; i_elt_ratio_mul = mul[RW-1:0]; i_elt_ratio_shift = sh[5:0];
            for (L = 0; L < 16; L = L + 1) begin
                cb = $random; sb = $random;
                i_conv_res[L*8 +: 8] = cb;
                i_skip_res[L*8 +: 8] = sb;
            end
            i_vld = 1;
            @(posedge clk);            // combinational o_res settles with these inputs
            #1;
            for (L = 0; L < 16; L = L + 1) begin
                cb = i_conv_res[L*8 +: 8];
                sb = i_skip_res[L*8 +: 8];
                gb = o_res[L*8 +: 8];
                if (signed_mode) eb = exp_signed(cb, sb, zb, ren, mul, sh);
                else             eb = exp_unsigned(cb, sb);
                if (gb !== eb) begin
                    errors = errors + 1;
                    $display("MISMATCH mode=%0d ren=%0d zp=%0d mul=%0d sh=%0d lane=%0d conv=%0d skip=%0d got=%0d exp=%0d",
                             signed_mode, ren, $signed(zb), mul, sh, L, $signed(cb), $signed(sb), $signed(gb), $signed(eb));
                end
            end
            i_vld = 0;
        end
    endtask

    initial begin
        i_conv_res=0; i_skip_res=0; i_eltwise_en=0; i_signed_mode=0;
        i_elt_ratio_en=0; i_elt_zp=0; i_elt_ratio_mul=1; i_elt_ratio_shift=0; i_vld=0;
        repeat (3) @(posedge clk); rst_n = 1; @(posedge clk);

        // 1) unsigned MNIST-legacy path (ratio must NOT affect it)
        for (t=0; t<20; t=t+1) run_vec(1'b0, 1'b1, 8'd50, 67752, 16);
        // 2) legacy signed (ratio_en=0): conv + skip - zp
        for (t=0; t<20; t=t+1) run_vec(1'b1, 1'b0, 8'h86, 1, 0);
        // 3) identity ratio (mul=1<<sh) must equal skip-zp
        for (t=0; t<20; t=t+1) run_vec(1'b1, 1'b1, 8'h86, (1<<16), 16);
        // 4) real YOLO ratios vs CPU add_word
        for (t=0; t<40; t=t+1) run_vec(1'b1, 1'b1, 8'h82, 67752, 16); // c2f2 ADD0 (zp -126)
        for (t=0; t<40; t=t+1) run_vec(1'b1, 1'b1, 8'h87, 71211, 16); // c2f4 ADD0 (zp -121)
        for (t=0; t<40; t=t+1) run_vec(1'b1, 1'b1, 8'h87, 42797, 16); // c2f4 ADD1 (mul<1)
        for (t=0; t<40; t=t+1) run_vec(1'b1, 1'b1, 8'h85, 45977, 16); // c2f8 ADD0 (zp -123)

        if (errors == 0) $display("ALL TESTS PASSED.");
        else             $display("ERROR! %0d mismatches", errors);
        $finish;
    end
endmodule
