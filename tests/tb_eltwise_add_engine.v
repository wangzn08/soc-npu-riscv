`timescale 1ns/1ps
// ===================================================================
// Directed test for a generic Act-SRAM signed INT8 eltwise add engine.
//
// The operation matches the YOLO C2f residual math but the engine itself is
// generic over two Act-SRAM tensors:
//   out = sat_s8( s8(src1) + round((s8(src0)-s8(zp))*mul >> shift) )
// ===================================================================
module tb_eltwise_add_engine;
    localparam ADDR_W = 10;
    localparam DATA_W = 128;

    reg clk = 1, rst_n = 0;
    reg i_trig;
    reg [ADDR_W-1:0] i_src0_base, i_src1_base, i_dst_base;
    reg [15:0] i_len;
    reg [7:0] i_zp;
    reg i_ratio_en;
    reg [16:0] i_ratio_mul;
    reg [5:0] i_ratio_shift;
    wire [ADDR_W-1:0] o_addr;
    wire o_en, o_we, o_busy, o_done;
    wire [DATA_W-1:0] o_wdata;
    wire [DATA_W-1:0] i_rdata;

    reg [DATA_W-1:0] mem [0:255];
    integer i, lane, errors;
    reg [7:0] got, expb;

    always #5 clk = ~clk;
    assign i_rdata = mem[o_addr];
    always @(posedge clk) if (o_en && o_we) mem[o_addr] <= o_wdata;

    eltwise_add_engine #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) dut (
        .clk(clk), .rst_n(rst_n), .i_trig(i_trig),
        .i_src0_base(i_src0_base), .i_src1_base(i_src1_base),
        .i_dst_base(i_dst_base), .i_len(i_len),
        .i_zp(i_zp), .i_ratio_en(i_ratio_en),
        .i_ratio_mul(i_ratio_mul), .i_ratio_shift(i_ratio_shift),
        .o_addr(o_addr), .o_en(o_en), .o_we(o_we), .o_wdata(o_wdata),
        .i_rdata(i_rdata), .o_busy(o_busy), .o_done(o_done)
    );

    function integer s8; input [7:0] b; begin s8 = b[7] ? (b - 256) : b; end endfunction
    function [7:0] sat8; input integer v; begin
        sat8 = (v > 127) ? 8'h7F : (v < -128) ? 8'h80 : v[7:0];
    end endfunction
    function integer round_shift; input integer v; input integer sh; begin
        if (sh == 0) round_shift = v;
        else round_shift = (v + (1 << (sh-1))) >>> sh;
    end endfunction
    function [7:0] exp_lane;
        input [7:0] src0; input [7:0] src1;
        integer term;
        begin
            if (i_ratio_en) term = round_shift((s8(src0)-s8(i_zp)) * i_ratio_mul, i_ratio_shift);
            else            term = s8(src0) - s8(i_zp);
            exp_lane = sat8(s8(src1) + term);
        end
    endfunction

    task start_engine;
        begin
            @(negedge clk); i_trig = 1'b1;
            @(negedge clk); i_trig = 1'b0;
        end
    endtask

    initial begin
        errors = 0;
        for (i = 0; i < 256; i = i + 1) mem[i] = 128'h0;
        i_trig = 0;
        i_src0_base = 10'd8;
        i_src1_base = 10'd32;
        i_dst_base  = 10'd64;
        i_len = 16'd9;
        i_zp = 8'h84;
        i_ratio_en = 1'b1;
        i_ratio_mul = 17'd67752;
        i_ratio_shift = 6'd16;

        for (i = 0; i < 9; i = i + 1)
            for (lane = 0; lane < 16; lane = lane + 1) begin
                mem[i_src0_base+i][lane*8 +: 8] = (8'h80 + i*7 + lane*3);
                mem[i_src1_base+i][lane*8 +: 8] = (8'h90 + i*5 - lane*2);
            end

        repeat (4) @(negedge clk); rst_n = 1; repeat (2) @(negedge clk);
        start_engine();
        for (i = 0; i < 200 && !o_done; i = i + 1) @(negedge clk);
        if (!o_done) begin $display("TB_ELTWISE_ADD_ENGINE FAIL timeout"); $finish; end
        repeat (2) @(negedge clk);

        for (i = 0; i < 9; i = i + 1)
            for (lane = 0; lane < 16; lane = lane + 1) begin
                got = mem[i_dst_base+i][lane*8 +: 8];
                expb = exp_lane(mem[i_src0_base+i][lane*8 +: 8], mem[i_src1_base+i][lane*8 +: 8]);
                if (got !== expb) begin
                    errors = errors + 1;
                    $display("FAIL word=%0d lane=%0d got=%0d exp=%0d",
                             i, lane, $signed(got), $signed(expb));
                end
            end

        if (errors == 0) $display("TB_ELTWISE_ADD_ENGINE PASS");
        else $display("TB_ELTWISE_ADD_ENGINE FAIL errors=%0d", errors);
        $finish;
    end

    initial begin #200000 $display("TB_ELTWISE_ADD_ENGINE TIMEOUT"); $finish; end
endmodule
