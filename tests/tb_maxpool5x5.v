`timescale 1ns/1ps
// ===================================================================
// Directed test for a generic Act-SRAM 5x5 stride-1 maxpool engine.
//
// Layout:
//   addr = base + ic_group * H*W + y*W + x
//
// The engine is intentionally generic: it pools any tile-major INT8 tensor
// already in Act SRAM, independent of YOLO layer numbering.
// ===================================================================
module tb_maxpool5x5;
    localparam ADDR_W = 10;
    localparam DATA_W = 128;

    reg clk = 1, rst_n = 0;
    reg i_trig;
    reg [ADDR_W-1:0] i_src_base, i_dst_base;
    reg [15:0] i_w, i_h, i_ic_groups;
    wire [ADDR_W-1:0] o_addr;
    wire o_en, o_we, o_busy, o_done;
    wire [DATA_W-1:0] o_wdata;
    wire [DATA_W-1:0] i_rdata;

    reg [DATA_W-1:0] mem [0:255];
    reg signed [7:0] src_lane [0:1][0:3][0:3][0:15];
    integer i, g, y, x, lane, ky, kx, yy, xx, errors;
    integer src_addr, dst_addr;
    reg signed [7:0] best, v;
    reg [127:0] exp_word;

    always #5 clk = ~clk;
    assign i_rdata = mem[o_addr];

    always @(posedge clk) begin
        if (o_en && o_we)
            mem[o_addr] <= o_wdata;
    end

    maxpool5x5 #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .i_trig(i_trig),
        .i_src_base(i_src_base),
        .i_dst_base(i_dst_base),
        .i_w(i_w),
        .i_h(i_h),
        .i_ic_groups(i_ic_groups),
        .o_addr(o_addr),
        .o_en(o_en),
        .o_we(o_we),
        .o_wdata(o_wdata),
        .i_rdata(i_rdata),
        .o_busy(o_busy),
        .o_done(o_done)
    );

    task start_engine;
        begin
            @(negedge clk); i_trig = 1'b1;
            @(negedge clk); i_trig = 1'b0;
        end
    endtask

    function signed [7:0] pix;
        input integer gg, yy0, xx0, ll;
        begin
            pix = src_lane[gg][yy0][xx0][ll];
        end
    endfunction

    initial begin
        errors = 0;
        for (i = 0; i < 256; i = i + 1) mem[i] = 128'h0;

        i_trig = 0;
        i_src_base = 10'd4;
        i_dst_base = 10'd64;
        i_w = 16'd4;
        i_h = 16'd4;
        i_ic_groups = 16'd2;

        for (g = 0; g < 2; g = g + 1)
            for (y = 0; y < 4; y = y + 1)
                for (x = 0; x < 4; x = x + 1) begin
                    src_addr = i_src_base + g * 16 + y * 4 + x;
                    for (lane = 0; lane < 16; lane = lane + 1) begin
                        src_lane[g][y][x][lane] = $signed(-8'sd70 + g * 8'sd20 + y * 8'sd9 + x * 8'sd3 + lane[7:0]);
                        mem[src_addr][lane*8 +: 8] = src_lane[g][y][x][lane];
                    end
                end

        repeat (4) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);

        start_engine();
        for (i = 0; i < 1400 && !o_done; i = i + 1) @(negedge clk);
        if (!o_done) begin
            $display("TB_MAXPOOL5X5 FAIL timeout");
            $finish;
        end
        repeat (2) @(negedge clk);

        for (g = 0; g < 2; g = g + 1)
            for (y = 0; y < 4; y = y + 1)
                for (x = 0; x < 4; x = x + 1) begin
                    exp_word = 128'h0;
                    for (lane = 0; lane < 16; lane = lane + 1) begin
                        best = -8'sd128;
                        for (ky = -2; ky <= 2; ky = ky + 1)
                            for (kx = -2; kx <= 2; kx = kx + 1) begin
                                yy = y + ky;
                                xx = x + kx;
                                if (yy >= 0 && yy < 4 && xx >= 0 && xx < 4) begin
                                    v = pix(g, yy, xx, lane);
                                    if (v > best) best = v;
                                end
                            end
                        exp_word[lane*8 +: 8] = best[7:0];
                    end
                    dst_addr = i_dst_base + g * 16 + y * 4 + x;
                    if (mem[dst_addr] !== exp_word) begin
                        errors = errors + 1;
                        $display("FAIL g=%0d y=%0d x=%0d dst=%0d exp=%032h got=%032h",
                                 g, y, x, dst_addr, exp_word, mem[dst_addr]);
                    end
                end

        if (errors == 0) $display("TB_MAXPOOL5X5 PASS");
        else $display("TB_MAXPOOL5X5 FAIL errors=%0d", errors);
        $finish;
    end

    initial begin
        #200000;
        $display("TB_MAXPOOL5X5 TIMEOUT");
        $finish;
    end
endmodule
