`timescale 1ns/1ps
// ===================================================================
// Directed test for the shared Act-SRAM nearest-neighbor 2x upsample
// engine. Layout is the same tile-major layout consumed by the NPU FSM:
//
//   addr = base + ic_group * H*W + y*W + x
//
// The engine reads each source word from Act SRAM Port B and writes it to
// the 2x2 output block in the destination tensor:
//
//   dst(g, 2y+dy, 2x+dx) = src(g, y, x), dy/dx in {0,1}
// ===================================================================
module tb_upsample2x;
    localparam ADDR_W = 10;
    localparam DATA_W = 128;

    reg clk = 1, rst_n = 0;
    reg i_trig;
    reg [ADDR_W-1:0] i_src_base, i_dst_base;
    reg [15:0] i_in_w, i_in_h, i_ic_groups;
    wire [ADDR_W-1:0] o_addr;
    wire o_en, o_we, o_busy, o_done;
    wire [DATA_W-1:0] o_wdata;
    wire [DATA_W-1:0] i_rdata;

    reg [DATA_W-1:0] mem [0:255];
    integer i, g, y, x, dy, dx, errors;
    integer src_addr, dst_addr;
    reg [DATA_W-1:0] exp;

    always #5 clk = ~clk;
    assign i_rdata = mem[o_addr];

    always @(posedge clk) begin
        if (o_en && o_we)
            mem[o_addr] <= o_wdata;
    end

    upsample2x #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .i_trig(i_trig),
        .i_src_base(i_src_base),
        .i_dst_base(i_dst_base),
        .i_in_w(i_in_w),
        .i_in_h(i_in_h),
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

    initial begin
        errors = 0;
        for (i = 0; i < 256; i = i + 1) mem[i] = 128'h0;

        i_trig = 0;
        i_src_base = 10'd4;
        i_dst_base = 10'd64;
        i_in_w = 16'd3;
        i_in_h = 16'd2;
        i_ic_groups = 16'd2;

        // Source tensor: two IC groups, 2x3 each. Fill every 128-bit word
        // with a recognizable byte pattern.
        for (g = 0; g < 2; g = g + 1)
            for (y = 0; y < 2; y = y + 1)
                for (x = 0; x < 3; x = x + 1) begin
                    src_addr = i_src_base + g * 6 + y * 3 + x;
                    mem[src_addr] = {16{8'(8'h10 + g * 8'h40 + y * 8'h08 + x[7:0])}};
                end

        repeat (4) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);

        start_engine();
        for (i = 0; i < 300 && !o_done; i = i + 1) @(negedge clk);
        if (!o_done) begin
            $display("TB_UPSAMPLE2X FAIL timeout");
            $finish;
        end
        repeat (2) @(negedge clk);

        for (g = 0; g < 2; g = g + 1)
            for (y = 0; y < 2; y = y + 1)
                for (x = 0; x < 3; x = x + 1)
                    for (dy = 0; dy < 2; dy = dy + 1)
                        for (dx = 0; dx < 2; dx = dx + 1) begin
                            src_addr = i_src_base + g * 6 + y * 3 + x;
                            dst_addr = i_dst_base + g * 24 + (2*y + dy) * 6 + (2*x + dx);
                            exp = mem[src_addr];
                            if (mem[dst_addr] !== exp) begin
                                errors = errors + 1;
                                $display("FAIL g=%0d y=%0d x=%0d dy=%0d dx=%0d dst=%0d exp=%032h got=%032h",
                                         g, y, x, dy, dx, dst_addr, exp, mem[dst_addr]);
                            end
                        end

        if (errors == 0) $display("TB_UPSAMPLE2X PASS");
        else $display("TB_UPSAMPLE2X FAIL errors=%0d", errors);
        $finish;
    end
endmodule
