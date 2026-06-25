`timescale 1ns/1ps

module tb_upsample2x;
    localparam ADDR_W = 8;
    localparam DATA_W = 128;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg trig = 1'b0;
    reg [ADDR_W-1:0] src_base = 8'd20;
    reg [ADDR_W-1:0] dst_base = 8'd0;
    wire [ADDR_W-1:0] addr;
    wire en;
    wire we;
    wire [DATA_W-1:0] wdata;
    wire [DATA_W-1:0] rdata;
    wire busy;
    wire done;

    reg [DATA_W-1:0] mem [0:255];
    integer i;

    assign rdata = mem[addr];

    upsample2x #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .i_trig(trig),
        .i_src_base(src_base),
        .i_dst_base(dst_base),
        .i_in_w(16'd2),
        .i_in_h(16'd2),
        .i_ic_groups(16'd1),
        .o_addr(addr),
        .o_en(en),
        .o_we(we),
        .o_wdata(wdata),
        .i_rdata(rdata),
        .o_busy(busy),
        .o_done(done)
    );

    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (en && we)
            mem[addr] <= wdata;
    end

    initial begin
        for (i = 0; i < 256; i = i + 1)
            mem[i] = 128'h0;

        mem[20] = 128'h00000000000000000000000000000011;
        mem[21] = 128'h00000000000000000000000000000022;
        mem[22] = 128'h00000000000000000000000000000033;
        mem[23] = 128'h00000000000000000000000000000044;

        repeat (3) @(posedge clk);
        rst_n <= 1'b1;
        @(posedge clk);
        trig <= 1'b1;
        @(posedge clk);
        trig <= 1'b0;

        wait (done);
        @(posedge clk);

        if (mem[0]  !== 128'h11 || mem[1]  !== 128'h11 ||
            mem[4]  !== 128'h11 || mem[5]  !== 128'h11 ||
            mem[2]  !== 128'h22 || mem[3]  !== 128'h22 ||
            mem[6]  !== 128'h22 || mem[7]  !== 128'h22 ||
            mem[8]  !== 128'h33 || mem[9]  !== 128'h33 ||
            mem[12] !== 128'h33 || mem[13] !== 128'h33 ||
            mem[10] !== 128'h44 || mem[11] !== 128'h44 ||
            mem[14] !== 128'h44 || mem[15] !== 128'h44) begin
            $display("TB_UPSAMPLE2X FAIL");
            $finish;
        end

        $display("TB_UPSAMPLE2X PASS");
        $finish;
    end
endmodule
