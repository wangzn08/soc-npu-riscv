`timescale 1ns/1ps
// ===================================================================
// npu_top integration test for the shared Act-SRAM 2x upsample engine.
//
// This drives the same AXI-Lite register path firmware will use, then
// checks that Act SRAM Port B performs the tile-major nearest-neighbor
// expansion:
//   dst(g, 2y+dy, 2x+dx) = src(g, y, x)
// ===================================================================
module tb_npu_upsample2x;
    localparam REG_ADDR_W = 10;
    localparam REG_DATA_W = 32;

    reg clk = 1'b1;
    reg rst_n = 1'b0;

    reg                    s_axi_awvalid; wire s_axi_awready;
    reg  [REG_ADDR_W-1:0]  s_axi_awaddr;
    reg                    s_axi_wvalid;  wire s_axi_wready;
    reg  [REG_DATA_W-1:0]  s_axi_wdata;
    reg  [3:0]             s_axi_wstrb;
    wire                   s_axi_bvalid;  reg  s_axi_bready;
    wire [1:0]             s_axi_bresp;
    reg                    s_axi_arvalid; wire s_axi_arready;
    reg  [REG_ADDR_W-1:0]  s_axi_araddr;
    wire                   s_axi_rvalid;  reg  s_axi_rready;
    wire [REG_DATA_W-1:0]  s_axi_rdata;
    wire [1:0]             s_axi_rresp;

    wire                   m_axi_arvalid; reg m_axi_arready = 1'b1;
    wire [3:0]             m_axi_arid;    wire [31:0] m_axi_araddr;
    wire [7:0]             m_axi_arlen;   wire [2:0]  m_axi_arsize; wire [1:0] m_axi_arburst;
    reg                    m_axi_rvalid = 1'b0; wire m_axi_rready;
    reg  [3:0]             m_axi_rid = 4'd0; reg [127:0] m_axi_rdata = 128'd0; reg [1:0] m_axi_rresp = 2'd0; reg m_axi_rlast = 1'b0;
    wire                   m_axi_awvalid; reg m_axi_awready = 1'b1;
    wire [3:0]             m_axi_awid;    wire [31:0] m_axi_awaddr;
    wire [7:0]             m_axi_awlen;   wire [2:0]  m_axi_awsize; wire [1:0] m_axi_awburst;
    wire                   m_axi_wvalid;  reg m_axi_wready = 1'b1;
    wire [127:0]           m_axi_wdata;   wire [15:0] m_axi_wstrb; wire m_axi_wlast;
    reg                    m_axi_bvalid = 1'b0; wire m_axi_bready;
    reg  [3:0]             m_axi_bid = 4'd0; reg [1:0] m_axi_bresp = 2'd0;

    wire irq_done, dma_rd_done, dma_wr_done;

    npu_top dut (
        .clk(clk), .rst_n(rst_n),
        .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready), .s_axi_awaddr(s_axi_awaddr),
        .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready), .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb),
        .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready), .s_axi_bresp(s_axi_bresp),
        .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready), .s_axi_araddr(s_axi_araddr),
        .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready), .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready), .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready), .m_axi_rid(m_axi_rid), .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp), .m_axi_rlast(m_axi_rlast),
        .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready), .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready), .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready), .m_axi_bid(m_axi_bid), .m_axi_bresp(m_axi_bresp),
        .irq_done(irq_done), .dma_rd_done(dma_rd_done), .dma_wr_done(dma_wr_done)
    );

    always #5 clk = ~clk;

    localparam R_DMA_STATUS  = 10'h140;
    localparam R_DMA_RD_BASE = 10'h12C;
    localparam R_DMA_WR_BASE = 10'h13C;
    localparam R_UP_CFG0     = 10'h3C0;
    localparam R_UP_CFG1     = 10'h3C4;
    localparam R_UP_TRIG     = 10'h3C8;

    localparam SRC_BASE = 10'd4;
    localparam DST_BASE = 10'd64;

    integer i, g, y, x, dy, dx, errors;
    integer src_addr, dst_addr;
    reg [31:0] rd;
    reg [127:0] exp;

    function [127:0] fill_byte(input [7:0] b);
        begin
            fill_byte = {16{b}};
        end
    endfunction

    task reg_write(input [9:0] addr, input [31:0] data);
        begin
            @(negedge clk);
            s_axi_awvalid = 1'b1; s_axi_awaddr = addr;
            s_axi_wvalid  = 1'b1; s_axi_wdata  = data; s_axi_wstrb = 4'hF;
            s_axi_bready  = 1'b1;
            @(negedge clk);
            s_axi_awvalid = 1'b0; s_axi_wvalid = 1'b0;
        end
    endtask

    task reg_read(input [9:0] addr, output [31:0] data);
        begin
            @(negedge clk);
            s_axi_arvalid = 1'b1; s_axi_araddr = addr; s_axi_rready = 1'b1;
            @(negedge clk);
            s_axi_arvalid = 1'b0;
            while (!s_axi_rvalid) @(negedge clk);
            data = s_axi_rdata;
            @(negedge clk);
        end
    endtask

    task act_poke(input integer a, input [127:0] d);
        begin
            dut.u_act_sram.u_bram.mem[a] = d;
        end
    endtask

    initial begin
        errors = 0;
        s_axi_awvalid = 1'b0; s_axi_wvalid = 1'b0; s_axi_arvalid = 1'b0;
        s_axi_awaddr = 10'd0; s_axi_wdata = 32'd0; s_axi_wstrb = 4'd0;
        s_axi_bready = 1'b0; s_axi_rready = 1'b1; s_axi_araddr = 10'd0;

        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        for (i = 0; i < 160; i = i + 1)
            act_poke(i, 128'd0);

        for (g = 0; g < 2; g = g + 1)
            for (y = 0; y < 2; y = y + 1)
                for (x = 0; x < 3; x = x + 1) begin
                    src_addr = SRC_BASE + g * 6 + y * 3 + x;
                    act_poke(src_addr, fill_byte(8'h10 + g * 8'h40 + y * 8'h08 + x[7:0]));
                end

        reg_write(R_DMA_RD_BASE, SRC_BASE);
        reg_write(R_DMA_WR_BASE, DST_BASE);
        reg_write(R_UP_CFG0, {16'd2, 16'd3});  // in_h=2, in_w=3
        reg_write(R_UP_CFG1, 32'd2);           // ic_groups=2
        reg_write(R_UP_TRIG, 32'h1);

        for (i = 0; i < 300; i = i + 1) begin
            reg_read(R_DMA_STATUS, rd);
            if (rd[4]) i = 300;
        end
        if (!rd[4]) begin
            $display("TB_NPU_UPSAMPLE2X FAIL timeout/status rd=%08h", rd);
            $finish;
        end

        repeat (2) @(negedge clk);

        for (g = 0; g < 2; g = g + 1)
            for (y = 0; y < 2; y = y + 1)
                for (x = 0; x < 3; x = x + 1)
                    for (dy = 0; dy < 2; dy = dy + 1)
                        for (dx = 0; dx < 2; dx = dx + 1) begin
                            src_addr = SRC_BASE + g * 6 + y * 3 + x;
                            dst_addr = DST_BASE + g * 24 + (2*y + dy) * 6 + (2*x + dx);
                            exp = dut.u_act_sram.u_bram.mem[src_addr];
                            if (dut.u_act_sram.u_bram.mem[dst_addr] !== exp) begin
                                errors = errors + 1;
                                $display("  [FAIL] g=%0d y=%0d x=%0d dy=%0d dx=%0d dst=%0d exp=%032h got=%032h",
                                         g, y, x, dy, dx, dst_addr, exp, dut.u_act_sram.u_bram.mem[dst_addr]);
                            end
                        end

        if (errors == 0) $display("TB_NPU_UPSAMPLE2X PASS");
        else $display("TB_NPU_UPSAMPLE2X FAIL errors=%0d", errors);
        $finish;
    end
endmodule
