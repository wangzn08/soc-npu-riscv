`timescale 1ns/1ps
// ===================================================================
// npu_top integration test harness.
//
// Drives npu_top directly: writes config registers over the AXI4-Lite
// slave, preloads Act/Wgt SRAM by hierarchical poke (DMA bypassed, master
// tied idle), starts the op, waits for irq_done, then reads Out SRAM by
// hierarchical peek and compares to a software golden.
//
// SRAM hierarchy:  dut.u_act_sram.u_bram.mem / u_wgt_sram / u_out_sram
//   ping bank PING_BASE=0  => logical addr == physical addr.
//
// Weight layout (wgt_reader header): word(oc,icg,ko) =
//   oc*ic_groups*KO + icg*KO + ko ; 128-bit word = 16 INT8, ic innermost.
//
// Test GEMM_KNOWN validates the harness on the existing GEMM path:
//   IC=16, OC=16, act[ic]=1, wgt[oc][ic] = (ic <= oc) ? 1 : 0
//   => out[oc] = sum_ic act*wgt = oc+1  (bias 0, scale 1, shift 0, ReLU)
// ===================================================================
module tb_npu_integ;
    localparam REG_ADDR_W = 10, REG_DATA_W = 32;

    reg clk = 1, rst_n = 0;

    // AXI4-Lite slave (config)
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

    // AXI master (DDR) — tied idle (DMA unused; SRAM preloaded directly)
    wire                   m_axi_arvalid; reg m_axi_arready = 1'b1;
    wire [3:0]             m_axi_arid;    wire [31:0] m_axi_araddr;
    wire [7:0]             m_axi_arlen;   wire [2:0]  m_axi_arsize; wire [1:0] m_axi_arburst;
    reg                    m_axi_rvalid = 1'b0; wire m_axi_rready;
    reg  [3:0]             m_axi_rid = 0; reg [127:0] m_axi_rdata = 0; reg [1:0] m_axi_rresp = 0; reg m_axi_rlast = 0;
    wire                   m_axi_awvalid; reg m_axi_awready = 1'b1;
    wire [3:0]             m_axi_awid;    wire [31:0] m_axi_awaddr;
    wire [7:0]             m_axi_awlen;   wire [2:0]  m_axi_awsize; wire [1:0] m_axi_awburst;
    wire                   m_axi_wvalid;  reg m_axi_wready = 1'b1;
    wire [127:0]           m_axi_wdata;   wire [15:0] m_axi_wstrb; wire m_axi_wlast;
    reg                    m_axi_bvalid = 1'b0; wire m_axi_bready;
    reg  [3:0]             m_axi_bid = 0; reg [1:0] m_axi_bresp = 0;

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

    integer errors = 0;

    // ---- AXI-Lite single register write (param_regfile is always-ready) ----
    task reg_write(input [9:0] addr, input [31:0] data);
        begin
            @(negedge clk);
            s_axi_awvalid = 1; s_axi_awaddr = addr;
            s_axi_wvalid  = 1; s_axi_wdata  = data; s_axi_wstrb = 4'hF;
            s_axi_bready  = 1;
            @(negedge clk);
            s_axi_awvalid = 0; s_axi_wvalid = 0;
        end
    endtask

    // ---- SRAM hierarchical poke (ping bank, logical==physical) ----
    task act_poke(input integer a, input [127:0] d); begin dut.u_act_sram.u_bram.mem[a] = d; end endtask
    task wgt_poke(input integer a, input [127:0] d); begin dut.u_wgt_sram.u_bram.mem[a] = d; end endtask

    // ---- Wait for op completion ----
    task wait_done(input integer max);
        integer i;
        begin
            i = 0;
            while (!irq_done && i < max) begin @(posedge clk); i = i + 1; end
            if (!irq_done) $display("  [FAIL] timeout waiting for irq_done");
        end
    endtask

    // CTRL bits
    localparam C_START = 32'h1, C_PINGPONG = 32'h2, C_RELU = 32'h20, C_GEMM = 32'h80;
    localparam C_PW = 32'h4000;  // CTRL[14] pw_en
    // Register byte offsets
    localparam R_CTRL=10'h00, R_INW=10'h20, R_INH=10'h24, R_IC=10'h28, R_OC=10'h2C,
               R_KERN=10'h30, R_STR=10'h34, R_ACTA=10'h08, R_WGTA=10'h10, R_OUTA=10'h18;
    function [9:0] R_BIAS (input integer ch); R_BIAS  = 10'h40 + ch*4; endfunction
    function [9:0] R_SCALE(input integer ch); R_SCALE = 10'h80 + ch*4; endfunction
    function [9:0] R_SHIFT(input integer ch); R_SHIFT = 10'hC0 + ch*4; endfunction

    integer ic, oc, p;
    reg [127:0] actw, wgtw, outw;
    reg [7:0] got, exp;

    initial begin
        s_axi_awvalid=0; s_axi_wvalid=0; s_axi_arvalid=0; s_axi_bready=0; s_axi_rready=1;
        s_axi_awaddr=0; s_axi_wdata=0; s_axi_wstrb=0; s_axi_araddr=0;
        repeat (4) @(negedge clk); rst_n = 1; repeat (2) @(negedge clk);

        // ---- Preload Act SRAM: act[ic] = 1, one IC-tile word at addr 0 ----
        actw = 128'd0;
        for (ic = 0; ic < 16; ic = ic + 1) actw[ic*8 +: 8] = 8'd1;
        act_poke(0, actw);

        // ---- Preload Wgt SRAM: word(oc) = {ic<=oc ? 1 : 0} ----
        for (oc = 0; oc < 16; oc = oc + 1) begin
            wgtw = 128'd0;
            for (ic = 0; ic < 16; ic = ic + 1)
                if (ic <= oc) wgtw[ic*8 +: 8] = 8'd1;
            wgt_poke(oc, wgtw);
        end

        // ---- Configure a 16x16 GEMM (ping bank) ----
        reg_write(R_INW, 1);  reg_write(R_INH, 1);
        reg_write(R_IC, 16);  reg_write(R_OC, 16);
        reg_write(R_KERN, (1<<8)|1); reg_write(R_STR, (1<<8)|1);
        reg_write(R_ACTA, 0); reg_write(R_WGTA, 0); reg_write(R_OUTA, 0);
        for (oc = 0; oc < 16; oc = oc + 1) begin
            reg_write(R_BIAS(oc), 0);
            reg_write(R_SCALE(oc), 1);
            reg_write(R_SHIFT(oc), 0);
        end

        // ---- Start GEMM (ping bank, no PINGPONG), wait ----
        reg_write(R_CTRL, C_START | C_GEMM | C_RELU);
        wait_done(5000);
        repeat (5) @(negedge clk);

        // ---- Read Out SRAM word 0 and check out[oc] == oc+1 ----
        outw = dut.u_out_sram.u_bram.mem[0];
        for (oc = 0; oc < 16; oc = oc + 1) begin
            got = outw[oc*8 +: 8];
            exp = oc + 1;
            if (got !== exp) begin
                $display("  [FAIL] GEMM out[%0d]=%0d exp=%0d", oc, got, exp);
                errors = errors + 1;
            end
        end
        if (errors == 0) $display("  [PASS] GEMM_KNOWN: out[oc]=oc+1");

        // ================= TEST: 1x1 pointwise (pw_en) =================
        // in 2x2, IC=16, OC=16. act[p][ic]=p+1 (all ic); wgt[oc][ic]=(ic<(oc%4+1))?1:0
        // => out[p][oc] = (oc%4+1)*(p+1), one word per pixel at out addr p.
        for (p = 0; p < 4; p = p + 1) begin
            actw = 128'd0;
            for (ic = 0; ic < 16; ic = ic + 1) actw[ic*8 +: 8] = p + 1;
            act_poke(p, actw);
        end
        for (oc = 0; oc < 16; oc = oc + 1) begin
            wgtw = 128'd0;
            for (ic = 0; ic < 16; ic = ic + 1)
                if (ic < (oc % 4 + 1)) wgtw[ic*8 +: 8] = 8'd1;
            wgt_poke(oc, wgtw);
        end
        reg_write(R_INW, 2);  reg_write(R_INH, 2);
        reg_write(R_IC, 16);  reg_write(R_OC, 16);
        reg_write(R_KERN, (1<<8)|1); reg_write(R_STR, (1<<8)|1);
        reg_write(R_ACTA, 0); reg_write(R_WGTA, 0); reg_write(R_OUTA, 0);
        for (oc = 0; oc < 16; oc = oc + 1) begin
            reg_write(R_BIAS(oc), 0); reg_write(R_SCALE(oc), 1); reg_write(R_SHIFT(oc), 0);
        end
        reg_write(R_CTRL, C_START | C_PW | C_RELU);
        wait_done(20000);
        repeat (5) @(negedge clk);
        for (p = 0; p < 4; p = p + 1) begin
            outw = dut.u_out_sram.u_bram.mem[p];
            for (oc = 0; oc < 16; oc = oc + 1) begin
                got = outw[oc*8 +: 8];
                exp = (oc % 4 + 1) * (p + 1);
                if (got !== exp) begin
                    $display("  [FAIL] PW out[p=%0d][oc=%0d]=%0d exp=%0d", p, oc, got, exp);
                    errors = errors + 1;
                end
            end
        end
        if (errors == 0) $display("  [PASS] PW_1x1: out[p][oc]=(oc%%4+1)*(p+1)");

        if (errors == 0) $display("TB_NPU_INTEG PASS");
        else $display("TB_NPU_INTEG FAIL errors=%0d", errors);
        $finish;
    end

    initial begin #2000000 $display("TB_NPU_INTEG FAIL global timeout"); $finish; end
endmodule
