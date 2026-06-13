`timescale 1ns/1ps
// ---------------------------------------------------------------------------
// Directed test for axi_upsizer_32_128: a 32-bit single-beat master (the CPU
// side) writing/reading through the upsizer into a 128-bit shared memory.
//
// Key checks:
//   * 4 separate 32-bit writes to the 4 lanes of one 128-bit word each land in
//     their own lane and do NOT clobber the others  (WSTRB lane steering).
//   * read-back muxes the correct 32-bit word out of the 128-bit beat.
//   * a partial-strobe (byte) write only touches the addressed bytes.
// ---------------------------------------------------------------------------
module tb_axi_upsizer;

    reg clk = 1'b0, resetn = 1'b0;
    integer err = 0;
    reg [31:0] rd;

    always #5 clk = ~clk;

    // 32-bit CPU-side
    reg         s_awvalid=0; wire s_awready; reg [31:0] s_awaddr=0;
    reg         s_wvalid=0;  wire s_wready;  reg [31:0] s_wdata=0; reg [3:0] s_wstrb=0; reg s_wlast=0;
    wire        s_bvalid;    reg  s_bready=0; wire [1:0] s_bresp;
    reg         s_arvalid=0; wire s_arready; reg [31:0] s_araddr=0;
    wire        s_rvalid;    reg  s_rready=0; wire [31:0] s_rdata; wire [1:0] s_rresp;

    // 128-bit memory-side
    wire        m_awvalid, m_awready; wire [31:0] m_awaddr; wire [7:0] m_awlen; wire [2:0] m_awsize; wire [1:0] m_awburst;
    wire        m_wvalid,  m_wready;  wire [127:0] m_wdata; wire [15:0] m_wstrb; wire m_wlast;
    wire        m_bvalid,  m_bready;  wire [1:0] m_bresp;
    wire        m_arvalid, m_arready; wire [31:0] m_araddr; wire [7:0] m_arlen; wire [2:0] m_arsize; wire [1:0] m_arburst;
    wire        m_rvalid,  m_rready;  wire [127:0] m_rdata; wire [1:0] m_rresp; wire m_rlast;

    axi_upsizer_32_128 dut (
        .clk(clk), .resetn(resetn),
        .s_awvalid(s_awvalid), .s_awready(s_awready), .s_awaddr(s_awaddr), .s_awsize(3'd2), .s_awburst(2'b01),
        .s_wvalid(s_wvalid), .s_wready(s_wready), .s_wdata(s_wdata), .s_wstrb(s_wstrb), .s_wlast(s_wlast),
        .s_bvalid(s_bvalid), .s_bready(s_bready), .s_bresp(s_bresp),
        .s_arvalid(s_arvalid), .s_arready(s_arready), .s_araddr(s_araddr), .s_arsize(3'd2), .s_arburst(2'b01),
        .s_rvalid(s_rvalid), .s_rready(s_rready), .s_rdata(s_rdata), .s_rresp(s_rresp),
        .m_awvalid(m_awvalid), .m_awready(m_awready), .m_awaddr(m_awaddr), .m_awlen(m_awlen), .m_awsize(m_awsize), .m_awburst(m_awburst),
        .m_wvalid(m_wvalid), .m_wready(m_wready), .m_wdata(m_wdata), .m_wstrb(m_wstrb), .m_wlast(m_wlast),
        .m_bvalid(m_bvalid), .m_bready(m_bready), .m_bresp(m_bresp),
        .m_arvalid(m_arvalid), .m_arready(m_arready), .m_araddr(m_araddr), .m_arlen(m_arlen), .m_arsize(m_arsize), .m_arburst(m_arburst),
        .m_rvalid(m_rvalid), .m_rready(m_rready), .m_rdata(m_rdata), .m_rresp(m_rresp), .m_rlast(m_rlast)
    );

    axi_full_slave_v1_0_S00_AXI #(
        .C_S_AXI_ID_WIDTH(1), .C_S_AXI_DATA_WIDTH(128), .C_S_AXI_ADDR_WIDTH(24)
    ) mem (
        .S_AXI_ACLK(clk), .S_AXI_ARESETN(resetn),
        .S_AXI_AWID(1'b0), .S_AXI_AWADDR(m_awaddr[23:0]), .S_AXI_AWLEN(m_awlen), .S_AXI_AWSIZE(m_awsize),
        .S_AXI_AWBURST(m_awburst), .S_AXI_AWLOCK(1'b0), .S_AXI_AWCACHE(4'd0), .S_AXI_AWPROT(3'd0),
        .S_AXI_AWQOS(4'd0), .S_AXI_AWREGION(4'd0), .S_AXI_AWVALID(m_awvalid), .S_AXI_AWREADY(m_awready),
        .S_AXI_WDATA(m_wdata), .S_AXI_WSTRB(m_wstrb), .S_AXI_WLAST(m_wlast), .S_AXI_WVALID(m_wvalid), .S_AXI_WREADY(m_wready),
        .S_AXI_BRESP(m_bresp), .S_AXI_BVALID(m_bvalid), .S_AXI_BREADY(m_bready),
        .S_AXI_ARID(1'b0), .S_AXI_ARADDR(m_araddr[23:0]), .S_AXI_ARLEN(m_arlen), .S_AXI_ARSIZE(m_arsize),
        .S_AXI_ARBURST(m_arburst), .S_AXI_ARLOCK(1'b0), .S_AXI_ARCACHE(4'd0), .S_AXI_ARPROT(3'd0),
        .S_AXI_ARQOS(4'd0), .S_AXI_ARREGION(4'd0), .S_AXI_ARVALID(m_arvalid), .S_AXI_ARREADY(m_arready),
        .S_AXI_RDATA(m_rdata), .S_AXI_RRESP(m_rresp), .S_AXI_RLAST(m_rlast), .S_AXI_RVALID(m_rvalid), .S_AXI_RREADY(m_rready)
    );

    task cpu_write(input [31:0] addr, input [31:0] data, input [3:0] strb);
        begin
            @(posedge clk); #1;
            s_awvalid=1; s_awaddr=addr;
            s_wvalid=1;  s_wdata=data; s_wstrb=strb; s_wlast=1; s_bready=1;
            // hold AW and W until each is accepted (upsizer fast path may take
            // both in the same cycle, or WCAP fallback in separate cycles)
            while (s_awvalid || s_wvalid) begin
                @(posedge clk);
                if (s_awvalid && s_awready) s_awvalid = 1'b0;
                if (s_wvalid  && s_wready ) s_wvalid  = 1'b0;
            end
            @(posedge clk); while(!s_bvalid) @(posedge clk);
            #1; s_bready=0;
        end
    endtask

    task cpu_read(input [31:0] addr, output [31:0] data);
        begin
            @(posedge clk); #1;
            s_arvalid=1; s_araddr=addr; s_rready=1;
            @(posedge clk); while(!s_arready) @(posedge clk);
            #1; s_arvalid=0;
            @(posedge clk); while(!s_rvalid) @(posedge clk);
            data = s_rdata;
            #1; s_rready=0;
        end
    endtask

    task chk(input [31:0] got, input [31:0] exp, input [127:0] tag);
        begin
            if (got !== exp) begin
                $display("[%0t] MISMATCH %0s: got %h exp %h", $time, tag, got, exp);
                err = err + 1;
            end
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        #1; resetn = 1'b1;
        repeat (2) @(posedge clk);

        // 4 lanes of one 128-bit word @0x100 -- must not clobber each other
        cpu_write(32'h100, 32'hAAAAAAAA, 4'hF);
        cpu_write(32'h104, 32'hBBBBBBBB, 4'hF);
        cpu_write(32'h108, 32'hCCCCCCCC, 4'hF);
        cpu_write(32'h10C, 32'hDDDDDDDD, 4'hF);
        cpu_read(32'h100, rd); chk(rd, 32'hAAAAAAAA, "lane0");
        cpu_read(32'h104, rd); chk(rd, 32'hBBBBBBBB, "lane1");
        cpu_read(32'h108, rd); chk(rd, 32'hCCCCCCCC, "lane2");
        cpu_read(32'h10C, rd); chk(rd, 32'hDDDDDDDD, "lane3");

        // a different 128-bit word
        cpu_write(32'h200, 32'h12345678, 4'hF);
        cpu_read(32'h200, rd); chk(rd, 32'h12345678, "word2");
        // lane0 of 0x100 still intact?
        cpu_read(32'h100, rd); chk(rd, 32'hAAAAAAAA, "lane0_again");

        // partial strobe: byte-lane write (low 2 bytes only) to a fresh word
        cpu_write(32'h300, 32'hFFFFFFFF, 4'h3);
        cpu_read(32'h300, rd); chk(rd, 32'h0000FFFF, "partial_strb");

        repeat (4) @(posedge clk);
        if (err == 0) $display("==== TB_AXI_UPSIZER: ALL PASS ====");
        else          $display("==== TB_AXI_UPSIZER: %0d ERROR(S) ====", err);
        $finish;
    end

    initial begin #200000; $display("*** WATCHDOG TIMEOUT ***"); $finish; end

endmodule
