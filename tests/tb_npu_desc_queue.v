`timescale 1ns/1ps

module tb_npu_desc_queue;
    localparam REG_ADDR_W = 12;
    localparam REG_DATA_W = 32;

    reg clk = 1'b1;
    reg rst_n = 1'b0;

    reg                    s_axi_awvalid; wire s_axi_awready;
    reg  [REG_ADDR_W-1:0]  s_axi_awaddr;
    reg                    s_axi_wvalid;  wire s_axi_wready;
    reg  [REG_DATA_W-1:0]  s_axi_wdata;
    reg  [3:0]             s_axi_wstrb;
    wire                   s_axi_bvalid;  reg s_axi_bready;
    wire [1:0]             s_axi_bresp;
    reg                    s_axi_arvalid; wire s_axi_arready;
    reg  [REG_ADDR_W-1:0]  s_axi_araddr;
    wire                   s_axi_rvalid;  reg s_axi_rready;
    wire [REG_DATA_W-1:0]  s_axi_rdata;
    wire [1:0]             s_axi_rresp;

    wire                   m_axi_arvalid; reg m_axi_arready = 1'b1;
    wire [3:0]             m_axi_arid;    wire [31:0] m_axi_araddr;
    wire [7:0]             m_axi_arlen;   wire [2:0] m_axi_arsize; wire [1:0] m_axi_arburst;
    reg                    m_axi_rvalid = 1'b0; wire m_axi_rready;
    reg  [3:0]             m_axi_rid = 4'd0; reg [127:0] m_axi_rdata = 128'd0;
    reg  [1:0]             m_axi_rresp = 2'b00; reg m_axi_rlast = 1'b0;
    wire                   m_axi_awvalid; reg m_axi_awready = 1'b1;
    wire [3:0]             m_axi_awid;    wire [31:0] m_axi_awaddr;
    wire [7:0]             m_axi_awlen;   wire [2:0] m_axi_awsize; wire [1:0] m_axi_awburst;
    wire                   m_axi_wvalid;  reg m_axi_wready = 1'b1;
    wire [127:0]           m_axi_wdata;   wire [15:0] m_axi_wstrb; wire m_axi_wlast;
    reg                    m_axi_bvalid = 1'b0; wire m_axi_bready;
    reg  [3:0]             m_axi_bid = 4'd0; reg [1:0] m_axi_bresp = 2'b00;

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

    reg [127:0] ddr [0:63];
    reg [7:0] rd_base;
    reg [7:0] rd_cnt;
    reg [7:0] rd_last_cnt;
    integer errors = 0;
    integer wait_i;
    integer ic, oc;
    reg [31:0] rd_data;
    reg [127:0] actw, wgtw, outw;
    reg [7:0] got, exp;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_rvalid <= 1'b0;
            m_axi_rlast <= 1'b0;
            rd_base <= 8'd0;
            rd_cnt <= 8'd0;
            rd_last_cnt <= 8'd0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                rd_base <= m_axi_araddr[9:4];
                rd_cnt <= 8'd0;
                rd_last_cnt <= m_axi_arlen;
                m_axi_rvalid <= 1'b1;
                m_axi_rlast <= (m_axi_arlen == 8'd0);
            end else if (m_axi_rvalid && m_axi_rready) begin
                if (rd_cnt == rd_last_cnt) begin
                    m_axi_rvalid <= 1'b0;
                    m_axi_rlast <= 1'b0;
                end else begin
                    rd_cnt <= rd_cnt + 8'd1;
                    m_axi_rlast <= (rd_cnt + 8'd1 == rd_last_cnt);
                end
            end
        end
    end

    always @(*) begin
        m_axi_rdata = ddr[rd_base + rd_cnt];
    end

    task reg_write(input [11:0] addr, input [31:0] data);
        begin
            @(negedge clk);
            s_axi_awvalid = 1'b1; s_axi_awaddr = addr;
            s_axi_wvalid  = 1'b1; s_axi_wdata  = data; s_axi_wstrb = 4'hF;
            s_axi_bready  = 1'b1;
            @(negedge clk);
            s_axi_awvalid = 1'b0; s_axi_wvalid = 1'b0;
            while (!s_axi_bvalid) @(negedge clk);
            @(negedge clk);
            s_axi_bready = 1'b0;
        end
    endtask

    task reg_read(input [11:0] addr, output [31:0] data);
        begin
            @(negedge clk);
            s_axi_arvalid = 1'b1; s_axi_araddr = addr; s_axi_rready = 1'b1;
            @(negedge clk);
            s_axi_arvalid = 1'b0;
            while (!s_axi_rvalid) @(negedge clk);
            data = s_axi_rdata;
            @(negedge clk);
            s_axi_rready = 1'b0;
        end
    endtask

    task act_poke(input integer a, input [127:0] d); begin dut.u_act_sram.u_bram.mem[a] = d; end endtask
    task wgt_poke(input integer a, input [127:0] d); begin dut.u_wgt_sram.u_bram.mem[a] = d; end endtask
    task out_clear(input integer a); begin dut.u_out_sram.u_bram.mem[a] = 128'd0; end endtask

    initial begin
        s_axi_awvalid=0; s_axi_wvalid=0; s_axi_arvalid=0; s_axi_bready=0; s_axi_rready=0;
        s_axi_awaddr=0; s_axi_wdata=0; s_axi_wstrb=0; s_axi_araddr=0;

        ddr[0] = {32'h0, 32'h4000_0100, 32'h0, 32'h0000_0101}; // DMA_DDR_TO_ACT
        ddr[1] = {32'd2, 32'h0, 32'h0, 32'd4};
        ddr[2] = 128'h0;
        ddr[3] = 128'h0;
        ddr[4] = {32'h0, 32'h0, 32'h0, 32'h0000_0108}; // STOP_IRQ
        ddr[5] = 128'h0;
        ddr[6] = 128'h0;
        ddr[7] = 128'h0;
        ddr[16] = 128'h00112233445566778899aabbccddeeff;
        ddr[17] = 128'hffeeddccbbaa99887766554433221100;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (5) @(posedge clk);

        reg_write(12'h400, 32'h4000_0000);
        reg_write(12'h404, 32'h0);
        reg_write(12'h408, 32'd2);
        reg_write(12'h40C, 32'h5); // start + irq_en

        wait_i = 0;
        while (!irq_done && wait_i < 200) begin
            @(posedge clk);
            wait_i = wait_i + 1;
        end
        if (!irq_done) begin
            $display("FAIL timeout waiting descriptor irq");
            errors = errors + 1;
        end

        reg_read(12'h410, rd_data);
        if ((rd_data & 32'h2) == 0 || (rd_data & 32'h4) != 0) begin
            $display("FAIL desc status=0x%08h", rd_data);
            errors = errors + 1;
        end
        reg_read(12'h414, rd_data);
        if (rd_data != 32'd1) begin
            $display("FAIL desc pc=%0d", rd_data);
            errors = errors + 1;
        end
        if (dut.u_act_sram.u_bram.mem[4] !== 128'h00112233445566778899aabbccddeeff) begin
            $display("FAIL act[4]=0x%032h", dut.u_act_sram.u_bram.mem[4]);
            errors = errors + 1;
        end
        if (dut.u_act_sram.u_bram.mem[5] !== 128'hffeeddccbbaa99887766554433221100) begin
            $display("FAIL act[5]=0x%032h", dut.u_act_sram.u_bram.mem[5]);
            errors = errors + 1;
        end

        reg_write(12'h40C, 32'hC); // clear_done + irq_en
        repeat (5) @(posedge clk);

        actw = 128'd0;
        for (ic = 0; ic < 16; ic = ic + 1)
            actw[ic*8 +: 8] = 8'd1;
        act_poke(0, actw);
        for (oc = 0; oc < 16; oc = oc + 1) begin
            wgtw = 128'd0;
            for (ic = 0; ic < 16; ic = ic + 1)
                if (ic <= oc)
                    wgtw[ic*8 +: 8] = 8'd1;
            wgt_poke(oc, wgtw);
        end

        ddr[32] = {32'h0, 32'h0, 32'h0, 32'h0020_0107}; // GEMM + ReLU
        ddr[33] = {32'h0, 32'h0, 32'h0, 32'd8};          // out=8
        ddr[34] = {32'h4000_0300, 32'h0001_0101, 32'h0010_0010, 32'h0001_0001};
        ddr[35] = {32'h0, 32'h0, 32'h0, 32'd16};         // qparam_count=16
        ddr[36] = {32'h0, 32'h0, 32'h0, 32'h0000_0108}; // STOP_IRQ
        ddr[37] = 128'h0;
        ddr[38] = 128'h0;
        ddr[39] = 128'h0;
        for (oc = 0; oc < 16; oc = oc + 1)
            ddr[48 + oc] = {26'd0, 6'd0, 32'd0, 32'd1, 32'd0};

        reg_write(12'h400, 32'h4000_0200);
        reg_write(12'h404, 32'h0);
        reg_write(12'h408, 32'd2);
        reg_write(12'h40C, 32'h5); // start + irq_en

        wait_i = 0;
        while (!irq_done && wait_i < 5000) begin
            @(posedge clk);
            wait_i = wait_i + 1;
        end
        if (!irq_done) begin
            $display("FAIL timeout waiting GEMM descriptor irq");
            errors = errors + 1;
        end

        reg_read(12'h410, rd_data);
        if ((rd_data & 32'h2) == 0 || (rd_data & 32'h4) != 0) begin
            $display("FAIL GEMM desc status=0x%08h", rd_data);
            errors = errors + 1;
        end
        outw = dut.u_out_sram.u_bram.mem[8];
        for (oc = 0; oc < 16; oc = oc + 1) begin
            got = outw[oc*8 +: 8];
            exp = oc + 1;
            if (got !== exp) begin
                $display("FAIL GEMM desc out[%0d]=%0d exp=%0d", oc, got, exp);
                errors = errors + 1;
            end
        end

        reg_write(12'h40C, 32'hC); // clear_done + irq_en
        repeat (5) @(posedge clk);

        actw = 128'd0;
        for (ic = 0; ic < 16; ic = ic + 1)
            actw[ic*8 +: 8] = 8'd2;
        act_poke(8192, actw);
        for (oc = 0; oc < 16; oc = oc + 1) begin
            wgtw = 128'd0;
            for (ic = 0; ic < 16; ic = ic + 1)
                if (ic < 4)
                    wgtw[ic*8 +: 8] = 8'd1;
            wgt_poke(8192 + oc, wgtw);
        end
        out_clear(12);

        ddr[40] = {32'h0, 32'h0, 32'h0, 32'h0422_0107}; // GEMM + PONG + ReLU + reduce
        ddr[41] = {32'h0, 32'h0, 32'h0, 32'd12};         // out=12
        ddr[42] = {32'h4000_0300, 32'h0001_0101, 32'h0010_0010, 32'h0001_0001};
        ddr[43] = {32'h0, 32'h0, 32'h0, 32'd16};         // qparam_count=16
        ddr[44] = {32'h0, 32'h0, 32'h0, 32'h0000_0108}; // STOP_IRQ
        ddr[45] = 128'h0;
        ddr[46] = 128'h0;
        ddr[47] = 128'h0;

        reg_write(12'h400, 32'h4000_0280);
        reg_write(12'h404, 32'h0);
        reg_write(12'h408, 32'd2);
        reg_write(12'h40C, 32'h5); // start + irq_en

        wait_i = 0;
        while (!irq_done && wait_i < 5000) begin
            @(posedge clk);
            wait_i = wait_i + 1;
        end
        if (!irq_done) begin
            $display("FAIL timeout waiting PONG GEMM descriptor irq");
            errors = errors + 1;
        end

        reg_read(12'h410, rd_data);
        if ((rd_data & 32'h2) == 0 || (rd_data & 32'h4) != 0) begin
            $display("FAIL PONG GEMM desc status=0x%08h", rd_data);
            errors = errors + 1;
        end
        outw = dut.u_out_sram.u_bram.mem[12];
        for (oc = 0; oc < 16; oc = oc + 1) begin
            got = outw[oc*8 +: 8];
            exp = 8;
            if (got !== exp) begin
                $display("FAIL PONG GEMM desc out[%0d]=%0d exp=%0d", oc, got, exp);
                errors = errors + 1;
            end
        end

        if (errors == 0)
            $display("NPU DESC QUEUE TEST PASS");
        else
            $display("NPU DESC QUEUE TEST FAIL errors=%0d", errors);
        $finish;
    end
endmodule
