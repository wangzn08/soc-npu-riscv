`timescale 1ns/1ps

module tb_descriptor_engine;
    reg clk = 1'b1;
    reg rst_n = 1'b0;

    reg [31:0] desc_base_lo = 32'h4000_0000;
    reg [31:0] desc_base_hi = 32'd0;
    reg [15:0] desc_count = 16'd2;
    reg desc_start = 1'b0;
    reg desc_abort = 1'b0;
    reg desc_irq_en = 1'b1;
    reg desc_clear_done = 1'b0;
    reg global_idle = 1'b1;

    wire busy, done, err, aborted, irq;
    wire [15:0] pc;
    wire [7:0] err_code;
    wire dma_rd_req, dma_wr_req, copy_trig, expand_trig;
    wire [31:0] dma_rd_ddr_addr, dma_wr_ddr_addr;
    wire [15:0] dma_rd_len, dma_wr_len;
    wire [13:0] dma_rd_sram_base, dma_wr_sram_base;
    wire dma_sram_sel, dma_out_rd_sel;
    wire [1:0] dma_rd_sram_sel;
    wire dma_act_ping_sel, dma_wgt_ping_sel, dma_out_ping_sel;

    wire arvalid;
    reg arready = 1'b1;
    wire [31:0] araddr;
    wire [7:0] arlen;
    wire [2:0] arsize;
    wire [1:0] arburst;
    reg rvalid = 1'b0;
    wire rready;
    reg [127:0] rdata = 128'd0;
    reg [1:0] rresp = 2'b00;
    reg rlast = 1'b0;

    descriptor_engine dut (
        .clk(clk),
        .rst_n(rst_n),
        .i_desc_base_lo(desc_base_lo),
        .i_desc_base_hi(desc_base_hi),
        .i_desc_count(desc_count),
        .i_desc_start(desc_start),
        .i_desc_abort(desc_abort),
        .i_desc_irq_en(desc_irq_en),
        .i_desc_clear_done(desc_clear_done),
        .i_global_idle(global_idle),
        .o_busy(busy),
        .o_done(done),
        .o_err(err),
        .o_aborted(aborted),
        .o_pc(pc),
        .o_err_code(err_code),
        .o_irq(irq),
        .m_axi_arvalid(arvalid),
        .m_axi_arready(arready),
        .m_axi_araddr(araddr),
        .m_axi_arlen(arlen),
        .m_axi_arsize(arsize),
        .m_axi_arburst(arburst),
        .m_axi_rvalid(rvalid),
        .m_axi_rready(rready),
        .m_axi_rdata(rdata),
        .m_axi_rresp(rresp),
        .m_axi_rlast(rlast),
        .o_dma_rd_req(dma_rd_req),
        .o_dma_wr_req(dma_wr_req),
        .o_dma_rd_ddr_addr(dma_rd_ddr_addr),
        .o_dma_wr_ddr_addr(dma_wr_ddr_addr),
        .o_dma_rd_len(dma_rd_len),
        .o_dma_wr_len(dma_wr_len),
        .o_dma_rd_sram_base(dma_rd_sram_base),
        .o_dma_wr_sram_base(dma_wr_sram_base),
        .o_dma_sram_sel(dma_sram_sel),
        .o_dma_out_rd_sel(dma_out_rd_sel),
        .o_dma_rd_sram_sel(dma_rd_sram_sel),
        .o_dma_act_ping_sel(dma_act_ping_sel),
        .o_dma_wgt_ping_sel(dma_wgt_ping_sel),
        .o_dma_out_ping_sel(dma_out_ping_sel),
        .o_copy_trig(copy_trig),
        .o_expand_trig(expand_trig),
        .i_dma_rd_done(1'b0),
        .i_dma_wr_done(1'b0),
        .i_copy_done(1'b0),
        .i_expand_done(1'b0)
    );

    always #5 clk = ~clk;

    reg [127:0] mem [0:15];
    reg [7:0] rd_base;
    reg [1:0] rd_cnt;
    integer errors = 0;
    integer wait_i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rvalid <= 1'b0;
            rlast <= 1'b0;
            rd_base <= 8'd0;
            rd_cnt <= 2'd0;
        end else begin
            if (arvalid && arready) begin
                rd_base <= araddr[9:4];
                rd_cnt <= 2'd0;
                rvalid <= 1'b1;
                rlast <= 1'b0;
            end else if (rvalid && rready) begin
                if (rd_cnt == 2'd3) begin
                    rvalid <= 1'b0;
                    rlast <= 1'b0;
                end else begin
                    rd_cnt <= rd_cnt + 2'd1;
                    rlast <= (rd_cnt == 2'd2);
                end
            end
        end
    end

    always @(*) begin
        rdata = mem[rd_base + rd_cnt];
    end

    task pulse_start;
        begin
            @(negedge clk);
            desc_start = 1'b1;
            @(negedge clk);
            desc_start = 1'b0;
        end
    endtask

    initial begin
        mem[0] = {32'h0, 32'h0, 32'h0, 32'h0000_0100}; // NOP
        mem[1] = 128'h0;
        mem[2] = 128'h0;
        mem[3] = 128'h0;
        mem[4] = {32'h0, 32'h0, 32'h0, 32'h0000_0108}; // STOP_IRQ
        mem[5] = 128'h0;
        mem[6] = 128'h0;
        mem[7] = 128'h0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        pulse_start();

        wait_i = 0;
        while (!done && !err && wait_i < 100) begin
            @(posedge clk);
            wait_i = wait_i + 1;
        end

        if (!done || err || pc != 16'd1 || !irq) begin
            $display("FAIL nop_stop done=%0b err=%0b pc=%0d irq=%0b err_code=%0d",
                     done, err, pc, irq, err_code);
            errors = errors + 1;
        end

        if (errors == 0)
            $display("DESC_ENGINE TEST PASS");
        else
            $display("DESC_ENGINE TEST FAIL errors=%0d", errors);
        $finish;
    end
endmodule
