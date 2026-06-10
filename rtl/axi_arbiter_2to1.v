// 2:1 AXI4‑Full 仲裁器，固定优先级：Master1 高于 Master0
// 不支持 outstanding 事务（必须等待当前事务完成）
module axi_arbiter_2to1 #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32,
    parameter ID_WIDTH   = 1
) (
    input  wire        clk,
    input  wire        resetn,

    // Master 0 (CPU, 来自桥接器)
    input  wire        m0_axi_awvalid,
    output wire        m0_axi_awready,
    input  wire [ADDR_WIDTH-1:0] m0_axi_awaddr,
    input  wire [7:0]  m0_axi_awlen,
    input  wire [2:0]  m0_axi_awsize,
    input  wire [1:0]  m0_axi_awburst,
    input  wire        m0_axi_wvalid,
    output wire        m0_axi_wready,
    input  wire [DATA_WIDTH-1:0] m0_axi_wdata,
    input  wire [3:0]  m0_axi_wstrb,
    input  wire        m0_axi_wlast,
    output wire        m0_axi_bvalid,
    input  wire        m0_axi_bready,
    output wire [1:0]  m0_axi_bresp,
    input  wire        m0_axi_arvalid,
    output wire        m0_axi_arready,
    input  wire [ADDR_WIDTH-1:0] m0_axi_araddr,
    input  wire [7:0]  m0_axi_arlen,
    input  wire [2:0]  m0_axi_arsize,
    input  wire [1:0]  m0_axi_arburst,
    output wire        m0_axi_rvalid,
    input  wire        m0_axi_rready,
    output wire [DATA_WIDTH-1:0] m0_axi_rdata,
    output wire [1:0]  m0_axi_rresp,
    output wire        m0_axi_rlast,

    // Master 1 (NPU, 优先级高)
    input  wire        m1_axi_awvalid,
    output wire        m1_axi_awready,
    input  wire [ADDR_WIDTH-1:0] m1_axi_awaddr,
    input  wire [7:0]  m1_axi_awlen,
    input  wire [2:0]  m1_axi_awsize,
    input  wire [1:0]  m1_axi_awburst,
    input  wire        m1_axi_wvalid,
    output wire        m1_axi_wready,
    input  wire [DATA_WIDTH-1:0] m1_axi_wdata,
    input  wire [3:0]  m1_axi_wstrb,
    input  wire        m1_axi_wlast,
    output wire        m1_axi_bvalid,
    input  wire        m1_axi_bready,
    output wire [1:0]  m1_axi_bresp,
    input  wire        m1_axi_arvalid,
    output wire        m1_axi_arready,
    input  wire [ADDR_WIDTH-1:0] m1_axi_araddr,
    input  wire [7:0]  m1_axi_arlen,
    input  wire [2:0]  m1_axi_arsize,
    input  wire [1:0]  m1_axi_arburst,
    output wire        m1_axi_rvalid,
    input  wire        m1_axi_rready,
    output wire [DATA_WIDTH-1:0] m1_axi_rdata,
    output wire [1:0]  m1_axi_rresp,
    output wire        m1_axi_rlast,

    // 共享从机接口
    output reg         s_axi_awvalid,
    input  wire        s_axi_awready,
    output reg  [ADDR_WIDTH-1:0] s_axi_awaddr,
    output reg  [7:0]  s_axi_awlen,
    output reg  [2:0]  s_axi_awsize,
    output reg  [1:0]  s_axi_awburst,
    output reg         s_axi_wvalid,
    input  wire        s_axi_wready,
    output reg  [DATA_WIDTH-1:0] s_axi_wdata,
    output reg  [3:0]  s_axi_wstrb,
    output reg         s_axi_wlast,
    input  wire        s_axi_bvalid,
    output wire        s_axi_bready,
    input  wire [1:0]  s_axi_bresp,
    output reg         s_axi_arvalid,
    input  wire        s_axi_arready,
    output reg  [ADDR_WIDTH-1:0] s_axi_araddr,
    output reg  [7:0]  s_axi_arlen,
    output reg  [2:0]  s_axi_arsize,
    output reg  [1:0]  s_axi_arburst,
    input  wire        s_axi_rvalid,
    output wire        s_axi_rready,
    input  wire [DATA_WIDTH-1:0] s_axi_rdata,
    input  wire [1:0]  s_axi_rresp,
    input  wire        s_axi_rlast
);

    // ------------- 写事务仲裁 -------------
    reg        wr_grant;          // 0: master0, 1: master1
    reg        wr_active;         // 写事务进行中
    reg [7:0]  wr_len_remaining;  // 剩余写传输计数（仅用于追踪完成）

    // 写地址选择
    always @* begin
        s_axi_awvalid = 1'b0;
        s_axi_awaddr  = {ADDR_WIDTH{1'b0}};
        s_axi_awlen   = 8'b0;
        s_axi_awsize  = 3'b0;
        s_axi_awburst = 2'b0;

        if (wr_active) begin
            case (wr_grant)
                0: begin
                    s_axi_awvalid = m0_axi_awvalid;
                    s_axi_awaddr  = m0_axi_awaddr;
                    s_axi_awlen   = m0_axi_awlen;
                    s_axi_awsize  = m0_axi_awsize;
                    s_axi_awburst = m0_axi_awburst;
                end
                1: begin
                    s_axi_awvalid = m1_axi_awvalid;
                    s_axi_awaddr  = m1_axi_awaddr;
                    s_axi_awlen   = m1_axi_awlen;
                    s_axi_awsize  = m1_axi_awsize;
                    s_axi_awburst = m1_axi_awburst;
                end
            endcase
        end else begin
            // 无活跃事务，按优先级选择新请求
            if (m1_axi_awvalid) begin
                s_axi_awvalid = 1'b1;
                s_axi_awaddr  = m1_axi_awaddr;
                s_axi_awlen   = m1_axi_awlen;
                s_axi_awsize  = m1_axi_awsize;
                s_axi_awburst = m1_axi_awburst;
            end else if (m0_axi_awvalid) begin
                s_axi_awvalid = 1'b1;
                s_axi_awaddr  = m0_axi_awaddr;
                s_axi_awlen   = m0_axi_awlen;
                s_axi_awsize  = m0_axi_awsize;
                s_axi_awburst = m0_axi_awburst;
            end
        end
    end

    // 写地址 ready 反馈
    assign m0_axi_awready = (wr_active && wr_grant == 0) ? s_axi_awready : (!wr_active && !m1_axi_awvalid && s_axi_awready);
    assign m1_axi_awready = (wr_active && wr_grant == 1) ? s_axi_awready : (!wr_active && s_axi_awready);

    // 写数据通道选择
    always @* begin
        s_axi_wvalid = 1'b0;
        s_axi_wdata  = {DATA_WIDTH{1'b0}};
        s_axi_wstrb  = 4'b0;
        s_axi_wlast  = 1'b0;
        if (wr_active) begin
            case (wr_grant)
                0: begin
                    s_axi_wvalid = m0_axi_wvalid;
                    s_axi_wdata  = m0_axi_wdata;
                    s_axi_wstrb  = m0_axi_wstrb;
                    s_axi_wlast  = m0_axi_wlast;
                end
                1: begin
                    s_axi_wvalid = m1_axi_wvalid;
                    s_axi_wdata  = m1_axi_wdata;
                    s_axi_wstrb  = m1_axi_wstrb;
                    s_axi_wlast  = m1_axi_wlast;
                end
            endcase
        end else begin
            // 空闲时不允许写数据
            s_axi_wvalid = 1'b0;
        end
    end
    assign m0_axi_wready = (wr_active && wr_grant == 0) ? s_axi_wready : 1'b0;
    assign m1_axi_wready = (wr_active && wr_grant == 1) ? s_axi_wready : 1'b0;

    // 写响应反馈
    assign m0_axi_bvalid = (wr_active && wr_grant == 0) ? s_axi_bvalid : 1'b0;
    assign m0_axi_bresp  = s_axi_bresp;
    assign m1_axi_bvalid = (wr_active && wr_grant == 1) ? s_axi_bvalid : 1'b0;
    assign m1_axi_bresp  = s_axi_bresp;
    assign s_axi_bready  = (wr_active && ((wr_grant == 0 && m0_axi_bready) || (wr_grant == 1 && m1_axi_bready))) ? 1'b1 : 1'b0;

    // 写状态机
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            wr_active <= 1'b0;
            wr_grant  <= 1'b0;
        end else begin
            if (!wr_active) begin
                // 发起新写事务
                if (m1_axi_awvalid) begin
                    wr_active <= 1'b1;
                    wr_grant  <= 1'b1;
                end else if (m0_axi_awvalid) begin
                    wr_active <= 1'b1;
                    wr_grant  <= 1'b0;
                end
            end else begin
                // 当前写事务是否完成：收到 B 响应
                if (s_axi_bvalid && s_axi_bready) begin
                    wr_active <= 1'b0;
                end
            end
        end
    end

    // ------------- 读事务仲裁 -------------
    reg        rd_grant;
    reg        rd_active;

    // 读地址选择
    always @* begin
        s_axi_arvalid = 1'b0;
        s_axi_araddr  = {ADDR_WIDTH{1'b0}};
        s_axi_arlen   = 8'b0;
        s_axi_arsize  = 3'b0;
        s_axi_arburst = 2'b0;

        if (rd_active) begin
            case (rd_grant)
                0: begin
                    s_axi_arvalid = m0_axi_arvalid;
                    s_axi_araddr  = m0_axi_araddr;
                    s_axi_arlen   = m0_axi_arlen;
                    s_axi_arsize  = m0_axi_arsize;
                    s_axi_arburst = m0_axi_arburst;
                end
                1: begin
                    s_axi_arvalid = m1_axi_arvalid;
                    s_axi_araddr  = m1_axi_araddr;
                    s_axi_arlen   = m1_axi_arlen;
                    s_axi_arsize  = m1_axi_arsize;
                    s_axi_arburst = m1_axi_arburst;
                end
            endcase
        end else begin
            if (m1_axi_arvalid) begin
                s_axi_arvalid = 1'b1;
                s_axi_araddr  = m1_axi_araddr;
                s_axi_arlen   = m1_axi_arlen;
                s_axi_arsize  = m1_axi_arsize;
                s_axi_arburst = m1_axi_arburst;
            end else if (m0_axi_arvalid) begin
                s_axi_arvalid = 1'b1;
                s_axi_araddr  = m0_axi_araddr;
                s_axi_arlen   = m0_axi_arlen;
                s_axi_arsize  = m0_axi_arsize;
                s_axi_arburst = m0_axi_arburst;
            end
        end
    end

    assign m0_axi_arready = (rd_active && rd_grant == 0) ? s_axi_arready : (!rd_active && !m1_axi_arvalid && s_axi_arready);
    assign m1_axi_arready = (rd_active && rd_grant == 1) ? s_axi_arready : (!rd_active && s_axi_arready);

    // 读数据反馈
    assign m0_axi_rvalid = (rd_active && rd_grant == 0) ? s_axi_rvalid : 1'b0;
    assign m0_axi_rdata  = s_axi_rdata;
    assign m0_axi_rresp  = s_axi_rresp;
    assign m0_axi_rlast  = (rd_active && rd_grant == 0) ? s_axi_rlast : 1'b0;
    assign m1_axi_rvalid = (rd_active && rd_grant == 1) ? s_axi_rvalid : 1'b0;
    assign m1_axi_rdata  = s_axi_rdata;
    assign m1_axi_rresp  = s_axi_rresp;
    assign m1_axi_rlast  = (rd_active && rd_grant == 1) ? s_axi_rlast : 1'b0;
    assign s_axi_rready  = (rd_active && s_axi_rvalid && ((rd_grant == 0 && m0_axi_rready) || (rd_grant == 1 && m1_axi_rready))) ? 1'b1 : 1'b0;

    // 读状态机
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            rd_active <= 1'b0;
            rd_grant  <= 1'b0;
        end else begin
            if (!rd_active) begin
                if (m1_axi_arvalid) begin
                    rd_active <= 1'b1;
                    rd_grant  <= 1'b1;
                end else if (m0_axi_arvalid) begin
                    rd_active <= 1'b1;
                    rd_grant  <= 1'b0;
                end
            end else begin
                // 读事务完成条件：收到最后一个数据（RLAST）
                if (s_axi_rvalid && s_axi_rready && s_axi_rlast) begin
                    rd_active <= 1'b0;
                end
            end
        end
    end

    // synthesis translate_off
    `ifdef DEBUG
    always @(posedge clk) begin
        if (m1_axi_arvalid && !rd_active)
            $display("ARB_RD: m1 arvalid addr=%0h len=%0d", m1_axi_araddr, m1_axi_arlen);
        if (m0_axi_arvalid && !rd_active && !m1_axi_arvalid)
            $display("ARB_RD: m0 arvalid addr=%0h len=%0d", m0_axi_araddr, m0_axi_arlen);
        if (m1_axi_awvalid && !wr_active)
            $display("ARB_WR: m1 awvalid addr=%0h len=%0d", m1_axi_awaddr, m1_axi_awlen);
        if (m0_axi_awvalid && !wr_active && !m1_axi_awvalid)
            $display("ARB_WR: m0 awvalid addr=%0h len=%0d", m0_axi_awaddr, m0_axi_awlen);
        if (s_axi_arvalid && s_axi_arready)
            $display("ARB_RD_ACCEPT: araddr=%0h arlen=%0d rd_active=%0b rd_grant=%0b",
                     s_axi_araddr, s_axi_arlen, rd_active, rd_grant);
        if (s_axi_awvalid && s_axi_awready)
            $display("ARB_WR_ACCEPT: awaddr=%0h awlen=%0d wr_active=%0b wr_grant=%0b",
                     s_axi_awaddr, s_axi_awlen, wr_active, wr_grant);
        if (s_axi_wvalid && s_axi_wready)
            $display("ARB_WR_DATA: wdata=%0h wstrb=%0h wlast=%0b",
                     s_axi_wdata, s_axi_wstrb, s_axi_wlast);
        if (s_axi_rvalid && s_axi_rready)
            $display("ARB_RD_DATA: rdata=%0h rlast=%0b rd_grant=%0b",
                     s_axi_rdata, s_axi_rlast, rd_grant);
    end
    `endif
    // synthesis translate_on

endmodule