// 将单拍 AXI‑Lite 请求转换为 AXI‑Full 单拍事务
module axi_lite_to_axi_full (
    input  wire        clk,
    input  wire        resetn,

    // AXI‑Lite 从接口
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [31:0] s_axi_awaddr,
    input  wire [2:0]  s_axi_awprot,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    output wire        s_axi_bvalid,
    input  wire        s_axi_bready,
    output wire [1:0]  s_axi_bresp,
    input  wire        s_axi_arvalid,
    output wire        s_axi_arready,
    input  wire [31:0] s_axi_araddr,
    input  wire [2:0]  s_axi_arprot,
    output wire        s_axi_rvalid,
    input  wire        s_axi_rready,
    output wire [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,

    // AXI‑Full 主接口（单拍，无突发）
    output reg         m_axi_awvalid,
    input  wire        m_axi_awready,
    output reg  [31:0] m_axi_awaddr,
    output wire [2:0]  m_axi_awsize,
    output wire [1:0]  m_axi_awburst,
    output reg         m_axi_wvalid,
    input  wire        m_axi_wready,
    output reg  [31:0] m_axi_wdata,
    output reg  [3:0]  m_axi_wstrb,
    output wire        m_axi_wlast,
    input  wire        m_axi_bvalid,
    output wire        m_axi_bready,
    input  wire [1:0]  m_axi_bresp,
    output reg         m_axi_arvalid,
    input  wire        m_axi_arready,
    output reg  [31:0] m_axi_araddr,
    output wire [2:0]  m_axi_arsize,
    output wire [1:0]  m_axi_arburst,
    input  wire        m_axi_rvalid,
    output wire        m_axi_rready,
    input  wire [31:0] m_axi_rdata,
    input  wire [1:0]  m_axi_rresp
);

    // 固定 Full 传输参数 (单拍，4字节，INCR)
    assign m_axi_awsize  = 3'b010;
    assign m_axi_awburst = 2'b01;
    assign m_axi_arsize  = 3'b010;
    assign m_axi_arburst = 2'b01;
    assign m_axi_wlast   = 1'b1;

    // ---- 写事务转换 ----
    assign s_axi_awready = m_axi_awready;
    assign s_axi_wready  = m_axi_wready;
    assign s_axi_bvalid  = m_axi_bvalid;
    assign s_axi_bresp   = m_axi_bresp;
    assign m_axi_bready  = s_axi_bready;

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            m_axi_awvalid <= 1'b0;
            m_axi_awaddr  <= 32'b0;
            m_axi_wvalid  <= 1'b0;
            m_axi_wdata   <= 32'b0;
            m_axi_wstrb   <= 4'b0;
        end else begin
            // 写地址通道
            if (!m_axi_awvalid && s_axi_awvalid) begin
                m_axi_awvalid <= 1'b1;
                m_axi_awaddr  <= s_axi_awaddr;
            end else if (m_axi_awvalid && m_axi_awready) begin
                m_axi_awvalid <= 1'b0;
            end

            // 写数据通道
            if (!m_axi_wvalid && s_axi_wvalid) begin
                m_axi_wvalid <= 1'b1;
                m_axi_wdata  <= s_axi_wdata;
                m_axi_wstrb  <= s_axi_wstrb;
            end else if (m_axi_wvalid && m_axi_wready) begin
                m_axi_wvalid <= 1'b0;
            end
        end
    end

    // ---- 读事务转换 ----
    assign s_axi_arready = m_axi_arready;
    assign s_axi_rvalid  = m_axi_rvalid;
    assign s_axi_rdata   = m_axi_rdata;
    assign s_axi_rresp   = m_axi_rresp;
    assign m_axi_rready  = s_axi_rready;

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            m_axi_arvalid <= 1'b0;
            m_axi_araddr  <= 32'b0;
        end else begin
            if (!m_axi_arvalid && s_axi_arvalid) begin
                m_axi_arvalid <= 1'b1;
                m_axi_araddr  <= s_axi_araddr;
            end else if (m_axi_arvalid && m_axi_arready) begin
                m_axi_arvalid <= 1'b0;
            end
        end
    end

endmodule