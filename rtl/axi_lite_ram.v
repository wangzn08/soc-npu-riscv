// 16MB 单端口 RAM，支持 AXI‑Lite 接口（字数组，便于 $readmemh）
module axi_lite_ram #(
    parameter MEM_SIZE_BYTES = 16 * 1024 * 1024,   // 16MB
    parameter ADDR_WIDTH     = 24                  // 2^24 = 16MB
) (
    input  wire               clk,
    input  wire               resetn,

    // 写地址通道
    input  wire [ADDR_WIDTH-1:0] s_axi_awaddr,
    input  wire                  s_axi_awvalid,
    output wire                  s_axi_awready,

    // 写数据通道
    input  wire [31:0]           s_axi_wdata,
    input  wire [3:0]            s_axi_wstrb,
    input  wire                  s_axi_wvalid,
    output wire                  s_axi_wready,

    // 写响应通道
    output wire [1:0]            s_axi_bresp,
    output wire                  s_axi_bvalid,
    input  wire                  s_axi_bready,

    // 读地址通道
    input  wire [ADDR_WIDTH-1:0] s_axi_araddr,
    input  wire                  s_axi_arvalid,
    output wire                  s_axi_arready,

    // 读数据通道
    output wire [31:0]           s_axi_rdata,
    output wire [1:0]            s_axi_rresp,
    output wire                  s_axi_rvalid,
    input  wire                  s_axi_rready
);

    // 字数组 (每个元素 32 位，共 MEM_SIZE_BYTES/4 个字)
    localparam WORD_COUNT = MEM_SIZE_BYTES / 4;
    localparam WORD_ADDR_WIDTH = ADDR_WIDTH - 2;

    reg [31:0] mem [0:WORD_COUNT-1];
    reg [ADDR_WIDTH-1:0] raddr;

    wire [WORD_ADDR_WIDTH-1:0] waddr_word = s_axi_awaddr[ADDR_WIDTH-1:2];

    // 写地址/数据握手：同时有效时接受
    wire aw_hs = s_axi_awvalid && s_axi_awready;
    wire w_hs  = s_axi_wvalid  && s_axi_wready;

    reg aw_done, w_done;
    wire write_done = aw_done && w_done;

    assign s_axi_awready = !aw_done;
    assign s_axi_wready  = !w_done;

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            aw_done <= 1'b0;
            w_done  <= 1'b0;
        end else begin
            if (aw_hs) aw_done <= 1'b1;
            if (w_hs)  w_done  <= 1'b1;
            if (write_done) begin
                aw_done <= 1'b0;
                w_done  <= 1'b0;
            end
        end
    end

    // 写操作（支持字节写使能）
    always @(posedge clk) begin
        if (write_done) begin
            if (s_axi_wstrb[0]) mem[waddr_word][ 7: 0] <= s_axi_wdata[ 7: 0];
            if (s_axi_wstrb[1]) mem[waddr_word][15: 8] <= s_axi_wdata[15: 8];
            if (s_axi_wstrb[2]) mem[waddr_word][23:16] <= s_axi_wdata[23:16];
            if (s_axi_wstrb[3]) mem[waddr_word][31:24] <= s_axi_wdata[31:24];
        end
    end

    // 写响应：write_done 后一拍
    reg bvalid_r;
    always @(posedge clk or negedge resetn) begin
        if (!resetn)
            bvalid_r <= 1'b0;
        else if (write_done)
            bvalid_r <= 1'b1;
        else if (bvalid_r && s_axi_bready)
            bvalid_r <= 1'b0;
    end

    assign s_axi_bresp  = 2'b00;
    assign s_axi_bvalid = bvalid_r;

    // 读通道：组合逻辑接受，寄存器输出
    assign s_axi_arready = 1'b1;
    assign s_axi_rresp   = 2'b00;

    reg rvalid_r;
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            rvalid_r <= 1'b0;
            raddr    <= {ADDR_WIDTH{1'b0}};
        end else begin
            if (s_axi_arvalid && s_axi_arready) begin
                rvalid_r <= 1'b1;
                raddr    <= s_axi_araddr;
            end else if (rvalid_r && s_axi_rready)
                rvalid_r <= 1'b0;
        end
    end

    assign s_axi_rvalid = rvalid_r;
    assign s_axi_rdata  = mem[raddr[ADDR_WIDTH-1:2]];


endmodule
