`timescale 1 ns / 1 ps

module axi_sys #(
    parameter SHARED_BASE    = 32'h4000_0000,  // 共享内存基地址
    parameter FIRMWARE_FILE  = "firmware/build/firmware7"
) (
    input  wire        clk,
    input  wire        resetn,

    // 状态监控
    output wire        txn_done,
    output wire        error,
    output reg         tests_passed
);

    // -----------------------------------------------------------------
    // CPU 实例 (PicoRV32 + AXI‑Lite 适配器)
    // -----------------------------------------------------------------
    wire        cpu_axi_awvalid;
    wire        cpu_axi_awready;
    wire [31:0] cpu_axi_awaddr;
    wire [2:0]  cpu_axi_awprot;
    wire        cpu_axi_wvalid;
    wire        cpu_axi_wready;
    wire [31:0] cpu_axi_wdata;
    wire [3:0]  cpu_axi_wstrb;
    wire        cpu_axi_bvalid;
    wire        cpu_axi_bready;
    wire        cpu_axi_arvalid;
    wire        cpu_axi_arready;
    wire [31:0] cpu_axi_araddr;
    wire [2:0]  cpu_axi_arprot;
    wire        cpu_axi_rvalid;
    wire        cpu_axi_rready;
    wire [31:0] cpu_axi_rdata;

    reg         npu_irq_latched;   // NPU done IRQ latch (declared early for CPU irq port)

    picorv32_axi #(
        .ENABLE_COUNTERS     (1),
        .ENABLE_COUNTERS64   (1),
        .ENABLE_REGS_16_31   (1),
        .ENABLE_REGS_DUALPORT(1),
        .TWO_STAGE_SHIFT     (1),
        .BARREL_SHIFTER      (0),
        .TWO_CYCLE_COMPARE   (0),
        .TWO_CYCLE_ALU       (0),
        .COMPRESSED_ISA      (1),
        .CATCH_MISALIGN      (1),
        .CATCH_ILLINSN       (0),
        .ENABLE_PCPI         (0),
        .ENABLE_MUL          (1),
        .ENABLE_FAST_MUL     (0),
        .ENABLE_DIV          (1),
        .ENABLE_IRQ          (1),
        .ENABLE_TRACE        (0),
        .REGS_INIT_ZERO      (0),
        .PROGADDR_RESET      (32'h0000_0000),
        .STACKADDR           (32'hffff_ffff)
    ) cpu (
        .clk                (clk),
        .resetn             (resetn),
        .trap               (),
        // AXI‑Lite 主接口
        .mem_axi_awvalid    (cpu_axi_awvalid),
        .mem_axi_awready    (cpu_axi_awready),
        .mem_axi_awaddr     (cpu_axi_awaddr),
        .mem_axi_awprot     (cpu_axi_awprot),
        .mem_axi_wvalid     (cpu_axi_wvalid),
        .mem_axi_wready     (cpu_axi_wready),
        .mem_axi_wdata      (cpu_axi_wdata),
        .mem_axi_wstrb      (cpu_axi_wstrb),
        .mem_axi_bvalid     (cpu_axi_bvalid),
        .mem_axi_bready     (cpu_axi_bready),
        .mem_axi_arvalid    (cpu_axi_arvalid),
        .mem_axi_arready    (cpu_axi_arready),
        .mem_axi_araddr     (cpu_axi_araddr),
        .mem_axi_arprot     (cpu_axi_arprot),
        .mem_axi_rvalid     (cpu_axi_rvalid),
        .mem_axi_rready     (cpu_axi_rready),
        .mem_axi_rdata      (cpu_axi_rdata),
        .irq                ({29'b0, npu_irq_latched, 2'b0}),  // bit 3 = NPU done
        .eoi                (),
        .trace_valid        (),
        .trace_data         ()
    );

// -----------------------------------------------------------------
    // 地址解码：私有内存 (0x0000_0000 ~ 0x00FF_FFFF) 和 共享内存 (0x4000_0000~)
    // -----------------------------------------------------------------

    // ---- 1. 彻底独立的通道解码 ----
    wire aw_sel_priv   = (cpu_axi_awaddr[31:24] == 8'h00);
    wire aw_sel_shared = (cpu_axi_awaddr[31:28] == 4'h4);

    wire ar_sel_priv   = (cpu_axi_araddr[31:24] == 8'h00);
    wire ar_sel_shared = (cpu_axi_araddr[31:28] == 4'h4);

    // MMIO 地址译码
    wire aw_sel_uart   = (cpu_axi_awaddr == 32'h1000_0000);
    wire aw_sel_test   = (cpu_axi_awaddr == 32'h2000_0000);
    wire aw_sel_npu    = (cpu_axi_awaddr[31:12] == 20'h30000); // 0x3000_0000 ~ 0x3000_0FFF
    wire aw_sel_mmio   = aw_sel_uart || aw_sel_test || aw_sel_npu;

    wire ar_sel_uart   = (cpu_axi_araddr == 32'h1000_0000);
    wire ar_sel_test   = (cpu_axi_araddr == 32'h2000_0000);
    wire ar_sel_npu    = (cpu_axi_araddr[31:12] == 20'h30000); // 0x3000_0000 ~ 0x3000_0FFF
    wire ar_sel_mmio   = ar_sel_uart || ar_sel_test || ar_sel_npu;

    // ---- 内部连线声明 ----
    wire priv_awready, priv_wready, priv_bvalid, priv_arready, priv_rvalid;
    wire [31:0] priv_rdata;
    wire [1:0]  priv_bresp, priv_rresp;

    wire cpu_shared_awvalid, cpu_shared_awready;
    wire [31:0] cpu_shared_awaddr;
    wire [2:0]  cpu_shared_awprot;
    wire cpu_shared_wvalid, cpu_shared_wready;
    wire [31:0] cpu_shared_wdata;
    wire [3:0]  cpu_shared_wstrb;
    wire cpu_shared_bvalid, cpu_shared_bready;
    wire [1:0]  cpu_shared_bresp;
    wire cpu_shared_arvalid, cpu_shared_arready;
    wire [31:0] cpu_shared_araddr;
    wire [2:0]  cpu_shared_arprot;
    wire cpu_shared_rvalid, cpu_shared_rready;
    wire [31:0] cpu_shared_rdata;
    wire [1:0]  cpu_shared_rresp;

    // ---- 2. 路由到共享桥接器 (写用 aw_sel_shared, 读用 ar_sel_shared) ----
    assign cpu_shared_awvalid = aw_sel_shared ? cpu_axi_awvalid : 1'b0;
    assign cpu_shared_awaddr  = cpu_axi_awaddr;
    assign cpu_shared_awprot  = cpu_axi_awprot;
    assign cpu_shared_wvalid  = aw_sel_shared ? cpu_axi_wvalid : 1'b0;
    assign cpu_shared_wdata   = cpu_axi_wdata;
    assign cpu_shared_wstrb   = cpu_axi_wstrb;
    assign cpu_shared_bready  = aw_sel_shared ? cpu_axi_bready : 1'b0;

    assign cpu_shared_arvalid = ar_sel_shared ? cpu_axi_arvalid : 1'b0;
    assign cpu_shared_araddr  = cpu_axi_araddr;
    assign cpu_shared_arprot  = cpu_axi_arprot;
    assign cpu_shared_rready  = ar_sel_shared ? cpu_axi_rready : 1'b0;

    // -----------------------------------------------------------------
    // MMIO 信号声明（提前声明供 NPU 逻辑使用）
    // -----------------------------------------------------------------
    reg  [31:0] mmio_wdata;
    reg         mmio_aw_valid;
    reg         mmio_w_valid;
    wire        mmio_access = aw_sel_mmio;
    wire        mmio_read   = ar_sel_mmio;

    // -----------------------------------------------------------------
    // NPU Wrapper 寄存器接口 (0x3000_0000 ~ 0x3000_0FFF, 4KB)
    // CPU MMIO writes are forwarded to the NPU's param_regfile.
    // CPU MMIO reads are served by the NPU's register read port.
    // -----------------------------------------------------------------
    wire        npu_irq_done;

    reg         npu_reg_wr_en;
    reg  [9:0]  npu_reg_wr_addr;
    reg  [31:0] npu_reg_wr_data;
    reg         npu_reg_rd_en;
    reg  [9:0]  npu_reg_rd_addr;
    wire [31:0] npu_reg_rd_data;
    wire        npu_reg_rd_data_valid;
    wire        npu_reg_wr_done;

    // Latch NPU irq_done pulse so IRQ line stays high until CPU acknowledges
    always @(posedge clk or negedge resetn) begin
        if (!resetn)
            npu_irq_latched <= 1'b0;
        else if (npu_irq_done)
            npu_irq_latched <= 1'b1;
        else if (npu_reg_wr_en && npu_reg_wr_addr == 10'h000)
            npu_irq_latched <= 1'b0;  // cleared on any CTRL write (start or clear_done)
    end

    // MMIO 地址锁存
    reg [31:0] mmio_aw_addr;
    reg        mmio_aw_is_npu;
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            mmio_aw_addr   <= 32'b0;
            mmio_aw_is_npu <= 1'b0;
        end else if (cpu_axi_awvalid && cpu_axi_awready && aw_sel_npu) begin
            mmio_aw_addr   <= cpu_axi_awaddr;
            mmio_aw_is_npu <= 1'b1;
        end else if (cpu_axi_awvalid && cpu_axi_awready && mmio_access) begin
            mmio_aw_is_npu <= 1'b0;
        end
    end

    // -----------------------------------------------------------------
    // MMIO 写处理 (UART 0x10000000 / TEST 0x20000000 / NPU 0x3000_0000)
    // -----------------------------------------------------------------
    reg         mmio_bvalid;
    reg         mmio_rvalid;
    reg         npu_read_active;  // declared early for use in MUX

    // NPU register read data: latch when valid, hold until CPU accepts.
    reg [31:0] npu_rdata_lat;
    always @(posedge clk or negedge resetn) begin
        if (!resetn)
            npu_rdata_lat <= 32'b0;
        else if (npu_reg_rd_data_valid && npu_read_active)
            npu_rdata_lat <= npu_reg_rd_data;
    end

    // ---- 3. 从各个从机返回到 CPU (MUX 选择器) ----
    assign cpu_axi_awready = aw_sel_priv   ? priv_awready :
                             aw_sel_shared ? cpu_shared_awready :
                             aw_sel_mmio   ? 1'b1 : 1'b0;
    assign cpu_axi_wready  = aw_sel_priv   ? priv_wready :
                             aw_sel_shared ? cpu_shared_wready :
                             aw_sel_mmio   ? 1'b1 : 1'b0;
    assign cpu_axi_bvalid  = aw_sel_priv   ? priv_bvalid :
                             aw_sel_shared ? cpu_shared_bvalid :
                             aw_sel_mmio   ? mmio_bvalid : 1'b0;

    assign cpu_axi_arready = ar_sel_priv   ? priv_arready :
                             ar_sel_shared ? cpu_shared_arready :
                             ar_sel_mmio   ? 1'b1 : 1'b0;
    assign cpu_axi_rvalid  = ar_sel_priv   ? priv_rvalid :
                             ar_sel_shared ? cpu_shared_rvalid :
                             ar_sel_mmio   ? mmio_rvalid : 1'b0;
    assign cpu_axi_rdata   = ar_sel_priv   ? priv_rdata :
                             ar_sel_shared ? cpu_shared_rdata :
                             npu_read_active ? npu_rdata_lat : 32'b0;

    // MMIO 写地址/数据锁存
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            mmio_aw_valid <= 1'b0;
            mmio_w_valid  <= 1'b0;
        end else begin
            if (cpu_axi_awvalid && cpu_axi_awready && mmio_access)
                mmio_aw_valid <= 1'b1;
            else if (mmio_aw_valid && mmio_w_valid)
                mmio_aw_valid <= 1'b0;

            if (cpu_axi_wvalid && cpu_axi_wready && mmio_access) begin
                mmio_w_valid <= 1'b1;
                mmio_wdata   <= cpu_axi_wdata;
            end else if (mmio_aw_valid && mmio_w_valid)
                mmio_w_valid <= 1'b0;
        end
    end

    // MMIO 写响应
    // For NPU writes, wait until wrapper confirms write completion
    reg npu_write_pending;
    always @(posedge clk or negedge resetn) begin
        if (!resetn)
            npu_write_pending <= 1'b0;
        else if (mmio_aw_valid && mmio_w_valid && mmio_aw_is_npu && !mmio_bvalid)
            npu_write_pending <= 1'b1;
        else if (npu_reg_wr_done)
            npu_write_pending <= 1'b0;
    end

    always @(posedge clk or negedge resetn) begin
        if (!resetn)
            mmio_bvalid <= 1'b0;
        else if (mmio_aw_valid && mmio_w_valid && !mmio_bvalid && !mmio_aw_is_npu)
            mmio_bvalid <= 1'b1;
        else if (npu_reg_wr_done && npu_write_pending)
            mmio_bvalid <= 1'b1;
        else if (mmio_bvalid && cpu_axi_bready)
            mmio_bvalid <= 1'b0;
    end

    // MMIO 读响应
    // For NPU reads:
    //   1. npu_read_active goes high when read request accepted
    //   2. npu_reg_rd_data_valid fires combinationally (1 cycle later) with correct data
    //   3. mmio_rvalid is set on the dv cycle; npu_read_active stays high so the MUX
    //      continues to present npu_reg_rd_data
    //   4. When CPU accepts (rvalid && rready), both mmio_rvalid and npu_read_active clear
    reg  npu_data_ready;  // set when NPU data valid, held until CPU accepts
    always @(posedge clk or negedge resetn) begin
        if (!resetn)
            npu_read_active <= 1'b0;
        else if (cpu_axi_arvalid && cpu_axi_arready && ar_sel_npu && !mmio_rvalid)
            npu_read_active <= 1'b1;
        else if (mmio_rvalid && cpu_axi_rready && npu_read_active)
            npu_read_active <= 1'b0;
    end

    always @(posedge clk or negedge resetn) begin
        if (!resetn)
            npu_data_ready <= 1'b0;
        else if (npu_reg_rd_data_valid && npu_read_active)
            npu_data_ready <= 1'b1;
        else if (mmio_rvalid && cpu_axi_rready)
            npu_data_ready <= 1'b0;
    end

    always @(posedge clk or negedge resetn) begin
        if (!resetn)
            mmio_rvalid <= 1'b0;
        else if (cpu_axi_arvalid && cpu_axi_arready && mmio_read && !mmio_rvalid && !npu_read_active && !ar_sel_npu)
            mmio_rvalid <= 1'b1;
        else if (npu_data_ready && npu_read_active)
            mmio_rvalid <= 1'b1;
        else if (mmio_rvalid && cpu_axi_rready)
            mmio_rvalid <= 1'b0;
    end

    // NPU register read trigger
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            npu_reg_rd_en   <= 1'b0;
            npu_reg_rd_addr <= 10'b0;
        end else begin
            npu_reg_rd_en <= (cpu_axi_arvalid && cpu_axi_arready && ar_sel_npu && !mmio_rvalid);
            if (cpu_axi_arvalid && cpu_axi_arready && ar_sel_npu && !mmio_rvalid)
                npu_reg_rd_addr <= cpu_axi_araddr[11:2];  // byte addr → word addr
        end
    end

    // synthesis translate_off
    `ifdef DEBUG
    // Debug: trace NPU write path
    always @(posedge clk) begin
        if (cpu_axi_awvalid && cpu_axi_awready && aw_sel_npu)
            $display("AXI_SYS_AW: NPU AW handshake addr=0x%08h", cpu_axi_awaddr);
        if (cpu_axi_wvalid && cpu_axi_wready && mmio_access)
            $display("AXI_SYS_W: MMIO W handshake data=0x%08h aw_sel_npu=%0b addr=0x%08h aw_valid=%0b w_valid=%0b",
                     cpu_axi_wdata, aw_sel_npu, cpu_axi_awaddr, mmio_aw_valid, mmio_w_valid);
        if (mmio_aw_valid && mmio_w_valid && mmio_aw_is_npu)
            $display("AXI_SYS_MMIO: both latched, npu addr=0x%08h data=0x%08h", mmio_aw_addr, mmio_wdata);
        if (npu_reg_wr_en)
            $display("AXI_SYS_FWD: npu_reg_wr_en addr=0x%03h data=0x%08h", npu_reg_wr_addr, npu_reg_wr_data);
        // Specific trace for DMA len register (0x128 = word addr 0x04A)
        if (cpu_axi_wvalid && cpu_axi_wready && cpu_axi_awaddr == 32'h3000_0128)
            $display("AXI_SYS_DMALEN: CPU writes 0x128 data=0x%08h", cpu_axi_wdata);
        if (npu_reg_wr_en && npu_reg_wr_addr == 10'h04A)
            $display("AXI_SYS_DMALEN_FWD: forwarded data=0x%08h", npu_reg_wr_data);
    end
    `endif
    // synthesis translate_on

    // MMIO 写效果：UART 输出 / test pass / NPU 寄存器写（转发给 wrapper）
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            tests_passed    <= 1'b0;
            npu_reg_wr_en   <= 1'b0;
            npu_reg_wr_addr <= 10'b0;
            npu_reg_wr_data <= 32'b0;
        end else begin
            npu_reg_wr_en <= 1'b0;  // 默认清除，单周期脉冲
            if (mmio_aw_valid && mmio_w_valid) begin
                if (aw_sel_uart) begin
                    $write("%c", mmio_wdata[7:0]);
`ifndef VERILATOR
                    $fflush();
`endif
                end else if (aw_sel_test) begin
                    if (mmio_wdata == 32'd123456789)
                        tests_passed <= 1'b1;
                end else if (mmio_aw_is_npu) begin
                    // Forward write to NPU wrapper
                    npu_reg_wr_en   <= 1'b1;
                    npu_reg_wr_addr <= mmio_aw_addr[11:2];  // byte addr → word addr
                    npu_reg_wr_data <= mmio_wdata;
                end
            end
        end
    end

    // ---- 4. 实例化私有内存 (握手逻辑也必须严格区分读写) ----
    axi_lite_ram priv_mem (
        .clk             (clk),
        .resetn          (resetn),
        .s_axi_awaddr    (cpu_axi_awaddr[23:0]),
        .s_axi_awvalid   (cpu_axi_awvalid && aw_sel_priv),
        .s_axi_awready   (priv_awready),
        .s_axi_wdata     (cpu_axi_wdata),
        .s_axi_wstrb     (cpu_axi_wstrb),
        .s_axi_wvalid    (cpu_axi_wvalid && aw_sel_priv),
        .s_axi_wready    (priv_wready),
        .s_axi_bresp     (priv_bresp),
        .s_axi_bvalid    (priv_bvalid),
        .s_axi_bready    (cpu_axi_bready && aw_sel_priv),

        .s_axi_araddr    (cpu_axi_araddr[23:0]),
        .s_axi_arvalid   (cpu_axi_arvalid && ar_sel_priv),
        .s_axi_arready   (priv_arready),
        .s_axi_rdata     (priv_rdata),
        .s_axi_rresp     (priv_rresp),
        .s_axi_rvalid    (priv_rvalid),
        .s_axi_rready    (cpu_axi_rready && ar_sel_priv)
    );

    // 加载 firmware 到私有内存（支持 +firmware= 命令行覆盖）
    initial begin : firmware_load
        reg [256*8-1:0] fw_file;
        if ($value$plusargs("firmware=%s", fw_file))
            $readmemh(fw_file, priv_mem.mem);
        else
            $readmemh({FIRMWARE_FILE, ".hex"}, priv_mem.mem);
    end

    // -----------------------------------------------------------------
    // AXI‑Lite → AXI‑Full 桥接器 (仅用于共享路径)
    // -----------------------------------------------------------------
    wire bridge_awvalid, bridge_awready;
    wire [31:0] bridge_awaddr;
    wire [2:0]  bridge_awsize;
    wire [1:0]  bridge_awburst;
    wire bridge_wvalid, bridge_wready;
    wire [31:0] bridge_wdata;
    wire [3:0]  bridge_wstrb;
    wire bridge_wlast;
    wire bridge_bvalid, bridge_bready;
    wire [1:0]  bridge_bresp;
    wire bridge_arvalid, bridge_arready;
    wire [31:0] bridge_araddr;
    wire [2:0]  bridge_arsize;
    wire [1:0]  bridge_arburst;
    wire bridge_rvalid, bridge_rready;
    wire [31:0] bridge_rdata;
    wire [1:0]  bridge_rresp;

    axi_lite_to_axi_full bridge (
        .clk            (clk),
        .resetn         (resetn),
        .s_axi_awvalid  (cpu_shared_awvalid),
        .s_axi_awready  (cpu_shared_awready),
        .s_axi_awaddr   (cpu_shared_awaddr),
        .s_axi_awprot   (cpu_shared_awprot),
        .s_axi_wvalid   (cpu_shared_wvalid),
        .s_axi_wready   (cpu_shared_wready),
        .s_axi_wdata    (cpu_shared_wdata),
        .s_axi_wstrb    (cpu_shared_wstrb),
        .s_axi_bvalid   (cpu_shared_bvalid),
        .s_axi_bready   (cpu_shared_bready),
        .s_axi_bresp    (cpu_shared_bresp),
        .s_axi_arvalid  (cpu_shared_arvalid),
        .s_axi_arready  (cpu_shared_arready),
        .s_axi_araddr   (cpu_shared_araddr),
        .s_axi_arprot   (cpu_shared_arprot),
        .s_axi_rvalid   (cpu_shared_rvalid),
        .s_axi_rready   (cpu_shared_rready),
        .s_axi_rdata    (cpu_shared_rdata),
        .s_axi_rresp    (cpu_shared_rresp),
        .m_axi_awvalid  (bridge_awvalid),
        .m_axi_awready  (bridge_awready),
        .m_axi_awaddr   (bridge_awaddr),
        .m_axi_awsize   (bridge_awsize),
        .m_axi_awburst  (bridge_awburst),
        .m_axi_wvalid   (bridge_wvalid),
        .m_axi_wready   (bridge_wready),
        .m_axi_wdata    (bridge_wdata),
        .m_axi_wstrb    (bridge_wstrb),
        .m_axi_wlast    (bridge_wlast),
        .m_axi_bvalid   (bridge_bvalid),
        .m_axi_bready   (bridge_bready),
        .m_axi_bresp    (bridge_bresp),
        .m_axi_arvalid  (bridge_arvalid),
        .m_axi_arready  (bridge_arready),
        .m_axi_araddr   (bridge_araddr),
        .m_axi_arsize   (bridge_arsize),
        .m_axi_arburst  (bridge_arburst),
        .m_axi_rvalid   (bridge_rvalid),
        .m_axi_rready   (bridge_rready),
        .m_axi_rdata    (bridge_rdata),
        .m_axi_rresp    (bridge_rresp)
    );

    // -----------------------------------------------------------------
    // NPU Wrapper 实例 (16×16 systolic NPU + 128→32 宽度转换)
    // -----------------------------------------------------------------
    wire        npu_m_awvalid, npu_m_awready;
    wire [31:0] npu_m_awaddr;
    wire [7:0]  npu_m_awlen;
    wire [2:0]  npu_m_awsize;
    wire [1:0]  npu_m_awburst;
    wire        npu_m_wvalid, npu_m_wready;
    wire [31:0] npu_m_wdata;
    wire        npu_m_wlast;
    wire        npu_m_bvalid, npu_m_bready;
    wire [1:0]  npu_m_bresp;
    wire        npu_m_arvalid, npu_m_arready;
    wire [31:0] npu_m_araddr;
    wire [7:0]  npu_m_arlen;
    wire [2:0]  npu_m_arsize;
    wire [1:0]  npu_m_arburst;
    wire        npu_m_rvalid, npu_m_rready;
    wire [31:0] npu_m_rdata;
    wire        npu_m_rlast;
    wire [1:0]  npu_m_rresp;

    npu_axi_wrapper #(
        .NPU_AXI_DATA_W (128),
        .SOC_AXI_DATA_W (32),
        .AXI_ADDR_W     (32),
        .AXI_ID_W       (4),
        .AXI_LEN_W      (8),
        .REG_ADDR_W      (10)
    ) npu_wrapper (
        .clk                (clk),
        .rst_n              (resetn),
        // Register interface
        .reg_wr_en          (npu_reg_wr_en),
        .reg_wr_addr        (npu_reg_wr_addr),
        .reg_wr_data        (npu_reg_wr_data),
        .reg_rd_en          (npu_reg_rd_en),
        .reg_rd_addr        (npu_reg_rd_addr),
        .reg_rd_data        (npu_reg_rd_data),
        .reg_rd_data_valid  (npu_reg_rd_data_valid),
        .reg_wr_done        (npu_reg_wr_done),
        // AXI4-Full master (32-bit)
        .m_axi_awvalid      (npu_m_awvalid),
        .m_axi_awready      (npu_m_awready),
        .m_axi_awaddr       (npu_m_awaddr),
        .m_axi_awlen        (npu_m_awlen),
        .m_axi_awsize       (npu_m_awsize),
        .m_axi_awburst      (npu_m_awburst),
        .m_axi_wvalid       (npu_m_wvalid),
        .m_axi_wready       (npu_m_wready),
        .m_axi_wdata        (npu_m_wdata),
        .m_axi_wlast        (npu_m_wlast),
        .m_axi_bvalid       (npu_m_bvalid),
        .m_axi_bready       (npu_m_bready),
        .m_axi_bresp        (npu_m_bresp),
        .m_axi_arvalid      (npu_m_arvalid),
        .m_axi_arready      (npu_m_arready),
        .m_axi_araddr       (npu_m_araddr),
        .m_axi_arlen        (npu_m_arlen),
        .m_axi_arsize       (npu_m_arsize),
        .m_axi_arburst      (npu_m_arburst),
        .m_axi_rvalid       (npu_m_rvalid),
        .m_axi_rready       (npu_m_rready),
        .m_axi_rdata        (npu_m_rdata),
        .m_axi_rlast        (npu_m_rlast),
        .m_axi_rresp        (npu_m_rresp),
        // Interrupt
        .irq_done           (npu_irq_done)
    );

    // 状态输出
    assign txn_done = npu_irq_done;
    assign error    = 1'b0;

    // -----------------------------------------------------------------
    // 仲裁器：CPU (Master0) 和 NPU (Master1) -> 共享内存 (Slave)
    // -----------------------------------------------------------------
    wire        arb_awvalid, arb_awready;
    wire [31:0] arb_awaddr;
    wire [7:0]  arb_awlen;
    wire [2:0]  arb_awsize;
    wire [1:0]  arb_awburst;
    wire        arb_wvalid, arb_wready;
    wire [31:0] arb_wdata;
    wire [3:0]  arb_wstrb;
    wire        arb_wlast;
    wire        arb_bvalid, arb_bready;
    wire [1:0]  arb_bresp;
    wire        arb_arvalid, arb_arready;
    wire [31:0] arb_araddr;
    wire [7:0]  arb_arlen;
    wire [2:0]  arb_arsize;
    wire [1:0]  arb_arburst;
    wire        arb_rvalid, arb_rready;
    wire [31:0] arb_rdata;
    wire        arb_rlast;
    wire [1:0]  arb_rresp;

    axi_arbiter_2to1 arbiter (
        .clk             (clk),
        .resetn          (resetn),
        // Master 0 (CPU 桥接器)
        .m0_axi_awvalid  (bridge_awvalid),
        .m0_axi_awready  (bridge_awready),
        .m0_axi_awaddr   (bridge_awaddr),
        .m0_axi_awlen    (8'd0),
        .m0_axi_awsize   (bridge_awsize),
        .m0_axi_awburst  (bridge_awburst),
        .m0_axi_wvalid   (bridge_wvalid),
        .m0_axi_wready   (bridge_wready),
        .m0_axi_wdata    (bridge_wdata),
        .m0_axi_wstrb    (bridge_wstrb),
        .m0_axi_wlast    (bridge_wlast),
        .m0_axi_bvalid   (bridge_bvalid),
        .m0_axi_bready   (bridge_bready),
        .m0_axi_bresp    (bridge_bresp),
        .m0_axi_arvalid  (bridge_arvalid),
        .m0_axi_arready  (bridge_arready),
        .m0_axi_araddr   (bridge_araddr),
        .m0_axi_arlen    (8'd0),
        .m0_axi_arsize   (bridge_arsize),
        .m0_axi_arburst  (bridge_arburst),
        .m0_axi_rvalid   (bridge_rvalid),
        .m0_axi_rready   (bridge_rready),
        .m0_axi_rdata    (bridge_rdata),
        .m0_axi_rresp    (bridge_rresp),
        .m0_axi_rlast    (),
        // Master 1 (NPU)
        .m1_axi_awvalid  (npu_m_awvalid),
        .m1_axi_awready  (npu_m_awready),
        .m1_axi_awaddr   (npu_m_awaddr),
        .m1_axi_awlen    (npu_m_awlen),
        .m1_axi_awsize   (npu_m_awsize),
        .m1_axi_awburst  (npu_m_awburst),
        .m1_axi_wvalid   (npu_m_wvalid),
        .m1_axi_wready   (npu_m_wready),
        .m1_axi_wdata    (npu_m_wdata),
        .m1_axi_wstrb    (4'b1111),
        .m1_axi_wlast    (npu_m_wlast),
        .m1_axi_bvalid   (npu_m_bvalid),
        .m1_axi_bready   (npu_m_bready),
        .m1_axi_bresp    (npu_m_bresp),
        .m1_axi_arvalid  (npu_m_arvalid),
        .m1_axi_arready  (npu_m_arready),
        .m1_axi_araddr   (npu_m_araddr),
        .m1_axi_arlen    (npu_m_arlen),
        .m1_axi_arsize   (npu_m_arsize),
        .m1_axi_arburst  (npu_m_arburst),
        .m1_axi_rvalid   (npu_m_rvalid),
        .m1_axi_rready   (npu_m_rready),
        .m1_axi_rdata    (npu_m_rdata),
        .m1_axi_rresp    (npu_m_rresp),
        .m1_axi_rlast    (npu_m_rlast),
        // 输出到共享从机
        .s_axi_awvalid   (arb_awvalid),
        .s_axi_awready   (arb_awready),
        .s_axi_awaddr    (arb_awaddr),
        .s_axi_awlen     (arb_awlen),
        .s_axi_awsize    (arb_awsize),
        .s_axi_awburst   (arb_awburst),
        .s_axi_wvalid    (arb_wvalid),
        .s_axi_wready    (arb_wready),
        .s_axi_wdata     (arb_wdata),
        .s_axi_wstrb     (arb_wstrb),
        .s_axi_wlast     (arb_wlast),
        .s_axi_bvalid    (arb_bvalid),
        .s_axi_bready    (arb_bready),
        .s_axi_bresp     (arb_bresp),
        .s_axi_arvalid   (arb_arvalid),
        .s_axi_arready   (arb_arready),
        .s_axi_araddr    (arb_araddr),
        .s_axi_arlen     (arb_arlen),
        .s_axi_arsize    (arb_arsize),
        .s_axi_arburst   (arb_arburst),
        .s_axi_rvalid    (arb_rvalid),
        .s_axi_rready    (arb_rready),
        .s_axi_rdata     (arb_rdata),
        .s_axi_rresp     (arb_rresp),
        .s_axi_rlast     (arb_rlast)
    );

    // -----------------------------------------------------------------
    // 共享内存 (AXI‑Full 从机, 16MB, 地址偏移 SHARED_BASE)
    // 注意：写地址和读地址必须独立连接，避免并发事务时地址被覆盖
    // -----------------------------------------------------------------

    // synthesis translate_off
    // always @(posedge clk) begin
    //     if (arb_arvalid && arb_arready)
    //         $display("SYS_RD_ACCEPT: arb_araddr=%0h shared_addr_off=%0h", arb_araddr, shared_addr_off);
    //     if (arb_rvalid && arb_rready)
    //         $display("SYS_RD_DATA: arb_rdata=%0h arb_rlast=%0b", arb_rdata, arb_rlast);
    // end
    // synthesis translate_on

    axi_full_slave_v1_0_S00_AXI #(
        .C_S_AXI_DATA_WIDTH(32),
        .C_S_AXI_ADDR_WIDTH(24)
    ) shared_mem (
        .S_AXI_ACLK     (clk),
        .S_AXI_ARESETN  (resetn),
        .S_AXI_AWID     (1'b0),
        .S_AXI_AWADDR   (arb_awaddr[23:0]),
        .S_AXI_AWLEN    (arb_awlen),
        .S_AXI_AWSIZE   (arb_awsize),
        .S_AXI_AWBURST  (arb_awburst),
        .S_AXI_AWLOCK   (1'b0),
        .S_AXI_AWCACHE  (4'b0011),
        .S_AXI_AWPROT   (3'b0),
        .S_AXI_AWQOS    (4'b0),
        .S_AXI_AWREGION (4'b0),
        .S_AXI_AWUSER   (1'b0),
        .S_AXI_AWVALID  (arb_awvalid),
        .S_AXI_AWREADY  (arb_awready),
        .S_AXI_WDATA    (arb_wdata),
        .S_AXI_WSTRB    (arb_wstrb),
        .S_AXI_WLAST    (arb_wlast),
        .S_AXI_WUSER    (1'b0),
        .S_AXI_WVALID   (arb_wvalid),
        .S_AXI_WREADY   (arb_wready),
        .S_AXI_BID      (),
        .S_AXI_BRESP    (arb_bresp),
        .S_AXI_BUSER    (),
        .S_AXI_BVALID   (arb_bvalid),
        .S_AXI_BREADY   (arb_bready),
        .S_AXI_ARID     (1'b0),
        .S_AXI_ARADDR   (arb_araddr[23:0]),
        .S_AXI_ARLEN    (arb_arlen),
        .S_AXI_ARSIZE   (arb_arsize),
        .S_AXI_ARBURST  (arb_arburst),
        .S_AXI_ARLOCK   (1'b0),
        .S_AXI_ARCACHE  (4'b0011),
        .S_AXI_ARPROT   (3'b0),
        .S_AXI_ARQOS    (4'b0),
        .S_AXI_ARREGION (4'b0),
        .S_AXI_ARUSER   (1'b0),
        .S_AXI_ARVALID  (arb_arvalid),
        .S_AXI_ARREADY  (arb_arready),
        .S_AXI_RID      (),
        .S_AXI_RDATA    (arb_rdata),
        .S_AXI_RRESP    (arb_rresp),
        .S_AXI_RLAST    (arb_rlast),
        .S_AXI_RUSER    (),
        .S_AXI_RVALID   (arb_rvalid),
        .S_AXI_RREADY   (arb_rready)
    );

endmodule
