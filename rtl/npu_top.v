// Filename: npu_top.v
// -------------------------------------------------------------------
// NPU_TOP — top-level integration of the 16×16 Edge Vision NPU.
//
// Instantiates:
//   param_regfile         — AXI-Lite config/status register file
//   top_controller_fsm    — Main sequencer & address generators
//   act_sram_wrapper      — Activation SRAM (256 KB Ping-Pong)
//   wgt_sram_wrapper      — Weight SRAM (256 KB Ping-Pong)
//   out_sram_wrapper      — Output SRAM (128 KB Ping-Pong)
//   im2col_line_buffer    — Sliding-window Im2Col converter
//   wgt_reader            — Weight pre-fetch & assembly
//   systolic_16x16        — 16×16 systolic array (4096 MACs/cycle)
//   post_process_top      — Bias / Quant / ReLU / MaxPool pipeline
//   vector_alu            — Element-wise skip-connection add
//   axi_dma               — AXI4-Full DMA engine
//
// External interfaces:
//   clk, rst_n
//   AXI4-Lite slave  (configuration — connects to param_regfile)
//   AXI4-Full master (DDR access — connects to axi_dma)
//   irq_done          (layer-complete interrupt)
// -------------------------------------------------------------------

module npu_top #(
    parameter AXI_ADDR_W    = 32,
    parameter AXI_DATA_W    = 128,
    parameter AXI_ID_W      = 4,
    parameter AXI_LEN_W     = 8,
    parameter REG_ADDR_W    = 10,
    parameter REG_DATA_W    = 32,
    parameter SRAM_ADDR_W   = 14,
    parameter OUT_SRAM_ADDR_W = 13,
    parameter ACT_DATA_W    = 128,
    parameter PSUM_WIDTH    = 32,
    parameter ACT_WIDTH     = 8,
    parameter NUM_CH        = 16,
    parameter ARRAY_ROWS    = 16,
    parameter ARRAY_COLS    = 16,
    parameter ACT_BUS_W     = ARRAY_ROWS * ACT_DATA_W,  // 2048
    parameter WGT_BUS_W     = ARRAY_COLS * ACT_DATA_W,  // 2048
    parameter MAX_WIDTH     = 256
) (
    input  wire                             clk,
    input  wire                             rst_n,

    // === AXI4-Lite Slave (Configuration) ===
    input  wire                             s_axi_awvalid,
    output wire                             s_axi_awready,
    input  wire [REG_ADDR_W-1:0]            s_axi_awaddr,

    input  wire                             s_axi_wvalid,
    output wire                             s_axi_wready,
    input  wire [REG_DATA_W-1:0]            s_axi_wdata,
    input  wire [3:0]                       s_axi_wstrb,

    output wire                             s_axi_bvalid,
    input  wire                             s_axi_bready,
    output wire [1:0]                       s_axi_bresp,

    input  wire                             s_axi_arvalid,
    output wire                             s_axi_arready,
    input  wire [REG_ADDR_W-1:0]            s_axi_araddr,

    output wire                             s_axi_rvalid,
    input  wire                             s_axi_rready,
    output wire [REG_DATA_W-1:0]            s_axi_rdata,
    output wire [1:0]                       s_axi_rresp,

    // === AXI4-Full Master (DDR) ===
    // Read Address
    output wire                             m_axi_arvalid,
    input  wire                             m_axi_arready,
    output wire [AXI_ID_W-1:0]              m_axi_arid,
    output wire [AXI_ADDR_W-1:0]            m_axi_araddr,
    output wire [AXI_LEN_W-1:0]             m_axi_arlen,
    output wire [2:0]                       m_axi_arsize,
    output wire [1:0]                       m_axi_arburst,

    // Read Data
    input  wire                             m_axi_rvalid,
    output wire                             m_axi_rready,
    input  wire [AXI_ID_W-1:0]              m_axi_rid,
    input  wire [AXI_DATA_W-1:0]            m_axi_rdata,
    input  wire [1:0]                       m_axi_rresp,
    input  wire                             m_axi_rlast,

    // Write Address
    output wire                             m_axi_awvalid,
    input  wire                             m_axi_awready,
    output wire [AXI_ID_W-1:0]              m_axi_awid,
    output wire [AXI_ADDR_W-1:0]            m_axi_awaddr,
    output wire [AXI_LEN_W-1:0]             m_axi_awlen,
    output wire [2:0]                       m_axi_awsize,
    output wire [1:0]                       m_axi_awburst,

    // Write Data
    output wire                             m_axi_wvalid,
    input  wire                             m_axi_wready,
    output wire [AXI_DATA_W-1:0]            m_axi_wdata,
    output wire [AXI_DATA_W/8-1:0]          m_axi_wstrb,
    output wire                             m_axi_wlast,

    // Write Response
    input  wire                             m_axi_bvalid,
    output wire                             m_axi_bready,
    input  wire [AXI_ID_W-1:0]              m_axi_bid,
    input  wire [1:0]                       m_axi_bresp,

    // === Interrupt ===
    output wire                             irq_done,

    // === DMA completion signals ===
    output wire                             dma_rd_done,
    output wire                             dma_wr_done
);

    // ===================================================================
    // Parameter Register File signals
    // ===================================================================
    wire                            cfg_start;
    wire                            cfg_ping_pong_sel;
    wire                            cfg_pool_en;
    wire                            cfg_eltwise_en;
    wire                            cfg_clear_done;
    wire                            cfg_relu_en;
    wire                            cfg_out_ping_sel;   // NPU write bank for Out SRAM (CTRL[6])
    wire                            cfg_gemm_en;        // GEMM/FC mode (CTRL[7])
    wire                            cfg_hw_pad;         // hardware padding (CTRL[8])
    wire                            cfg_row_par_en;     // CTRL[9]: 16-row spatial parallelism
    wire                            cfg_copy_trig;      // 0x154: on-chip copy trigger pulse
    wire                            cfg_expand_trig;    // 0x158: img_expand trigger pulse
    wire                            expand_done;
    wire                            expand_busy;
    wire [SRAM_ADDR_W-1:0]          expand_addr;
    wire                            expand_en;
    wire                            expand_we;
    wire [ACT_DATA_W-1:0]           expand_wdata;
    wire                            copy_done;
    wire                            copy_busy;
    wire [SRAM_ADDR_W-1:0]          copy_out_rd_addr;
    wire                            copy_out_rd_en;
    wire [SRAM_ADDR_W-1:0]          copy_act_wr_addr;
    wire                            copy_act_wr_en;
    wire [ACT_DATA_W-1:0]           copy_act_wr_data;
    wire [7:0]                      cfg_pad_w;          // NPU_PAD[7:0]
    wire [7:0]                      cfg_pad_h;          // NPU_PAD[15:8]
    wire                            status_done_irq;
    wire                            status_busy;
    wire                            dma_rd_err;
    wire                            dma_wr_err;

    wire [SRAM_ADDR_W-1:0]          cfg_act_addr_ping;
    wire [SRAM_ADDR_W-1:0]          cfg_act_addr_pong;
    wire [SRAM_ADDR_W-1:0]          cfg_wgt_addr_ping;
    wire [SRAM_ADDR_W-1:0]          cfg_wgt_addr_pong;
    wire [SRAM_ADDR_W-1:0]          cfg_out_addr_ping;
    wire [SRAM_ADDR_W-1:0]          cfg_out_addr_pong;

    wire [15:0]                     cfg_dim_in_w;
    wire [15:0]                     cfg_dim_in_h;
    wire [15:0]                     cfg_dim_in_c;
    wire [15:0]                     cfg_dim_out_c;
    wire [7:0]                      cfg_kh;
    wire [7:0]                      cfg_kw;
    wire [7:0]                      cfg_sx;
    wire [7:0]                      cfg_sy;

    wire [31:0]                     cfg_bias_addr;
    wire [31:0]                     cfg_scale_addr;

    wire [ARRAY_COLS-1:0][PSUM_WIDTH-1:0] cfg_bias_val;
    wire [ARRAY_COLS-1:0][31:0]           cfg_scale_mul;
    wire [ARRAY_COLS-1:0][5:0]            cfg_scale_shift;

    wire [9:0]                      cfg_act_ic_tile;
    wire [9:0]                      cfg_act_oc_tile;
    wire [15:0]                     cfg_out_x_base;
    wire [15:0]                     cfg_out_y_base;
    wire [15:0]                     cfg_total_ops_h;
    wire [15:0]                     cfg_total_ops_w;

    // DMA control wires from register file
    wire                            cfg_dma_rd_req;
    wire [31:0]                     cfg_dma_rd_ddr_addr;
    wire [15:0]                     cfg_dma_rd_len;
    wire [SRAM_ADDR_W-1:0]          cfg_dma_rd_sram_base;
    wire                            cfg_dma_wr_req;
    wire [31:0]                     cfg_dma_wr_ddr_addr;
    wire [15:0]                     cfg_dma_wr_len;
    wire [SRAM_ADDR_W-1:0]          cfg_dma_wr_sram_base;

    // DMA done signals: driven by axi_dma, exposed as output ports for CPU IRQ

    // DMA SRAM select: 0=Act SRAM, 1=Wgt SRAM
    wire                            cfg_dma_sram_sel;

    // DMA read path control
    wire                            cfg_dma_out_rd_sel;   // 0=skip path, 1=DMA owns Out SRAM Port B
    wire [1:0]                      cfg_dma_rd_sram_sel;  // 0=Out, 1=Act, 2=Wgt
    wire                            cfg_dma_act_ping_sel; // DMA Act SRAM buffer select
    wire                            cfg_dma_wgt_ping_sel; // DMA Wgt SRAM buffer select
    wire                            cfg_dma_out_ping_sel; // DMA Out SRAM read bank (decoupled from NPU write bank)

    // ===================================================================
    // Instantiate Parameter Register File
    // ===================================================================
    param_regfile #(
        .NUM_OC       (ARRAY_COLS),
        .ADDR_W       (REG_ADDR_W),
        .DATA_W       (REG_DATA_W),
        .PSUM_WIDTH   (PSUM_WIDTH),
        .SRAM_ADDR_W  (SRAM_ADDR_W)
    ) u_param_regfile (
        .clk              (clk),
        .rst_n            (rst_n),
        // AXI-Lite
        .s_axi_awvalid    (s_axi_awvalid),
        .s_axi_awready    (s_axi_awready),
        .s_axi_awaddr     (s_axi_awaddr),
        .s_axi_wvalid     (s_axi_wvalid),
        .s_axi_wready     (s_axi_wready),
        .s_axi_wdata      (s_axi_wdata),
        .s_axi_wstrb      (s_axi_wstrb),
        .s_axi_bvalid     (s_axi_bvalid),
        .s_axi_bready     (s_axi_bready),
        .s_axi_bresp      (s_axi_bresp),
        .s_axi_arvalid    (s_axi_arvalid),
        .s_axi_arready    (s_axi_arready),
        .s_axi_araddr     (s_axi_araddr),
        .s_axi_rvalid     (s_axi_rvalid),
        .s_axi_rready     (s_axi_rready),
        .s_axi_rdata      (s_axi_rdata),
        .s_axi_rresp      (s_axi_rresp),
        // NPU internal
        .o_start          (cfg_start),
        .o_ping_pong_sel  (cfg_ping_pong_sel),
        .o_pool_en        (cfg_pool_en),
        .o_eltwise_en     (cfg_eltwise_en),
        .o_relu_en        (cfg_relu_en),
        .o_out_ping_sel   (cfg_out_ping_sel),
        .o_gemm_en        (cfg_gemm_en),
        .o_hw_pad         (cfg_hw_pad),
        .o_row_par_en     (cfg_row_par_en),
        .o_pad_w          (cfg_pad_w),
        .o_pad_h          (cfg_pad_h),
        .i_done_irq       (status_done_irq),
        .i_busy           (status_busy),
        .i_dma_rd_err     (dma_rd_err),
        .i_dma_wr_err     (dma_wr_err),
        .o_clear_done     (cfg_clear_done),
        .o_act_addr_ping  (cfg_act_addr_ping),
        .o_act_addr_pong  (cfg_act_addr_pong),
        .o_wgt_addr_ping  (cfg_wgt_addr_ping),
        .o_wgt_addr_pong  (cfg_wgt_addr_pong),
        .o_out_addr_ping  (cfg_out_addr_ping),
        .o_out_addr_pong  (cfg_out_addr_pong),
        .o_dim_in_w       (cfg_dim_in_w),
        .o_dim_in_h       (cfg_dim_in_h),
        .o_dim_in_c       (cfg_dim_in_c),
        .o_dim_out_c      (cfg_dim_out_c),
        .o_kernel_kh      (cfg_kh),
        .o_kernel_kw      (cfg_kw),
        .o_stride_sx      (cfg_sx),
        .o_stride_sy      (cfg_sy),
        .o_bias_addr      (cfg_bias_addr),
        .o_scale_addr     (cfg_scale_addr),
        .o_bias_val       (cfg_bias_val),
        .o_scale_mul      (cfg_scale_mul),
        .o_scale_shift    (cfg_scale_shift),
        .o_act_ic_tile    (cfg_act_ic_tile),
        .o_act_oc_tile    (cfg_act_oc_tile),
        .o_out_x_base     (cfg_out_x_base),
        .o_out_y_base     (cfg_out_y_base),
        .o_total_ops_h    (cfg_total_ops_h),
        .o_total_ops_w    (cfg_total_ops_w),
        // DMA control
        .o_dma_rd_req      (cfg_dma_rd_req),
        .o_dma_rd_ddr_addr (cfg_dma_rd_ddr_addr),
        .o_dma_rd_len      (cfg_dma_rd_len),
        .o_dma_rd_sram_base(cfg_dma_rd_sram_base),
        .o_dma_wr_req      (cfg_dma_wr_req),
        .o_dma_wr_ddr_addr (cfg_dma_wr_ddr_addr),
        .o_dma_wr_len      (cfg_dma_wr_len),
        .o_dma_wr_sram_base(cfg_dma_wr_sram_base),
        .i_dma_rd_done     (dma_rd_done),
        .i_dma_wr_done     (dma_wr_done),
        .o_dma_sram_sel    (cfg_dma_sram_sel),
        .o_dma_out_rd_sel  (cfg_dma_out_rd_sel),
        .o_dma_rd_sram_sel (cfg_dma_rd_sram_sel),
        .o_dma_act_ping_sel(cfg_dma_act_ping_sel),
        .o_dma_wgt_ping_sel(cfg_dma_wgt_ping_sel),
        .o_dma_out_ping_sel(cfg_dma_out_ping_sel),
        .o_copy_trig       (cfg_copy_trig),
        .i_copy_done       (copy_done),
        .o_expand_trig     (cfg_expand_trig),
        .i_expand_done     (expand_done)
    );

    // ===================================================================
    // SRAM wrappers
    // ===================================================================
    // Act SRAM: Port A = NPU read (FSM/im2col), Port B = DMA
    wire [SRAM_ADDR_W-1:0]   act_sram_addra;
    wire                     act_sram_ena;
    wire [ACT_DATA_W-1:0]    act_sram_doa;

    wire [SRAM_ADDR_W-1:0]   act_sram_addrb;
    wire                     act_sram_enb;
    wire                     act_sram_web;
    wire [ACT_DATA_W-1:0]    act_sram_dib;
    wire [ACT_DATA_W-1:0]    act_sram_dob;

    act_sram_wrapper #(
        .DATA_W (ACT_DATA_W),
        .ADDR_W (SRAM_ADDR_W)
    ) u_act_sram (
        .clk           (clk),
        .rst_n         (rst_n),
        .ena           (act_sram_ena),
        .addra         (act_sram_addra),
        .doa           (act_sram_doa),
        .enb           (act_sram_enb),
        .web           (act_sram_web),
        .addrb         (act_sram_addrb),
        .dib           (act_sram_dib),
        .dob           (act_sram_dob),
        .npu_ping_sel  (cfg_ping_pong_sel),
        .dma_ping_sel  (cfg_dma_act_ping_sel)
    );

    // Wgt SRAM: Port A = NPU read (wgt_reader), Port B = DMA
    wire [SRAM_ADDR_W-1:0]   wgt_sram_addra;
    wire                     wgt_sram_ena;
    wire [ACT_DATA_W-1:0]    wgt_sram_doa;

    wire [SRAM_ADDR_W-1:0]   wgt_sram_addrb;
    wire                     wgt_sram_enb;
    wire                     wgt_sram_web;
    wire [ACT_DATA_W-1:0]    wgt_sram_dib;
    wire [ACT_DATA_W-1:0]    wgt_sram_dob;

    wgt_sram_wrapper #(
        .DATA_W (ACT_DATA_W),
        .ADDR_W (SRAM_ADDR_W)
    ) u_wgt_sram (
        .clk           (clk),
        .rst_n         (rst_n),
        .ena           (wgt_sram_ena),
        .addra         (wgt_sram_addra),
        .doa           (wgt_sram_doa),
        .enb           (wgt_sram_enb),
        .web           (wgt_sram_web),
        .addrb         (wgt_sram_addrb),
        .dib           (wgt_sram_dib),
        .dob           (wgt_sram_dob),
        .npu_ping_sel  (cfg_ping_pong_sel),
        .dma_ping_sel  (cfg_dma_wgt_ping_sel)
    );

    // Out SRAM: Port A = NPU write (post-process output), Port B = DMA
    wire [OUT_SRAM_ADDR_W-1:0] out_sram_addra;
    wire                       out_sram_ena;
    wire                       out_sram_wea;
    wire [ACT_DATA_W-1:0]      out_sram_dia;
    wire [ACT_DATA_W-1:0]      out_sram_doa;

    wire [OUT_SRAM_ADDR_W-1:0] out_sram_addrb;
    wire                       out_sram_enb;
    wire                       out_sram_web;
    wire [ACT_DATA_W-1:0]      out_sram_dib;
    wire [ACT_DATA_W-1:0]      out_sram_dob;

    out_sram_wrapper #(
        .DATA_W (ACT_DATA_W),
        .ADDR_W (OUT_SRAM_ADDR_W)
    ) u_out_sram (
        .clk           (clk),
        .rst_n         (rst_n),
        .ena           (out_sram_ena),
        .wea           (out_sram_wea),
        .addra         (out_sram_addra),
        .dia           (out_sram_dia),
        .doa           (out_sram_doa),
        .enb           (out_sram_enb),
        .web           (out_sram_web),
        .addrb         (out_sram_addrb),
        .dib           (out_sram_dib),
        .dob           (out_sram_dob),
        .npu_ping_sel  (cfg_out_ping_sel),     // NPU write bank for Out SRAM (CTRL[6]), independent of global ping_pong
        .dma_ping_sel  (cfg_dma_out_ping_sel) // DMA Out SRAM read bank is independent of NPU write bank
    );

    // synthesis translate_off
    `ifdef DEBUG
    always @(posedge clk) begin
        if (act_sram_ena)
            $display("ACT_SRAM_RD: addr=%0h data=%0h ping=%0b",
                     act_sram_addra, act_sram_doa, cfg_ping_pong_sel);
        if (wgt_sram_ena)
            $display("WGT_SRAM_RD: addr=%0h data=%0h ping=%0b",
                     wgt_sram_addra, wgt_sram_doa, cfg_ping_pong_sel);
    end
    `endif
    // synthesis translate_on

    // ===================================================================
    // Top Controller FSM
    // ===================================================================
    wire [SRAM_ADDR_W-1:0]      fsm_act_sram_addr;
    wire                        fsm_act_sram_en;

    wire                        fsm_wgt_start_prefetch;
    wire [9:0]                  fsm_wgt_oc_base;
    wire [9:0]                  fsm_wgt_ic_group;
    wire [15:0]                 fsm_wgt_ic_groups_total;
    wire                        fsm_wgt_prefetch_done;
    wire [SRAM_ADDR_W-1:0]      fsm_wgt_base;
    wire                        fsm_prefetch_all;
    wire [3:0]                  fsm_wgt_ic_sel;

    wire                        fsm_im2col_row_start;
    wire                        fsm_im2col_win_advance;
    wire                        fsm_im2col_win_freeze;
    wire [3:0]                  fsm_im2col_offset_sel;
    wire [ACT_DATA_W-1:0]       fsm_im2col_pixel_data;
    wire                        fsm_im2col_pixel_vld;
    wire                        fsm_border;
    wire [3:0]                  fsm_im2col_load_tile;
    wire [15:0]                 fsm_im2col_group_base;
    wire                        fsm_im2col_sweep_advance;
    wire                        fsm_im2col_win_vld;
    wire [15:0]                 fsm_im2col_win_x;
    wire [15:0]                 fsm_im2col_win_y;

    wire                        fsm_array_vld;
    wire                        fsm_array_k_end;
    wire                        fsm_array_drain_en;

    wire                        fsm_pp_start;
    wire [15:0]                 fsm_group_size;
    wire [15:0]                 fsm_group_base;
    wire [SRAM_ADDR_W-1:0]      fsm_out_wr_addr;
    wire                        fsm_out_wr_en;
    wire [15:0]                 fsm_out_x;
    wire [15:0]                 fsm_out_y;
    wire                        fsm_pp_done;

    wire                        fsm_in_drain;
    wire                        fsm_in_post;
    wire                        fsm_busy;
    wire                        fsm_done_irq;

    wire [9:0]                  fsm_cur_ic_tile;
    wire [9:0]                  fsm_cur_oc_tile;
    wire [15:0]                 fsm_cur_ox;
    wire [15:0]                 fsm_cur_oy;

    top_controller_fsm #(
        .SRAM_ADDR_W    (SRAM_ADDR_W),
        .DATA_W         (ACT_DATA_W)
    ) u_controller (
        .clk                  (clk),
        .rst_n                (rst_n),
        .i_start              (cfg_start),
        .i_ping_pong_sel      (cfg_ping_pong_sel),
        .i_pool_en            (cfg_pool_en),
        .i_eltwise_en         (cfg_eltwise_en),
        .i_gemm_en            (cfg_gemm_en),
        .i_hw_pad             (cfg_hw_pad),
        .i_pad_w              (cfg_pad_w),
        .i_pad_h              (cfg_pad_h),
        .i_act_base_ping      (cfg_act_addr_ping),
        .i_act_base_pong      (cfg_act_addr_pong),
        .i_wgt_base_ping      (cfg_wgt_addr_ping),
        .i_wgt_base_pong      (cfg_wgt_addr_pong),
        .i_out_base_ping      (cfg_out_addr_ping),
        .i_out_base_pong      (cfg_out_addr_pong),
        .i_dim_in_w           (cfg_dim_in_w),
        .i_dim_in_h           (cfg_dim_in_h),
        .i_dim_in_c           (cfg_dim_in_c),
        .i_dim_out_c          (cfg_dim_out_c),
        .i_kernel_kh          (cfg_kh),
        .i_kernel_kw          (cfg_kw),
        .i_stride_sx          (cfg_sx),
        .i_stride_sy          (cfg_sy),

        .o_act_sram_addr      (fsm_act_sram_addr),
        .o_act_sram_en        (fsm_act_sram_en),
        .i_act_sram_data      (act_sram_doa),

        .o_wgt_start_prefetch (fsm_wgt_start_prefetch),
        .o_wgt_oc_base        (fsm_wgt_oc_base),
        .o_wgt_ic_group       (fsm_wgt_ic_group),
        .o_wgt_ic_groups_total(fsm_wgt_ic_groups_total),
        .o_wgt_base           (fsm_wgt_base),
        .o_prefetch_all       (fsm_prefetch_all),
        .o_wgt_ic_sel         (fsm_wgt_ic_sel),
        .i_wgt_prefetch_done  (fsm_wgt_prefetch_done),

        .o_im2col_row_start   (fsm_im2col_row_start),
        .o_im2col_win_advance (fsm_im2col_win_advance),
        .o_im2col_win_freeze  (fsm_im2col_win_freeze),
        .o_im2col_offset_sel  (fsm_im2col_offset_sel),
        .o_im2col_pixel_data  (fsm_im2col_pixel_data),
        .o_im2col_pixel_vld   (fsm_im2col_pixel_vld),
        .o_border             (fsm_border),
        .i_row_par_en         (cfg_row_par_en),
        .o_im2col_load_tile   (fsm_im2col_load_tile),
        .o_im2col_group_base  (fsm_im2col_group_base),
        .o_group_size         (fsm_group_size),
        .o_group_base         (fsm_group_base),
        .i_im2col_win_vld     (fsm_im2col_win_vld),
        .i_im2col_win_x       (fsm_im2col_win_x),
        .i_im2col_win_y       (fsm_im2col_win_y),

        .o_array_vld          (fsm_array_vld),
        .o_array_k_end        (fsm_array_k_end),
        .o_array_drain_en     (fsm_array_drain_en),
        .o_in_drain           (fsm_in_drain),
        .o_in_post            (fsm_in_post),

        .o_pp_start           (fsm_pp_start),
        .o_out_wr_addr        (fsm_out_wr_addr),
        .o_out_wr_en          (fsm_out_wr_en),
        .o_out_x              (fsm_out_x),
        .o_out_y              (fsm_out_y),
        .i_pp_done            (fsm_pp_done),

        .o_busy               (fsm_busy),
        .o_done_irq           (fsm_done_irq),

        .o_cur_ic_tile        (fsm_cur_ic_tile),
        .o_cur_oc_tile        (fsm_cur_oc_tile),
        .o_cur_ox             (fsm_cur_ox),
        .o_cur_oy             (fsm_cur_oy)
    );

    // Connect FSM to Act SRAM Port A
    assign act_sram_ena   = fsm_act_sram_en;
    assign act_sram_addra = fsm_act_sram_addr;

    // Status signals
    assign status_done_irq = fsm_done_irq;
    assign status_busy     = fsm_busy;
    assign irq_done        = fsm_done_irq;

    // ===================================================================
    // Weight Reader
    // ===================================================================
    wire [SRAM_ADDR_W-1:0]   wgt_reader_sram_addr;
    wire                     wgt_reader_sram_en;
    wire [WGT_BUS_W-1:0]     wgt_reader_wgt;
    wire                     wgt_reader_wgt_vld;

    // Runtime kernel-offset count for wgt_reader (conv 3x3 -> 9, GEMM 1x1 -> 1)
    wire [7:0] cfg_kernel_offsets = {4'd0, cfg_kh[3:0]} * {4'd0, cfg_kw[3:0]};

    wgt_reader #(
        .NUM_OC         (ARRAY_COLS),
        .SRAM_ADDR_W    (SRAM_ADDR_W)
    ) u_wgt_reader (
        .clk               (clk),
        .rst_n             (rst_n),
        .o_sram_addr       (wgt_reader_sram_addr),
        .o_sram_en         (wgt_reader_sram_en),
        .i_sram_data       (wgt_sram_doa),
        .i_start_prefetch  (fsm_wgt_start_prefetch),
        .i_oc_base         (fsm_wgt_oc_base),
        .i_ic_group        (fsm_wgt_ic_group),
        .i_ic_groups_total (fsm_wgt_ic_groups_total),
        .i_wgt_base        (fsm_wgt_base),
        .i_kernel_offsets  (cfg_kernel_offsets),
        .i_prefetch_all    (fsm_prefetch_all),
        .i_wgt_ic_sel      (fsm_wgt_ic_sel),
        .o_prefetch_done   (fsm_wgt_prefetch_done),
        .i_wgt_offset      (fsm_im2col_offset_sel),
        .o_wgt             (wgt_reader_wgt),
        .o_wgt_vld         (wgt_reader_wgt_vld)
    );

    // Connect wgt_reader to Wgt SRAM Port A
    assign wgt_sram_ena   = wgt_reader_sram_en;
    assign wgt_sram_addra = wgt_reader_sram_addr;

    // ===================================================================
    // Im2Col Line Buffer
    // ===================================================================
    wire [ACT_BUS_W-1:0]  im2col_act_window;
    wire                  im2col_win_vld;
    wire [15:0]           im2col_win_x;
    wire [15:0]           im2col_win_y;
    wire                  im2col_at_left_edge;
    wire                  im2col_at_right_edge;
    wire                  im2col_at_top_edge;
    wire                  im2col_at_bottom_edge;

    // -----------------------------------------------------------------
    // Multi IC-tile im2col (single timeline): one line buffer stores all IC
    // tiles per column; the FSM streams a column's tiles one-per-cycle
    // (load_tile) and pulses win_advance once per column.  The block keeps one
    // shared cur_x/rd_ptr/win_valid so all tiles stay column-aligned (no skew).
    // During CALC the systolic reads the window for the current IC tile.
    // ic_groups==1 → identical to the original single-tile design.
    // -----------------------------------------------------------------
    localparam ICG_MAX = 4;
    wire [3:0] cfg_ic_groups = (cfg_dim_in_c + 16'd15) >> 4;  // 1..ICG_MAX

    // SRAM Read Latency Compensation
    reg fsm_im2col_pixel_vld_d;
    reg fsm_im2col_win_advance_d;
    reg [3:0] fsm_im2col_load_tile_d;
    reg fsm_border_d;   // hw-pad border flag, delayed to match the 1-cycle SRAM read
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm_im2col_pixel_vld_d   <= 1'b0;
            fsm_im2col_win_advance_d <= 1'b0;
            fsm_im2col_load_tile_d   <= 4'd0;
            fsm_border_d             <= 1'b0;
        end else begin
            fsm_im2col_pixel_vld_d   <= fsm_im2col_pixel_vld;
            fsm_im2col_win_advance_d <= fsm_im2col_win_advance;
            fsm_im2col_load_tile_d   <= fsm_im2col_load_tile;
            fsm_border_d             <= fsm_border;
        end
    end

    // hw-pad: inject zero for border pixels (aligned with the delayed pixel data)
    wire [ACT_DATA_W-1:0] im2col_pixel_data_mux =
        fsm_border_d ? {ACT_DATA_W{1'b0}} : fsm_im2col_pixel_data;

    im2col_line_buffer #(
        .MAX_WIDTH   (MAX_WIDTH),
        .ARRAY_ROWS  (ARRAY_ROWS),
        .ICG_MAX     (ICG_MAX)
    ) u_im2col (
        .clk             (clk),
        .rst_n           (rst_n),
        .i_pixel_data    (im2col_pixel_data_mux),
        .i_pixel_vld     (fsm_im2col_pixel_vld_d),
        .i_pixel_tile    (fsm_im2col_load_tile_d),
        .i_ic_groups     (cfg_ic_groups),
        .i_win_tile      (fsm_cur_ic_tile[7:4]),
        .i_width         (cfg_dim_in_w),
        .i_height        (cfg_dim_in_h),
        .i_row_start     (fsm_im2col_row_start),
        .i_win_advance   (fsm_im2col_win_advance_d),
        .i_win_freeze    (fsm_im2col_win_freeze),
        .i_offset_sel    (fsm_im2col_offset_sel),
        .i_row_par_en    (cfg_row_par_en),
        .i_group_base    (fsm_im2col_group_base),
        .o_act_window    (im2col_act_window),
        .o_win_vld       (im2col_win_vld),
        .o_win_x         (im2col_win_x),
        .o_win_y         (im2col_win_y),
        .o_at_left_edge  (im2col_at_left_edge),
        .o_at_right_edge (im2col_at_right_edge),
        .o_at_top_edge   (im2col_at_top_edge),
        .o_at_bottom_edge(im2col_at_bottom_edge)
    );

    // Feedback to FSM
    assign fsm_im2col_win_vld = im2col_win_vld;
    assign fsm_im2col_win_x   = im2col_win_x;
    assign fsm_im2col_win_y   = im2col_win_y;

    // ===================================================================
    // Systolic 16×16 Array
    // ===================================================================
    wire [ARRAY_COLS-1:0][PSUM_WIDTH-1:0] array_psum_col;

    systolic_16x16 #(
        .ARRAY_ROWS  (ARRAY_ROWS),
        .ARRAY_COLS  (ARRAY_COLS)
    ) u_systolic (
        .clk         (clk),
        .rst_n       (rst_n),
        // GEMM mode: replicate the current IC-tile activation word to all 16
        // rows (same replication im2col does for conv) — every PE in a column
        // computes the identical dot product, so drain/POST capture is
        // phase-independent. Conv mode: im2col window as before.
        .i_act       (cfg_gemm_en ? {ARRAY_ROWS{act_sram_doa}} : im2col_act_window),
        .i_wgt       (wgt_reader_wgt),
        .o_psum_col  (array_psum_col),
        .i_vld       (fsm_array_vld),
        .i_k_end     (fsm_array_k_end),
        .i_drain_en  (fsm_array_drain_en)
    );

    // synthesis translate_off
    `ifdef C4DBG
    always @(posedge clk) begin
        if (fsm_array_vld && fsm_cur_ox == 16'd8 && fsm_cur_oy == 16'd8)
            $display("C4DBG ic_tile=%0d ko=%0d act_lo32=%08h act_hi32=%08h",
                     fsm_cur_ic_tile, fsm_im2col_offset_sel,
                     im2col_act_window[31:0], im2col_act_window[127:96]);
    end
    `endif
    `ifdef DEBUG
    always @(posedge clk) begin
        if (fsm_array_vld) begin
            $display("SYSTOLIC: act[0]=%0h wgt[0]=%0h psum_out[0]=%0h k_end=%0b drain=%0b",
                     im2col_act_window[0], wgt_reader_wgt[0], array_psum_col[0],
                     fsm_array_k_end, fsm_array_drain_en);
        end
    end
    // PIXFEED: nonzero pixel fed into im2col line buffer
    always @(posedge clk) begin
        if (fsm_im2col_pixel_vld_d && (fsm_im2col_pixel_data[7:0] != 8'd0))
            $display("PIXFEED: data=%0d row_start=%0b win_adv=%0b freeze=%0b winvld=%0b winx=%0d winy=%0d",
                     fsm_im2col_pixel_data[7:0], fsm_im2col_row_start,
                     fsm_im2col_win_advance_d, fsm_im2col_win_freeze,
                     im2col_win_vld, im2col_win_x, im2col_win_y);
        if (fsm_array_vld && (im2col_act_window[7:0] != 8'd0))
            $display("WINNZ: act_lo8=%0d ko=%0d", im2col_act_window[7:0], fsm_im2col_offset_sel);
    end

    // Trace im2col window data during CALC_KERNEL (first few cycles per tile)
    reg [15:0] dbg_ic_tile_cnt;
    reg [3:0]  dbg_kern_cycle;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dbg_ic_tile_cnt <= 16'd0;
            dbg_kern_cycle  <= 4'd0;
        end else begin
            if (fsm_array_k_end) begin
                dbg_kern_cycle  <= 4'd0;
                dbg_ic_tile_cnt <= dbg_ic_tile_cnt + 16'd1;
            end else if (fsm_array_vld) begin
                dbg_kern_cycle <= dbg_kern_cycle + 4'd1;
            end
            // Reset counters on new OC tile (detected by ic_tile wrapping to 0)
            if (fsm_cur_ic_tile == 10'd0 && fsm_array_vld && dbg_kern_cycle == 4'd1)
                dbg_ic_tile_cnt <= 16'd0;
        end
    end

    always @(posedge clk) begin
        if (fsm_array_vld && dbg_kern_cycle < 4'd3) begin
            $display("IM2COL_DBG: oc_tile=%0d ic_tile=%0d ox=%0d oy=%0d ko=%0d act_lo32=%0h",
                     fsm_cur_oc_tile, fsm_cur_ic_tile, fsm_cur_ox, fsm_cur_oy,
                     dbg_kern_cycle, im2col_act_window[31:0]);
        end
    end
    `endif
    // synthesis translate_on

    // ===================================================================
    // Post-Processing Top
    // ===================================================================
    // Conv output width for pooling row boundary detection
    // ===================================================================
    wire [15:0] conv_out_w = (cfg_dim_in_w - {8'd0, cfg_kw}) / {8'd0, cfg_sx} + 16'd1;

    // ===================================================================
    // Post-Processing Pipeline
    // ===================================================================
    wire [ACT_DATA_W-1:0] pp_feat;
    wire                  pp_feat_vld;

    wire                  rp_pool_done;   // row-par pool replay complete (FSM advance)

    // Valid signal for post-process from FSM (during DRAIN + pipeline flush)
    wire pp_input_vld;
    assign pp_input_vld = fsm_array_drain_en || fsm_pp_start;

    post_process_top #(
        .NUM_OC      (ARRAY_COLS),
        .MAX_WIDTH   (MAX_WIDTH)
    ) u_post_process (
        .clk           (clk),
        .rst_n         (rst_n),
        .i_start       (cfg_start),
        .i_psum        (array_psum_col),
        .i_psum_vld    (pp_input_vld),
        .i_bias        (cfg_bias_val),
        .i_scale_mul   (cfg_scale_mul),
        .i_scale_shift (cfg_scale_shift),
        .i_width       (conv_out_w),
        .i_pool_en     (cfg_pool_en),
        .i_relu_en     (cfg_relu_en),
        .i_in_drain    (fsm_in_drain),
        .i_in_post     (fsm_in_post),
        .i_row_par_en  (cfg_row_par_en),
        .i_group_size  (fsm_group_size),
        .o_rp_pool_done(rp_pool_done),
        .o_feat        (pp_feat),
        .o_feat_vld    (pp_feat_vld)
    );

    // ===================================================================
    // Vector ALU (Eltwise Add)
    // ===================================================================
    wire [ACT_DATA_W-1:0] alu_res;
    wire                  alu_vld;

    vector_alu #(
        .NUM_CH  (NUM_CH),
        .DATA_W  (ACT_DATA_W)
    ) u_vector_alu (
        .clk           (clk),
        .rst_n         (rst_n),
        .i_conv_res    (pp_feat),
        .i_skip_res    (out_sram_dob),         // Skip data from Out SRAM Port B
        .i_eltwise_en  (cfg_eltwise_en),
        .i_vld         (pp_feat_vld),
        .o_res         (alu_res),
        .o_vld         (alu_vld)
    );

    // ===================================================================
    // Out SRAM write (Port A) — from post-process/ALU pipeline
    // ===================================================================
    // Pooled output address counter: increments only when pool_vld writes,
    // so pooled outputs are stored contiguously.
    reg [SRAM_ADDR_W-1:0] pool_out_addr_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            pool_out_addr_cnt <= {SRAM_ADDR_W{1'b0}};
        else if (cfg_start)
            pool_out_addr_cnt <= {SRAM_ADDR_W{1'b0}};
        else if (out_sram_ena)
            pool_out_addr_cnt <= pool_out_addr_cnt + {{SRAM_ADDR_W-1{1'b0}}, 1'b1};
    end

    // ---- Row-parallel non-pool Out-SRAM write sequencer ----
    // Drain emits row 15 first -> row 0 last (pe_core). Array row r holds output
    // column (group_base + r). We count post-process valids within a conv point
    // and map the k-th valid (k=0..15) to column = group_base + (15 - k), writing
    // only columns inside [group_base, group_base + group_size).
    reg  [4:0]  rp_vld_cnt;        // counts pp_feat_vld within a conv point
    reg         rp_active;         // 1 while draining a row-par non-pool group
    reg         rp_in_drain_d;     // delayed fsm_in_drain (rising-edge detect)
    // Arm at DRAIN start, not S_POST: the post-process pipeline (~6 stages) emits
    // most of the 16 drained valids DURING S_DRAIN, before S_POST is entered, so
    // arming at fsm_pp_start would miss them (S_POST would hang waiting on rp_done).
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rp_vld_cnt    <= 5'd0;
            rp_active     <= 1'b0;
            rp_in_drain_d <= 1'b0;
        end else begin
            rp_in_drain_d <= fsm_in_drain;
            if (fsm_in_drain && !rp_in_drain_d) begin   // rising edge: drain start
                rp_vld_cnt <= 5'd0;
                rp_active  <= cfg_row_par_en & ~cfg_pool_en;
            end else if (rp_active && pp_feat_vld) begin
                if (rp_vld_cnt == 5'd15)
                    rp_active <= 1'b0;                   // 16 drained pixels captured
                rp_vld_cnt <= rp_vld_cnt + 5'd1;
            end
        end
    end
    wire [15:0] rp_col = fsm_group_base + (16'd15 - {11'd0, rp_vld_cnt});
    wire        rp_col_valid = rp_active && pp_feat_vld
                             && (rp_col >= fsm_group_base)
                             && (rp_col <  fsm_group_base + fsm_group_size);
    // fsm_out_wr_addr = out_base + oc_off + cur_oy*stride + cur_ox(=group_base).
    // Strip group_base, add the per-pixel rp_col.
    wire [SRAM_ADDR_W-1:0] rp_wr_addr = fsm_out_wr_addr
                             - fsm_group_base[SRAM_ADDR_W-1:0]
                             + rp_col[SRAM_ADDR_W-1:0];
    wire rp_wr_en = cfg_row_par_en & ~cfg_pool_en & rp_col_valid;
    wire rp_done  = rp_active && (rp_vld_cnt == 5'd15) && pp_feat_vld;

    // In pool mode the Out-SRAM write is self-timed by the pooler's output
    // valid (pool_vld = pp_feat_vld) with the contiguous pool_out_addr_cnt;
    // in non-pool mode it is FSM-driven exactly as before.
    assign out_sram_ena   = cfg_pool_en    ? pp_feat_vld
                          : cfg_row_par_en ? rp_wr_en
                          :                  fsm_out_wr_en;
    assign out_sram_wea   = out_sram_ena;
    assign out_sram_addra = cfg_pool_en    ? pool_out_addr_cnt[OUT_SRAM_ADDR_W-1:0]
                          : cfg_row_par_en ? rp_wr_addr[OUT_SRAM_ADDR_W-1:0]
                          :                  fsm_out_wr_addr[OUT_SRAM_ADDR_W-1:0];
    assign out_sram_dia   = alu_res;

    // synthesis translate_off
    `ifdef DEBUG
    always @(posedge clk) begin
        if (fsm_out_wr_en)
            $display("OUTSRAM_WR: addr=%0h data=%0h ping=%0b",
                     out_sram_addra, alu_res, cfg_ping_pong_sel);
        if (pp_feat_vld)
            $display("POST_DBG: feat_vld=%0b pp_lo32=%0h alu_lo32=%0h",
                     pp_feat_vld, pp_feat[31:0], alu_res[31:0]);
    end
    `endif
    // synthesis translate_on

    // Post-process done: feed back to FSM.  In pool mode advance once per conv
    // point (pp_start pulses at S_POST entry) so the FSM does not stall on the
    // 3/4 of points where 2x2 pooling emits no output; the Out-SRAM write is
    // decoupled and self-timed by pp_feat_vld above.  Non-pool keeps alu_vld.
    assign fsm_pp_done = (cfg_pool_en && cfg_row_par_en) ? rp_pool_done
                       : cfg_pool_en                     ? fsm_pp_start
                       : cfg_row_par_en                  ? rp_done
                       :                                   alu_vld;

    // ===================================================================
    // DMA write port mux: Act SRAM or Wgt SRAM based on cfg_dma_sram_sel
    // ===================================================================
    wire [SRAM_ADDR_W-1:0]   dma_sram_wr_addr;
    wire                     dma_sram_wr_en;
    wire [ACT_DATA_W-1:0]    dma_sram_wr_data;

    // DMA SRAM read wires (from DMA engine)
    wire [SRAM_ADDR_W-1:0] dma_sram_rd_addr;
    wire                   dma_sram_rd_en;
    wire [ACT_DATA_W-1:0]  dma_sram_rd_data;

    // Out SRAM Port B DMA-side wires
    wire [OUT_SRAM_ADDR_W-1:0] out_sram_addrb_dma;
    wire                       out_sram_enb_dma;

    assign out_sram_addrb_dma = dma_sram_rd_addr[OUT_SRAM_ADDR_W-1:0];
    assign out_sram_enb_dma   = dma_sram_rd_en;

    // Act SRAM Port B: DMA writes (sel=0) or DMA reads (rd_sram_sel=1)
    // DMA read/write are mutually exclusive; write has priority
    wire act_dma_wr_active = dma_sram_wr_en & ~cfg_dma_sram_sel;
    wire act_dma_rd_active = dma_sram_rd_en & (cfg_dma_rd_sram_sel == 2'd1);

    // Copy engine takes Act Port B while busy (writes); else the DMA muxes.
    assign act_sram_addrb = expand_busy ? expand_addr
                          : copy_busy ? copy_act_wr_addr
                          : act_dma_wr_active ? dma_sram_wr_addr : dma_sram_rd_addr;
    assign act_sram_enb   = expand_busy ? expand_en
                          : copy_busy ? copy_act_wr_en
                          : (act_dma_wr_active | act_dma_rd_active);
    assign act_sram_dib   = expand_busy ? expand_wdata
                          : copy_busy ? copy_act_wr_data : dma_sram_wr_data;
    assign act_sram_web   = expand_busy ? expand_we
                          : copy_busy ? copy_act_wr_en : act_dma_wr_active;

    // Wgt SRAM Port B: DMA writes (sel=1) or DMA reads (rd_sram_sel=2)
    wire wgt_dma_wr_active = dma_sram_wr_en & cfg_dma_sram_sel;
    wire wgt_dma_rd_active = dma_sram_rd_en & (cfg_dma_rd_sram_sel == 2'd2);

    assign wgt_sram_addrb = wgt_dma_wr_active ? dma_sram_wr_addr : dma_sram_rd_addr;
    assign wgt_sram_enb   = wgt_dma_wr_active | wgt_dma_rd_active;
    assign wgt_sram_dib   = dma_sram_wr_data;
    assign wgt_sram_web   = wgt_dma_wr_active;

    // Out SRAM Port B: mux between skip-read path and DMA read
    // Skip path: reads previous layer's output for eltwise addition
    wire skip_rd_en = cfg_eltwise_en & pp_feat_vld & ~cfg_dma_out_rd_sel;
    wire [OUT_SRAM_ADDR_W-1:0] skip_rd_addr = fsm_out_wr_addr[OUT_SRAM_ADDR_W-1:0];

    // Out SRAM Port B address/en mux
    wire [OUT_SRAM_ADDR_W-1:0] out_sram_addrb_mux;
    wire                       out_sram_enb_mux;

    assign out_sram_addrb_mux = cfg_dma_out_rd_sel ? out_sram_addrb_dma : skip_rd_addr;
    assign out_sram_enb_mux   = cfg_dma_out_rd_sel ? out_sram_enb_dma   : skip_rd_en;

    // Copy engine takes Out Port B while busy (reads); else the skip/DMA mux.
    assign out_sram_addrb = copy_busy ? copy_out_rd_addr : out_sram_addrb_mux;
    assign out_sram_enb   = copy_busy ? copy_out_rd_en   : out_sram_enb_mux;
    assign out_sram_web   = 1'b0;
    assign out_sram_dib   = {ACT_DATA_W{1'b0}};

    // DMA SRAM read data mux: select source based on cfg_dma_rd_sram_sel
    // 0=Out SRAM, 1=Act SRAM, 2=Wgt SRAM
    assign dma_sram_rd_data = (cfg_dma_rd_sram_sel == 2'd1) ? act_sram_dob :
                              (cfg_dma_rd_sram_sel == 2'd2) ? wgt_sram_dob :
                                                               out_sram_dob;

    // synthesis translate_off
    `ifdef DEBUG
    always @(posedge clk) begin
        if (out_sram_enb_mux && cfg_dma_out_rd_sel)
            $display("OUTSRAM_RD: addr=%0h data=%0h ping=%0b dma_ping=%0b",
                     out_sram_addrb_mux, out_sram_dob, cfg_ping_pong_sel, ~cfg_ping_pong_sel);
    end
    `endif
    // synthesis translate_on

    // ===================================================================
    // On-chip Out->Act copy engine (SRAM residency). src/dst/len reuse the DMA
    // SRAM-base/len registers; banks follow cfg_dma_out_ping_sel /
    // cfg_dma_act_ping_sel via the SRAM wrappers' dma_ping_sel. i_len = full
    // word count (cfg_dma_rd_len). Time-shares Port B with axi_dma (mutually
    // exclusive — firmware never overlaps them).
    // ===================================================================
    sram_copy #(
        .ADDR_W (SRAM_ADDR_W),
        .DATA_W (ACT_DATA_W)
    ) u_sram_copy (
        .clk           (clk),
        .rst_n         (rst_n),
        .i_trig        (cfg_copy_trig),
        .i_src_base    (cfg_dma_rd_sram_base),
        .i_dst_base    (cfg_dma_wr_sram_base),
        .i_len         (cfg_dma_rd_len),
        .o_out_rd_addr (copy_out_rd_addr),
        .o_out_rd_en   (copy_out_rd_en),
        .i_out_rd_data (out_sram_dob),
        .o_act_wr_addr (copy_act_wr_addr),
        .o_act_wr_en   (copy_act_wr_en),
        .o_act_wr_data (copy_act_wr_data),
        .o_busy        (copy_busy),
        .o_done        (copy_done)
    );

    // ===================================================================
    // Image expand engine (Conv1 input formatting): Act scratch (packed bytes)
    // -> Act output region (zero-extended 16-ch words). Shares Act Port B
    // (read+write on different cycles), highest priority via expand_busy.
    // ===================================================================
    img_expand #(
        .ADDR_W (SRAM_ADDR_W),
        .DATA_W (ACT_DATA_W)
    ) u_img_expand (
        .clk        (clk),
        .rst_n      (rst_n),
        .i_trig     (cfg_expand_trig),
        .i_src_base (cfg_dma_rd_sram_base),
        .i_dst_base (cfg_dma_wr_sram_base),
        .i_n_out    (cfg_dma_rd_len),
        .o_addr     (expand_addr),
        .o_en       (expand_en),
        .o_we       (expand_we),
        .o_wdata    (expand_wdata),
        .i_rdata    (act_sram_dob),
        .o_busy     (expand_busy),
        .o_done     (expand_done)
    );

    // ===================================================================
    // AXI DMA — connect to SRAM Port B (DDR ↔ SRAM)
    // ===================================================================
    axi_dma #(
        .AXI_ADDR_W  (AXI_ADDR_W),
        .AXI_DATA_W  (AXI_DATA_W),
        .AXI_ID_W    (AXI_ID_W),
        .AXI_LEN_W   (AXI_LEN_W),
        .SRAM_ADDR_W (SRAM_ADDR_W),
        .SRAM_DATA_W (ACT_DATA_W)
    ) u_axi_dma (
        .clk               (clk),
        .rst_n             (rst_n),

        .i_dma_rd_req      (cfg_dma_rd_req),
        .i_dma_rd_ddr_addr (cfg_dma_rd_ddr_addr),
        .i_dma_rd_len      (cfg_dma_rd_len[7:0]),
        .i_dma_rd_sram_base(cfg_dma_rd_sram_base),
        .o_dma_rd_done     (dma_rd_done),
        .o_dma_rd_err      (dma_rd_err),

        .i_dma_wr_req      (cfg_dma_wr_req),
        .i_dma_wr_ddr_addr (cfg_dma_wr_ddr_addr),
        .i_dma_wr_len      (cfg_dma_wr_len[7:0]),
        .i_dma_wr_sram_base(cfg_dma_wr_sram_base),
        .o_dma_wr_done     (dma_wr_done),
        .o_dma_wr_err      (dma_wr_err),

        // SRAM write (DDR read → SRAM write) — muxed to Act or Wgt SRAM
        .o_sram_wr_addr    (dma_sram_wr_addr),
        .o_sram_wr_en      (dma_sram_wr_en),
        .o_sram_wr_data    (dma_sram_wr_data),

        // SRAM read (SRAM read → DDR write) — muxed from Out/Act/Wgt SRAM
        .o_sram_rd_addr    (dma_sram_rd_addr),
        .o_sram_rd_en      (dma_sram_rd_en),
        .i_sram_rd_data    (dma_sram_rd_data),

        // AXI4-Full Master
        .m_axi_arvalid   (m_axi_arvalid),
        .m_axi_arready   (m_axi_arready),
        .m_axi_arid      (m_axi_arid),
        .m_axi_araddr    (m_axi_araddr),
        .m_axi_arlen     (m_axi_arlen),
        .m_axi_arsize    (m_axi_arsize),
        .m_axi_arburst   (m_axi_arburst),
        .m_axi_rvalid    (m_axi_rvalid),
        .m_axi_rready    (m_axi_rready),
        .m_axi_rid       (m_axi_rid),
        .m_axi_rdata     (m_axi_rdata),
        .m_axi_rresp     (m_axi_rresp),
        .m_axi_rlast     (m_axi_rlast),
        .m_axi_awvalid   (m_axi_awvalid),
        .m_axi_awready   (m_axi_awready),
        .m_axi_awid      (m_axi_awid),
        .m_axi_awaddr    (m_axi_awaddr),
        .m_axi_awlen     (m_axi_awlen),
        .m_axi_awsize    (m_axi_awsize),
        .m_axi_awburst   (m_axi_awburst),
        .m_axi_wvalid    (m_axi_wvalid),
        .m_axi_wready    (m_axi_wready),
        .m_axi_wdata     (m_axi_wdata),
        .m_axi_wstrb     (m_axi_wstrb),
        .m_axi_wlast     (m_axi_wlast),
        .m_axi_bvalid    (m_axi_bvalid),
        .m_axi_bready    (m_axi_bready),
        .m_axi_bid       (m_axi_bid),
        .m_axi_bresp     (m_axi_bresp)
    );

endmodule
