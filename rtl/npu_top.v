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
    parameter REG_ADDR_W    = 12,
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
    parameter MAX_WIDTH     = 512  // im2col line-buffer width cap; 512 covers 320-wide YOLO stem (conv0)
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
    wire [7:0]                      cfg_clip_max;
    wire                            cfg_pool_avg;
    wire [SRAM_ADDR_W-1:0]          cfg_skip_base;
    wire                            cfg_elt_signed;
    wire [7:0]                      cfg_elt_zp;
    wire                            cfg_elt_ratio_en;
    wire [16:0]                     cfg_elt_ratio_mul;
    wire [5:0]                      cfg_elt_ratio_shift;
    wire                            cfg_pw_en;
    wire                            cfg_dw_en;
    wire                            cfg_gpool_en;
    wire                            cfg_silu_en;
    wire                            cfg_silu_requant_en;
    wire [15:0]                     cfg_silu_requant_mul;
    wire [5:0]                      cfg_silu_requant_shift;
    wire [7:0]                      cfg_silu_requant_zp;
    wire [31:0]                     cfg_gavg_mul;
    wire [5:0]                      cfg_gavg_shift;
    wire                            fsm_last_spatial;
    wire                            cfg_out_ping_sel;   // NPU write bank for Out SRAM (CTRL[6])
    wire                            cfg_gemm_en;        // GEMM/FC mode (CTRL[7])
    wire                            cfg_hw_pad;         // hardware padding (CTRL[8])
    wire                            cfg_row_par_en;     // CTRL[9]: 16-row spatial parallelism
    wire                            cfg_gemm_reduce;    // CTRL[10]: GEMM 16-row IC-reduction
    wire                            cfg_row_block_en;   // CTRL[11]: row-block packing (#4)
    wire                            cfg_oc_single;      // CTRL[12]: all-OC in one start (decision O)
    wire                            cfg_int32_out;      // CTRL[13]: raw INT32 output (decision Q)
    wire [2:0]                      cfg_oc_tiles_total; // #16-OC tiles (1..4), assigned below
    wire [2:0]                      fsm_oc_tile_sel;     // active OC-tile from FSM (oc_single OC-inner loop)
    wire [2:0]                      fsm_wgt_oc_tile_sel = fsm_oc_tile_sel;
    wire [3:0]                       fsm_rows_per_grp;   // #4: R output rows packed
    wire                            cfg_copy_trig;      // 0x154: on-chip copy trigger pulse
    wire                            cfg_expand_trig;    // 0x158: img_expand trigger pulse
    wire                            cfg_upsample_trig;  // 0x3C8: upsample2x trigger pulse
    wire [15:0]                     cfg_upsample_in_w;
    wire [15:0]                     cfg_upsample_in_h;
    wire [15:0]                     cfg_upsample_ic_groups;
    wire                            expand_done;
    wire                            expand_busy;
    wire [SRAM_ADDR_W-1:0]          expand_addr;
    wire                            expand_en;
    wire                            expand_we;
    wire [ACT_DATA_W-1:0]           expand_wdata;
    wire                            upsample_done;
    wire                            upsample_busy;
    wire [SRAM_ADDR_W-1:0]          upsample_addr;
    wire                            upsample_en;
    wire                            upsample_we;
    wire [ACT_DATA_W-1:0]           upsample_wdata;
    wire                            cfg_maxpool5_trig;
    wire                            maxpool5_done;
    wire                            maxpool5_busy;
    wire [SRAM_ADDR_W-1:0]          maxpool5_addr;
    wire                            maxpool5_en;
    wire                            maxpool5_we;
    wire [ACT_DATA_W-1:0]           maxpool5_wdata;
    wire                            cfg_eltwise_trig;
    wire                            eltwise_done;
    wire                            eltwise_busy;
    wire [SRAM_ADDR_W-1:0]          eltwise_addr;
    wire                            eltwise_en;
    wire                            eltwise_we;
    wire [ACT_DATA_W-1:0]           eltwise_wdata;
    // DFL expectation engine
    wire                            cfg_dfl_trig;
    wire [SRAM_ADDR_W-1:0]          cfg_dfl_src;
    wire [SRAM_ADDR_W-1:0]          cfg_dfl_dst;
    wire [15:0]                     cfg_dfl_cnt;
    wire                            cfg_dfl_wload_en;
    wire [3:0]                      cfg_dfl_wload_idx;
    wire [15:0]                     cfg_dfl_wload_val;
    wire                            cfg_dfl_eload_en;
    wire [7:0]                      cfg_dfl_eload_idx;
    wire [15:0]                     cfg_dfl_eload_val;
    wire                            cfg_sigmoid_en;
    wire                            cfg_sigm_load_en;
    wire [7:0]                      cfg_sigm_load_idx;
    wire [7:0]                      cfg_sigm_load_val;
    wire                            cfg_silu_exact_en;
    wire                            cfg_ic_stream;      // CTRL[23]: large-IC conv streaming
    wire                            cfg_silu_load_en;
    wire [7:0]                      cfg_silu_load_idx;
    wire [7:0]                      cfg_silu_load_val;
    wire                            dfl_done;
    wire                            dfl_busy;
    wire [SRAM_ADDR_W-1:0]          dfl_addr;
    wire                            dfl_en;
    wire                            dfl_we;
    wire [ACT_DATA_W-1:0]           dfl_wdata;
    wire                            copy_done;
    wire                            copy_busy;
    wire [SRAM_ADDR_W-1:0]          copy_out_rd_addr;
    wire                            copy_out_rd_en;
    wire [SRAM_ADDR_W-1:0]          copy_act_wr_addr;
    wire                            copy_act_wr_en;
    wire [ACT_DATA_W-1:0]           copy_act_wr_data;
    wire [7:0]                      cfg_pad_w;          // NPU_PAD[7:0]
    wire [7:0]                      cfg_pad_h;          // NPU_PAD[15:8]
    wire [7:0]                      cfg_pad_value;      // NPU_PAD_VALUE[7:0]
    wire                            status_done_irq;
    wire                            status_busy;
    wire                            npu_done_irq;
    wire                            npu_busy_visible;
    wire                            dma_rd_err;
    wire                            dma_wr_err;
    wire [31:0]                     desc_base_lo;
    wire [31:0]                     desc_base_hi;
    wire [15:0]                     desc_count;
    wire                            desc_start;
    wire                            desc_abort;
    wire                            desc_irq_en;
    wire                            desc_clear_done;
    wire                            desc_busy;
    wire                            desc_done;
    wire                            desc_err;
    wire                            desc_aborted;
    wire [15:0]                     desc_pc;
    wire [7:0]                      desc_err_code;
    wire                            desc_irq;
    wire                            desc_arvalid;
    wire                            desc_arready;
    wire [AXI_ADDR_W-1:0]           desc_araddr;
    wire [7:0]                      desc_arlen;
    wire [2:0]                      desc_arsize;
    wire [1:0]                      desc_arburst;
    wire                            desc_rvalid;
    wire                            desc_rready;
    wire [AXI_DATA_W-1:0]           desc_rdata;
    wire [1:0]                      desc_rresp;
    wire                            desc_rlast;

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
    assign cfg_oc_tiles_total = (cfg_dim_out_c + 16'd15) >> 4;  // decision O: #16-OC tiles
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
    wire                            desc_dma_rd_req;
    wire                            desc_dma_wr_req;
    wire [31:0]                     desc_dma_rd_ddr_addr;
    wire [31:0]                     desc_dma_wr_ddr_addr;
    wire [15:0]                     desc_dma_rd_len;
    wire [15:0]                     desc_dma_wr_len;
    wire [SRAM_ADDR_W-1:0]          desc_dma_rd_sram_base;
    wire [SRAM_ADDR_W-1:0]          desc_dma_wr_sram_base;
    wire                            desc_dma_sram_sel;
    wire                            desc_dma_out_rd_sel;
    wire [1:0]                      desc_dma_rd_sram_sel;
    wire                            desc_dma_act_ping_sel;
    wire                            desc_dma_wgt_ping_sel;
    wire                            desc_dma_out_ping_sel;
    wire                            desc_copy_trig;
    wire                            desc_expand_trig;
    wire                            desc_qparam_we;
    wire [5:0]                      desc_qparam_idx;
    wire [31:0]                     desc_qparam_bias;
    wire [31:0]                     desc_qparam_scale;
    wire [5:0]                      desc_qparam_shift;

    reg [31:0] desc_bias_val [0:63];
    reg [31:0] desc_scale_mul [0:63];
    reg [5:0]  desc_scale_shift [0:63];

    always @(posedge clk) begin
        if (desc_qparam_we) begin
            desc_bias_val[desc_qparam_idx] <= desc_qparam_bias;
            desc_scale_mul[desc_qparam_idx] <= desc_qparam_scale;
            desc_scale_shift[desc_qparam_idx] <= desc_qparam_shift;
        end
    end

    wire                            run_dma_rd_req       = desc_busy ? desc_dma_rd_req       : cfg_dma_rd_req;
    wire [31:0]                     run_dma_rd_ddr_addr  = desc_busy ? desc_dma_rd_ddr_addr  : cfg_dma_rd_ddr_addr;
    wire [15:0]                     run_dma_rd_len       = desc_busy ? desc_dma_rd_len       : cfg_dma_rd_len;
    wire [SRAM_ADDR_W-1:0]          run_dma_rd_sram_base = desc_busy ? desc_dma_rd_sram_base : cfg_dma_rd_sram_base;
    wire                            run_dma_wr_req       = desc_busy ? desc_dma_wr_req       : cfg_dma_wr_req;
    wire [31:0]                     run_dma_wr_ddr_addr  = desc_busy ? desc_dma_wr_ddr_addr  : cfg_dma_wr_ddr_addr;
    wire [15:0]                     run_dma_wr_len       = desc_busy ? desc_dma_wr_len       : cfg_dma_wr_len;
    wire [SRAM_ADDR_W-1:0]          run_dma_wr_sram_base = desc_busy ? desc_dma_wr_sram_base : cfg_dma_wr_sram_base;
    wire                            run_dma_sram_sel     = desc_busy ? desc_dma_sram_sel     : cfg_dma_sram_sel;
    wire                            run_dma_out_rd_sel   = desc_busy ? desc_dma_out_rd_sel   : cfg_dma_out_rd_sel;
    wire [1:0]                      run_dma_rd_sram_sel  = desc_busy ? desc_dma_rd_sram_sel  : cfg_dma_rd_sram_sel;
    wire                            run_dma_act_ping_sel = desc_busy ? desc_dma_act_ping_sel : cfg_dma_act_ping_sel;
    wire                            run_dma_wgt_ping_sel = desc_busy ? desc_dma_wgt_ping_sel : cfg_dma_wgt_ping_sel;
    wire                            run_dma_out_ping_sel = desc_busy ? desc_dma_out_ping_sel : cfg_dma_out_ping_sel;
    wire                            run_copy_trig        = desc_busy ? desc_copy_trig        : cfg_copy_trig;
    wire                            run_expand_trig      = desc_busy ? desc_expand_trig      : cfg_expand_trig;

    // ===================================================================
    // Performance counter event strobes (fed to param_regfile @0x3A4+).
    // fsm_busy / fsm_array_vld are declared HERE (earlier than the FSM signal
    // block below) because the param_regfile instance taps them above their FSM
    // connection -- declaring at first use avoids an implicit-net redeclaration.
    // ===================================================================
    wire fsm_array_vld;
    wire fsm_busy;

    wire perf_rd_beat = m_axi_rvalid && m_axi_rready;  // one read data beat
    wire perf_wr_beat = m_axi_wvalid && m_axi_wready;  // one write data beat

    reg  perf_rd_busy_r, perf_wr_busy_r;               // burst-outstanding flags
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_rd_busy_r <= 1'b0;
            perf_wr_busy_r <= 1'b0;
        end else begin
            if (m_axi_arvalid && m_axi_arready)
                perf_rd_busy_r <= 1'b1;
            else if (m_axi_rvalid && m_axi_rready && m_axi_rlast)
                perf_rd_busy_r <= 1'b0;

            if (m_axi_awvalid && m_axi_awready)
                perf_wr_busy_r <= 1'b1;
            else if (m_axi_bvalid && m_axi_bready)
                perf_wr_busy_r <= 1'b0;
        end
    end

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
        .o_gemm_reduce    (cfg_gemm_reduce),
        .o_row_block_en   (cfg_row_block_en),
        .o_oc_single      (cfg_oc_single),
        .o_int32_out      (cfg_int32_out),
        .o_pool_avg       (cfg_pool_avg),
        .o_pw_en          (cfg_pw_en),
        .o_dw_en          (cfg_dw_en),
        .o_gpool_en       (cfg_gpool_en),
        .o_silu_en        (cfg_silu_en),
        .o_silu_requant_en(cfg_silu_requant_en),
        .o_silu_requant_mul(cfg_silu_requant_mul),
        .o_silu_requant_shift(cfg_silu_requant_shift),
        .o_silu_requant_zp(cfg_silu_requant_zp),
        .o_gavg_mul       (cfg_gavg_mul),
        .o_gavg_shift     (cfg_gavg_shift),
        .o_pad_w          (cfg_pad_w),
        .o_pad_h          (cfg_pad_h),
        .o_pad_value      (cfg_pad_value),
        .o_clip_max       (cfg_clip_max),
        .o_skip_base      (cfg_skip_base),
        .o_elt_signed     (cfg_elt_signed),
        .o_elt_zp         (cfg_elt_zp),
        .o_elt_ratio_en   (cfg_elt_ratio_en),
        .o_elt_ratio_mul  (cfg_elt_ratio_mul),
        .o_elt_ratio_shift(cfg_elt_ratio_shift),
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
        .i_oc_tile_sel    (fsm_oc_tile_sel),
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
        .i_expand_done     (expand_done),
        .o_upsample_trig   (cfg_upsample_trig),
        .i_upsample_done   (upsample_done),
        .o_maxpool5_trig   (cfg_maxpool5_trig),
        .i_maxpool5_done   (maxpool5_done),
        .o_eltwise_trig    (cfg_eltwise_trig),
        .i_eltwise_done    (eltwise_done),
        .o_upsample_in_w   (cfg_upsample_in_w),
        .o_upsample_in_h   (cfg_upsample_in_h),
        .o_upsample_ic_groups(cfg_upsample_ic_groups),
        .o_desc_base_lo    (desc_base_lo),
        .o_desc_base_hi    (desc_base_hi),
        .o_desc_count      (desc_count),
        .o_desc_start      (desc_start),
        .o_desc_abort      (desc_abort),
        .o_desc_irq_en     (desc_irq_en),
        .o_desc_clear_done (desc_clear_done),
        .i_desc_busy       (desc_busy),
        .i_desc_done       (desc_done),
        .i_desc_err        (desc_err),
        .i_desc_aborted    (desc_aborted),
        .i_desc_pc         (desc_pc),
        .i_desc_err_code   (desc_err_code),
        .i_dfl_done        (dfl_done),
        .o_dfl_src         (cfg_dfl_src),
        .o_dfl_dst         (cfg_dfl_dst),
        .o_dfl_cnt         (cfg_dfl_cnt),
        .o_dfl_trig        (cfg_dfl_trig),
        .o_dfl_wload_en    (cfg_dfl_wload_en),
        .o_dfl_wload_idx   (cfg_dfl_wload_idx),
        .o_dfl_wload_val   (cfg_dfl_wload_val),
        .o_dfl_eload_en    (cfg_dfl_eload_en),
        .o_dfl_eload_idx   (cfg_dfl_eload_idx),
        .o_dfl_eload_val   (cfg_dfl_eload_val),
        .o_sigmoid_en      (cfg_sigmoid_en),
        .o_sigm_load_en    (cfg_sigm_load_en),
        .o_sigm_load_idx   (cfg_sigm_load_idx),
        .o_sigm_load_val   (cfg_sigm_load_val),
        .o_silu_exact_en   (cfg_silu_exact_en),
        .o_ic_stream       (cfg_ic_stream),
        .o_silu_load_en    (cfg_silu_load_en),
        .o_silu_load_idx   (cfg_silu_load_idx),
        .o_silu_load_val   (cfg_silu_load_val),
        .i_perf_busy       (fsm_busy),
        .i_perf_arr_active (fsm_array_vld),
        .i_perf_rd_beat    (perf_rd_beat),
        .i_perf_wr_beat    (perf_wr_beat),
        .i_perf_rd_busy    (perf_rd_busy_r),
        .i_perf_wr_busy    (perf_wr_busy_r)
    );

    descriptor_engine #(
        .ADDR_W      (AXI_ADDR_W),
        .SRAM_ADDR_W (SRAM_ADDR_W)
    ) u_descriptor_engine (
        .clk                (clk),
        .rst_n              (rst_n),
        .i_desc_base_lo     (desc_base_lo),
        .i_desc_base_hi     (desc_base_hi),
        .i_desc_count       (desc_count),
        .i_desc_start       (desc_start),
        .i_desc_abort       (desc_abort),
        .i_desc_irq_en      (desc_irq_en),
        .i_desc_clear_done  (desc_clear_done),
        .i_global_idle      (~npu_busy_visible),
        .o_busy             (desc_busy),
        .o_done             (desc_done),
        .o_err              (desc_err),
        .o_aborted          (desc_aborted),
        .o_pc               (desc_pc),
        .o_err_code         (desc_err_code),
        .o_irq              (desc_irq),
        .m_axi_arvalid      (desc_arvalid),
        .m_axi_arready      (desc_arready),
        .m_axi_araddr       (desc_araddr),
        .m_axi_arlen        (desc_arlen),
        .m_axi_arsize       (desc_arsize),
        .m_axi_arburst      (desc_arburst),
        .m_axi_rvalid       (desc_rvalid),
        .m_axi_rready       (desc_rready),
        .m_axi_rdata        (desc_rdata),
        .m_axi_rresp        (desc_rresp),
        .m_axi_rlast        (desc_rlast),
        .o_dma_rd_req       (desc_dma_rd_req),
        .o_dma_wr_req       (desc_dma_wr_req),
        .o_dma_rd_ddr_addr  (desc_dma_rd_ddr_addr),
        .o_dma_wr_ddr_addr  (desc_dma_wr_ddr_addr),
        .o_dma_rd_len       (desc_dma_rd_len),
        .o_dma_wr_len       (desc_dma_wr_len),
        .o_dma_rd_sram_base (desc_dma_rd_sram_base),
        .o_dma_wr_sram_base (desc_dma_wr_sram_base),
        .o_dma_sram_sel     (desc_dma_sram_sel),
        .o_dma_out_rd_sel   (desc_dma_out_rd_sel),
        .o_dma_rd_sram_sel  (desc_dma_rd_sram_sel),
        .o_dma_act_ping_sel (desc_dma_act_ping_sel),
        .o_dma_wgt_ping_sel (desc_dma_wgt_ping_sel),
        .o_dma_out_ping_sel (desc_dma_out_ping_sel),
        .o_copy_trig        (desc_copy_trig),
        .o_expand_trig      (desc_expand_trig),
        .i_dma_rd_done      (dma_rd_done),
        .i_dma_wr_done      (dma_wr_done),
        .i_copy_done        (copy_done),
        .i_expand_done      (expand_done),
        .o_qparam_we        (desc_qparam_we),
        .o_qparam_idx       (desc_qparam_idx),
        .o_qparam_bias      (desc_qparam_bias),
        .o_qparam_scale     (desc_qparam_scale),
        .o_qparam_shift     (desc_qparam_shift)
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
        .dma_ping_sel  (run_dma_act_ping_sel)
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
        .dma_ping_sel  (run_dma_wgt_ping_sel)
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
        .dma_ping_sel  (run_dma_out_ping_sel) // DMA Out SRAM read bank is independent of NPU write bank
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
    wire [4:0]                  fsm_im2col_load_tile;
    wire [15:0]                 fsm_im2col_group_base;
    wire                        fsm_im2col_sweep_advance;
    wire                        fsm_im2col_win_vld;
    wire [15:0]                 fsm_im2col_win_x;
    wire [15:0]                 fsm_im2col_win_y;

    wire                        fsm_array_k_end;
    wire                        fsm_array_drain_en;

    // GEMM-reduce (decision M)
    wire [ARRAY_ROWS*ARRAY_COLS*128-1:0] wgt_reader_plane;
    wire                        wgt_reader_plane_done;
    wire                        fsm_wgt_plane_trig;
    wire [3:0]                  fsm_superstep;
    wire                        fsm_rdc_act_we;     // capture act_sram_doa → act_row
    wire [3:0]                  fsm_rdc_act_idx;    // which row (0..15)
    wire                        fsm_rdc_act_clear;  // clear act_row at super-step start

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
        .i_gemm_reduce        (cfg_gemm_reduce),
        .i_row_block_en       (cfg_row_block_en),
        .i_oc_single          (cfg_oc_single),
        .i_ic_stream          (cfg_ic_stream),
        .i_pw_en              (cfg_pw_en),
        .o_oc_tile_sel        (fsm_oc_tile_sel),
        .o_rows_per_grp       (fsm_rows_per_grp),
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
        .o_wgt_plane_trig     (fsm_wgt_plane_trig),
        .o_superstep          (fsm_superstep),
        .i_plane_done         (wgt_reader_plane_done),
        .o_rdc_act_we         (fsm_rdc_act_we),
        .o_rdc_act_idx        (fsm_rdc_act_idx),
        .o_rdc_act_clear      (fsm_rdc_act_clear),
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
        .o_cur_oy             (fsm_cur_oy),
        .o_last_spatial       (fsm_last_spatial)
    );

    // Connect FSM to Act SRAM Port A
    assign act_sram_ena   = fsm_act_sram_en;
    assign act_sram_addra = fsm_act_sram_addr;

    // Status signals
    assign status_done_irq = npu_done_irq;
    assign status_busy     = npu_busy_visible;
    assign irq_done        = npu_done_irq | desc_irq;

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
        .i_oc_single       (cfg_oc_single),
        .i_oc_tiles_total  (cfg_oc_tiles_total),
        .i_oc_tile_sel     (fsm_wgt_oc_tile_sel),
        .i_wgt_offset      (fsm_im2col_offset_sel),
        .o_wgt             (wgt_reader_wgt),
        .o_wgt_vld         (wgt_reader_wgt_vld),
        .i_gemm_reduce     (cfg_gemm_reduce),
        .i_plane_trig      (fsm_wgt_plane_trig),
        .i_superstep       (fsm_superstep),
        .o_wgt_plane       (wgt_reader_plane),
        .o_plane_done      (wgt_reader_plane_done)
    );

    // ===================================================================
    // Depthwise engine (CTRL[15] dw_en): channel-parallel MAC bypassing the
    // array. Prefetches KO depthwise weight words from Wgt SRAM Port A during
    // LOAD_ROW (wgt_reader idle then), then accumulates the im2col window taps.
    // ===================================================================
    wire [ARRAY_COLS-1:0][PSUM_WIDTH-1:0] dw_psum;
    wire                     dw_vld;
    wire [SRAM_ADDR_W-1:0]   dw_wgt_rd_addr;
    wire                     dw_wgt_rd_en;
    wire                     dw_wgt_active;
    // (depthwise_engine instance is below, after im2col declares im2col_act_window)

    // Connect wgt_reader to Wgt SRAM Port A; depthwise borrows it during prefetch.
    assign wgt_sram_ena   = dw_wgt_active ? dw_wgt_rd_en   : wgt_reader_sram_en;
    assign wgt_sram_addra = dw_wgt_active ? dw_wgt_rd_addr : wgt_reader_sram_addr;

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
    localparam ICG_MAX = 16; // im2col window holds 16 IC tiles (256 ch): lets icg<=16 3x3
                             // convs run the resident path instead of CPU ic_stream psum.
    wire [4:0] cfg_ic_groups = (cfg_dim_in_c + 16'd15) >> 4;  // 1..ICG_MAX

    // SRAM Read Latency Compensation
    reg fsm_im2col_pixel_vld_d;
    reg fsm_im2col_win_advance_d;
    reg [4:0] fsm_im2col_load_tile_d;
    reg fsm_border_d;   // hw-pad border flag, delayed to match the 1-cycle SRAM read
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm_im2col_pixel_vld_d   <= 1'b0;
            fsm_im2col_win_advance_d <= 1'b0;
            fsm_im2col_load_tile_d   <= 5'd0;
            fsm_border_d             <= 1'b0;
        end else begin
            fsm_im2col_pixel_vld_d   <= fsm_im2col_pixel_vld;
            fsm_im2col_win_advance_d <= fsm_im2col_win_advance;
            fsm_im2col_load_tile_d   <= fsm_im2col_load_tile;
            fsm_border_d             <= fsm_border;
        end
    end

    // hw-pad: inject configured border byte (default 0, YOLO uses input zero-point).
    wire [ACT_DATA_W-1:0] im2col_pixel_data_mux =
        fsm_border_d ? {16{cfg_pad_value}} : fsm_im2col_pixel_data;

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
        .i_win_tile      ({1'b0, fsm_cur_ic_tile[7:4]}),
        .i_width         (cfg_dim_in_w),
        .i_height        (cfg_dim_in_h),
        .i_row_start     (fsm_im2col_row_start),
        .i_win_advance   (fsm_im2col_win_advance_d),
        .i_win_freeze    (fsm_im2col_win_freeze),
        .i_offset_sel    (fsm_im2col_offset_sel),
        .i_row_par_en    (cfg_row_par_en),
        .i_group_base    (fsm_im2col_group_base),
        .i_row_block_en  (cfg_row_block_en),
        .i_group_size    (fsm_group_size),
        .i_rows_per_grp  (fsm_rows_per_grp),
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
    // GEMM-reduce activation row register (decision M): 16 distinct IC-tile
    // words, one per array row, loaded from Act SRAM by the FSM super-step
    // sequencer. Cleared at super-step start so out-of-range rows feed 0.
    // ===================================================================
    reg [127:0] act_row [0:ARRAY_ROWS-1];
    integer ar_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (ar_i = 0; ar_i < ARRAY_ROWS; ar_i = ar_i + 1)
                act_row[ar_i] <= 128'd0;
        end else begin
            if (fsm_rdc_act_clear)
                for (ar_i = 0; ar_i < ARRAY_ROWS; ar_i = ar_i + 1)
                    act_row[ar_i] <= 128'd0;
            else if (fsm_rdc_act_we)
                act_row[fsm_rdc_act_idx] <= act_sram_doa[127:0];
        end
    end
    wire [ARRAY_ROWS*128-1:0] act_row_bus;
    genvar arg;
    generate
        for (arg = 0; arg < ARRAY_ROWS; arg = arg + 1) begin : gen_act_row_bus
            assign act_row_bus[arg*128 +: 128] = act_row[arg];
        end
    endgenerate

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
        .i_act       (cfg_gemm_reduce       ? act_row_bus
                    : (cfg_gemm_en|cfg_pw_en) ? {ARRAY_ROWS{act_sram_doa}}
                    :                         im2col_act_window),
        .i_wgt       (wgt_reader_wgt),
        .i_wgt_plane (wgt_reader_plane),
        .i_reduce    (cfg_gemm_reduce),
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
    wire [15:0] conv_out_h = (cfg_dim_in_h - {8'd0, cfg_kh}) / {8'd0, cfg_sy} + 16'd1;
    // Decision O: pooled words per OC-tile = (conv_out_w/2)*(conv_out_h/2). Used to
    // place each OC tile's pooled output tile-major in the single-start (oc_single).
    wire [SRAM_ADDR_W-1:0] pool_tile_words =
         (conv_out_w[SRAM_ADDR_W-1:0] >> 1) * (conv_out_h[SRAM_ADDR_W-1:0] >> 1);

    // ===================================================================
    // Post-Processing Pipeline
    // ===================================================================
    wire [ACT_DATA_W-1:0] pp_feat;
    wire                  pp_feat_vld;

    wire                  rp_pool_done;   // row-par pool replay complete (FSM advance)
    wire [1:0]            pp_pool_tile;   // decision P: tile of the pooled output (aligned w/ pp_feat_vld)
    wire [NUM_CH*32-1:0] pp_feat32;      // decision Q: raw INT32 post-process output (16×INT32)

    // Valid signal for post-process from FSM (during DRAIN + pipeline flush)
    // Depthwise engine instance (im2col_act_window now in scope). Prefetches KO
    // weight words from Wgt Port A during LOAD_ROW, then channel-parallel MACs the
    // im2col taps; o_psum is muxed into post_process when cfg_dw_en.
    depthwise_engine #(
        .NUM_CH(ARRAY_COLS), .PSUM_WIDTH(PSUM_WIDTH), .SRAM_ADDR_W(SRAM_ADDR_W)
    ) u_depthwise (
        .clk(clk), .rst_n(rst_n),
        .i_start(cfg_start & cfg_dw_en),   // only borrow Wgt port / accumulate in depthwise mode
        .i_wgt_base(cfg_ping_pong_sel ? cfg_wgt_addr_pong : cfg_wgt_addr_ping),
        .o_wgt_rd_addr(dw_wgt_rd_addr),
        .o_wgt_rd_en(dw_wgt_rd_en),
        .o_wgt_active(dw_wgt_active),
        .i_wgt_rd_data(wgt_sram_doa),
        .i_calc_vld(fsm_array_vld),
        .i_offset(fsm_im2col_offset_sel),
        .i_act(im2col_act_window[127:0]),
        .i_k_end(fsm_array_k_end),
        .o_psum(dw_psum),
        .o_vld(dw_vld)
    );

    wire pp_input_vld;
    assign pp_input_vld = fsm_array_drain_en || fsm_pp_start;

    post_process_top #(
        .NUM_OC      (ARRAY_COLS),
        .MAX_WIDTH   (MAX_WIDTH)
    ) u_post_process (
        .clk           (clk),
        .rst_n         (rst_n),
        .i_start       (cfg_start),
        .i_psum        (cfg_dw_en ? dw_psum : array_psum_col),  // depthwise bypasses the array
        .i_psum_vld    (pp_input_vld),
        .i_bias        (cfg_bias_val),
        .i_scale_mul   (cfg_scale_mul),
        .i_scale_shift (cfg_scale_shift),
        .i_width       (conv_out_w),
        .i_pool_en     (cfg_pool_en),
        .i_pool_avg    (cfg_pool_avg),
        .i_relu_en     (cfg_relu_en),
        .i_silu_en     (cfg_silu_en),
        .i_silu_requant_en(cfg_silu_requant_en),
        .i_silu_requant_mul(cfg_silu_requant_mul),
        .i_silu_requant_shift(cfg_silu_requant_shift),
        .i_silu_requant_zp(cfg_silu_requant_zp),
        .i_clip_max    (cfg_clip_max),
        .i_sigmoid_en  (cfg_sigmoid_en),
        .i_sigm_load_en(cfg_sigm_load_en),
        .i_sigm_load_idx(cfg_sigm_load_idx),
        .i_sigm_load_val(cfg_sigm_load_val),
        .i_silu_exact_en(cfg_silu_exact_en),
        .i_silu_load_en(cfg_silu_load_en),
        .i_silu_load_idx(cfg_silu_load_idx),
        .i_silu_load_val(cfg_silu_load_val),
        .i_in_drain    (fsm_in_drain),
        .i_in_post     (fsm_in_post),
        .i_row_par_en  (cfg_row_par_en),
        .i_group_size  (fsm_group_size),
        .i_rows_per_grp(fsm_rows_per_grp),
        .i_oc_tile     (fsm_oc_tile_sel[1:0]),
        // INT32 psum accumulate modes: tied OFF here (wired by a later task);
        // ACC_NONE = byte-identical legacy post-process.
        .i_acc_mode     (2'd0),
        .i_psum_readback({(16*32){1'b0}}),
        .o_pool_tile   (pp_pool_tile),
        .o_rp_pool_done(rp_pool_done),
        .o_feat        (pp_feat),
        .o_feat_vld    (pp_feat_vld),
        .o_feat32      (pp_feat32)
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
        .i_elt_ratio_en   (cfg_elt_ratio_en),
        .i_elt_ratio_mul  (cfg_elt_ratio_mul),
        .i_elt_ratio_shift(cfg_elt_ratio_shift),
        .i_signed_mode (cfg_elt_signed),
        .i_elt_zp      (cfg_elt_zp),
        .i_vld         (pp_feat_vld),
        .o_res         (alu_res),
        .o_vld         (alu_vld)
    );

    // ===================================================================
    // Out SRAM write (Port A) — from post-process/ALU pipeline
    // ===================================================================
    // Pooled output address counter: increments only when pool_vld writes, so
    // pooled outputs are stored contiguously WITHIN an OC tile.
    // Decision O/P (oc_single): the OC-inner loop emits pool outputs group-major /
    // tile-inner — each OC tile is revisited once per group-row, interleaved with the
    // other tiles. So the within-tile counter must be PER OC-tile (persist across a
    // tile's interleaved visits, NOT reset on every tile switch). pool_wr_addr adds a
    // tile-major base (oc_t*pool_tile_words). oc_single off ⇒ only tile 0's counter,
    // base 0, reset on cfg_start ⇒ byte-identical to the legacy single counter.
    localparam POOL_NTILES = 4;
    reg [SRAM_ADDR_W-1:0] pool_out_cnt [0:POOL_NTILES-1];
    // pp_pool_tile is registered in the pooler aligned with pp_feat_vld, so the
    // write address uses the tile of the pixel actually being written (not the FSM's
    // possibly-advanced oc_t). 0 when oc_single off.
    wire [1:0] pool_act_tile = cfg_oc_single ? pp_pool_tile : 2'd0;
    integer pti;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            for (pti = 0; pti < POOL_NTILES; pti = pti + 1)
                pool_out_cnt[pti] <= {SRAM_ADDR_W{1'b0}};
        else if (cfg_start)
            for (pti = 0; pti < POOL_NTILES; pti = pti + 1)
                pool_out_cnt[pti] <= {SRAM_ADDR_W{1'b0}};
        else if (out_sram_ena)
            pool_out_cnt[pool_act_tile] <= pool_out_cnt[pool_act_tile]
                                         + {{SRAM_ADDR_W-1{1'b0}}, 1'b1};
    end
    // Tile-major pooled write address (oc_single adds the per-tile base).
    wire [SRAM_ADDR_W-1:0] pool_wr_addr = pool_out_cnt[pool_act_tile]
         + (cfg_oc_single ? ({{(SRAM_ADDR_W-2){1'b0}}, pool_act_tile} * pool_tile_words)
                          : {SRAM_ADDR_W{1'b0}});

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
    // Drained array row r = 15 - rp_vld_cnt. #4 row-block: r → block b=r/gs
    // (output row cur_oy+b), col c=r%gs; legacy (R=1): b=0, c=r.
    wire [4:0]  rb_arr_row = 5'd15 - rp_vld_cnt;
    wire        rb_b = cfg_row_block_en && ({11'd0, rb_arr_row} >= fsm_group_size);
    wire [15:0] rb_c = {11'd0, rb_arr_row} - (rb_b ? fsm_group_size : 16'd0);
    wire [15:0] rp_col = fsm_group_base
                       + (cfg_row_block_en ? rb_c : {11'd0, rb_arr_row});
    wire        rp_col_valid = rp_active && pp_feat_vld
                             && (rp_col >= fsm_group_base)
                             && (rp_col <  fsm_group_base + fsm_group_size);
    // fsm_out_wr_addr = out_base + oc_off + cur_oy*stride + cur_ox(=group_base).
    // Strip group_base, add the per-pixel rp_col; row-block adds b*out_row_stride
    // for the block's output row (cur_oy+b).
    wire [SRAM_ADDR_W-1:0] rp_wr_addr = fsm_out_wr_addr
                             - fsm_group_base[SRAM_ADDR_W-1:0]
                             + rp_col[SRAM_ADDR_W-1:0]
                             + (rb_b ? conv_out_w[SRAM_ADDR_W-1:0] : {SRAM_ADDR_W{1'b0}});
    wire rp_wr_en = cfg_row_par_en & ~cfg_pool_en & rp_col_valid;
    wire rp_done  = rp_active && (rp_vld_cnt == 5'd15) && pp_feat_vld;

    // ---- Decision Q: raw INT32 output write sequencer (GEMM/final-FC) ----
    // 16 OC × INT32 = 512 bits don't fit one 128-bit Out word, so latch the INT32
    // post-process result (pp_feat32) at the single GEMM write pulse and serialize
    // it into 4 Out-SRAM words (4×INT32 each) at base..base+3. Off ⇒ unused ⇒
    // byte-identical (the INT8 path drives out_sram_* exactly as before).
    reg [NUM_CH*32-1:0]   i32_buf;
    reg [1:0]             i32_cnt;
    reg                   i32_active;
    reg [SRAM_ADDR_W-1:0] i32_base;
    reg                   i32_done_pending;
    reg                   i32_done_irq_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            i32_buf <= {NUM_CH*32{1'b0}}; i32_cnt <= 2'd0;
            i32_active <= 1'b0; i32_base <= {SRAM_ADDR_W{1'b0}};
        end else if (cfg_int32_out && fsm_out_wr_en && !i32_active) begin
            i32_buf    <= pp_feat32;                    // 16×INT32 latched at the GEMM write
            // FC (single position): base = addr. Spatial INT32 (ic_stream, large-IC
            // streaming partials): each output position owns 4 words, so space the
            // base x4 to avoid overlap with the next position. Positions in a conv
            // sweep are >>4 cycles apart, so the 4-cycle serialize never drops.
            i32_base   <= cfg_ic_stream ? {fsm_out_wr_addr[SRAM_ADDR_W-3:0], 2'b00}
                                        : fsm_out_wr_addr;
            i32_cnt    <= 2'd0;
            i32_active <= 1'b1;
        end else if (i32_active) begin
            if (i32_cnt == 2'd3) i32_active <= 1'b0;
            i32_cnt <= i32_cnt + 2'd1;
        end
    end
    wire                  i32_wr_en   = i32_active;
    wire [SRAM_ADDR_W-1:0] i32_wr_addr = i32_base + {{(SRAM_ADDR_W-2){1'b0}}, i32_cnt};
    wire [ACT_DATA_W-1:0] i32_wr_data = i32_buf[i32_cnt*ACT_DATA_W +: ACT_DATA_W];
    wire                  i32_last_write = i32_active && (i32_cnt == 2'd3);

    // In INT32-output mode the FSM can reach DONE before the 4-word serializer
    // has committed all logits to Out SRAM. Delay the externally visible done
    // pulse until the final serializer write is on the SRAM write edge.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            i32_done_pending <= 1'b0;
            i32_done_irq_r   <= 1'b0;
        end else begin
            i32_done_irq_r <= 1'b0;
            if (cfg_start) begin
                i32_done_pending <= 1'b0;
            end

            if (cfg_int32_out && fsm_done_irq) begin
                if (!i32_active || i32_last_write) begin
                    i32_done_pending <= 1'b0;
                    i32_done_irq_r   <= 1'b1;
                end else begin
                    i32_done_pending <= 1'b1;
                end
            end else if (i32_done_pending && i32_last_write) begin
                i32_done_pending <= 1'b0;
                i32_done_irq_r   <= 1'b1;
            end
        end
    end

    assign npu_done_irq     = cfg_int32_out ? i32_done_irq_r : fsm_done_irq;
    assign npu_busy_visible = fsm_busy | (cfg_int32_out & (i32_active | i32_done_pending));

    // -------------------------------------------------------------------
    // Global average pooling (CTRL[17] gpool_en). Accumulates the per-position
    // post-process output across the OC tile's spatial sweep and emits one mean
    // word at the last position. Used in plain (non-pool/non-row_par/non-int32)
    // legacy multi-start mode: one NPU start per 16-OC tile. gpool_en off ⇒ block
    // idle and the write mux below is unchanged ⇒ byte-identical.
    // -------------------------------------------------------------------
    wire [ACT_DATA_W-1:0] gavg_feat;
    wire                  gavg_feat_vld;
    wire gpool_feed_vld = cfg_gpool_en & fsm_out_wr_en;                    // one per output position
    wire gpool_last     = cfg_gpool_en & fsm_out_wr_en & fsm_last_spatial; // last position of tile

    global_avg #(.NUM_CH(ARRAY_COLS)) u_global_avg (
        .clk(clk), .rst_n(rst_n),
        .i_start(cfg_start),
        .i_feat(alu_res),
        .i_feat_vld(gpool_feed_vld),
        .i_last(gpool_last),
        .i_avg_mul(cfg_gavg_mul),
        .i_avg_shift(cfg_gavg_shift),
        .o_feat(gavg_feat),
        .o_feat_vld(gavg_feat_vld)
    );

    // Latch the tile's first output address; the single mean word lands there.
    reg [OUT_SRAM_ADDR_W-1:0] gpool_addr;
    reg                       gpool_addr_set;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gpool_addr     <= {OUT_SRAM_ADDR_W{1'b0}};
            gpool_addr_set <= 1'b0;
        end else if (cfg_start) begin
            gpool_addr_set <= 1'b0;
        end else if (cfg_gpool_en && fsm_out_wr_en && !gpool_addr_set) begin
            gpool_addr     <= fsm_out_wr_addr[OUT_SRAM_ADDR_W-1:0];
            gpool_addr_set <= 1'b1;
        end
    end

    // In pool mode the Out-SRAM write is self-timed by the pooler's output
    // valid (pool_vld = pp_feat_vld) with the contiguous pool_out_addr_cnt;
    // in non-pool mode it is FSM-driven exactly as before.
    assign out_sram_ena   = cfg_gpool_en   ? gavg_feat_vld
                          : cfg_int32_out  ? i32_wr_en
                          : cfg_pool_en    ? pp_feat_vld
                          : cfg_row_par_en ? rp_wr_en
                          :                  fsm_out_wr_en;
    assign out_sram_wea   = out_sram_ena;
    assign out_sram_addra = cfg_gpool_en   ? gpool_addr
                          : cfg_int32_out  ? i32_wr_addr[OUT_SRAM_ADDR_W-1:0]
                          : cfg_pool_en    ? pool_wr_addr[OUT_SRAM_ADDR_W-1:0]
                          : cfg_row_par_en ? rp_wr_addr[OUT_SRAM_ADDR_W-1:0]
                          :                  fsm_out_wr_addr[OUT_SRAM_ADDR_W-1:0];
    assign out_sram_dia   = cfg_gpool_en   ? gavg_feat
                          : cfg_int32_out  ? i32_wr_data : alu_res;

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
    wire act_dma_wr_active = dma_sram_wr_en & ~run_dma_sram_sel;
    wire act_dma_rd_active = dma_sram_rd_en & (run_dma_rd_sram_sel == 2'd1);

    // dfl_unit / maxpool5x5 / eltwise_add / img_expand / upsample2x / sram_copy take Act Port B while busy; else DMA muxes.
    assign act_sram_addrb = dfl_busy ? dfl_addr
                          : maxpool5_busy ? maxpool5_addr
                          : eltwise_busy ? eltwise_addr
                          : expand_busy ? expand_addr
                          : upsample_busy ? upsample_addr
                          : copy_busy ? copy_act_wr_addr
                          : act_dma_wr_active ? dma_sram_wr_addr : dma_sram_rd_addr;
    assign act_sram_enb   = dfl_busy ? dfl_en
                          : maxpool5_busy ? maxpool5_en
                          : eltwise_busy ? eltwise_en
                          : expand_busy ? expand_en
                          : upsample_busy ? upsample_en
                          : copy_busy ? copy_act_wr_en
                          : (act_dma_wr_active | act_dma_rd_active);
    assign act_sram_dib   = dfl_busy ? dfl_wdata
                          : maxpool5_busy ? maxpool5_wdata
                          : eltwise_busy ? eltwise_wdata
                          : expand_busy ? expand_wdata
                          : upsample_busy ? upsample_wdata
                          : copy_busy ? copy_act_wr_data : dma_sram_wr_data;
    assign act_sram_web   = dfl_busy ? dfl_we
                          : maxpool5_busy ? maxpool5_we
                          : eltwise_busy ? eltwise_we
                          : expand_busy ? expand_we
                          : upsample_busy ? upsample_we
                          : copy_busy ? copy_act_wr_en : act_dma_wr_active;

    // Wgt SRAM Port B: DMA writes (sel=1) or DMA reads (rd_sram_sel=2)
    wire wgt_dma_wr_active = dma_sram_wr_en & run_dma_sram_sel;
    wire wgt_dma_rd_active = dma_sram_rd_en & (run_dma_rd_sram_sel == 2'd2);

    assign wgt_sram_addrb = wgt_dma_wr_active ? dma_sram_wr_addr : dma_sram_rd_addr;
    assign wgt_sram_enb   = wgt_dma_wr_active | wgt_dma_rd_active;
    assign wgt_sram_dib   = dma_sram_wr_data;
    assign wgt_sram_web   = wgt_dma_wr_active;

    // Out SRAM Port B: mux between skip-read path and DMA read
    // Skip path: reads previous layer's output for eltwise addition
    wire skip_rd_en = cfg_eltwise_en & pp_feat_vld & ~run_dma_out_rd_sel;
    // Residual skip source: configurable base (NPU_SKIP_BASE) + the write-position
    // offset. cfg_skip_base=0 ⇒ same-address legacy in-place accumulate.
    wire [OUT_SRAM_ADDR_W-1:0] skip_rd_addr =
             cfg_skip_base[OUT_SRAM_ADDR_W-1:0] + fsm_out_wr_addr[OUT_SRAM_ADDR_W-1:0];

    // Out SRAM Port B address/en mux
    wire [OUT_SRAM_ADDR_W-1:0] out_sram_addrb_mux;
    wire                       out_sram_enb_mux;

    assign out_sram_addrb_mux = run_dma_out_rd_sel ? out_sram_addrb_dma : skip_rd_addr;
    assign out_sram_enb_mux   = run_dma_out_rd_sel ? out_sram_enb_dma   : skip_rd_en;

    // sram_copy takes Out Port B while busy (reads); else the skip/DMA mux.
    assign out_sram_addrb = copy_busy ? copy_out_rd_addr : out_sram_addrb_mux;
    assign out_sram_enb   = copy_busy ? copy_out_rd_en   : out_sram_enb_mux;
    assign out_sram_web   = 1'b0;
    assign out_sram_dib   = {ACT_DATA_W{1'b0}};

    // DMA SRAM read data mux: select source based on cfg_dma_rd_sram_sel
    // 0=Out SRAM, 1=Act SRAM, 2=Wgt SRAM
    assign dma_sram_rd_data = (run_dma_rd_sram_sel == 2'd1) ? act_sram_dob :
                              (run_dma_rd_sram_sel == 2'd2) ? wgt_sram_dob :
                                                               out_sram_dob;

    // synthesis translate_off
    `ifdef DEBUG
    always @(posedge clk) begin
        if (out_sram_enb_mux && run_dma_out_rd_sel)
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
        .i_trig        (run_copy_trig),
        .i_src_base    (run_dma_rd_sram_base),
        .i_dst_base    (run_dma_wr_sram_base),
        .i_len         (run_dma_rd_len),
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
        .i_trig     (run_expand_trig),
        .i_src_base (run_dma_rd_sram_base),
        .i_dst_base (run_dma_wr_sram_base),
        .i_n_out    (run_dma_rd_len),
        .o_addr     (expand_addr),
        .o_en       (expand_en),
        .o_we       (expand_we),
        .o_wdata    (expand_wdata),
        .i_rdata    (act_sram_dob),
        .o_busy     (expand_busy),
        .o_done     (expand_done)
    );

    // ===================================================================
    // Shared Act-SRAM 2x nearest-neighbor upsample engine for YOLO FPN/PAN.
    // Source/destination bases reuse the DMA SRAM-base registers; dimensions
    // come from NPU_UPSAMPLE_CFG*. Same hardware path, default idle for MNIST.
    // ===================================================================
    upsample2x #(
        .ADDR_W (SRAM_ADDR_W),
        .DATA_W (ACT_DATA_W)
    ) u_upsample2x (
        .clk         (clk),
        .rst_n       (rst_n),
        .i_trig      (cfg_upsample_trig),
        .i_src_base  (cfg_dma_rd_sram_base),
        .i_dst_base  (cfg_dma_wr_sram_base),
        .i_in_w      (cfg_upsample_in_w),
        .i_in_h      (cfg_upsample_in_h),
        .i_ic_groups (cfg_upsample_ic_groups),
        .o_addr      (upsample_addr),
        .o_en        (upsample_en),
        .o_we        (upsample_we),
        .o_wdata     (upsample_wdata),
        .i_rdata     (act_sram_dob),
        .o_busy      (upsample_busy),
        .o_done      (upsample_done)
    );

    // ===================================================================
    // Shared Act-SRAM 5x5 stride-1 signed maxpool engine. Source/destination
    // bases reuse the DMA SRAM-base registers; dimensions come from the same
    // generic spatial-engine cfg as upsample2x. Default idle for MNIST.
    // ===================================================================
    maxpool5x5 #(
        .ADDR_W (SRAM_ADDR_W),
        .DATA_W (ACT_DATA_W)
    ) u_maxpool5x5 (
        .clk         (clk),
        .rst_n       (rst_n),
        .i_trig      (cfg_maxpool5_trig),
        .i_src_base  (cfg_dma_rd_sram_base),
        .i_dst_base  (cfg_dma_wr_sram_base),
        .i_w         (cfg_upsample_in_w),
        .i_h         (cfg_upsample_in_h),
        .i_ic_groups (cfg_upsample_ic_groups),
        .o_addr      (maxpool5_addr),
        .o_en        (maxpool5_en),
        .o_we        (maxpool5_we),
        .o_wdata     (maxpool5_wdata),
        .i_rdata     (act_sram_dob),
        .o_busy      (maxpool5_busy),
        .o_done      (maxpool5_done)
    );

    // ===================================================================
    // Shared Act-SRAM signed INT8 eltwise-add engine. Source0 reuses the DMA
    // read SRAM base, source1 uses NPU_SKIP_BASE, destination reuses DMA write
    // SRAM base, and length reuses DMA_RD_LEN. Default idle for MNIST.
    // ===================================================================
    eltwise_add_engine #(
        .ADDR_W (SRAM_ADDR_W),
        .DATA_W (ACT_DATA_W)
    ) u_eltwise_add_engine (
        .clk           (clk),
        .rst_n         (rst_n),
        .i_trig        (cfg_eltwise_trig),
        .i_src0_base   (cfg_dma_rd_sram_base),
        .i_src1_base   (cfg_skip_base),
        .i_dst_base    (cfg_dma_wr_sram_base),
        .i_len         (cfg_dma_rd_len),
        .i_zp          (cfg_elt_zp),
        .i_ratio_en    (cfg_elt_ratio_en),
        .i_ratio_mul   (cfg_elt_ratio_mul),
        .i_ratio_shift (cfg_elt_ratio_shift),
        .o_addr        (eltwise_addr),
        .o_en          (eltwise_en),
        .o_we          (eltwise_we),
        .o_wdata       (eltwise_wdata),
        .i_rdata       (act_sram_dob),
        .o_busy        (eltwise_busy),
        .o_done        (eltwise_done)
    );

    // ===================================================================
    // Shared Act-SRAM DFL expectation engine for YOLO detect-head decode.
    // Reads 16xINT8 logit words from Act SRAM, writes packed Q8.8 distances.
    // Default idle for MNIST. (CTRL[21] sigmoid_en routes to post_process.)
    // ===================================================================
    dfl_unit #(
        .ADDR_W (SRAM_ADDR_W),
        .DATA_W (ACT_DATA_W)
    ) u_dfl_unit (
        .clk         (clk),
        .rst_n       (rst_n),
        .i_trig      (cfg_dfl_trig),
        .i_src_base  (cfg_dfl_src),
        .i_dst_base  (cfg_dfl_dst),
        .i_cnt       (cfg_dfl_cnt),
        .i_wload_en  (cfg_dfl_wload_en),
        .i_wload_idx (cfg_dfl_wload_idx),
        .i_wload_val (cfg_dfl_wload_val),
        .i_eload_en  (cfg_dfl_eload_en),
        .i_eload_idx (cfg_dfl_eload_idx),
        .i_eload_val (cfg_dfl_eload_val),
        .o_addr      (dfl_addr),
        .o_en        (dfl_en),
        .o_we        (dfl_we),
        .o_wdata     (dfl_wdata),
        .i_rdata     (act_sram_dob),
        .o_busy      (dfl_busy),
        .o_done      (dfl_done)
    );

    // ===================================================================
    // AXI DMA — connect to SRAM Port B (DDR ↔ SRAM)
    // ===================================================================
    wire                         dma_axi_arvalid;
    wire                         dma_axi_arready;
    wire [AXI_ID_W-1:0]          dma_axi_arid;
    wire [AXI_ADDR_W-1:0]        dma_axi_araddr;
    wire [AXI_LEN_W-1:0]         dma_axi_arlen;
    wire [2:0]                   dma_axi_arsize;
    wire [1:0]                   dma_axi_arburst;
    wire                         dma_axi_rvalid;
    wire                         dma_axi_rready;
    wire [AXI_ID_W-1:0]          dma_axi_rid;
    wire [AXI_DATA_W-1:0]        dma_axi_rdata;
    wire [1:0]                   dma_axi_rresp;
    wire                         dma_axi_rlast;
    wire                         dma_axi_awvalid;
    wire                         dma_axi_awready;
    wire [AXI_ID_W-1:0]          dma_axi_awid;
    wire [AXI_ADDR_W-1:0]        dma_axi_awaddr;
    wire [AXI_LEN_W-1:0]         dma_axi_awlen;
    wire [2:0]                   dma_axi_awsize;
    wire [1:0]                   dma_axi_awburst;
    wire                         dma_axi_wvalid;
    wire                         dma_axi_wready;
    wire [AXI_DATA_W-1:0]        dma_axi_wdata;
    wire [AXI_DATA_W/8-1:0]      dma_axi_wstrb;
    wire                         dma_axi_wlast;
    wire                         dma_axi_bvalid;
    wire                         dma_axi_bready;
    wire [AXI_ID_W-1:0]          dma_axi_bid;
    wire [1:0]                   dma_axi_bresp;

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

        .i_dma_rd_req      (run_dma_rd_req),
        .i_dma_rd_ddr_addr (run_dma_rd_ddr_addr),
        .i_dma_rd_len      (run_dma_rd_len[7:0]),
        .i_dma_rd_sram_base(run_dma_rd_sram_base),
        .o_dma_rd_done     (dma_rd_done),
        .o_dma_rd_err      (dma_rd_err),

        .i_dma_wr_req      (run_dma_wr_req),
        .i_dma_wr_ddr_addr (run_dma_wr_ddr_addr),
        .i_dma_wr_len      (run_dma_wr_len[7:0]),
        .i_dma_wr_sram_base(run_dma_wr_sram_base),
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
        .m_axi_arvalid   (dma_axi_arvalid),
        .m_axi_arready   (dma_axi_arready),
        .m_axi_arid      (dma_axi_arid),
        .m_axi_araddr    (dma_axi_araddr),
        .m_axi_arlen     (dma_axi_arlen),
        .m_axi_arsize    (dma_axi_arsize),
        .m_axi_arburst   (dma_axi_arburst),
        .m_axi_rvalid    (dma_axi_rvalid),
        .m_axi_rready    (dma_axi_rready),
        .m_axi_rid       (dma_axi_rid),
        .m_axi_rdata     (dma_axi_rdata),
        .m_axi_rresp     (dma_axi_rresp),
        .m_axi_rlast     (dma_axi_rlast),
        .m_axi_awvalid   (dma_axi_awvalid),
        .m_axi_awready   (dma_axi_awready),
        .m_axi_awid      (dma_axi_awid),
        .m_axi_awaddr    (dma_axi_awaddr),
        .m_axi_awlen     (dma_axi_awlen),
        .m_axi_awsize    (dma_axi_awsize),
        .m_axi_awburst   (dma_axi_awburst),
        .m_axi_wvalid    (dma_axi_wvalid),
        .m_axi_wready    (dma_axi_wready),
        .m_axi_wdata     (dma_axi_wdata),
        .m_axi_wstrb     (dma_axi_wstrb),
        .m_axi_wlast     (dma_axi_wlast),
        .m_axi_bvalid    (dma_axi_bvalid),
        .m_axi_bready    (dma_axi_bready),
        .m_axi_bid       (dma_axi_bid),
        .m_axi_bresp     (dma_axi_bresp)
    );

    wire desc_axi_rd_sel = desc_arvalid | desc_rready;
    assign m_axi_arvalid   = desc_axi_rd_sel ? desc_arvalid : dma_axi_arvalid;
    assign m_axi_arid      = desc_axi_rd_sel ? {AXI_ID_W{1'b0}} : dma_axi_arid;
    assign m_axi_araddr    = desc_axi_rd_sel ? desc_araddr : dma_axi_araddr;
    assign m_axi_arlen     = desc_axi_rd_sel ? desc_arlen : dma_axi_arlen;
    assign m_axi_arsize    = desc_axi_rd_sel ? desc_arsize : dma_axi_arsize;
    assign m_axi_arburst   = desc_axi_rd_sel ? desc_arburst : dma_axi_arburst;
    assign desc_arready    = desc_axi_rd_sel ? m_axi_arready : 1'b0;
    assign dma_axi_arready = desc_axi_rd_sel ? 1'b0 : m_axi_arready;

    assign desc_rvalid     = desc_axi_rd_sel ? m_axi_rvalid : 1'b0;
    assign desc_rdata      = m_axi_rdata;
    assign desc_rresp      = m_axi_rresp;
    assign desc_rlast      = m_axi_rlast;
    assign dma_axi_rvalid  = desc_axi_rd_sel ? 1'b0 : m_axi_rvalid;
    assign dma_axi_rid     = m_axi_rid;
    assign dma_axi_rdata   = m_axi_rdata;
    assign dma_axi_rresp   = m_axi_rresp;
    assign dma_axi_rlast   = m_axi_rlast;
    assign m_axi_rready    = desc_axi_rd_sel ? desc_rready : dma_axi_rready;

    assign m_axi_awvalid   = dma_axi_awvalid;
    assign m_axi_awid      = dma_axi_awid;
    assign m_axi_awaddr    = dma_axi_awaddr;
    assign m_axi_awlen     = dma_axi_awlen;
    assign m_axi_awsize    = dma_axi_awsize;
    assign m_axi_awburst   = dma_axi_awburst;
    assign dma_axi_awready = m_axi_awready;
    assign m_axi_wvalid    = dma_axi_wvalid;
    assign m_axi_wdata     = dma_axi_wdata;
    assign m_axi_wstrb     = dma_axi_wstrb;
    assign m_axi_wlast     = dma_axi_wlast;
    assign dma_axi_wready  = m_axi_wready;
    assign dma_axi_bvalid  = m_axi_bvalid;
    assign dma_axi_bid     = m_axi_bid;
    assign dma_axi_bresp   = m_axi_bresp;
    assign m_axi_bready    = dma_axi_bready;

endmodule
