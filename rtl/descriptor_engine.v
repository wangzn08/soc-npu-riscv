// -------------------------------------------------------------------
// Hardware descriptor queue engine, v1 skeleton.
//
// V1 command format is fixed at 16 x 32-bit words. The engine fetches one
// descriptor as four 128-bit AXI beats, executes it, then advances linearly.
// This skeleton supports NOP and STOP_IRQ; later tasks add DMA/NPU ops.
// -------------------------------------------------------------------

module descriptor_engine #(
    parameter ADDR_W      = 32,
    parameter SRAM_ADDR_W = 14
) (
    input  wire             clk,
    input  wire             rst_n,

    input  wire [31:0]      i_desc_base_lo,
    input  wire [31:0]      i_desc_base_hi,
    input  wire [15:0]      i_desc_count,
    input  wire             i_desc_start,
    input  wire             i_desc_abort,
    input  wire             i_desc_irq_en,
    input  wire             i_desc_clear_done,
    input  wire             i_global_idle,

    output reg              o_busy,
    output reg              o_done,
    output reg              o_err,
    output reg              o_aborted,
    output reg [15:0]       o_pc,
    output reg [7:0]        o_err_code,
    output wire             o_irq,

    output reg              m_axi_arvalid,
    input  wire             m_axi_arready,
    output reg [ADDR_W-1:0] m_axi_araddr,
    output reg [7:0]        m_axi_arlen,
    output wire [2:0]       m_axi_arsize,
    output wire [1:0]       m_axi_arburst,
    input  wire             m_axi_rvalid,
    output reg              m_axi_rready,
    input  wire [127:0]     m_axi_rdata,
    input  wire [1:0]       m_axi_rresp,
    input  wire             m_axi_rlast

    ,output reg             o_dma_rd_req
    ,output reg             o_dma_wr_req
    ,output reg [31:0]      o_dma_rd_ddr_addr
    ,output reg [31:0]      o_dma_wr_ddr_addr
    ,output reg [15:0]      o_dma_rd_len
    ,output reg [15:0]      o_dma_wr_len
    ,output reg [SRAM_ADDR_W-1:0] o_dma_rd_sram_base
    ,output reg [SRAM_ADDR_W-1:0] o_dma_wr_sram_base
    ,output reg             o_dma_sram_sel
    ,output reg             o_dma_out_rd_sel
    ,output reg [1:0]       o_dma_rd_sram_sel
    ,output reg             o_dma_act_ping_sel
    ,output reg             o_dma_wgt_ping_sel
    ,output reg             o_dma_out_ping_sel
    ,output reg             o_copy_trig
    ,output reg             o_expand_trig
    ,input  wire            i_dma_rd_done
    ,input  wire            i_dma_wr_done
    ,input  wire            i_copy_done
    ,input  wire            i_expand_done
    ,output reg             o_qparam_we
    ,output reg [5:0]       o_qparam_idx
    ,output reg [31:0]      o_qparam_bias
    ,output reg [31:0]      o_qparam_scale
    ,output reg [5:0]       o_qparam_shift
    ,output reg             o_npu_start
    ,output reg [31:0]      o_ctrl_flags
    ,output reg [15:0]      o_in_w
    ,output reg [15:0]      o_in_h
    ,output reg [15:0]      o_in_c
    ,output reg [15:0]      o_out_c
    ,output reg [7:0]       o_kernel_kh
    ,output reg [7:0]       o_kernel_kw
    ,output reg [7:0]       o_stride_sx
    ,output reg [7:0]       o_stride_sy
    ,output reg [7:0]       o_pad_w
    ,output reg [7:0]       o_pad_h
    ,output reg [SRAM_ADDR_W-1:0] o_act_addr
    ,output reg [SRAM_ADDR_W-1:0] o_wgt_addr
    ,output reg [SRAM_ADDR_W-1:0] o_out_addr
    ,input  wire            i_npu_done
    // YOLO spatial movement engines (Act SRAM <-> Act SRAM). Trigger like the
    // copy/expand engines and wait on their level-type done.
    ,output reg             o_upsample_trig
    ,input  wire            i_upsample_done
    ,output reg             o_maxpool5_trig
    ,input  wire            i_maxpool5_done
    // Exact-SiLU LUT streaming load (OP_LUT_LOAD): one (idx,val) per cycle,
    // mirrors the CPU NPU_SILU_LOAD MMIO write so YOLO convs can carry their
    // own per-layer LUT inside the descriptor program.
    ,output reg             o_silu_load_en
    ,output reg [7:0]       o_silu_load_idx
    ,output reg [7:0]       o_silu_load_val
    // Per-layer activation config (OP_ACTIVATION_CFG): latched, persists for
    // the following CONV2D/GEMM ops until the next ACTIVATION_CFG. Defaults
    // (pad 0, no SiLU, clip 127) keep MNIST byte-identical.
    ,output reg [7:0]       o_act_pad_value
    ,output reg             o_act_silu_exact_en
    ,output reg             o_act_silu_en
    ,output reg             o_act_silu_requant_en
    ,output reg [15:0]      o_act_silu_requant_mul
    ,output reg [5:0]       o_act_silu_requant_shift
    ,output reg [7:0]       o_act_silu_requant_zp
    ,output reg [7:0]       o_act_clip_max
);

    localparam OP_NOP      = 8'h00;
    localparam OP_DMA_DDR_TO_ACT       = 8'h01;
    localparam OP_DMA_ACT_TO_DDR       = 8'h02;
    localparam OP_DMA_OUT_TO_DDR       = 8'h03;
    localparam OP_IMG_EXPAND           = 8'h04;
    localparam OP_SRAM_COPY_OUT_TO_ACT = 8'h05;
    localparam OP_CONV2D               = 8'h06;
    localparam OP_GEMM                 = 8'h07;
    localparam OP_STOP_IRQ = 8'h08;
    localparam OP_UPSAMPLE2X = 8'h20; // 2x nearest-neighbour upsample (Act SRAM)
    localparam OP_MAXPOOL5X5 = 8'h21; // 5x5 stride-1 signed maxpool (Act SRAM)
    localparam OP_LUT_LOAD = 8'h24;   // load 256-entry exact-SiLU LUT from DDR
    localparam OP_ACT_CFG  = 8'h25;   // latch per-layer activation/requant config
    localparam VERSION     = 8'h01;

    localparam ERR_NONE          = 8'd0;
    localparam ERR_BAD_VERSION   = 8'd1;
    localparam ERR_BAD_OPCODE    = 8'd2;
    localparam ERR_BAD_COUNT     = 8'd4;
    localparam ERR_BAD_SHAPE     = 8'd6;
    localparam ERR_BUSY_AT_START = 8'd7;
    localparam ERR_AXI_DESC_READ = 8'd8;
    localparam ERR_AXI_QPARAM_READ = 8'd9;
    localparam ERR_AXI_LUT_READ  = 8'd11;

    localparam S_IDLE     = 4'd0;
    localparam S_FETCH_AR = 4'd1;
    localparam S_FETCH_R  = 4'd2;
    localparam S_DECODE   = 4'd3;
    localparam S_ADVANCE  = 4'd4;
    localparam S_START_OP = 4'd5;
    localparam S_WAIT_OP  = 4'd6;
    localparam S_QPARAM_AR = 4'd7;
    localparam S_QPARAM_R  = 4'd8;
    localparam S_DONE     = 4'd9;
    localparam S_ERROR    = 4'd10;
    localparam S_ABORT    = 4'd11;
    localparam S_NPU_START = 4'd12;
    localparam S_LUT_AR   = 4'd13;
    localparam S_LUT_R    = 4'd14;
    localparam S_LUT_WR   = 4'd15;

    localparam WAIT_NONE   = 3'd0;
    localparam WAIT_DMA_RD = 3'd1;
    localparam WAIT_DMA_WR = 3'd2;
    localparam WAIT_COPY   = 3'd3;
    localparam WAIT_EXPAND = 3'd4;
    localparam WAIT_NPU    = 3'd5;
    localparam WAIT_UPSAMPLE = 3'd6;
    localparam WAIT_MAXPOOL  = 3'd7;

    reg [3:0] state;
    reg [1:0] beat_idx;
    reg [31:0] desc_w [0:15];
    reg [2:0] wait_kind;
    reg       wait_guard;   // skip the first S_WAIT_OP cycle: the trig pulse is
                            // still in flight, so the engine's level-type done
                            // (copy/expand/dma) has not yet cleared its STALE
                            // value from the previous op. Sampling done here
                            // would falsely advance. fsm_done_irq is a pulse so
                            // this guard is harmless for WAIT_NPU.
    reg       qparam_loaded;
    reg [5:0] qparam_idx;
    reg [4:0] lut_beat;        // 0..15: which 16-byte beat of the 256-entry LUT
    reg [3:0] lut_sub;         // 0..15: byte within the current beat
    reg [127:0] lut_buf;       // latched LUT beat being streamed into the NPU
    integer i;

    wire [7:0] desc_op      = desc_w[0][7:0];
    wire [7:0] desc_version = desc_w[0][15:8];
    wire [31:0] desc_addr   = i_desc_base_lo + {14'd0, o_pc, 6'b0};
    wire [6:0] qparam_count = desc_w[12][6:0];
    // LUT lives at desc_w[2], 16 contiguous 128-bit beats (256 bytes).
    wire [31:0] lut_addr    = desc_w[2] + {23'd0, lut_beat, 4'b0};
    wire        qparam_needed = (desc_w[11] != 32'd0) && (qparam_count != 7'd0);
    wire [31:0] qparam_addr = desc_w[11] + {22'd0, qparam_idx, 4'b0};

    assign m_axi_arsize  = 3'd4; // 16 bytes per beat
    assign m_axi_arburst = 2'b01; // INCR
    assign o_irq = o_done & i_desc_irq_en;

    initial begin
        state = S_IDLE;
        o_busy = 1'b0;
        o_done = 1'b0;
        o_err = 1'b0;
        o_aborted = 1'b0;
        o_pc = 16'd0;
        o_err_code = ERR_NONE;
        m_axi_arvalid = 1'b0;
        m_axi_araddr = {ADDR_W{1'b0}};
        m_axi_arlen = 8'd0;
        m_axi_rready = 1'b0;
        beat_idx = 2'd0;
        wait_kind = WAIT_NONE;
        wait_guard = 1'b0;
        o_dma_rd_req = 1'b0;
        o_dma_wr_req = 1'b0;
        o_dma_rd_ddr_addr = 32'd0;
        o_dma_wr_ddr_addr = 32'd0;
        o_dma_rd_len = 16'd0;
        o_dma_wr_len = 16'd0;
        o_dma_rd_sram_base = {SRAM_ADDR_W{1'b0}};
        o_dma_wr_sram_base = {SRAM_ADDR_W{1'b0}};
        o_dma_sram_sel = 1'b0;
        o_dma_out_rd_sel = 1'b0;
        o_dma_rd_sram_sel = 2'd0;
        o_dma_act_ping_sel = 1'b0;
        o_dma_wgt_ping_sel = 1'b0;
        o_dma_out_ping_sel = 1'b0;
        o_copy_trig = 1'b0;
        o_expand_trig = 1'b0;
        o_qparam_we = 1'b0;
        o_qparam_idx = 6'd0;
        o_qparam_bias = 32'd0;
        o_qparam_scale = 32'd0;
        o_qparam_shift = 6'd0;
        o_npu_start = 1'b0;
        o_ctrl_flags = 32'd0;
        o_in_w = 16'd0;
        o_in_h = 16'd0;
        o_in_c = 16'd0;
        o_out_c = 16'd0;
        o_kernel_kh = 8'd0;
        o_kernel_kw = 8'd0;
        o_stride_sx = 8'd0;
        o_stride_sy = 8'd0;
        o_pad_w = 8'd0;
        o_pad_h = 8'd0;
        o_act_addr = {SRAM_ADDR_W{1'b0}};
        o_wgt_addr = {SRAM_ADDR_W{1'b0}};
        o_out_addr = {SRAM_ADDR_W{1'b0}};
        qparam_loaded = 1'b0;
        qparam_idx = 6'd0;
        lut_beat = 5'd0;
        lut_sub = 4'd0;
        lut_buf = 128'd0;
        o_silu_load_en = 1'b0;
        o_silu_load_idx = 8'd0;
        o_silu_load_val = 8'd0;
        o_act_pad_value = 8'd0;
        o_act_silu_exact_en = 1'b0;
        o_act_silu_en = 1'b0;
        o_act_silu_requant_en = 1'b0;
        o_act_silu_requant_mul = 16'd0;
        o_act_silu_requant_shift = 6'd0;
        o_act_silu_requant_zp = 8'd0;
        o_act_clip_max = 8'd127;
        o_upsample_trig = 1'b0;
        o_maxpool5_trig = 1'b0;
    end

    task set_error;
        input [7:0] code;
        begin
            o_err      <= 1'b1;
            o_err_code <= code;
            o_busy     <= 1'b0;
            state      <= S_ERROR;
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            o_busy <= 1'b0;
            o_done <= 1'b0;
            o_err <= 1'b0;
            o_aborted <= 1'b0;
            o_pc <= 16'd0;
            o_err_code <= ERR_NONE;
            m_axi_arvalid <= 1'b0;
            m_axi_araddr <= {ADDR_W{1'b0}};
            m_axi_arlen <= 8'd0;
            m_axi_rready <= 1'b0;
            beat_idx <= 2'd0;
            wait_kind <= WAIT_NONE;
            wait_guard <= 1'b0;
            o_dma_rd_req <= 1'b0;
            o_dma_wr_req <= 1'b0;
            o_dma_rd_ddr_addr <= 32'd0;
            o_dma_wr_ddr_addr <= 32'd0;
            o_dma_rd_len <= 16'd0;
            o_dma_wr_len <= 16'd0;
            o_dma_rd_sram_base <= {SRAM_ADDR_W{1'b0}};
            o_dma_wr_sram_base <= {SRAM_ADDR_W{1'b0}};
            o_dma_sram_sel <= 1'b0;
            o_dma_out_rd_sel <= 1'b0;
            o_dma_rd_sram_sel <= 2'd0;
            o_dma_act_ping_sel <= 1'b0;
            o_dma_wgt_ping_sel <= 1'b0;
            o_dma_out_ping_sel <= 1'b0;
            o_copy_trig <= 1'b0;
            o_expand_trig <= 1'b0;
            o_qparam_we <= 1'b0;
            o_qparam_idx <= 6'd0;
            o_qparam_bias <= 32'd0;
            o_qparam_scale <= 32'd0;
            o_qparam_shift <= 6'd0;
            o_npu_start <= 1'b0;
            o_ctrl_flags <= 32'd0;
            o_in_w <= 16'd0;
            o_in_h <= 16'd0;
            o_in_c <= 16'd0;
            o_out_c <= 16'd0;
            o_kernel_kh <= 8'd0;
            o_kernel_kw <= 8'd0;
            o_stride_sx <= 8'd0;
            o_stride_sy <= 8'd0;
            o_pad_w <= 8'd0;
            o_pad_h <= 8'd0;
            o_act_addr <= {SRAM_ADDR_W{1'b0}};
            o_wgt_addr <= {SRAM_ADDR_W{1'b0}};
            o_out_addr <= {SRAM_ADDR_W{1'b0}};
            qparam_loaded <= 1'b0;
            qparam_idx <= 6'd0;
            lut_beat <= 5'd0;
            lut_sub <= 4'd0;
            lut_buf <= 128'd0;
            o_silu_load_en <= 1'b0;
            o_silu_load_idx <= 8'd0;
            o_silu_load_val <= 8'd0;
            o_act_pad_value <= 8'd0;
            o_act_silu_exact_en <= 1'b0;
            o_act_silu_en <= 1'b0;
            o_act_silu_requant_en <= 1'b0;
            o_act_silu_requant_mul <= 16'd0;
            o_act_silu_requant_shift <= 6'd0;
            o_act_silu_requant_zp <= 8'd0;
            o_act_clip_max <= 8'd127;
            o_upsample_trig <= 1'b0;
            o_maxpool5_trig <= 1'b0;
            for (i = 0; i < 16; i = i + 1)
                desc_w[i] <= 32'd0;
        end else begin
            o_dma_rd_req <= 1'b0;
            o_dma_wr_req <= 1'b0;
            o_copy_trig <= 1'b0;
            o_expand_trig <= 1'b0;
            o_qparam_we <= 1'b0;
            o_silu_load_en <= 1'b0;
            o_upsample_trig <= 1'b0;
            o_maxpool5_trig <= 1'b0;
            o_npu_start <= 1'b0;

            if (i_desc_clear_done) begin
                o_done <= 1'b0;
                if (!o_busy) begin
                    o_err <= 1'b0;
                    o_aborted <= 1'b0;
                    o_err_code <= ERR_NONE;
                end
            end

            if (i_desc_abort && o_busy) begin
                m_axi_arvalid <= 1'b0;
                m_axi_rready <= 1'b0;
                o_busy <= 1'b0;
                o_aborted <= 1'b1;
                state <= S_ABORT;
            end else begin
                case (state)
                S_IDLE: begin
                    m_axi_arvalid <= 1'b0;
                    m_axi_rready <= 1'b0;
                    if (i_desc_start) begin
                        o_done <= 1'b0;
                        o_err <= 1'b0;
                        o_aborted <= 1'b0;
                        o_err_code <= ERR_NONE;
                        o_pc <= 16'd0;
                        if (i_desc_count == 16'd0) begin
                            set_error(ERR_BAD_COUNT);
                        end else if (!i_global_idle) begin
                            set_error(ERR_BUSY_AT_START);
                        end else begin
                            o_busy <= 1'b1;
                            state <= S_FETCH_AR;
                        end
                    end
                end

                S_FETCH_AR: begin
                    m_axi_arvalid <= 1'b1;
                    m_axi_araddr <= desc_addr[ADDR_W-1:0];
                    m_axi_arlen <= 8'd3;
                    if (m_axi_arvalid && m_axi_arready) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready <= 1'b1;
                        beat_idx <= 2'd0;
                        state <= S_FETCH_R;
                    end
                end

                S_FETCH_R: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        if (m_axi_rresp != 2'b00) begin
                            m_axi_rready <= 1'b0;
                            set_error(ERR_AXI_DESC_READ);
                        end else begin
                            desc_w[{beat_idx, 2'b00} + 0] <= m_axi_rdata[31:0];
                            desc_w[{beat_idx, 2'b00} + 1] <= m_axi_rdata[63:32];
                            desc_w[{beat_idx, 2'b00} + 2] <= m_axi_rdata[95:64];
                            desc_w[{beat_idx, 2'b00} + 3] <= m_axi_rdata[127:96];
                            if (m_axi_rlast) begin
                                m_axi_rready <= 1'b0;
                                qparam_loaded <= 1'b0;
                                state <= S_DECODE;
                            end else begin
                                beat_idx <= beat_idx + 2'd1;
                            end
                        end
                    end
                end

                S_DECODE: begin
                    if (desc_version != VERSION) begin
                        set_error(ERR_BAD_VERSION);
                    end else if (qparam_needed && !qparam_loaded) begin
                        qparam_idx <= 6'd0;
                        state <= S_QPARAM_AR;
                    end else if (desc_op == OP_NOP) begin
                        state <= S_ADVANCE;
                    end else if (desc_op == OP_STOP_IRQ) begin
                        o_busy <= 1'b0;
                        o_done <= 1'b1;
                        state <= S_DONE;
                    end else if (desc_op == OP_DMA_DDR_TO_ACT ||
                                 desc_op == OP_DMA_ACT_TO_DDR ||
                                 desc_op == OP_DMA_OUT_TO_DDR ||
                                 desc_op == OP_IMG_EXPAND ||
                                 desc_op == OP_SRAM_COPY_OUT_TO_ACT) begin
                        if (desc_w[7] == 32'd0)
                            set_error(ERR_BAD_SHAPE);
                        else
                            state <= S_START_OP;
                    end else if (desc_op == OP_CONV2D || desc_op == OP_GEMM) begin
                        if (desc_w[8] == 32'd0 || desc_w[9] == 32'd0)
                            set_error(ERR_BAD_SHAPE);
                        else
                            state <= S_START_OP;
                    end else if (desc_op == OP_UPSAMPLE2X ||
                                 desc_op == OP_MAXPOOL5X5) begin
                        if (desc_w[8] == 32'd0 || desc_w[9] == 32'd0)
                            set_error(ERR_BAD_SHAPE);
                        else
                            state <= S_START_OP;
                    end else if (desc_op == OP_LUT_LOAD) begin
                        if (desc_w[2] == 32'd0) begin
                            set_error(ERR_BAD_SHAPE);
                        end else begin
                            lut_beat <= 5'd0;
                            lut_sub  <= 4'd0;
                            state    <= S_LUT_AR;
                        end
                    end else if (desc_op == OP_ACT_CFG) begin
                        // Latch per-layer activation config; persists for the
                        // following conv/gemm ops. w[3] mirrors NPU_SILU_REQUANT_CFG.
                        o_act_pad_value          <= desc_w[2][7:0];
                        o_act_silu_requant_mul   <= desc_w[3][15:0];
                        o_act_silu_requant_shift <= desc_w[3][21:16];
                        o_act_silu_requant_zp    <= desc_w[3][31:24];
                        o_act_silu_exact_en      <= desc_w[4][0];
                        o_act_silu_en            <= desc_w[4][1];
                        o_act_silu_requant_en    <= desc_w[4][2];
                        o_act_clip_max           <= (desc_w[5][7:0] == 8'd0) ? 8'd127
                                                                            : desc_w[5][7:0];
                        state <= S_ADVANCE;
                    end else begin
                        set_error(ERR_BAD_OPCODE);
                    end
                end

                S_START_OP: begin
                    // Arm the wait guard so the first S_WAIT_OP cycle ignores the
                    // STALE level-done from the previous op (engines clear their
                    // done one cycle after they sample the trig/req we issue here).
                    wait_guard <= 1'b1;
                    case (desc_op)
                    OP_DMA_DDR_TO_ACT: begin
                        o_dma_sram_sel <= 1'b0;
                        o_dma_out_rd_sel <= 1'b0;
                        o_dma_rd_sram_sel <= 2'd1;
                        o_dma_act_ping_sel <= desc_w[1][0];
                        o_dma_rd_ddr_addr <= desc_w[2];
                        o_dma_rd_len <= desc_w[7][15:0] - 16'd1;
                        o_dma_rd_sram_base <= desc_w[4][SRAM_ADDR_W-1:0];
                        o_dma_rd_req <= 1'b1;
                        wait_kind <= WAIT_DMA_RD;
                        state <= S_WAIT_OP;
                    end
                    OP_DMA_ACT_TO_DDR: begin
                        o_dma_out_rd_sel <= 1'b0;
                        o_dma_rd_sram_sel <= 2'd1;
                        o_dma_act_ping_sel <= desc_w[1][0];
                        o_dma_wr_ddr_addr <= desc_w[4];
                        o_dma_wr_len <= desc_w[7][15:0] - 16'd1;
                        o_dma_wr_sram_base <= desc_w[2][SRAM_ADDR_W-1:0];
                        o_dma_wr_req <= 1'b1;
                        wait_kind <= WAIT_DMA_WR;
                        state <= S_WAIT_OP;
                    end
                    OP_DMA_OUT_TO_DDR: begin
                        o_dma_out_rd_sel <= 1'b1;
                        o_dma_rd_sram_sel <= 2'd0;
                        o_dma_wr_ddr_addr <= desc_w[4];
                        o_dma_wr_len <= desc_w[7][15:0] - 16'd1;
                        o_dma_wr_sram_base <= desc_w[2][SRAM_ADDR_W-1:0];
                        o_dma_out_ping_sel <= desc_w[1][2];
                        o_dma_wr_req <= 1'b1;
                        wait_kind <= WAIT_DMA_WR;
                        state <= S_WAIT_OP;
                    end
                    OP_IMG_EXPAND: begin
                        o_dma_act_ping_sel <= desc_w[1][0];
                        o_dma_wgt_ping_sel <= desc_w[1][1];
                        o_dma_out_ping_sel <= desc_w[1][2];
                        o_dma_rd_sram_base <= desc_w[2][SRAM_ADDR_W-1:0];
                        o_dma_wr_sram_base <= desc_w[4][SRAM_ADDR_W-1:0];
                        o_dma_rd_len <= desc_w[7][15:0];
                        o_expand_trig <= 1'b1;
                        wait_kind <= WAIT_EXPAND;
                        state <= S_WAIT_OP;
                    end
                    OP_SRAM_COPY_OUT_TO_ACT: begin
                        o_dma_act_ping_sel <= desc_w[1][0];
                        o_dma_wgt_ping_sel <= desc_w[1][1];
                        o_dma_out_ping_sel <= desc_w[1][2];
                        o_dma_rd_sram_base <= desc_w[2][SRAM_ADDR_W-1:0];
                        o_dma_wr_sram_base <= desc_w[4][SRAM_ADDR_W-1:0];
                        o_dma_rd_len <= desc_w[7][15:0];
                        o_copy_trig <= 1'b1;
                        wait_kind <= WAIT_COPY;
                        state <= S_WAIT_OP;
                    end
                    OP_CONV2D, OP_GEMM: begin
                        o_ctrl_flags <= {desc_w[1][15:0], desc_w[0][31:16]} |
                                        (desc_op == OP_GEMM ? 32'h0000_0080 : 32'd0);
                        o_act_addr <= desc_w[2][SRAM_ADDR_W-1:0];
                        o_wgt_addr <= desc_w[3][SRAM_ADDR_W-1:0];
                        o_out_addr <= desc_w[4][SRAM_ADDR_W-1:0];
                        o_in_w <= desc_w[8][15:0];
                        o_in_h <= desc_w[8][31:16];
                        o_in_c <= desc_w[9][15:0];
                        o_out_c <= desc_w[9][31:16];
                        o_kernel_kw <= desc_w[10][7:0];
                        o_kernel_kh <= desc_w[10][15:8];
                        o_stride_sx <= desc_w[10][23:16];
                        o_stride_sy <= desc_w[10][23:16];
                        o_pad_w <= desc_w[10][31:24];
                        o_pad_h <= desc_w[10][31:24];
                        state <= S_NPU_START;
                    end
                    OP_UPSAMPLE2X: begin
                        // src/dst reuse the DMA SRAM-base regs; dims reuse the
                        // conv in_w/in_h/in_c regs (engine derives ic_groups).
                        o_dma_act_ping_sel <= desc_w[1][0];
                        o_dma_rd_sram_base <= desc_w[2][SRAM_ADDR_W-1:0];
                        o_dma_wr_sram_base <= desc_w[4][SRAM_ADDR_W-1:0];
                        o_in_w <= desc_w[8][15:0];
                        o_in_h <= desc_w[8][31:16];
                        o_in_c <= desc_w[9][15:0];
                        o_upsample_trig <= 1'b1;
                        wait_kind <= WAIT_UPSAMPLE;
                        state <= S_WAIT_OP;
                    end
                    OP_MAXPOOL5X5: begin
                        o_dma_act_ping_sel <= desc_w[1][0];
                        o_dma_rd_sram_base <= desc_w[2][SRAM_ADDR_W-1:0];
                        o_dma_wr_sram_base <= desc_w[4][SRAM_ADDR_W-1:0];
                        o_in_w <= desc_w[8][15:0];
                        o_in_h <= desc_w[8][31:16];
                        o_in_c <= desc_w[9][15:0];
                        o_maxpool5_trig <= 1'b1;
                        wait_kind <= WAIT_MAXPOOL;
                        state <= S_WAIT_OP;
                    end
                    default: set_error(ERR_BAD_OPCODE);
                    endcase
                end

                S_NPU_START: begin
                    o_npu_start <= 1'b1;
                    wait_kind <= WAIT_NPU;
                    state <= S_WAIT_OP;
                end

                S_QPARAM_AR: begin
                    m_axi_arvalid <= 1'b1;
                    m_axi_araddr <= qparam_addr[ADDR_W-1:0];
                    m_axi_arlen <= 8'd0;
                    if (m_axi_arvalid && m_axi_arready) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready <= 1'b1;
                        state <= S_QPARAM_R;
                    end
                end

                S_QPARAM_R: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        if (m_axi_rresp != 2'b00) begin
                            m_axi_rready <= 1'b0;
                            set_error(ERR_AXI_QPARAM_READ);
                        end else begin
                            o_qparam_we <= 1'b1;
                            o_qparam_idx <= qparam_idx;
                            o_qparam_bias <= m_axi_rdata[31:0];
                            o_qparam_scale <= m_axi_rdata[63:32];
                            o_qparam_shift <= m_axi_rdata[101:96];
                            m_axi_rready <= 1'b0;
                            if ({1'b0, qparam_idx} + 7'd1 >= qparam_count) begin
                                qparam_loaded <= 1'b1;
                                state <= S_DECODE;
                            end else begin
                                qparam_idx <= qparam_idx + 6'd1;
                                state <= S_QPARAM_AR;
                            end
                        end
                    end
                end

                // ---- OP_LUT_LOAD: stream a 256-entry exact-SiLU LUT from DDR ----
                // 16 single-beat reads (16 bytes each); each beat is then pushed
                // into the NPU one entry/cycle via the silu_load handshake.
                S_LUT_AR: begin
                    m_axi_arvalid <= 1'b1;
                    m_axi_araddr <= lut_addr[ADDR_W-1:0];
                    m_axi_arlen <= 8'd0;
                    if (m_axi_arvalid && m_axi_arready) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready <= 1'b1;
                        state <= S_LUT_R;
                    end
                end

                S_LUT_R: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        m_axi_rready <= 1'b0;
                        if (m_axi_rresp != 2'b00) begin
                            set_error(ERR_AXI_LUT_READ);
                        end else begin
                            lut_buf <= m_axi_rdata;
                            lut_sub <= 4'd0;
                            state   <= S_LUT_WR;
                        end
                    end
                end

                S_LUT_WR: begin
                    o_silu_load_en  <= 1'b1;
                    o_silu_load_idx <= {lut_beat[3:0], lut_sub};
                    o_silu_load_val <= lut_buf[lut_sub*8 +: 8];
                    if (lut_sub == 4'd15) begin
                        if (lut_beat == 5'd15) begin
                            state <= S_ADVANCE;
                        end else begin
                            lut_beat <= lut_beat + 5'd1;
                            state    <= S_LUT_AR;
                        end
                    end else begin
                        lut_sub <= lut_sub + 4'd1;
                    end
                end

                S_WAIT_OP: begin
                    if (wait_guard) begin
                        // First cycle after issuing the trig/req: the op's own
                        // done has not cleared its stale value yet. Skip it.
                        wait_guard <= 1'b0;
                    end else if ((wait_kind == WAIT_DMA_RD && i_dma_rd_done) ||
                        (wait_kind == WAIT_DMA_WR && i_dma_wr_done) ||
                        (wait_kind == WAIT_COPY   && i_copy_done) ||
                        (wait_kind == WAIT_EXPAND && i_expand_done) ||
                        (wait_kind == WAIT_UPSAMPLE && i_upsample_done) ||
                        (wait_kind == WAIT_MAXPOOL && i_maxpool5_done) ||
                        (wait_kind == WAIT_NPU    && i_npu_done)) begin
                        wait_kind <= WAIT_NONE;
                        state <= S_ADVANCE;
                    end
                end

                S_ADVANCE: begin
                    if (o_pc + 16'd1 >= i_desc_count) begin
                        o_busy <= 1'b0;
                        o_done <= 1'b1;
                        state <= S_DONE;
                    end else begin
                        o_pc <= o_pc + 16'd1;
                        state <= S_FETCH_AR;
                    end
                end

                S_DONE: begin
                    if (i_desc_start) begin
                        o_done <= 1'b0;
                        o_err <= 1'b0;
                        o_aborted <= 1'b0;
                        o_err_code <= ERR_NONE;
                        o_pc <= 16'd0;
                        if (i_desc_count == 16'd0) begin
                            set_error(ERR_BAD_COUNT);
                        end else if (!i_global_idle) begin
                            set_error(ERR_BUSY_AT_START);
                        end else begin
                            o_busy <= 1'b1;
                            state <= S_FETCH_AR;
                        end
                    end
                end

                S_ERROR: begin
                    if (i_desc_start) begin
                        o_done <= 1'b0;
                        o_err <= 1'b0;
                        o_aborted <= 1'b0;
                        o_err_code <= ERR_NONE;
                        o_pc <= 16'd0;
                        if (i_desc_count == 16'd0) begin
                            set_error(ERR_BAD_COUNT);
                        end else if (!i_global_idle) begin
                            set_error(ERR_BUSY_AT_START);
                        end else begin
                            o_busy <= 1'b1;
                            state <= S_FETCH_AR;
                        end
                    end
                end

                S_ABORT: begin
                    if (i_desc_start) begin
                        o_done <= 1'b0;
                        o_err <= 1'b0;
                        o_aborted <= 1'b0;
                        o_err_code <= ERR_NONE;
                        o_pc <= 16'd0;
                        if (i_desc_count == 16'd0) begin
                            set_error(ERR_BAD_COUNT);
                        end else if (!i_global_idle) begin
                            set_error(ERR_BUSY_AT_START);
                        end else begin
                            o_busy <= 1'b1;
                            state <= S_FETCH_AR;
                        end
                    end
                end

                default: state <= S_IDLE;
                endcase
            end
        end
    end

endmodule
