// Filename: param_regfile.v
// -------------------------------------------------------------------
// Parameter Register File — AXI4-Lite accessible control/status and
// dimension registers for the NPU.  16 output channels × bias + scale
// parameters are included as register arrays.
//
// Address map (byte-offset, 32-bit word-aligned):
//   0x00  CTRL           [0]start [1]ping_pong [2]pool_en [3]eltwise_en [4]clear_done [5]relu_en [6]out_ping [7]gemm_en
//   0x04  STATUS         [0]done_irq [1]busy [2]dma_rd_err [3]dma_wr_err
//   0x08  ACT_ADDR_PING  Act SRAM ping buffer base address (in SRAM word addr)
//   0x0C  ACT_ADDR_PONG
//   0x10  WGT_ADDR_PING
//   0x14  WGT_ADDR_PONG
//   0x18  OUT_ADDR_PING
//   0x1C  OUT_ADDR_PONG
//   0x20  DIM_IN_W       Input feature map width
//   0x24  DIM_IN_H       Input feature map height
//   0x28  DIM_IN_C       Input channels
//   0x2C  DIM_OUT_C      Output channels
//   0x30  KERNEL         [15:8]KH [7:0]KW
//   0x34  STRIDE         [15:8]SX [7:0]SY
//   0x38  BIAS_ADDR      Bias data base in DDR (or SRAM offset)
//   0x3C  SCALE_ADDR     Scale/q-param base address
//   0x40  BIAS_VAL_00    Bias value OC[0]  (if immediate, else addr-based)
//   ...   ...            16 bias vals at 0x40..0x7C
//   0x80  SCALE_MUL_00   Scale multiplier OC[0]
//   ...   ...            16 scale muls at 0x80..0xBC
//   0xC0  SCALE_SHFT_00  Scale shift OC[0]
//   ...   ...            16 scale shifts at 0xC0..0xFC
//   0x100 ACT_IC_TILE    Current IC tile base (FSM updates)
//   0x104 ACT_OC_TILE    Current OC tile base (FSM updates)
//   0x108 OUT_X_BASE     Current output X base
//   0x10C OUT_Y_BASE     Current output Y base
//   0x110 TOTAL_OPS_H    Output spatial height
//   0x114 TOTAL_OPS_W    Output spatial width
// -------------------------------------------------------------------

module param_regfile #(
    parameter NUM_OC        = 16,
    parameter MAX_OC_RESIDENT = 64,   // decision O: all-OC-resident bias/scale/shift capacity
    parameter ADDR_W        = 10,
    parameter DATA_W        = 32,
    parameter PSUM_WIDTH    = 32,
    parameter ACT_WIDTH     = 8,
    parameter SRAM_ADDR_W   = 14
) (
    input  wire                         clk,
    input  wire                         rst_n,

    // === AXI4-Lite Write Address Channel ===
    input  wire                         s_axi_awvalid,
    output wire                         s_axi_awready,
    input  wire [ADDR_W-1:0]            s_axi_awaddr,

    // === AXI4-Lite Write Data Channel ===
    input  wire                         s_axi_wvalid,
    output wire                         s_axi_wready,
    input  wire [DATA_W-1:0]            s_axi_wdata,
    input  wire [3:0]                   s_axi_wstrb,

    // === AXI4-Lite Write Response Channel ===
    output wire                         s_axi_bvalid,
    input  wire                         s_axi_bready,
    output wire [1:0]                   s_axi_bresp,

    // === AXI4-Lite Read Address Channel ===
    input  wire                         s_axi_arvalid,
    output wire                         s_axi_arready,
    input  wire [ADDR_W-1:0]            s_axi_araddr,

    // === AXI4-Lite Read Data Channel ===
    output wire                         s_axi_rvalid,
    input  wire                         s_axi_rready,
    output wire [DATA_W-1:0]            s_axi_rdata,
    output wire [1:0]                   s_axi_rresp,

    // === NPU internal read ports (combinational or registered) ===
    // Control
    output wire                         o_start,
    output wire                         o_ping_pong_sel,
    output wire                         o_pool_en,
    output wire                         o_eltwise_en,
    output wire                         o_relu_en,
    output wire                         o_out_ping_sel,  // CTRL[6]: NPU write bank for Out SRAM (independent of global ping_pong)
    output wire                         o_gemm_en,       // CTRL[7]: GEMM/FC mode (bypass im2col)
    output wire                         o_hw_pad,        // CTRL[8]: hardware padding (FSM border zero-inject)
    output wire                         o_row_par_en,    // CTRL[9]: 16-row spatial parallelism (task E)
    output wire                         o_gemm_reduce,   // CTRL[10]: GEMM 16-row IC-reduction (decision M)
    output wire                         o_row_block_en,  // CTRL[11]: row-block packing for narrow layers (#4)
    output wire                         o_oc_single,     // CTRL[12]: all-OC-tiles in one start (decision O / Level 2)
    output wire                         o_int32_out,     // CTRL[13]: raw INT32 output (decision Q, final FC)
    output wire                         o_pool_avg,      // CTRL[16]: 2x2 average pooling (vs max)
    output wire                         o_gpool_en,      // CTRL[17]: global average pooling
    output wire [31:0]                  o_gavg_mul,      // NPU_GAVG_CFG[25:0]: reciprocal multiplier (1/N)
    output wire [5:0]                   o_gavg_shift,    // NPU_GAVG_CFG[31:26]: reciprocal shift
    output wire [7:0]                   o_pad_w,         // NPU_PAD[7:0]: zero-pad columns each side
    output wire [7:0]                   o_pad_h,         // NPU_PAD[15:8]: zero-pad rows each side
    output wire [7:0]                   o_clip_max,      // NPU_CLIP_MAX (0x118): post-process upper clamp (default 127 = ReLU; ReLU6 = q(6.0))
    output wire [SRAM_ADDR_W-1:0]       o_skip_base,     // NPU_SKIP_BASE (0x11C): residual skip source Out-SRAM base (0 = same-addr legacy)

    // Status
    input  wire                         i_done_irq,
    input  wire                         i_busy,
    input  wire                         i_dma_rd_err,
    input  wire                         i_dma_wr_err,
    output wire                         o_clear_done,   // write-1-to-clear

    // SRAM base addresses
    output wire [SRAM_ADDR_W-1:0]       o_act_addr_ping,
    output wire [SRAM_ADDR_W-1:0]       o_act_addr_pong,
    output wire [SRAM_ADDR_W-1:0]       o_wgt_addr_ping,
    output wire [SRAM_ADDR_W-1:0]       o_wgt_addr_pong,
    output wire [SRAM_ADDR_W-1:0]       o_out_addr_ping,
    output wire [SRAM_ADDR_W-1:0]       o_out_addr_pong,

    // Dimensions
    output wire [15:0]                  o_dim_in_w,
    output wire [15:0]                  o_dim_in_h,
    output wire [15:0]                  o_dim_in_c,
    output wire [15:0]                  o_dim_out_c,
    output wire [7:0]                   o_kernel_kh,
    output wire [7:0]                   o_kernel_kw,
    output wire [7:0]                   o_stride_sx,
    output wire [7:0]                   o_stride_sy,

    // Bias / Scale base addresses
    output wire [31:0]                  o_bias_addr,
    output wire [31:0]                  o_scale_addr,

    // Active OC-tile selector (decision O): selects which 16-OC window of the
    // resident 64-entry bias/scale/shift is presented. 0 ⇒ legacy low-16 window.
    input  wire [2:0]                        i_oc_tile_sel,

    // Per-OC bias / scale (immediate values, the active OC-tile's 16 channels)
    output wire [NUM_OC-1:0][PSUM_WIDTH-1:0] o_bias_val,
    output wire [NUM_OC-1:0][31:0]           o_scale_mul,
    output wire [NUM_OC-1:0][5:0]            o_scale_shift,

    // Tile tracking (written by FSM, readable via AXI)
    output wire [9:0]                   o_act_ic_tile,
    output wire [9:0]                   o_act_oc_tile,
    output wire [15:0]                  o_out_x_base,
    output wire [15:0]                  o_out_y_base,

    // Derived output dimensions
    output wire [15:0]                  o_total_ops_h,
    output wire [15:0]                  o_total_ops_w,

    // DMA control outputs
    output wire                         o_dma_rd_req,
    output wire [31:0]                  o_dma_rd_ddr_addr,
    output wire [15:0]                  o_dma_rd_len,
    output wire [SRAM_ADDR_W-1:0]       o_dma_rd_sram_base,
    output wire                         o_dma_wr_req,
    output wire [31:0]                  o_dma_wr_ddr_addr,
    output wire [15:0]                  o_dma_wr_len,
    output wire [SRAM_ADDR_W-1:0]       o_dma_wr_sram_base,

    // DMA status inputs
    input  wire                         i_dma_rd_done,
    input  wire                         i_dma_wr_done,

    // DMA SRAM select: 0=Act SRAM, 1=Wgt SRAM
    output wire                         o_dma_sram_sel,

    // DMA read path control
    output wire                         o_dma_out_rd_sel,   // 0=skip path, 1=DMA owns Out SRAM Port B
    output wire [1:0]                   o_dma_rd_sram_sel,  // 0=Out SRAM, 1=Act SRAM, 2=Wgt SRAM

    // DMA ping/pong buffer select for SRAM writes
    output wire                         o_dma_act_ping_sel, // 0=Ping, 1=Pong — Act SRAM DMA write target
    output wire                         o_dma_wgt_ping_sel, // 0=Ping, 1=Pong — Wgt SRAM DMA write target
    output wire                         o_dma_out_ping_sel, // 0=Ping, 1=Pong — Out SRAM DMA read source (decoupled from NPU write bank)
    output wire                         o_copy_trig,        // 0x154 write: pulse to start on-chip Out->Act copy
    input  wire                         i_copy_done,        // copy engine: completion (level), exposed in STATUS[2]
    output wire                         o_expand_trig,      // 0x158 write: pulse to start img_expand
    input  wire                         i_expand_done,      // img_expand: completion (level), exposed in STATUS[3]

    // Performance counter event strobes (asserted for 1 cycle per event)
    input  wire                         i_perf_busy,        // NPU FSM busy
    input  wire                         i_perf_arr_active,  // systolic array accumulating (MAC cycle)
    input  wire                         i_perf_rd_beat,     // AXI read data beat
    input  wire                         i_perf_wr_beat,     // AXI write data beat
    input  wire                         i_perf_rd_busy,     // AXI read burst outstanding
    input  wire                         i_perf_wr_busy      // AXI read/write burst outstanding
);

    // -------------------------------------------------------------------
    // Register definitions
    // -------------------------------------------------------------------
    // Control register (0x00)
    reg        ctrl_start;
    reg        ctrl_ping_pong;
    reg        ctrl_pool_en;
    reg        ctrl_eltwise_en;
    reg        ctrl_clear_done;
    reg        ctrl_relu_en;
    reg        ctrl_out_ping;   // CTRL[6]: NPU write bank for Out SRAM
    reg        ctrl_gemm_en;    // CTRL[7]: GEMM/FC mode (bypass im2col)
    reg        ctrl_hw_pad;     // CTRL[8]: hardware padding
    reg        ctrl_row_par;   // CTRL[9]: 16-row spatial parallelism
    reg        ctrl_gemm_reduce; // CTRL[10]: GEMM 16-row IC-reduction
    reg        ctrl_row_block;   // CTRL[11]: row-block packing (#4)
    reg        ctrl_oc_single;   // CTRL[12]: all-OC-tiles in one start (decision O)
    reg        ctrl_int32_out;   // CTRL[13]: raw INT32 output (decision Q)
    reg        ctrl_pool_avg;    // CTRL[16]: 2x2 average pooling (vs max)
    reg        ctrl_gpool_en;    // CTRL[17]: global average pooling
    reg [25:0] gavg_mul;         // NPU_GAVG_CFG: reciprocal multiplier (1/N)
    reg [5:0]  gavg_shift;       // NPU_GAVG_CFG: reciprocal shift
    reg [15:0] pad_cfg;         // NPU_PAD: {pad_h[15:8], pad_w[7:0]}
    reg [7:0]  clip_max;        // NPU_CLIP_MAX: post-process upper clamp value
    reg [SRAM_ADDR_W-1:0] skip_base;  // NPU_SKIP_BASE: residual skip source base

    // Status
    wire       status_done;
    wire       status_busy;

    reg        done_irq_latched;   // latch done_irq until CTRL write

    // SRAM addresses
    reg [SRAM_ADDR_W-1:0] act_addr_ping;
    reg [SRAM_ADDR_W-1:0] act_addr_pong;
    reg [SRAM_ADDR_W-1:0] wgt_addr_ping;
    reg [SRAM_ADDR_W-1:0] wgt_addr_pong;
    reg [SRAM_ADDR_W-1:0] out_addr_ping;
    reg [SRAM_ADDR_W-1:0] out_addr_pong;

    // Dimensions
    reg [15:0] dim_in_w;
    reg [15:0] dim_in_h;
    reg [15:0] dim_in_c;
    reg [15:0] dim_out_c;
    reg [7:0]  kernel_kh;
    reg [7:0]  kernel_kw;
    reg [7:0]  stride_sx;
    reg [7:0]  stride_sy;

    // Bias/scale addresses
    reg [31:0] bias_addr;
    reg [31:0] scale_addr;

    // Per-OC parameters — sized to MAX_OC_RESIDENT (decision O); FSM/output uses
    // a NUM_OC-wide window (active OC-tile selected in oc_single mode).
    reg [MAX_OC_RESIDENT-1:0][PSUM_WIDTH-1:0] bias_val;
    reg [MAX_OC_RESIDENT-1:0][31:0]           scale_mul;
    reg [MAX_OC_RESIDENT-1:0][5:0]            scale_shift;

    // Tile tracking
    reg [9:0]  act_ic_tile;
    reg [9:0]  act_oc_tile;
    reg [15:0] out_x_base;
    reg [15:0] out_y_base;

    // Derived
    reg [15:0] total_ops_h;
    reg [15:0] total_ops_w;

    // DMA control registers
    reg [31:0] dma_rd_ddr_addr;
    reg [15:0] dma_rd_len;
    reg [SRAM_ADDR_W-1:0] dma_rd_sram_base;
    reg [31:0] dma_wr_ddr_addr;
    reg [15:0] dma_wr_len;
    reg [SRAM_ADDR_W-1:0] dma_wr_sram_base;
    reg        dma_rd_req_d;   // 1-cycle delayed pulse for DMA read request
    reg        dma_wr_req_d;   // 1-cycle delayed pulse for DMA write request
    reg        copy_trig_d;    // 1-cycle delayed pulse for on-chip Out->Act copy trigger
    reg        expand_trig_d;  // 1-cycle delayed pulse for img_expand trigger
    reg        dma_sram_sel;   // 0=Act SRAM, 1=Wgt SRAM for DMA write target
    reg        dma_out_rd_sel; // 0=skip path owns Out SRAM Port B, 1=DMA owns it
    reg [1:0]  dma_rd_sram_sel; // 0=Out SRAM, 1=Act SRAM, 2=Wgt SRAM for DMA read source
    reg        dma_act_ping_sel; // 0=Ping, 1=Pong — Act SRAM DMA write target
    reg        dma_wgt_ping_sel; // 0=Ping, 1=Pong — Wgt SRAM DMA write target
    reg        dma_out_ping_sel; // 0=Ping, 1=Pong — Out SRAM DMA read source (independent of NPU write bank)

    // -------------------------------------------------------------------
    // Write channel: decode address and update register
    // -------------------------------------------------------------------
    wire [ADDR_W-1:0] word_addr;
    assign word_addr = s_axi_awaddr[ADDR_W-1:0];

    wire wr_en;
    assign wr_en = s_axi_awvalid && s_axi_wvalid && s_axi_awready && s_axi_wready;

    // Latch done_irq: set on i_done_irq pulse, cleared on any CTRL write
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            done_irq_latched <= 1'b0;
        else if (i_done_irq)
            done_irq_latched <= 1'b1;
        else if (wr_en && (word_addr == 5'h00))
            done_irq_latched <= 1'b0;  // clear on CTRL write
    end

    // AXI write handshake
    assign s_axi_awready = 1'b1;
    assign s_axi_wready  = 1'b1;

    reg bvalid;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            bvalid <= 1'b0;
        else if (wr_en)
            bvalid <= 1'b1;
        else if (s_axi_bready)
            bvalid <= 1'b0;
    end
    assign s_axi_bvalid = bvalid;
    assign s_axi_bresp  = 2'b00;

    // Register write logic
    integer wi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_start      <= 1'b0;
            ctrl_ping_pong  <= 1'b0;
            ctrl_pool_en    <= 1'b0;
            ctrl_eltwise_en <= 1'b0;
            ctrl_clear_done <= 1'b0;
            ctrl_relu_en    <= 1'b1;    // ReLU enabled by default
            ctrl_out_ping   <= 1'b0;    // NPU writes Out SRAM Ping bank by default
            ctrl_gemm_en    <= 1'b0;    // GEMM/FC mode off by default (conv path)
            ctrl_hw_pad     <= 1'b0;    // hardware padding off by default
            ctrl_row_par    <= 1'b0;    // row-parallel off by default
            ctrl_gemm_reduce <= 1'b0;   // GEMM-reduce off by default
            ctrl_row_block   <= 1'b0;   // row-block packing off by default
            ctrl_oc_single   <= 1'b0;   // oc-single off by default
            ctrl_int32_out   <= 1'b0;   // int32 raw output off by default
            ctrl_pool_avg    <= 1'b0;   // average pooling off by default (max)
            ctrl_gpool_en    <= 1'b0;   // global average pooling off by default
            gavg_mul         <= 26'd0;
            gavg_shift       <= 6'd0;
            pad_cfg         <= 16'd0;
            clip_max        <= 8'd127;   // default = legacy ReLU clamp [0,127]
            skip_base       <= {SRAM_ADDR_W{1'b0}};  // default 0 = same-addr legacy residual
            act_addr_ping   <= {SRAM_ADDR_W{1'b0}};
            act_addr_pong   <= {SRAM_ADDR_W{1'b0}};
            wgt_addr_ping   <= {SRAM_ADDR_W{1'b0}};
            wgt_addr_pong   <= {SRAM_ADDR_W{1'b0}};
            out_addr_ping   <= {SRAM_ADDR_W{1'b0}};
            out_addr_pong   <= {SRAM_ADDR_W{1'b0}};
            dim_in_w        <= 16'd0;
            dim_in_h        <= 16'd0;
            dim_in_c        <= 16'd0;
            dim_out_c       <= 16'd0;
            kernel_kh       <= 8'd3;
            kernel_kw       <= 8'd3;
            stride_sx       <= 8'd1;
            stride_sy       <= 8'd1;
            bias_addr       <= 32'd0;
            scale_addr      <= 32'd0;
            act_ic_tile     <= 10'd0;
            act_oc_tile     <= 10'd0;
            out_x_base      <= 16'd0;
            out_y_base      <= 16'd0;
            total_ops_h     <= 16'd0;
            total_ops_w     <= 16'd0;
            dma_rd_ddr_addr  <= 32'd0;
            dma_rd_len       <= 8'd0;
            dma_rd_sram_base <= {SRAM_ADDR_W{1'b0}};
            dma_wr_ddr_addr  <= 32'd0;
            dma_wr_len       <= 8'd0;
            dma_wr_sram_base <= {SRAM_ADDR_W{1'b0}};
            dma_rd_req_d     <= 1'b0;
            dma_wr_req_d     <= 1'b0;
            copy_trig_d      <= 1'b0;
            expand_trig_d    <= 1'b0;
            dma_sram_sel     <= 1'b0;
            dma_out_rd_sel   <= 1'b0;
            dma_rd_sram_sel  <= 2'd0;
            dma_act_ping_sel <= 1'b0;  // Default: DMA writes to Ping (same as NPU default)
            dma_wgt_ping_sel <= 1'b0;
            dma_out_ping_sel <= 1'b0;  // Default: DMA reads Out SRAM Ping bank
            for (wi = 0; wi < MAX_OC_RESIDENT; wi = wi + 1) begin
                bias_val[wi]    <= 32'd0;
                scale_mul[wi]   <= 32'd0;
                scale_shift[wi] <= 6'd0;
            end
        end else begin
            // start bit auto-clears after one cycle
            ctrl_start      <= 1'b0;
            ctrl_clear_done <= 1'b0;

            // DMA request pulse: auto-clear after 1 cycle
            dma_rd_req_d <= 1'b0;
            dma_wr_req_d <= 1'b0;
            copy_trig_d  <= 1'b0;
            expand_trig_d <= 1'b0;

            if (wr_en) begin
                // synthesis translate_off
                `ifdef DEBUG
                $display("REG_WR: addr=0x%03h data=0x%08h", word_addr, s_axi_wdata);
                `endif
                // synthesis translate_on
                case (word_addr)
                    // CTRL
                    10'h00: begin
                        ctrl_start      <= s_axi_wdata[0];
                        ctrl_ping_pong  <= s_axi_wdata[1];
                        ctrl_pool_en    <= s_axi_wdata[2];
                        ctrl_eltwise_en <= s_axi_wdata[3];
                        ctrl_clear_done <= s_axi_wdata[4];
                        ctrl_relu_en    <= s_axi_wdata[5];
                        ctrl_out_ping   <= s_axi_wdata[6];
                        ctrl_gemm_en    <= s_axi_wdata[7];
                        ctrl_hw_pad     <= s_axi_wdata[8];
                        ctrl_row_par    <= s_axi_wdata[9];
                        ctrl_gemm_reduce <= s_axi_wdata[10];
                        ctrl_row_block   <= s_axi_wdata[11];
                        ctrl_oc_single   <= s_axi_wdata[12];
                        ctrl_int32_out   <= s_axi_wdata[13];
                        ctrl_pool_avg    <= s_axi_wdata[16];
                        ctrl_gpool_en    <= s_axi_wdata[17];
                    end
                    // STATUS is read-only (write ignored)
                    // 10'h04: (no action)

                    // SRAM base addresses
                    10'h08: act_addr_ping <= s_axi_wdata[SRAM_ADDR_W-1:0];
                    10'h0C: act_addr_pong <= s_axi_wdata[SRAM_ADDR_W-1:0];
                    10'h10: wgt_addr_ping <= s_axi_wdata[SRAM_ADDR_W-1:0];
                    10'h14: wgt_addr_pong <= s_axi_wdata[SRAM_ADDR_W-1:0];
                    10'h18: out_addr_ping <= s_axi_wdata[SRAM_ADDR_W-1:0];
                    10'h1C: out_addr_pong <= s_axi_wdata[SRAM_ADDR_W-1:0];

                    // Dimensions
                    10'h20: dim_in_w  <= s_axi_wdata[15:0];
                    10'h24: dim_in_h  <= s_axi_wdata[15:0];
                    10'h28: dim_in_c  <= s_axi_wdata[15:0];
                    10'h2C: dim_out_c <= s_axi_wdata[15:0];

                    // Kernel & Stride
                    10'h30: begin
                        kernel_kh <= s_axi_wdata[15:8];
                        kernel_kw <= s_axi_wdata[7:0];
                    end
                    10'h34: begin
                        stride_sx <= s_axi_wdata[15:8];
                        stride_sy <= s_axi_wdata[7:0];
                    end

                    // Bias / Scale base addresses
                    10'h38: bias_addr  <= s_axi_wdata;
                    10'h3C: scale_addr <= s_axi_wdata;

                    // Per-OC bias values (0x40 .. 0x7C)
                    10'h40, 10'h44, 10'h48, 10'h4C,
                    10'h50, 10'h54, 10'h58, 10'h5C,
                    10'h60, 10'h64, 10'h68, 10'h6C,
                    10'h70, 10'h74, 10'h78, 10'h7C:
                        bias_val[word_addr[6:2] - 5'd16] <= s_axi_wdata;

                    // Per-OC scale multipliers (0x80 .. 0xBC)
                    10'h80, 10'h84, 10'h88, 10'h8C,
                    10'h90, 10'h94, 10'h98, 10'h9C,
                    10'hA0, 10'hA4, 10'hA8, 10'hAC,
                    10'hB0, 10'hB4, 10'hB8, 10'hBC:
                        scale_mul[word_addr[6:2] - 5'd32] <= s_axi_wdata;

                    // Per-OC scale shifts (0xC0 .. 0xFC)
                    10'hC0, 10'hC4, 10'hC8, 10'hCC,
                    10'hD0, 10'hD4, 10'hD8, 10'hDC,
                    10'hE0, 10'hE4, 10'hE8, 10'hEC,
                    10'hF0, 10'hF4, 10'hF8, 10'hFC:
                        scale_shift[word_addr[6:2] - 5'd48] <= s_axi_wdata[5:0];

                    // Tile tracking
                    10'h100: act_ic_tile <= s_axi_wdata[9:0];
                    10'h104: act_oc_tile <= s_axi_wdata[9:0];
                    10'h108: out_x_base  <= s_axi_wdata[15:0];
                    10'h10C: out_y_base  <= s_axi_wdata[15:0];
                    10'h110: total_ops_h <= s_axi_wdata[15:0];
                    10'h114: total_ops_w <= s_axi_wdata[15:0];
                    10'h118: clip_max    <= s_axi_wdata[7:0];   // NPU_CLIP_MAX
                    10'h11C: skip_base   <= s_axi_wdata[SRAM_ADDR_W-1:0];  // NPU_SKIP_BASE

                    // DMA control registers (0x120-0x13C)
                    10'h120: dma_rd_req_d     <= 1'b1;  // Write triggers DMA read pulse
                    10'h124: dma_rd_ddr_addr   <= s_axi_wdata;
                    10'h128: dma_rd_len        <= s_axi_wdata[15:0];
                    10'h12C: dma_rd_sram_base  <= s_axi_wdata[SRAM_ADDR_W-1:0];
                    10'h130: dma_wr_req_d      <= 1'b1;  // Write triggers DMA write pulse
                    10'h134: dma_wr_ddr_addr   <= s_axi_wdata;
                    10'h138: dma_wr_len        <= s_axi_wdata[15:0];
                    10'h13C: dma_wr_sram_base  <= s_axi_wdata[SRAM_ADDR_W-1:0];
                    10'h144: dma_sram_sel       <= s_axi_wdata[0];
                    10'h148: begin
                        dma_out_rd_sel  <= s_axi_wdata[0];
                        dma_rd_sram_sel <= s_axi_wdata[2:1];
                    end
                    10'h14C: begin
                        dma_act_ping_sel <= s_axi_wdata[0];
                        dma_wgt_ping_sel <= s_axi_wdata[1];
                        dma_out_ping_sel <= s_axi_wdata[2];
                    end
                    10'h150: pad_cfg <= s_axi_wdata[15:0];  // {pad_h, pad_w}
                    10'h154: copy_trig_d <= 1'b1;            // trigger on-chip Out->Act copy
                    10'h158: expand_trig_d <= 1'b1;          // trigger img_expand
                    10'h15C: begin                            // NPU_GAVG_CFG
                        gavg_mul   <= s_axi_wdata[25:0];
                        gavg_shift <= s_axi_wdata[31:26];
                    end

                    default: ; // Ignore unmapped addresses
                endcase

                // ---- Decision O: extended per-OC params, indices 16..63 ----
                // bias[16..63]        @ 0x160..0x21C
                // scale_mul[16..63]   @ 0x220..0x2DC
                // scale_shift[16..63] @ 0x2E0..0x39C
                if (word_addr >= 10'h160 && word_addr <= 10'h21C)
                    bias_val[((word_addr - 10'h160) >> 2) + 6'd16]    <= s_axi_wdata;
                else if (word_addr >= 10'h220 && word_addr <= 10'h2DC)
                    scale_mul[((word_addr - 10'h220) >> 2) + 6'd16]   <= s_axi_wdata;
                else if (word_addr >= 10'h2E0 && word_addr <= 10'h39C)
                    scale_shift[((word_addr - 10'h2E0) >> 2) + 6'd16] <= s_axi_wdata[5:0];
            end
        end
    end

    // -------------------------------------------------------------------
    // Performance counters — read at 0x3A4+, cleared by a write to 0x3A0.
    // (Relocated above the resident bias/scale/shift region 0x160..0x39C that
    //  decision O added; the old 0x200 location now overlaps resident bias.)
    //   cyc_total : free-running time base since clear
    //   cyc_busy  : NPU FSM busy cycles
    //   cyc_arr   : systolic-array MAC cycles      -> array util & effective TOPS
    //   rd/wr_beats: AXI data beats                -> data volume
    //   rd/wr_busy : cycles a burst is outstanding -> bandwidth util = beats/busy
    // -------------------------------------------------------------------
    wire perf_clr = wr_en && (word_addr == 10'h3A0);

    reg [31:0] perf_cyc_total, perf_cyc_busy, perf_cyc_arr;
    reg [31:0] perf_rd_beats, perf_wr_beats, perf_rd_busy, perf_wr_busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || perf_clr) begin
            perf_cyc_total <= 32'd0; perf_cyc_busy <= 32'd0; perf_cyc_arr <= 32'd0;
            perf_rd_beats  <= 32'd0; perf_wr_beats <= 32'd0;
            perf_rd_busy   <= 32'd0; perf_wr_busy  <= 32'd0;
        end else begin
            perf_cyc_total <= perf_cyc_total + 32'd1;
            if (i_perf_busy)       perf_cyc_busy <= perf_cyc_busy + 32'd1;
            if (i_perf_arr_active) perf_cyc_arr  <= perf_cyc_arr  + 32'd1;
            if (i_perf_rd_beat)    perf_rd_beats <= perf_rd_beats + 32'd1;
            if (i_perf_wr_beat)    perf_wr_beats <= perf_wr_beats + 32'd1;
            if (i_perf_rd_busy)    perf_rd_busy  <= perf_rd_busy  + 32'd1;
            if (i_perf_wr_busy)    perf_wr_busy  <= perf_wr_busy  + 32'd1;
        end
    end

    // -------------------------------------------------------------------
    // Read channel
    // -------------------------------------------------------------------
    assign s_axi_arready = 1'b1;

    reg [DATA_W-1:0] rdata;
    reg              rvalid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rdata  <= 32'd0;
            rvalid <= 1'b0;
        end else begin
            if (s_axi_arvalid && s_axi_arready && !rvalid) begin
                rvalid <= 1'b1;
                case (s_axi_araddr[ADDR_W-1:0])
                    10'h00: rdata <= {14'd0, ctrl_gpool_en, ctrl_pool_avg, 2'd0, ctrl_int32_out, ctrl_oc_single, ctrl_row_block, ctrl_gemm_reduce, ctrl_row_par, ctrl_hw_pad, ctrl_gemm_en, ctrl_out_ping, ctrl_relu_en, ctrl_clear_done, ctrl_eltwise_en, ctrl_pool_en, ctrl_ping_pong, ctrl_start};
                    10'h04: rdata <= {28'd0, i_dma_wr_err, i_dma_rd_err, i_busy, done_irq_latched};
                    10'h08: rdata <= {{(32-SRAM_ADDR_W){1'b0}}, act_addr_ping};
                    10'h0C: rdata <= {{(32-SRAM_ADDR_W){1'b0}}, act_addr_pong};
                    10'h10: rdata <= {{(32-SRAM_ADDR_W){1'b0}}, wgt_addr_ping};
                    10'h14: rdata <= {{(32-SRAM_ADDR_W){1'b0}}, wgt_addr_pong};
                    10'h18: rdata <= {{(32-SRAM_ADDR_W){1'b0}}, out_addr_ping};
                    10'h1C: rdata <= {{(32-SRAM_ADDR_W){1'b0}}, out_addr_pong};
                    10'h20: rdata <= {16'd0, dim_in_w};
                    10'h24: rdata <= {16'd0, dim_in_h};
                    10'h28: rdata <= {16'd0, dim_in_c};
                    10'h2C: rdata <= {16'd0, dim_out_c};
                    10'h30: rdata <= {16'd0, kernel_kh, kernel_kw};
                    10'h34: rdata <= {16'd0, stride_sx, stride_sy};
                    10'h38: rdata <= bias_addr;
                    10'h3C: rdata <= scale_addr;

                    // Per-OC bias (0x40..0x7C)
                    10'h40, 10'h44, 10'h48, 10'h4C,
                    10'h50, 10'h54, 10'h58, 10'h5C,
                    10'h60, 10'h64, 10'h68, 10'h6C,
                    10'h70, 10'h74, 10'h78, 10'h7C:
                        rdata <= bias_val[s_axi_araddr[6:2] - 5'd16];

                    // Per-OC scale mul (0x80..0xBC)
                    10'h80, 10'h84, 10'h88, 10'h8C,
                    10'h90, 10'h94, 10'h98, 10'h9C,
                    10'hA0, 10'hA4, 10'hA8, 10'hAC,
                    10'hB0, 10'hB4, 10'hB8, 10'hBC:
                        rdata <= scale_mul[s_axi_araddr[6:2] - 5'd32];

                    // Per-OC scale shift (0xC0..0xFC)
                    10'hC0, 10'hC4, 10'hC8, 10'hCC,
                    10'hD0, 10'hD4, 10'hD8, 10'hDC,
                    10'hE0, 10'hE4, 10'hE8, 10'hEC,
                    10'hF0, 10'hF4, 10'hF8, 10'hFC:
                        rdata <= {26'd0, scale_shift[s_axi_araddr[6:2] - 5'd48]};

                    // Tile tracking
                    10'h100: rdata <= {22'd0, act_ic_tile};
                    10'h104: rdata <= {22'd0, act_oc_tile};
                    10'h108: rdata <= {16'd0, out_x_base};
                    10'h10C: rdata <= {16'd0, out_y_base};
                    10'h110: rdata <= {16'd0, total_ops_h};
                    10'h114: rdata <= {16'd0, total_ops_w};
                    10'h118: rdata <= {24'd0, clip_max};
                    10'h11C: rdata <= {{(32-SRAM_ADDR_W){1'b0}}, skip_base};
                    10'h15C: rdata <= {gavg_shift, gavg_mul};

                    // DMA status (0x140, read-only)
                    10'h140: rdata <= {28'd0, i_expand_done, i_copy_done, i_dma_wr_done, i_dma_rd_done};

                    // DMA register readback (0x124-0x13C)
                    10'h124: rdata <= dma_rd_ddr_addr;
                    10'h128: rdata <= {16'd0, dma_rd_len};
                    10'h12C: rdata <= {{(32-SRAM_ADDR_W){1'b0}}, dma_rd_sram_base};
                    10'h134: rdata <= dma_wr_ddr_addr;
                    10'h138: rdata <= {16'd0, dma_wr_len};
                    10'h13C: rdata <= {{(32-SRAM_ADDR_W){1'b0}}, dma_wr_sram_base};
                    10'h144: rdata <= {31'd0, dma_sram_sel};
                    10'h148: rdata <= {29'd0, dma_rd_sram_sel, dma_out_rd_sel};
                    10'h150: rdata <= {16'd0, pad_cfg};

                    default: rdata <= 32'd0;
                endcase

                // ---- Decision O: extended per-OC param readback (indices 16..63) ----
                if (s_axi_araddr[ADDR_W-1:0] >= 10'h160 && s_axi_araddr[ADDR_W-1:0] <= 10'h21C)
                    rdata <= bias_val[((s_axi_araddr[ADDR_W-1:0] - 10'h160) >> 2) + 6'd16];
                else if (s_axi_araddr[ADDR_W-1:0] >= 10'h220 && s_axi_araddr[ADDR_W-1:0] <= 10'h2DC)
                    rdata <= scale_mul[((s_axi_araddr[ADDR_W-1:0] - 10'h220) >> 2) + 6'd16];
                else if (s_axi_araddr[ADDR_W-1:0] >= 10'h2E0 && s_axi_araddr[ADDR_W-1:0] <= 10'h39C)
                    rdata <= {26'd0, scale_shift[((s_axi_araddr[ADDR_W-1:0] - 10'h2E0) >> 2) + 6'd16]};
                // Performance counters (relocated to 0x3A4..0x3BC, above resident params)
                else if (s_axi_araddr[ADDR_W-1:0] == 10'h3A4) rdata <= perf_cyc_total;
                else if (s_axi_araddr[ADDR_W-1:0] == 10'h3A8) rdata <= perf_cyc_busy;
                else if (s_axi_araddr[ADDR_W-1:0] == 10'h3AC) rdata <= perf_cyc_arr;
                else if (s_axi_araddr[ADDR_W-1:0] == 10'h3B0) rdata <= perf_rd_beats;
                else if (s_axi_araddr[ADDR_W-1:0] == 10'h3B4) rdata <= perf_wr_beats;
                else if (s_axi_araddr[ADDR_W-1:0] == 10'h3B8) rdata <= perf_rd_busy;
                else if (s_axi_araddr[ADDR_W-1:0] == 10'h3BC) rdata <= perf_wr_busy;
            end else if (s_axi_rready && rvalid) begin
                rvalid <= 1'b0;
            end
        end
    end

    assign s_axi_rvalid = rvalid;
    assign s_axi_rdata  = rdata;
    assign s_axi_rresp  = 2'b00;

    // -------------------------------------------------------------------
    // Output assignments
    // -------------------------------------------------------------------
    assign o_start        = ctrl_start;
    assign o_ping_pong_sel = ctrl_ping_pong;
    assign o_pool_en      = ctrl_pool_en;
    assign o_eltwise_en   = ctrl_eltwise_en;
    assign o_relu_en      = ctrl_relu_en;
    assign o_out_ping_sel = ctrl_out_ping;
    assign o_gemm_en      = ctrl_gemm_en;
    assign o_hw_pad       = ctrl_hw_pad;
    assign o_row_par_en   = ctrl_row_par;
    assign o_gemm_reduce  = ctrl_gemm_reduce;
    assign o_row_block_en = ctrl_row_block;
    assign o_oc_single    = ctrl_oc_single;
    assign o_int32_out    = ctrl_int32_out;
    assign o_pool_avg     = ctrl_pool_avg;
    assign o_gpool_en     = ctrl_gpool_en;
    assign o_gavg_mul     = {6'd0, gavg_mul};
    assign o_gavg_shift   = gavg_shift;
    assign o_pad_w        = pad_cfg[7:0];
    assign o_pad_h        = pad_cfg[15:8];
    assign o_clip_max     = clip_max;
    assign o_skip_base    = skip_base;
    assign o_clear_done   = ctrl_clear_done;

    assign o_act_addr_ping = act_addr_ping;
    assign o_act_addr_pong = act_addr_pong;
    assign o_wgt_addr_ping = wgt_addr_ping;
    assign o_wgt_addr_pong = wgt_addr_pong;
    assign o_out_addr_ping = out_addr_ping;
    assign o_out_addr_pong = out_addr_pong;

    assign o_dim_in_w   = dim_in_w;
    assign o_dim_in_h   = dim_in_h;
    assign o_dim_in_c   = dim_in_c;
    assign o_dim_out_c  = dim_out_c;
    assign o_kernel_kh  = kernel_kh;
    assign o_kernel_kw  = kernel_kw;
    assign o_stride_sx  = stride_sx;
    assign o_stride_sy  = stride_sy;

    assign o_bias_addr  = bias_addr;
    assign o_scale_addr = scale_addr;

    // Decision O: present the active OC-tile's NUM_OC-wide window (base = tile*16).
    // i_oc_tile_sel==0 ⇒ low-16 window ⇒ byte-identical to legacy.
    wire [6:0] oc_win_base = {i_oc_tile_sel, 4'd0};
    assign o_bias_val    = bias_val[oc_win_base +: NUM_OC];
    assign o_scale_mul   = scale_mul[oc_win_base +: NUM_OC];
    assign o_scale_shift = scale_shift[oc_win_base +: NUM_OC];

    assign o_act_ic_tile = act_ic_tile;
    assign o_act_oc_tile = act_oc_tile;
    assign o_out_x_base  = out_x_base;
    assign o_out_y_base  = out_y_base;

    assign o_total_ops_h = total_ops_h;
    assign o_total_ops_w = total_ops_w;

    // DMA control outputs
    assign o_dma_rd_req      = dma_rd_req_d;
    assign o_dma_rd_ddr_addr = dma_rd_ddr_addr;
    assign o_dma_rd_len      = dma_rd_len;
    assign o_dma_rd_sram_base = dma_rd_sram_base;
    assign o_dma_wr_req      = dma_wr_req_d;
    assign o_dma_wr_ddr_addr = dma_wr_ddr_addr;
    assign o_dma_wr_len      = dma_wr_len;
    assign o_dma_wr_sram_base = dma_wr_sram_base;
    assign o_dma_sram_sel     = dma_sram_sel;
    assign o_dma_out_rd_sel   = dma_out_rd_sel;
    assign o_dma_rd_sram_sel  = dma_rd_sram_sel;
    assign o_dma_act_ping_sel = dma_act_ping_sel;
    assign o_dma_wgt_ping_sel = dma_wgt_ping_sel;
    assign o_dma_out_ping_sel = dma_out_ping_sel;
    assign o_copy_trig        = copy_trig_d;
    assign o_expand_trig      = expand_trig_d;

endmodule
