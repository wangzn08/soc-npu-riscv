// Filename: top_controller_fsm.v
// -------------------------------------------------------------------
// Top-Level Controller FSM — orchestrates the full convolution pipeline:
//   IDLE → LOAD_ROW → WAIT_WIN → PREFETCH_WGT → CALC_KERNEL →
//   K_END → DRAIN → (loop IC tiles, spatial positions, OC tiles) → DONE
//
// Generates Act SRAM read addresses, controls im2col, wgt_reader,
// systolic array, and post-processing → Out SRAM write sequencing.
// -------------------------------------------------------------------

module top_controller_fsm #(
    parameter SRAM_ADDR_W    = 14,
    parameter DATA_W         = 128,
    parameter ACT_WIDTH      = 8,
    parameter NUM_CH_PER_GRP = 16,
    parameter KH              = 3,
    parameter KW              = 3,
    parameter KERNEL_OFFSETS = KH * KW,
    parameter MAX_OC_TILES   = 64,   // Up to 1024 output channels in tiles of 16
    parameter MAX_IC_TILES   = 64    // Up to 1024 input channels in tiles of 16
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // === From param_regfile ===
    input  wire                     i_start,
    input  wire                     i_ping_pong_sel,
    input  wire                     i_pool_en,
    input  wire                     i_eltwise_en,
    input  wire                     i_gemm_en,      // GEMM/FC mode: bypass im2col, act = vector
    input  wire                     i_hw_pad,       // hardware padding: read tile-major, inject border zeros
    input  wire                     i_row_par_en,   // 16-row spatial parallelism (task E)
    input  wire                     i_gemm_reduce,  // GEMM 16-row IC-reduction (decision M)
    input  wire                     i_row_block_en, // #4: row-block packing for narrow layers
    input  wire                     i_oc_single,    // decision O: all OC-tiles in one start (OC-inner loop)
    input  wire                     i_pw_en,        // 1x1 pointwise: bypass im2col, direct per-pixel feed
    input  wire [7:0]               i_pad_w,        // zero-pad columns each side
    input  wire [7:0]               i_pad_h,        // zero-pad rows each side
    input  wire [SRAM_ADDR_W-1:0]   i_act_base_ping,
    input  wire [SRAM_ADDR_W-1:0]   i_act_base_pong,
    input  wire [SRAM_ADDR_W-1:0]   i_wgt_base_ping,
    input  wire [SRAM_ADDR_W-1:0]   i_wgt_base_pong,
    input  wire [SRAM_ADDR_W-1:0]   i_out_base_ping,
    input  wire [SRAM_ADDR_W-1:0]   i_out_base_pong,
    input  wire [15:0]              i_dim_in_w,
    input  wire [15:0]              i_dim_in_h,
    input  wire [15:0]              i_dim_in_c,
    input  wire [15:0]              i_dim_out_c,
    input  wire [7:0]               i_kernel_kh,
    input  wire [7:0]               i_kernel_kw,
    input  wire [7:0]               i_stride_sx,
    input  wire [7:0]               i_stride_sy,

    // === Act SRAM read port ===
    output wire [SRAM_ADDR_W-1:0]   o_act_sram_addr,
    output wire                     o_act_sram_en,
    input  wire [DATA_W-1:0]        i_act_sram_data,

    // === Wgt Reader control ===
    output wire                     o_wgt_start_prefetch,
    output wire [9:0]               o_wgt_oc_base,
    output wire [9:0]               o_wgt_ic_group,
    output wire [15:0]              o_wgt_ic_groups_total,
    output wire [SRAM_ADDR_W-1:0]   o_wgt_base,
    output wire                     o_prefetch_all,  // 1 = reuse mode (prefetch all ic_groups once)
    output wire [3:0]               o_wgt_ic_sel,    // current ic_group for the weight output mux
    input  wire                     i_wgt_prefetch_done,

    // === Im2Col control ===
    output wire                     o_im2col_row_start,
    output wire                     o_im2col_win_advance,
    output wire                     o_im2col_win_freeze,
    output wire [3:0]               o_im2col_offset_sel,
    output wire [DATA_W-1:0]        o_im2col_pixel_data,
    output wire                     o_im2col_pixel_vld,
    output wire                     o_border,             // hw-pad: current pixel is a border zero
    output wire [3:0]               o_im2col_load_tile,   // IC tile being streamed during LOAD_ROW
    output wire [15:0]              o_im2col_group_base,  // first output column of the current 16-wide group
    output wire [15:0]              o_group_size,
    output wire [3:0]               o_rows_per_grp,   // #4: R output rows packed (1 = byte-identical)
    output wire [15:0]              o_group_base,   // = cur_ox during the write window
    input  wire                     i_im2col_win_vld,
    input  wire [15:0]              i_im2col_win_x,
    input  wire [15:0]              i_im2col_win_y,

    // === Systolic array control ===
    output wire                     o_array_vld,
    output wire                     o_array_k_end,
    output wire                     o_array_drain_en,

    // === GEMM-reduce sequencer (decision M) ===
    output wire                     o_wgt_plane_trig,  // pulse: prefetch a super-step's weight plane
    output wire [3:0]               o_superstep,       // super-step index s
    input  wire                     i_plane_done,      // wgt_reader plane prefetch complete
    output wire                     o_rdc_act_we,      // capture Act SRAM data → act_row[idx]
    output wire [3:0]               o_rdc_act_idx,     // act_row write index (0..15)
    output wire                     o_rdc_act_clear,   // clear act_row at super-step start

    // === Post-process / Out SRAM write ===
    output wire                     o_pp_start,
    output wire [SRAM_ADDR_W-1:0]   o_out_wr_addr,
    output wire                     o_out_wr_en,
    output wire [15:0]              o_out_x,
    output wire [15:0]              o_out_y,
    input  wire                     i_pp_done,

    // === State indicators for pooling control ===
    output wire                     o_in_drain,
    output wire                     o_in_post,

    // === Status outputs ===
    output wire                     o_busy,
    output wire                     o_done_irq,

    // === Tile tracking outputs (to param_regfile for debug) ===
    output wire [9:0]               o_cur_ic_tile,
    output wire [9:0]               o_cur_oc_tile,
    output wire [15:0]              o_cur_ox,
    output wire [15:0]              o_cur_oy,

    // === Decision O: active OC-tile selector (oc_single OC-inner loop) ===
    output wire [2:0]               o_oc_tile_sel,  // 0 when oc_single off (byte-identical)

    // === global avgpool: current output position is the last of this OC tile ===
    output wire                     o_last_spatial
);

    // -------------------------------------------------------------------
    // FSM state definitions
    // -------------------------------------------------------------------
    localparam S_IDLE         = 4'd0;
    localparam S_LOAD_ROW     = 4'd1;
    localparam S_WAIT_WIN     = 4'd2;
    localparam S_PREFETCH_WGT = 4'd3;
    localparam S_CALC_KERNEL  = 4'd4;
    localparam S_K_END        = 4'd5;
    localparam S_DRAIN        = 4'd6;
    localparam S_POST         = 4'd7;
    localparam S_NEXT_TILE    = 4'd8;
    localparam S_DONE         = 4'd9;
    localparam S_RDC_WLOAD    = 4'd10;   // GEMM-reduce: prefetch weight plane
    localparam S_RDC_ALOAD    = 4'd11;   // GEMM-reduce: load 16 act words
    localparam S_RDC_VLD      = 4'd12;   // GEMM-reduce: 1 accumulate cycle
    localparam S_WIN_STEP     = 4'd13;   // stride>1: extra im2col window advances per output column

    reg [3:0] state, state_next;
    reg [7:0] win_step_cnt;              // remaining extra win_advance pulses (stride_sx-1)

    // -------------------------------------------------------------------
    // Address & loop counters
    // -------------------------------------------------------------------
    reg [SRAM_ADDR_W-1:0] act_base_addr;
    reg [SRAM_ADDR_W-1:0] wgt_base_addr;
    reg [SRAM_ADDR_W-1:0] out_base_addr;

    reg [15:0] oc_tile;       // Current output channel tile (0, 16, 32...)
    reg [15:0] ic_tile;       // Current input channel tile (groups of 16)
    reg [15:0] ic_tiles_total;// Total IC tiles = ceil(i_dim_in_c / 16)
    reg [15:0] oc_tiles_total;// Total OC tiles = ceil(i_dim_out_c / 16)
    reg [15:0] out_w;         // Output feature map width
    reg [15:0] out_h;         // Output feature map height
    reg [15:0] out_spatial_size; // out_w * out_h, for OC tile offset
    reg [15:0] cur_oy;        // Current output Y
    reg [15:0] cur_ox;        // Current output X
    reg [15:0] cur_in_row;    // Current input row being loaded
    reg [15:0] cur_in_col;    // Current input column being loaded

    reg [3:0]  ko_cnt;        // Kernel offset counter (0..8)
    reg [4:0]  drain_cnt;     // Drain cycle counter (0..15)
    reg [15:0] load_col_cnt;  // Column load counter
    reg [15:0] pf_wait_cnt;   // Pre-fetch wait counter
    reg [15:0] row_base_in_row; // Starting cur_in_row for current output row
    reg [3:0]  load_tile;     // Current IC tile being streamed within a column (0..ic_groups-1)
    reg        wgt_loaded;    // 1 = this OC tile's weights are resident in wgt_buf (reuse mode)
    reg [2:0]  oc_t;          // decision O: active OC-tile in the oc_single OC-inner loop

    // GEMM-reduce (decision M)
    reg [3:0]  super_step;    // current super-step (each = 16 IC-tiles across 16 rows)
    reg [4:0]  aload_cnt;     // act-load counter (0..16)
    reg        plane_trig_r;  // 1-cycle weight-plane prefetch pulse

    // Derived
    wire [15:0] ic_groups;
    assign ic_groups = (i_dim_in_c + 16'd15) >> 4;  // ceil(IC/16)

    // GEMM-reduce derived (decision M)
    wire        reduce = i_gemm_en && i_gemm_reduce;
    wire [15:0] super_total = (ic_groups + 16'd15) >> 4;  // ceil(ic_groups/16)

    // Runtime kernel offsets (conv 3x3 -> 9, GEMM 1x1 -> 1); replaces KERNEL_OFFSETS
    wire [7:0] ko_total;
    assign ko_total = {4'd0, i_kernel_kh[3:0]} * {4'd0, i_kernel_kw[3:0]};

    // Weight-prefetch reuse: when a whole OC tile's weights fit the on-chip
    // buffer (ic_groups <= ICG_BUF), prefetch once per OC tile and reuse across
    // the spatial sweep. General over any model; GEMM excluded (handled separately).
    localparam ICG_BUF          = 4;
    localparam WGT_REUSE_SETTLE = 16'd2;  // im2col window settle when skipping prefetch
    // pointwise reuses weights across the spatial sweep (1x1 weights are position
    // independent), so it joins the per-OC-tile reuse path like conv ic_groups<=4.
    // BUT only when the whole OC tile's IC groups fit wgt_buf (ic_groups<=ICG_BUF).
    // For ic_groups>ICG_BUF, PW falls back to per-IC-group weight streaming (the
    // same non-reuse path conv uses): S_CALC_KERNEL already re-enters S_PREFETCH_WGT
    // per IC tile for PW, so dropping reuse here makes each IC group re-prefetch a
    // single group into wgt_buf[*][*][0] instead of overflowing the 4-deep buffer.
    // (Caller must NOT set oc_single for streamed PW; oc_single forces pf_all in
    // wgt_reader, which re-imposes full IC residency.)
    wire reuse_mode = !i_gemm_en && (ic_groups <= ICG_BUF);

    // Input stride in 128-bit words per row
    wire [SRAM_ADDR_W-1:0] act_row_stride;
    assign act_row_stride = i_dim_in_w * ic_groups;

    // Output stride
    wire [SRAM_ADDR_W-1:0] out_row_stride;
    assign out_row_stride = out_w;

    // Row-parallel: number of valid output columns in the current 16-wide group
    wire [15:0] rp_remaining = out_w - cur_ox;
    wire [15:0] group_size   = (i_row_par_en && rp_remaining > 16'd16) ? 16'd16
                             : (i_row_par_en)                          ? rp_remaining
                             :                                          16'd1;

    // #4 row-block packing: R output rows packed into the 16 array rows.
    // MAX_R=2; the gi→(block,col) mapping (im2col) is exact only when
    // 2*group_size==16, i.e. group_size==8 — so engage R=2 only there.
    // Any other layer keeps R=1 (byte-identical to decision I).
    wire [3:0] rows_per_grp = (i_row_block_en && group_size == 16'd8) ? 4'd2 : 4'd1;
    assign o_rows_per_grp = rows_per_grp;
    wire rb_active = (rows_per_grp > 4'd1);
    // Last input row to load for the current row-block: rows oy..oy+R+kh-2.
    // Input row to load up to before the current output row's window is ready.
    // Scaled by stride_sy so stride>1 loads stride_sy new rows per output row
    // (stride_sy==1 => identical to the legacy cur_oy + rows_per_grp + kh - 2).
    wire [15:0] lr_target = (cur_oy * {8'd0, i_stride_sy})
                          + {12'd0, rows_per_grp} + {8'd0, i_kernel_kh} - 16'd2;

    // -------------------------------------------------------------------
    // Output assignments
    // -------------------------------------------------------------------
    assign o_busy = (state != S_IDLE) && (state != S_DONE);

    // Act SRAM read
    wire [SRAM_ADDR_W-1:0] act_rd_addr;
    assign act_rd_addr = act_base_addr
                       + cur_in_row * act_row_stride
                       + cur_in_col * ic_groups
                       + load_tile;   // stream each IC tile of the column

    // GEMM: input vector word index = ic_tile/16 (ceil(IC/16) words at act_base)
    wire [SRAM_ADDR_W-1:0] gemm_act_addr;
    assign gemm_act_addr = act_base_addr + {{(SRAM_ADDR_W-12){1'b0}}, ic_tile[15:4]};

    // Pointwise (1x1): read the pixel at (cur_oy, cur_ox) for the current IC tile
    // directly from the same tile-major Act SRAM layout produced by conv/sram_copy:
    //   addr = ic_group * spatial + pixel.
    // IC=16 also worked with the old pixel-major formula, but IC>16 did not.
    wire [SRAM_ADDR_W-1:0] pw_spatial = i_dim_in_w * i_dim_in_h;
    wire [SRAM_ADDR_W-1:0] pw_pixel   = cur_oy * i_dim_in_w + cur_ox;
    wire [SRAM_ADDR_W-1:0] pw_act_addr = act_base_addr
                         + {{(SRAM_ADDR_W-12){1'b0}}, ic_tile[15:4]} * pw_spatial
                         + pw_pixel;

    // Hardware padding: read the previous layer's UNPADDED output tile-major and
    // inject border zeros, so im2col sees the same padded stream as before but no
    // CPU pad_activation is needed. unpad = i_dim_in_w/h - 2*pad (padded dims kept
    // in i_dim_in_w/h so the output geometry is unchanged).
    wire [15:0] unpad_w = i_dim_in_w - {7'd0, i_pad_w, 1'b0};
    wire [15:0] unpad_h = i_dim_in_h - {7'd0, i_pad_h, 1'b0};
    wire [SRAM_ADDR_W-1:0] unpad_spatial = unpad_w[SRAM_ADDR_W-1:0] * unpad_h[SRAM_ADDR_W-1:0];
    wire at_border = i_hw_pad &&
         ((cur_in_col <  {8'd0, i_pad_w}) || (cur_in_col >= i_dim_in_w - {8'd0, i_pad_w}) ||
          (cur_in_row <  {8'd0, i_pad_h}) || (cur_in_row >= i_dim_in_h - {8'd0, i_pad_h}));
    wire [SRAM_ADDR_W-1:0] tilemaj_addr = act_base_addr
         + load_tile * unpad_spatial
         + (cur_in_row - {8'd0, i_pad_h}) * unpad_w[SRAM_ADDR_W-1:0]
         + (cur_in_col - {8'd0, i_pad_w});
    assign o_border = at_border;

    // GEMM-reduce: row r of super-step s reads input-vector word (s*16 + r).
    wire [SRAM_ADDR_W-1:0] rdc_act_addr = act_base_addr
                         + {{(SRAM_ADDR_W-8){1'b0}}, super_step, 4'd0}   // s*16
                         + {{(SRAM_ADDR_W-5){1'b0}}, aload_cnt};
    assign o_act_sram_addr = reduce    ? rdc_act_addr
                           : i_pw_en   ? pw_act_addr
                           : i_gemm_en ? gemm_act_addr
                           : i_hw_pad  ? tilemaj_addr
                           :             act_rd_addr;
    // GEMM reads the IC-tile word during PREFETCH (1-cycle SRAM latency; the
    // sdp_bram holds doa until the next read, so the word stays stable through
    // CALC where it is replicated to all 16 array rows).
    assign o_act_sram_en   = reduce
                           ? ((state == S_RDC_ALOAD) && (aload_cnt < 5'd16))
                           : (i_gemm_en || i_pw_en)
                           ? ((state == S_PREFETCH_WGT) && (pf_wait_cnt == 16'd0))
                           : ((state == S_LOAD_ROW) && (cur_in_col < i_dim_in_w) && (load_col_cnt > 16'd0)
                              && (!i_hw_pad || !at_border));   // skip SRAM read on border

    // Im2Col pixel feed (pass through from SRAM during LOAD_ROW)
    assign o_im2col_pixel_data = i_act_sram_data;
    assign o_im2col_pixel_vld  = (state == S_LOAD_ROW) && (cur_in_col < i_dim_in_w) && (load_col_cnt > 16'd0);

    // Weight reader control
    // In reuse mode, only the OC tile's FIRST prefetch hits SRAM; later spatial
    // positions reuse wgt_buf (no re-trigger).
    assign o_wgt_start_prefetch  = (state == S_PREFETCH_WGT) && (pf_wait_cnt == 16'd0)
                                 && (!reuse_mode || !wgt_loaded);
    assign o_wgt_oc_base         = oc_tile[9:0];
    // wgt_reader wants the IC *group* index (0,1,2..), but ic_tile counts
    // channels (0,16,32..) → convert by >>4.
    assign o_wgt_ic_group        = ic_tile[9:0] >> 4;
    assign o_wgt_ic_groups_total = ic_groups;
    assign o_wgt_base            = wgt_base_addr;
    assign o_prefetch_all        = reuse_mode;
    assign o_wgt_ic_sel          = ic_tile[7:4];   // current ic_group (reuse output mux)

    // Array control
    assign o_array_vld     = (state == S_CALC_KERNEL) || (state == S_RDC_VLD);
    assign o_array_k_end   = (state == S_K_END);
    assign o_array_drain_en = (state == S_DRAIN);

    // GEMM-reduce sequencer outputs
    assign o_wgt_plane_trig = plane_trig_r;
    assign o_superstep      = super_step;
    assign o_rdc_act_clear  = (state == S_RDC_WLOAD);
    // ALOAD issues read at aload_cnt (0..15); data returns 1 cycle later, so
    // capture row (aload_cnt-1) when aload_cnt in 1..16.
    assign o_rdc_act_we     = (state == S_RDC_ALOAD) && (aload_cnt >= 5'd1) && (aload_cnt <= 5'd16);
    assign o_rdc_act_idx    = aload_cnt[4:0] - 5'd1;

    // State indicators for pooling control
    assign o_in_drain = (state == S_DRAIN);
    assign o_in_post  = (state == S_POST);

    // Im2Col window control
    assign o_im2col_row_start  = (state == S_LOAD_ROW) && (load_col_cnt == 16'd0) && (cur_in_col == 16'd0);
    assign o_im2col_win_freeze = (state == S_CALC_KERNEL) || (state == S_K_END)
                              || ((state == S_LOAD_ROW) && i_im2col_win_vld);
    // Window advance: once per COLUMN during LOAD (on the last IC tile, when the
    // whole column has been streamed into the line buffer), and once per spatial
    // step during the sweep (NEXT_TILE).  The im2col keeps one shared timeline
    // and shifts all IC tiles' windows together.  For ic_groups==1 the LOAD term
    // fires every pixel cycle, identical to the original single-tile design.
    assign o_im2col_win_advance = !i_gemm_en &&
                             (((state == S_LOAD_ROW) && (cur_in_col < i_dim_in_w)
                                     && !i_im2col_win_vld && (load_col_cnt > 16'd0)
                                     && ({12'd0, load_tile} == (ic_groups - 16'd1)))
                             || ((state == S_NEXT_TILE) && (ko_cnt == 4'd0))
                             || (state == S_WIN_STEP));
    assign o_im2col_offset_sel = ko_cnt;
    assign o_im2col_load_tile  = load_tile;
    assign o_im2col_group_base = cur_ox;
    assign o_group_size = group_size;
    assign o_group_base = cur_ox;

    // Active OC-tile index: oc_single uses the inner loop counter oc_t; otherwise
    // the outer oc_tile (0,16,32→idx 0,1,2). 0 when oc_single off ⇒ byte-identical.
    wire [5:0] active_oc_idx = i_oc_single ? {3'd0, oc_t} : oc_tile[9:4];
    assign o_oc_tile_sel = active_oc_idx[2:0];

    // global avgpool: this output position is the last spatial point of the tile
    // (used to trigger the single mean write). Valid during S_POST where cur_ox/oy
    // hold the current position. group_size=1, rows_per_grp=1 in the plain (non
    // row_par) mode global avgpool is used in.
    assign o_last_spatial = (cur_ox + group_size >= out_w) &&
                            (cur_oy + {12'd0, rows_per_grp} >= out_h);

    // Out SRAM write — include OC tile offset so tiles don't overwrite each other
    wire [SRAM_ADDR_W-1:0] out_wr_addr_calc;
    assign out_wr_addr_calc = out_base_addr
                            + active_oc_idx * out_spatial_size
                            + cur_oy * out_row_stride
                            + cur_ox;
    assign o_out_wr_addr = out_wr_addr_calc;
    assign o_out_wr_en   = (state == S_POST) && i_pp_done;
    assign o_out_x = cur_ox;
    assign o_out_y = cur_oy;
    assign o_pp_start = (state == S_POST) && (drain_cnt == 5'd0); // Pulse at start of POST

    // Tile tracking output
    assign o_cur_ic_tile = ic_tile[9:0];
    assign o_cur_oc_tile = oc_tile[9:0];
    assign o_cur_ox = cur_ox;
    assign o_cur_oy = cur_oy;

    // -------------------------------------------------------------------
    // FSM sequential logic
    // -------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            oc_tile      <= 16'd0;
            ic_tile      <= 16'd0;
            ic_tiles_total <= 16'd0;
            oc_tiles_total <= 16'd0;
            out_w        <= 16'd0;
            out_h        <= 16'd0;
            out_spatial_size <= 16'd0;
            cur_oy       <= 16'd0;
            cur_ox       <= 16'd0;
            cur_in_row   <= 16'd0;
            cur_in_col   <= 16'd0;
            ko_cnt       <= 4'd0;
            drain_cnt    <= 5'd0;
            load_col_cnt <= 16'd0;
            pf_wait_cnt  <= 16'd0;
            row_base_in_row <= 16'd0;
            load_tile    <= 4'd0;
            win_step_cnt <= 8'd0;
            wgt_loaded   <= 1'b0;
            oc_t         <= 3'd0;
            super_step   <= 4'd0;
            aload_cnt    <= 5'd0;
            plane_trig_r <= 1'b0;
            act_base_addr<= {SRAM_ADDR_W{1'b0}};
            wgt_base_addr<= {SRAM_ADDR_W{1'b0}};
            out_base_addr<= {SRAM_ADDR_W{1'b0}};
        end else begin
            case (state)

                // -------------------------------------------------------
                S_IDLE: begin
                    if (i_start) begin
                        // Latch configuration
                        act_base_addr  <= i_ping_pong_sel ? i_act_base_pong : i_act_base_ping;
                        wgt_base_addr  <= i_ping_pong_sel ? i_wgt_base_pong : i_wgt_base_ping;
                        out_base_addr  <= i_ping_pong_sel ? i_out_base_pong : i_out_base_ping;
                        ic_tiles_total <= ic_groups;
                        oc_tiles_total <= (i_dim_out_c + 16'd15) >> 4;
                        out_w          <= (i_dim_in_w - {8'd0, i_kernel_kw}) / {8'd0, i_stride_sx} + 16'd1;
                        out_h          <= (i_dim_in_h - {8'd0, i_kernel_kh}) / {8'd0, i_stride_sy} + 16'd1;
                        out_spatial_size <= ((i_dim_in_w - {8'd0, i_kernel_kw}) / {8'd0, i_stride_sx} + 16'd1)
                                          * ((i_dim_in_h - {8'd0, i_kernel_kh}) / {8'd0, i_stride_sy} + 16'd1);
                        // Reset tile counters
                        oc_tile    <= 16'd0;
                        ic_tile    <= 16'd0;
                        cur_oy     <= 16'd0;
                        cur_ox     <= 16'd0;
                        cur_in_row <= 16'd0;
                        cur_in_col <= 16'd0;
                        load_col_cnt <= 16'd0;
                        ko_cnt     <= 4'd0;
                        drain_cnt  <= 5'd0;
                        pf_wait_cnt<= 16'd0;
                        row_base_in_row <= 16'd0;
                        load_tile  <= 4'd0;
                        wgt_loaded <= 1'b0;   // force first prefetch of this layer
                        oc_t       <= 3'd0;   // decision O: start OC-inner loop at tile 0
                        super_step <= 4'd0;
                        aload_cnt  <= 5'd0;
                        // GEMM-reduce bypasses im2col AND the per-k-step GEMM loop;
                        // GEMM bypasses im2col line-load (S_LOAD_ROW/S_WAIT_WIN)
                        if (i_gemm_en && i_gemm_reduce) begin
                            plane_trig_r <= 1'b1;        // kick first super-step's plane prefetch
                            state <= S_RDC_WLOAD;
                        end else begin
                            state <= (i_gemm_en || i_pw_en) ? S_PREFETCH_WGT : S_LOAD_ROW;
                        end
                    end
                end

                // -------------------------------------------------------
                // LOAD_ROW: Read one row of input pixels from Act SRAM
                //           and feed them into the im2col line buffer.
                // -------------------------------------------------------
                S_LOAD_ROW: begin
                    if (cur_in_col < i_dim_in_w) begin
                        // Stream the IC tiles of the current column one per cycle.
                        // Skip the very first cycle (row_start settle); then read
                        // tile 0,1,..,ic_groups-1, advancing the column only after
                        // the last tile.  (ic_groups==1 ⇒ one read per column.)
                        if (load_col_cnt == 16'd0) begin
                            load_col_cnt <= 16'd1;
                        end else if ({12'd0, load_tile} >= ic_groups - 16'd1) begin
                            load_tile  <= 4'd0;
                            cur_in_col <= cur_in_col + 16'd1;
                        end else begin
                            load_tile  <= load_tile + 4'd1;
                        end
                    end else begin
                        // Row complete
                        cur_in_col   <= 16'd0;
                        load_col_cnt <= 16'd0;
                        load_tile    <= 4'd0;
                        // After loading enough rows (≥ kernel height),
                        // transition to window processing.
                        // IMPORTANT: do NOT increment cur_in_row here —
                        // NEXT_TILE will add stride when starting the next
                        // output row.  Incrementing here would double-count
                        // and push the SRAM address out of bounds.
                        // #4 row-block: load R+2 rows (up to lr_target) before a
                        // block's window; legacy loads kh rows (cur_in_row>=kh-1).
                        // Stop once enough input rows are resident for this output
                        // row's window. lr_target is stride-scaled; for cur_oy=0 it
                        // equals kh-1 (legacy first-row behavior) for both strides.
                        if (cur_in_row >= lr_target) begin
                            state <= S_WAIT_WIN;
                        end else if (cur_in_row < i_dim_in_h - 16'd1) begin
                            cur_in_row <= cur_in_row + 16'd1;
                        end
                    end
                end

                // -------------------------------------------------------
                // WAIT_WIN: Wait until im2col has formed a valid 3×3 window
                // Window advances automatically during LOAD_ROW (win_advance=pixel_vld).
                // After 3 full rows, window should be valid within 2 columns of row start.
                S_WAIT_WIN: begin
                    if (i_im2col_win_vld) begin
                        state <= S_PREFETCH_WGT;
                        pf_wait_cnt <= 16'd0;
                    end
                    // else: wait here — window will become valid as more columns are received
                    // (Should rarely loop; window is valid after ~3 pixels of the 3rd row)
                end

                // -------------------------------------------------------
                // PREFETCH_WGT: Wait for wgt_reader to load weights
                // -------------------------------------------------------
                S_PREFETCH_WGT: begin
                    pf_wait_cnt <= pf_wait_cnt + 16'd1;
                    if (reuse_mode && wgt_loaded) begin
                        // Weights already resident from this OC tile's first prefetch;
                        // just let the im2col window settle, then compute.
                        if (pf_wait_cnt >= WGT_REUSE_SETTLE) begin
                            ko_cnt <= 4'd0;
                            state  <= S_CALC_KERNEL;
                        end
                    end else if (i_wgt_prefetch_done) begin
                        if (reuse_mode) wgt_loaded <= 1'b1;
                        ko_cnt  <= 4'd0;
                        state   <= S_CALC_KERNEL;
                    end
                end

                // -------------------------------------------------------
                // CALC_KERNEL: 9 cycles, one per kernel offset
                // -------------------------------------------------------
                S_CALC_KERNEL: begin
                    if (ko_cnt == (ko_total[3:0] - 4'd1)) begin
                        ko_cnt <= 4'd0;
                        if (ic_tile + 16'd16 < i_dim_in_c) begin
                            // More IC tiles: accumulate across tiles, skip drain
                            ic_tile <= ic_tile + 16'd16;
                            if (i_pw_en) begin
                                // Pointwise reads Act SRAM directly; each IC group
                                // needs a fresh SRAM read even though weights are resident.
                                state       <= S_PREFETCH_WGT;
                                pf_wait_cnt <= 16'd0;
                            end else if (reuse_mode) begin
                                // Next ic_group already in wgt_buf; o_wgt_ic_sel
                                // follows ic_tile — recompute directly, no prefetch.
                                state <= S_CALC_KERNEL;
                            end else begin
                                state   <= S_PREFETCH_WGT;
                                pf_wait_cnt <= 16'd0;
                            end
                        end else begin
                            // Last IC tile: latch & drain
                            state  <= S_K_END;
                        end
                    end else begin
                        ko_cnt <= ko_cnt + 4'd1;
                    end
                end

                // -------------------------------------------------------
                // K_END: Pulse i_k_end to latch accumulators
                // -------------------------------------------------------
                S_K_END: begin
                    drain_cnt <= 5'd0;
                    state <= S_DRAIN;
                end

                // -------------------------------------------------------
                // DRAIN: Shift partial sums out (16 cycles)
                // -------------------------------------------------------
                S_DRAIN: begin
                    // Reduce drains 16 cycles too: the combinational column sum is
                    // stable while psum_shift holds, so post_process sees 16 identical
                    // valids — same pipeline-fill timing as legacy GEMM (which the
                    // S_POST write path relies on; a single valid mis-times the write).
                    if (drain_cnt == 5'd15) begin
                        drain_cnt <= 5'd0;
                        state <= S_POST;
                    end else begin
                        drain_cnt <= drain_cnt + 5'd1;
                    end
                end

                // -------------------------------------------------------
                // POST: Post-processing (bias + quant + relu + pool)
                // -------------------------------------------------------
                S_POST: begin
                    if (i_pp_done) begin
                        // All IC tiles already accumulated in CALC_KERNEL
                        ic_tile <= 16'd0;
                        // Decision O: OC-inner loop. If more OC tiles remain for this
                        // spatial group, advance oc_t and re-run CALC against the SAME
                        // (frozen) im2col window with the next tile's resident weights —
                        // no spatial advance, no window advance, no re-prefetch. Reuse
                        // the proven PREFETCH_WGT settle path (reuse_mode+wgt_loaded).
                        if (i_oc_single && ({13'd0, oc_t} + 16'd1 < oc_tiles_total)) begin
                            oc_t        <= oc_t + 3'd1;
                            pf_wait_cnt <= 16'd0;
                            state       <= S_PREFETCH_WGT;
                        end else begin
                            oc_t    <= 3'd0;   // restart OC-inner loop for the next group
                            state   <= S_NEXT_TILE;
                        end
                    end
                end

                // -------------------------------------------------------
                // NEXT_TILE: Advance spatial position or OC tile
                // -------------------------------------------------------
                S_NEXT_TILE: begin
                    if (cur_ox + group_size < out_w) begin
                        // Next group of output columns in current row
                        cur_ox <= cur_ox + group_size;
                        cur_in_col <= cur_in_col + group_size * {8'd0, i_stride_sx}; // slide by group*stride
                        // The window must shift stride_sx columns to the next output
                        // position. S_NEXT_TILE already pulses one win_advance; for
                        // non-row_par stride>1 issue the remaining (stride_sx-1) pulses
                        // in S_WIN_STEP. row_par feeds its 16 windows during LOAD, so
                        // it keeps the legacy single-pulse path (stride==1 only).
                        if (!i_row_par_en && i_stride_sx > 8'd1) begin
                            win_step_cnt <= i_stride_sx - 8'd1;
                            state <= S_WIN_STEP;
                        end else begin
                            state <= S_PREFETCH_WGT;
                        end
                        pf_wait_cnt <= 16'd0;
                    end else if (cur_oy + {12'd0, rows_per_grp} < out_h) begin
                        // Next row-block: advance cur_oy by R (=1 legacy).  The
                        // line buffer slides by loading R new rows in LOAD_ROW
                        // (cur_in_row += stride here; LOAD_ROW loads up to lr_target).
                        cur_ox <= 16'd0;
                        cur_oy <= cur_oy + {12'd0, rows_per_grp};
                        cur_in_col <= 16'd0;
                        // Advance one row past the last-loaded; LOAD_ROW then loads
                        // up to the stride-scaled lr_target (stride_sy new rows).
                        // stride_sy==1 => +1, identical to legacy.
                        cur_in_row <= cur_in_row + 16'd1;
                        // pointwise has no im2col rows to load; go straight to compute.
                        state <= i_pw_en ? S_PREFETCH_WGT : S_LOAD_ROW;
                        load_col_cnt <= 16'd0;
                        pf_wait_cnt  <= 16'd0;   // pw: re-arm the per-position act read on row change
                    end else begin
                        // All spatial positions done for this OC tile
                        cur_ox <= 16'd0;
                        cur_oy <= 16'd0;
                        cur_in_col <= 16'd0;
                        cur_in_row <= 16'd0;
                        // Decision O: in oc_single ALL OC tiles are computed per group
                        // (OC-inner), so spatial exhaustion means the layer is done.
                        if (!i_oc_single && oc_tile + 16'd16 < i_dim_out_c) begin
                            // Next OC tile (legacy multi-start path)
                            oc_tile <= oc_tile + 16'd16;
                            wgt_loaded <= 1'b0;   // new OC tile → re-prefetch its weights
                            state <= (i_gemm_en || i_pw_en) ? S_PREFETCH_WGT : S_LOAD_ROW;
                            load_col_cnt <= 16'd0;
                            pf_wait_cnt  <= 16'd0;
                        end else begin
                            state <= S_DONE;
                        end
                    end
                end

                // -------------------------------------------------------
                // WIN_STEP: stride>1 horizontal — issue the remaining
                // (stride_sx-1) im2col window advances before computing the
                // next output column. One pulse per cycle here (win_advance is
                // asserted combinationally while in this state).
                // -------------------------------------------------------
                S_WIN_STEP: begin
                    if (win_step_cnt <= 8'd1)
                        state <= S_PREFETCH_WGT;
                    else
                        win_step_cnt <= win_step_cnt - 8'd1;
                end

                // -------------------------------------------------------
                // GEMM-reduce super-step sequence (decision M)
                // -------------------------------------------------------
                S_RDC_WLOAD: begin
                    plane_trig_r <= 1'b0;          // 1-cycle pulse, then wait
                    if (i_plane_done) begin
                        aload_cnt <= 5'd0;
                        state     <= S_RDC_ALOAD;
                    end
                end

                S_RDC_ALOAD: begin
                    // Issue 16 act reads (cnt 0..15); capture completes at cnt 16.
                    if (aload_cnt == 5'd16) begin
                        state <= S_RDC_VLD;
                    end else begin
                        aload_cnt <= aload_cnt + 5'd1;
                    end
                end

                S_RDC_VLD: begin
                    // One accumulate cycle for this super-step.
                    if (super_step + 4'd1 < super_total[3:0]) begin
                        super_step   <= super_step + 4'd1;
                        plane_trig_r <= 1'b1;       // prefetch next super-step's plane
                        state        <= S_RDC_WLOAD;
                    end else begin
                        state <= S_K_END;           // all super-steps done → latch
                    end
                end

                // -------------------------------------------------------
                S_DONE: begin
                    // Wait for status register clear, then go to IDLE
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // -------------------------------------------------------------------
    // Done interrupt (pulse)
    // -------------------------------------------------------------------
    reg done_irq_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            done_irq_r <= 1'b0;
        else if (state == S_DONE)
            done_irq_r <= 1'b1;
        else
            done_irq_r <= 1'b0;
    end
    assign o_done_irq = done_irq_r;

endmodule
