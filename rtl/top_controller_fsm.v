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
    output wire [15:0]              o_group_base,   // = cur_ox during the write window
    input  wire                     i_im2col_win_vld,
    input  wire [15:0]              i_im2col_win_x,
    input  wire [15:0]              i_im2col_win_y,

    // === Systolic array control ===
    output wire                     o_array_vld,
    output wire                     o_array_k_end,
    output wire                     o_array_drain_en,

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
    output wire [15:0]              o_cur_oy
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

    reg [3:0] state, state_next;

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

    // Derived
    wire [15:0] ic_groups;
    assign ic_groups = (i_dim_in_c + 16'd15) >> 4;  // ceil(IC/16)

    // Runtime kernel offsets (conv 3x3 -> 9, GEMM 1x1 -> 1); replaces KERNEL_OFFSETS
    wire [7:0] ko_total;
    assign ko_total = {4'd0, i_kernel_kh[3:0]} * {4'd0, i_kernel_kw[3:0]};

    // Weight-prefetch reuse: when a whole OC tile's weights fit the on-chip
    // buffer (ic_groups <= ICG_BUF), prefetch once per OC tile and reuse across
    // the spatial sweep. General over any model; GEMM excluded (handled separately).
    localparam ICG_BUF          = 4;
    localparam WGT_REUSE_SETTLE = 16'd2;  // im2col window settle when skipping prefetch
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

    assign o_act_sram_addr = i_gemm_en ? gemm_act_addr
                           : i_hw_pad  ? tilemaj_addr
                           :             act_rd_addr;
    // GEMM reads the IC-tile word during PREFETCH (1-cycle SRAM latency; the
    // sdp_bram holds doa until the next read, so the word stays stable through
    // CALC where it is replicated to all 16 array rows).
    assign o_act_sram_en   = i_gemm_en
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
    assign o_array_vld     = (state == S_CALC_KERNEL);
    assign o_array_k_end   = (state == S_K_END);
    assign o_array_drain_en = (state == S_DRAIN);

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
                                     && (load_tile == (ic_groups[3:0] - 4'd1)))
                             || ((state == S_NEXT_TILE) && (ko_cnt == 4'd0)));
    assign o_im2col_offset_sel = ko_cnt;
    assign o_im2col_load_tile  = load_tile;
    assign o_im2col_group_base = cur_ox;
    assign o_group_size = group_size;
    assign o_group_base = cur_ox;

    // Out SRAM write — include OC tile offset so tiles don't overwrite each other
    wire [SRAM_ADDR_W-1:0] out_wr_addr_calc;
    assign out_wr_addr_calc = out_base_addr
                            + oc_tile[9:4] * out_spatial_size
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
            wgt_loaded   <= 1'b0;
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
                        // GEMM mode bypasses im2col line-load (S_LOAD_ROW/S_WAIT_WIN)
                        state <= i_gemm_en ? S_PREFETCH_WGT : S_LOAD_ROW;
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
                        end else if (load_tile >= ic_groups[3:0] - 4'd1) begin
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
                        if (cur_in_row >= {8'd0, i_kernel_kh} - 16'd1) begin
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
                            if (reuse_mode) begin
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
                        state   <= S_NEXT_TILE;
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
                        // Pre-fetch weights for IC tile 0 at new position
                        state <= S_PREFETCH_WGT;
                        pf_wait_cnt <= 16'd0;
                    end else if (cur_oy + 16'd1 < out_h) begin
                        // Next row
                        cur_ox <= 16'd0;
                        cur_oy <= cur_oy + 16'd1;
                        cur_in_col <= 16'd0;
                        cur_in_row <= cur_in_row + {8'd0, i_stride_sy};
                        state <= S_LOAD_ROW; // Need new row data
                        load_col_cnt <= 16'd0;
                    end else begin
                        // All spatial positions done for this OC tile
                        cur_ox <= 16'd0;
                        cur_oy <= 16'd0;
                        cur_in_col <= 16'd0;
                        cur_in_row <= 16'd0;
                        if (oc_tile + 16'd16 < i_dim_out_c) begin
                            // Next OC tile
                            oc_tile <= oc_tile + 16'd16;
                            wgt_loaded <= 1'b0;   // new OC tile → re-prefetch its weights
                            state <= i_gemm_en ? S_PREFETCH_WGT : S_LOAD_ROW;
                            load_col_cnt <= 16'd0;
                            pf_wait_cnt  <= 16'd0;
                        end else begin
                            state <= S_DONE;
                        end
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
