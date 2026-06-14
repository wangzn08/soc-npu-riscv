// Filename: im2col_line_buffer.v
// -------------------------------------------------------------------
// Im2Col Line Buffer — converts streaming pixel data into 3×3 sliding
// window kernel-offset vectors for the systolic array.
//
// MULTI IC-TILE (single timeline):
//   Activations with IC>16 are split into IC tiles of 16 channels.  The
//   FSM streams a column's tiles one-per-cycle (i_pixel_tile = 0..ic_groups-1)
//   and pulses i_win_advance once per column (on the last tile).  This block
//   keeps ic_groups separate 3×3 window register sets and ic_groups line-buffer
//   slots per column, but ONE shared cur_x / rd_ptr / wr_ptr / win_valid
//   timeline — so all tiles' windows are always column-aligned (no skew).
//   During CALC the systolic reads the window for i_win_tile.
//   For ic_groups==1 this is byte-identical to the original single-tile design.
//
// Dynamic zero-padding is applied at image boundaries.
// -------------------------------------------------------------------

module im2col_line_buffer #(
    parameter MAX_WIDTH        = 256,
    parameter ACT_WIDTH        = 8,
    parameter NUM_CH           = 16,
    parameter ACT_GROUP_W      = NUM_CH * ACT_WIDTH,   // 128
    parameter ARRAY_ROWS       = 16,
    parameter ACT_BUS_W        = ARRAY_ROWS * ACT_GROUP_W, // 2048
    parameter KERNEL_SIZE      = 3,
    parameter KERNEL_OFFSETS   = KERNEL_SIZE * KERNEL_SIZE, // 9
    parameter ADDR_W           = 8,   // log2(MAX_WIDTH)
    parameter ICG_MAX          = 4    // max IC tiles (64 channels / 16)
) (
    input  wire                         clk,
    input  wire                         rst_n,

    // Pixel input stream (from Act SRAM)
    input  wire [ACT_GROUP_W-1:0]       i_pixel_data,
    input  wire                         i_pixel_vld,
    input  wire [3:0]                   i_pixel_tile,   // IC tile of this pixel
    input  wire [3:0]                   i_ic_groups,    // number of IC tiles (1..ICG_MAX)
    input  wire [3:0]                   i_win_tile,     // IC tile whose window to output

    // Feature map geometry
    input  wire [15:0]                  i_width,
    input  wire [15:0]                  i_height,

    // Row start pulse: begins a new row, rotates line buffers
    input  wire                         i_row_start,

    // Window control
    input  wire                         i_win_advance,  // Shift window right 1 col (once/column)
    input  wire                         i_win_freeze,   // Hold window fixed
    input  wire [3:0]                   i_offset_sel,   // 0..8 kernel offset position
    input  wire                         i_row_par_en,   // task E: 16-wide slice mode
    input  wire [15:0]                  i_group_base,   // first output column of the 16-wide group
    input  wire                         i_row_block_en, // #4: pack R output rows into the array
    input  wire [15:0]                  i_group_size,   // columns per output row in the group
    input  wire [3:0]                   i_rows_per_grp, // R: output rows packed (1 = byte-identical)

    // Output: one kernel-offset activation vector, replicated 16×
    output wire [ACT_BUS_W-1:0]         o_act_window,
    output wire                         o_win_vld,

    output wire [15:0]                  o_win_x,
    output wire [15:0]                  o_win_y,

    output wire                         o_at_left_edge,
    output wire                         o_at_right_edge,
    output wire                         o_at_top_edge,
    output wire                         o_at_bottom_edge
);

    // -------------------------------------------------------------------
    // Line buffer storage — three row banks, each holding ICG_MAX tiles
    // per column.  lb_bankX[col][tile]
    // -------------------------------------------------------------------
    reg [ACT_GROUP_W-1:0] lb_bank0 [0:MAX_WIDTH-1][0:ICG_MAX-1];
    reg [ACT_GROUP_W-1:0] lb_bank1 [0:MAX_WIDTH-1][0:ICG_MAX-1];
    reg [ACT_GROUP_W-1:0] lb_bank2 [0:MAX_WIDTH-1][0:ICG_MAX-1];
    reg [ACT_GROUP_W-1:0] lb_bank3 [0:MAX_WIDTH-1][0:ICG_MAX-1];  // #4: 4th bank for R+2 rows

    reg [1:0] row_sel;
    reg [ADDR_W-1:0] wr_ptr;
    reg [ADDR_W-1:0] rd_ptr;
    reg [2:0] valid_rows;   // #4: up to 4 (was 2 bits / max 3)

    // 3×3 window register matrix, one set per IC tile
    reg [ACT_GROUP_W-1:0] win [0:ICG_MAX-1][0:2][0:2];

    reg [15:0] cur_x;
    reg [15:0] cur_y;
    reg [15:0] win_x;
    reg [15:0] win_y;
    reg win_valid;

    // Column complete = last IC tile of the column being streamed
    wire col_complete = i_pixel_vld && (i_pixel_tile == (i_ic_groups - 4'd1));

    integer init_i, init_g, gg, wr, wc;

    // -------------------------------------------------------------------
    // Line buffer init on reset (prevent X in simulation)
    // -------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (init_i = 0; init_i < MAX_WIDTH; init_i = init_i + 1)
                for (init_g = 0; init_g < ICG_MAX; init_g = init_g + 1) begin
                    lb_bank0[init_i][init_g] <= {ACT_GROUP_W{1'b0}};
                    lb_bank1[init_i][init_g] <= {ACT_GROUP_W{1'b0}};
                    lb_bank2[init_i][init_g] <= {ACT_GROUP_W{1'b0}};
                    lb_bank3[init_i][init_g] <= {ACT_GROUP_W{1'b0}};
                end
        end
    end

    // -------------------------------------------------------------------
    // Write incoming pixel into the current-row bank slot [wr_ptr][tile]
    // (always on valid pixel; wr_ptr held until the column completes)
    // -------------------------------------------------------------------
    always @(posedge clk) begin
        if (i_pixel_vld) begin
            case (row_sel)
                2'd0: lb_bank0[wr_ptr][i_pixel_tile] <= i_pixel_data;
                2'd1: lb_bank1[wr_ptr][i_pixel_tile] <= i_pixel_data;
                2'd2: lb_bank2[wr_ptr][i_pixel_tile] <= i_pixel_data;
                2'd3: lb_bank3[wr_ptr][i_pixel_tile] <= i_pixel_data;
            endcase
        end
    end

    // -------------------------------------------------------------------
    // Pointers, window shift, position tracking
    // -------------------------------------------------------------------
    reg [ACT_GROUP_W-1:0] cur_v, p1_v, p2_v;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_sel   <= 2'd0;
            wr_ptr    <= {ADDR_W{1'b0}};
            rd_ptr    <= {ADDR_W{1'b0}};
            valid_rows<= 2'd0;
            cur_x     <= 16'd0;
            cur_y     <= 16'd0;
            win_x     <= 16'd0;
            win_y     <= 16'd0;
            win_valid <= 1'b0;
            for (gg = 0; gg < ICG_MAX; gg = gg + 1)
                for (wr = 0; wr < 3; wr = wr + 1)
                    for (wc = 0; wc < 3; wc = wc + 1)
                        win[gg][wr][wc] <= {ACT_GROUP_W{1'b0}};
        end else begin

            // ---- Row start: rotate buffers, reset column ----
            if (i_row_start) begin
                // #4: rotate over 4 banks (R+2) when packing; else 3 (byte-identical).
                if (i_row_block_en)
                    row_sel <= (row_sel == 2'd3) ? 2'd0 : row_sel + 2'd1;
                else
                    row_sel <= (row_sel == 2'd2) ? 2'd0 : row_sel + 2'd1;
                if (valid_rows < (i_row_block_en ? 3'd4 : 3'd3))
                    valid_rows <= valid_rows + 3'd1;
                wr_ptr   <= {ADDR_W{1'b0}};
                rd_ptr   <= {ADDR_W{1'b0}};
                cur_x    <= 16'd0;
                cur_y    <= cur_y + 16'd1;
                win_x    <= 16'd0;
                win_y    <= win_y + 16'd1;
                win_valid<= 1'b0;
                for (gg = 0; gg < ICG_MAX; gg = gg + 1)
                    for (wr = 0; wr < 3; wr = wr + 1)
                        for (wc = 0; wc < 3; wc = wc + 1)
                            win[gg][wr][wc] <= {ACT_GROUP_W{1'b0}};
            end

            // ---- Line-buffer fill: advance write pointer once per column
            //      (on the last tile), independent of freeze so the full row
            //      is captured even after the window latches valid. ----
            if (col_complete) begin
                wr_ptr <= wr_ptr + {{ADDR_W-1{1'b0}}, 1'b1};
            end

            // ---- Window column tracking (gated by freeze) ----
            if (col_complete && !i_win_freeze) begin
                cur_x <= cur_x + 16'd1;
                if (cur_x >= 16'd2)
                    win_x <= cur_x - 16'd2;
                else
                    win_x <= 16'd0;
            end

            // ---- Window advance: shift all tiles' 3×3 matrices right ----
            if (i_win_advance && !i_win_freeze) begin
                for (gg = 0; gg < ICG_MAX; gg = gg + 1) begin
                    // shift left (col0<-col1, col1<-col2)
                    win[gg][0][0] <= win[gg][0][1];
                    win[gg][0][1] <= win[gg][0][2];
                    win[gg][1][0] <= win[gg][1][1];
                    win[gg][1][1] <= win[gg][1][2];
                    win[gg][2][0] <= win[gg][2][1];
                    win[gg][2][1] <= win[gg][2][2];

                    // new rightmost column from line buffers (per tile)
                    case (row_sel)
                        2'd0: begin cur_v = lb_bank0[rd_ptr][gg]; p1_v = lb_bank2[rd_ptr][gg]; p2_v = lb_bank1[rd_ptr][gg]; end
                        2'd1: begin cur_v = lb_bank1[rd_ptr][gg]; p1_v = lb_bank0[rd_ptr][gg]; p2_v = lb_bank2[rd_ptr][gg]; end
                        2'd2: begin cur_v = lb_bank2[rd_ptr][gg]; p1_v = lb_bank1[rd_ptr][gg]; p2_v = lb_bank0[rd_ptr][gg]; end
                        default: begin cur_v = {ACT_GROUP_W{1'b0}}; p1_v = {ACT_GROUP_W{1'b0}}; p2_v = {ACT_GROUP_W{1'b0}}; end
                    endcase

                    if (rd_ptr < i_width[ADDR_W-1:0]) begin
                        win[gg][0][2] <= (valid_rows >= 2'd3) ? p2_v : {ACT_GROUP_W{1'b0}};
                        win[gg][1][2] <= (valid_rows >= 2'd2) ? p1_v : {ACT_GROUP_W{1'b0}};
                        // current-row bypass: the tile written THIS column whose
                        // bank value is not yet visible is i_pixel_tile (the last);
                        // earlier tiles are already in the bank.
                        if ((gg == i_pixel_tile) && i_pixel_vld)
                            win[gg][2][2] <= i_pixel_data;
                        else
                            win[gg][2][2] <= (valid_rows >= 2'd2) ? cur_v : {ACT_GROUP_W{1'b0}};
                    end else begin
                        win[gg][0][2] <= {ACT_GROUP_W{1'b0}};
                        win[gg][1][2] <= {ACT_GROUP_W{1'b0}};
                        win[gg][2][2] <= {ACT_GROUP_W{1'b0}};
                    end
                end
                rd_ptr <= rd_ptr + {{ADDR_W-1{1'b0}}, 1'b1};
            end

            // ---- Window valid ----
            // Only evaluate at column completion (coincident with the column
            // advance), so the window isn't frozen one column early on the
            // multi-tile timeline (where cur_x advances only on col_complete,
            // leaving a gap before the next column's advance).
            if (i_row_start)
                win_valid <= 1'b0;
            // #4: row-block needs R+2 rows resident before a block's window is valid.
            else if (col_complete && cur_x >= 16'd2 &&
                     valid_rows >= (i_row_block_en ? 3'd4 : 3'd3))
                win_valid <= 1'b1;
        end
    end

    // -------------------------------------------------------------------
    // Offset selection (combinational): decode 0..8 → (row,col)
    // -------------------------------------------------------------------
    wire [3:0] offset_idx = i_offset_sel;
    wire [1:0] off_row_dec = (offset_idx < 4'd3) ? 2'd0 :
                             (offset_idx < 4'd6) ? 2'd1 : 2'd2;
    wire [1:0] off_col_dec = (offset_idx == 4'd0 || offset_idx == 4'd3 || offset_idx == 4'd6) ? 2'd0 :
                             (offset_idx == 4'd1 || offset_idx == 4'd4 || offset_idx == 4'd7) ? 2'd1 : 2'd2;

    wire [ACT_GROUP_W-1:0] selected_offset = win[i_win_tile][off_row_dec][off_col_dec];

    // Map the kernel-offset row (0=top,1=mid,2=bottom of the 3-row window) to a
    // physical line-buffer bank, mirroring the win[]-fill case(row_sel) mapping
    // (cur_v=bottom=bank[row_sel], p1_v=mid=bank[row_sel-1], p2_v=top=bank[row_sel-2]).
    function [1:0] bank_for_offrow;
        input [1:0] off_r;       // 0=top, 1=mid, 2=bottom
        input [1:0] rsel;        // row_sel
        reg   [1:0] bottom, mid, top;
        begin
            bottom = rsel;                                  // newest (this) row
            mid    = (rsel == 2'd0) ? 2'd2 : rsel - 2'd1;
            top    = (rsel == 2'd1) ? 2'd2 :
                     (rsel == 2'd0) ? 2'd1 : rsel - 2'd2;
            bank_for_offrow = (off_r == 2'd2) ? bottom :
                              (off_r == 2'd1) ? mid : top;
        end
    endfunction

    wire [1:0] rp_bank = bank_for_offrow(off_row_dec, row_sel);

    genvar gi;
    generate
        for (gi = 0; gi < ARRAY_ROWS; gi = gi + 1) begin : gen_window
            // #4 row-block: array row gi → block b = gi/group_size (output row
            // cur_oy+b), column-in-group c = gi%group_size. For MAX_R=2, b∈{0,1}.
            // Block b's 3-row window is the base window shifted DOWN by b input
            // rows, so its bank = (row_sel + b + off_row + 1) mod 4 (4-bank slide).
            wire        rb_b = i_row_block_en && (gi[15:0] >= i_group_size);
            wire [15:0] rb_c = rb_b ? (gi[15:0] - i_group_size) : gi[15:0];
            wire [1:0]  rb_bank = (row_sel + {1'b0, rb_b} + off_row_dec + 2'd1); // mod-4 (2-bit wrap)

            // Row-par: array row gi reads input column (group_base + off_col + gi).
            // Row-block overrides column to (group_base + off_col + c).
            wire [15:0] rp_col = i_group_base + {14'd0, off_col_dec}
                               + (i_row_block_en ? rb_c : gi[15:0]);
            wire [1:0]  rd_bank = i_row_block_en ? rb_bank : rp_bank;
            reg  [ACT_GROUP_W-1:0] rp_val;
            always @(*) begin
                if (rp_col >= i_width) begin
                    rp_val = {ACT_GROUP_W{1'b0}};
                end else begin
                    case (rd_bank)
                        2'd0: rp_val = lb_bank0[rp_col[ADDR_W-1:0]][i_win_tile];
                        2'd1: rp_val = lb_bank1[rp_col[ADDR_W-1:0]][i_win_tile];
                        2'd2: rp_val = lb_bank2[rp_col[ADDR_W-1:0]][i_win_tile];
                        2'd3: rp_val = lb_bank3[rp_col[ADDR_W-1:0]][i_win_tile];
                    endcase
                end
            end
            // Legacy (replicate) when row_par off — MUST stay byte-identical.
            assign o_act_window[gi*ACT_GROUP_W +: ACT_GROUP_W] =
                       i_row_par_en ? rp_val : selected_offset;
        end
    endgenerate

    assign o_win_vld = win_valid;
    assign o_win_x = win_x;
    assign o_win_y = win_y;
    assign o_at_left_edge   = (win_x == 16'd0);
    assign o_at_right_edge  = (win_x + 16'd3 >= i_width);
    assign o_at_top_edge    = (win_y == 16'd0);
    assign o_at_bottom_edge = (win_y + 16'd3 >= i_height);

endmodule
