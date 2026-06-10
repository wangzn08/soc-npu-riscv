// Filename: im2col_line_buffer.v
// -------------------------------------------------------------------
// Im2Col Line Buffer — converts streaming pixel data into 3×3 sliding
// window kernel-offset vectors for the systolic array.
//
// Maintains two previous-row line buffers and one current-row buffer
// (all simple-dual-port RAM style for BRAM inference).  A 3×3 register
// matrix forms the active window.  On each window-advance the matrix
// shifts right; on freeze, one of nine kernel-offset positions is
// output (128-bit activation replicated 16× to fill the 2048-bit bus).
//
// Dynamic zero-padding is applied at image boundaries (x<0, x>=Width,
// y<0, y>=Height).
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
    parameter ADDR_W           = 8   // log2(MAX_WIDTH), 256 → 8
) (
    input  wire                         clk,
    input  wire                         rst_n,

    // Pixel input stream (from Act SRAM)
    input  wire [ACT_GROUP_W-1:0]       i_pixel_data,
    input  wire                         i_pixel_vld,

    // Feature map geometry
    input  wire [15:0]                  i_width,
    input  wire [15:0]                  i_height,

    // Row start pulse: begins a new row, rotates line buffers
    input  wire                         i_row_start,

    // Window control
    input  wire                         i_win_advance,  // Shift window right 1 col
    input  wire                         i_win_freeze,   // Hold window fixed (during offset output)
    input  wire [3:0]                   i_offset_sel,   // 0..8 selects kernel offset position

    // Output: one kernel-offset activation vector, replicated 16×
    output wire [ACT_BUS_W-1:0]         o_act_window,
    output wire                         o_win_vld,

    // Window top-left coordinates (for the FSM)
    output wire [15:0]                  o_win_x,
    output wire [15:0]                  o_win_y,

    // Boundary flags
    output wire                         o_at_left_edge,
    output wire                         o_at_right_edge,
    output wire                         o_at_top_edge,
    output wire                         o_at_bottom_edge
);

    // -------------------------------------------------------------------
    // Line buffer storage — three buffers (cur, prev1, prev2)
    // Each stores one full row of 128-bit pixels.
    // On i_row_start the logical mapping rotates.
    // -------------------------------------------------------------------
    reg [ACT_GROUP_W-1:0] lb_bank0 [0:MAX_WIDTH-1];
    reg [ACT_GROUP_W-1:0] lb_bank1 [0:MAX_WIDTH-1];
    reg [ACT_GROUP_W-1:0] lb_bank2 [0:MAX_WIDTH-1];

    // Logical → physical mapping
    //   row_sel = 0: cur=bank0, prev1=bank1, prev2=bank2
    //   row_sel = 1: cur=bank1, prev1=bank2, prev2=bank0
    //   row_sel = 2: cur=bank2, prev1=bank0, prev2=bank1
    reg [1:0] row_sel;

    // Pointers
    reg [ADDR_W-1:0] wr_ptr;    // Write pointer into current-row bank
    reg [ADDR_W-1:0] rd_ptr;    // Read pointer for prev1/prev2 banks

    // Row validity: how many previous rows are valid (0, 1, or 2)
    reg [1:0] valid_rows;       // 0 = no row, 1 = 1 row, 2 = 2 rows

    // -------------------------------------------------------------------
    // 3×3 window register matrix
    // win[r][c]: row r (0=top=y-2, 1=mid=y-1, 2=bot=y), col c (0=left, 1=mid, 2=right)
    // -------------------------------------------------------------------
    reg [ACT_GROUP_W-1:0] win [0:2][0:2];

    // -------------------------------------------------------------------
    // Column and row position tracking
    // -------------------------------------------------------------------
    reg [15:0] cur_x;          // Current column being received (0..width-1)
    reg [15:0] cur_y;          // Current row being received (0..height-1)
    reg [15:0] win_x;          // Window top-left X (can be negative → clamped)
    reg [15:0] win_y;          // Window top-left Y

    // -------------------------------------------------------------------
    // Window-valid state
    // -------------------------------------------------------------------
    reg win_valid;              // Full 3×3 window is populated

    // -------------------------------------------------------------------
    // Read data from previous-row banks (combinational)
    // -------------------------------------------------------------------
    wire [ACT_GROUP_W-1:0] prev1_data;
    wire [ACT_GROUP_W-1:0] prev2_data;

    // Based on row_sel, map logical prev1/prev2/cur to physical banks
    reg [ACT_GROUP_W-1:0] prev1_data_r;
    reg [ACT_GROUP_W-1:0] prev2_data_r;
    reg [ACT_GROUP_W-1:0] cur_bank_data_r;

    always @(*) begin
        case (row_sel)
            2'd0: begin
                prev1_data_r = lb_bank2[rd_ptr];  // row_sel-1 = -1 → bank2
                prev2_data_r = lb_bank1[rd_ptr];  // row_sel-2 = -2 → bank1
                cur_bank_data_r = lb_bank0[rd_ptr]; // current write bank
            end
            2'd1: begin
                prev1_data_r = lb_bank0[rd_ptr];  // row_sel-1 = 0 → bank0
                prev2_data_r = lb_bank2[rd_ptr];  // row_sel-2 = -1 → bank2
                cur_bank_data_r = lb_bank1[rd_ptr]; // current write bank
            end
            2'd2: begin
                prev1_data_r = lb_bank1[rd_ptr];  // row_sel-1 = 1 → bank1
                prev2_data_r = lb_bank0[rd_ptr];  // row_sel-2 = 0 → bank0
                cur_bank_data_r = lb_bank2[rd_ptr]; // current write bank
            end
            default: begin
                prev1_data_r = {ACT_GROUP_W{1'b0}};
                prev2_data_r = {ACT_GROUP_W{1'b0}};
                cur_bank_data_r = {ACT_GROUP_W{1'b0}};
            end
        endcase
    end

    assign prev1_data = prev1_data_r;
    assign prev2_data = prev2_data_r;

    // Zero-padded versions for boundary handling
    wire [ACT_GROUP_W-1:0] prev1_padded;
    wire [ACT_GROUP_W-1:0] prev2_padded;

    assign prev2_padded = (valid_rows >= 3) ? prev2_data : {ACT_GROUP_W{1'b0}};
    assign prev1_padded = (valid_rows >= 2) ? prev1_data : {ACT_GROUP_W{1'b0}};

    // -------------------------------------------------------------------
    // Write to current-row bank (always write on valid pixel, regardless of freeze)
    // -------------------------------------------------------------------
    always @(posedge clk) begin
        if (i_pixel_vld) begin
            case (row_sel)
                2'd0: lb_bank0[wr_ptr] <= i_pixel_data;
                2'd1: lb_bank1[wr_ptr] <= i_pixel_data;
                2'd2: lb_bank2[wr_ptr] <= i_pixel_data;
            endcase
        end
    end

    // -------------------------------------------------------------------
    // Window shift logic and position tracking
    // -------------------------------------------------------------------
    // Initialize line buffer banks to zero on reset (prevent X in simulation)
    integer init_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (init_i = 0; init_i < MAX_WIDTH; init_i = init_i + 1) begin
                lb_bank0[init_i] <= {ACT_GROUP_W{1'b0}};
                lb_bank1[init_i] <= {ACT_GROUP_W{1'b0}};
                lb_bank2[init_i] <= {ACT_GROUP_W{1'b0}};
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        integer wr, wc;
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
            // Initialize window registers to zero
            for (wr = 0; wr < 3; wr = wr + 1) begin
                for (wc = 0; wc < 3; wc = wc + 1) begin
                    win[wr][wc] <= {ACT_GROUP_W{1'b0}};
                end
            end
        end else begin

            // ---- Row start: rotate buffers and reset column ----
            if (i_row_start) begin
                // Rotate logical mapping (modulo 3: 0→1→2→0)
                row_sel <= (row_sel == 2'd2) ? 2'd0 : row_sel + 2'd1;
                // Update valid_rows (cap at 3)
                if (valid_rows < 2'd3)
                    valid_rows <= valid_rows + 2'd1;
                // Reset column pointers
                wr_ptr   <= {ADDR_W{1'b0}};
                rd_ptr   <= {ADDR_W{1'b0}};
                cur_x    <= 16'd0;
                cur_y    <= cur_y + 16'd1;
                win_x    <= 16'd0;
                win_y    <= win_y + 16'd1;
                win_valid<= 1'b0;
                // Clear window registers to prevent stale data corruption
                for (wr = 0; wr < 3; wr = wr + 1) begin
                    for (wc = 0; wc < 3; wc = wc + 1) begin
                        win[wr][wc] <= {ACT_GROUP_W{1'b0}};
                    end
                end
            end

            // ---- Line-buffer fill: advance write pointer on EVERY valid
            //      pixel, independent of i_win_freeze.  Once the window
            //      latches valid it freezes at output x=0, but the rest of
            //      the input row must still be captured into the line buffer
            //      so the NEXT_TILE sweep (which reads buffered columns via
            //      rd_ptr) sees the full row instead of zeros. ----
            if (i_pixel_vld) begin
                wr_ptr <= wr_ptr + {{ADDR_W-1{1'b0}}, 1'b1};
            end

            // ---- Window column tracking: gated by freeze so the window
            //      holds at the current output position (x=0) until the FSM
            //      sweeps it right during compute. ----
            if (i_pixel_vld && !i_win_freeze) begin
                cur_x  <= cur_x + 16'd1;

                // When past the left-edge padding phase, track window position
                if (cur_x >= 16'd2) begin
                    win_x <= cur_x - 16'd2;
                end else begin
                    win_x <= 16'd0;  // Left-edge zero padding region
                end
            end

            // ---- Window advance: shift 3×3 matrix right ----
            if (i_win_advance && !i_win_freeze) begin
                // Shift columns left: col0 ← col1, col1 ← col2
                win[0][0] <= win[0][1];
                win[0][1] <= win[0][2];
                win[1][0] <= win[1][1];
                win[1][1] <= win[1][2];
                win[2][0] <= win[2][1];
                win[2][1] <= win[2][2];

                // New rightmost column: from line buffers (with zero-padding)
                // Boundary check: if rd_ptr is beyond image width, pad with zero
                if (rd_ptr < i_width[ADDR_W-1:0]) begin
                    win[0][2] <= prev2_padded;
                    win[1][2] <= prev1_padded;
                    // Bypass: when a new pixel arrives this cycle, the bank
                    // write is non-blocking so cur_bank_data_r is stale.
                    // Use i_pixel_data directly for the current row (row 2).
                    if (i_pixel_vld)
                        win[2][2] <= i_pixel_data;
                    else
                        win[2][2] <= (valid_rows >= 2'd2) ? cur_bank_data_r : {ACT_GROUP_W{1'b0}};
                end else begin
                    // Right-edge padding
                    win[0][2] <= {ACT_GROUP_W{1'b0}};
                    win[1][2] <= {ACT_GROUP_W{1'b0}};
                    win[2][2] <= {ACT_GROUP_W{1'b0}};
                end

                rd_ptr <= rd_ptr + {{ADDR_W-1{1'b0}}, 1'b1};
            end

            // ---- Window valid determination ----
            // Window is valid when we have 3 rows and >= 3 columns of data
            // i_row_start must unconditionally reset win_valid to prevent
            // stale cur_x from keeping win_valid high when a new row starts.
            if (i_row_start) begin
                win_valid <= 1'b0;
            end else if (valid_rows >= 2'd3 && cur_x >= 16'd2) begin
                win_valid <= 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------
    // Offset output selection (combinational)
    // Decode 0..8 → win[row][col]
    //   offset  row  col
    //     0      0    0    top-left
    //     1      0    1    top-center
    //     2      0    2    top-right
    //     3      1    0    mid-left
    //     4      1    1    mid-center
    //     5      1    2    mid-right
    //     6      2    0    bot-left
    //     7      2    1    bot-center
    //     8      2    2    bot-right
    // -------------------------------------------------------------------
    // Kernel-offset to (row, col) decode: offset / 3 = row, offset % 3 = col
    wire [3:0] offset_idx;
    assign offset_idx = i_offset_sel;

    wire [1:0] off_row_dec;
    wire [1:0] off_col_dec;
    assign off_row_dec = (offset_idx < 4'd3) ? 2'd0 :
                         (offset_idx < 4'd6) ? 2'd1 : 2'd2;
    assign off_col_dec = (offset_idx == 4'd0 || offset_idx == 4'd3 || offset_idx == 4'd6) ? 2'd0 :
                         (offset_idx == 4'd1 || offset_idx == 4'd4 || offset_idx == 4'd7) ? 2'd1 : 2'd2;

    wire [ACT_GROUP_W-1:0] selected_offset;
    assign selected_offset = win[off_row_dec][off_col_dec];

    // -------------------------------------------------------------------
    // Output assembly: replicate 128-bit offset 16× to 2048-bit bus
    // -------------------------------------------------------------------
    genvar gi;
    generate
        for (gi = 0; gi < ARRAY_ROWS; gi = gi + 1) begin : gen_repl
            assign o_act_window[gi*ACT_GROUP_W +: ACT_GROUP_W] = selected_offset;
        end
    endgenerate

    assign o_win_vld = win_valid;

    assign o_win_x = win_x;
    assign o_win_y = win_y;

    // Boundary flags for FSM awareness
    assign o_at_left_edge  = (win_x == 16'd0);
    assign o_at_right_edge = (win_x + 16'd3 >= i_width);
    assign o_at_top_edge   = (win_y == 16'd0);
    assign o_at_bottom_edge = (win_y + 16'd3 >= i_height);

endmodule
