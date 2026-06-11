// Filename: max_pooling_2x2.v
// -------------------------------------------------------------------
// Max Pooling 2x2, stride 2.  Streams a feature map row-major (one
// pixel per i_feat_vld) and emits one pooled pixel per 2x2 window.
//
// Window uses the current pixel as the bottom-right corner, so a valid
// stride-2 output exists exactly when the current pixel sits on an ODD
// row AND an ODD column (0-indexed):
//
//     window(r,c) = { (r-1,c-1) (r-1,c)
//                     (r,  c-1) (r,  c) }   emitted when r odd, c odd
//
// A one-row line buffer holds the previous row.  Row 0 never emits, so
// the line buffer is always fully written before it is read -> no stale
// / uninitialized reads.  o_pool and o_pool_vld are registered together
// so data and valid stay aligned.
//
// i_start pulses at the beginning of every NPU operation and resets the
// row/column phase so successive layers do not inherit each other's
// parity or neighbour registers.
//
// NOTE: assumes even feature-map width (28/16/8 for this deployment).
// -------------------------------------------------------------------

module max_pooling_2x2 #(
    parameter MAX_WIDTH = 256,
    parameter ACT_WIDTH = 8,
    parameter NUM_CH    = 16,
    parameter DATA_W    = NUM_CH * ACT_WIDTH,  // 128
    parameter ADDR_W    = 8                    // log2(MAX_WIDTH)
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              i_start,      // pulse: reset pooling phase at op start
    input  wire [DATA_W-1:0] i_feat,
    input  wire              i_feat_vld,
    input  wire [15:0]       i_width,
    output reg  [DATA_W-1:0] o_pool,
    output reg               o_pool_vld
);

    // One-row line buffer (previous row), asynchronous read.
    reg [DATA_W-1:0] line_buf [0:MAX_WIDTH-1];

    reg [15:0] col;       // current column, 0..width-1
    reg        row_odd;   // current row index is odd (1,3,5,...)

    // Registered neighbours, captured on the previous valid cycle.
    reg [DATA_W-1:0] cur_left;    // i_feat at column c-1, same row   (bottom-left)
    reg [DATA_W-1:0] above_left;  // line_buf at column c-1, prev row (top-left)

    // Previous-row pixel at the current column (top-right), async read.
    wire [DATA_W-1:0] above    = line_buf[col[ADDR_W-1:0]];
    wire              col_odd  = col[0];
    wire              last_col = (col == i_width - 16'd1);
    wire              emit     = i_feat_vld && row_odd && col_odd;

    // Per-channel 2x2 max of {above_left, above, cur_left, i_feat}.
    wire [DATA_W-1:0] win_max;
    genvar gi;
    generate
        for (gi = 0; gi < NUM_CH; gi = gi + 1) begin : gen_max
            wire [ACT_WIDTH-1:0] tl = above_left[gi*ACT_WIDTH +: ACT_WIDTH];
            wire [ACT_WIDTH-1:0] tr = above     [gi*ACT_WIDTH +: ACT_WIDTH];
            wire [ACT_WIDTH-1:0] bl = cur_left  [gi*ACT_WIDTH +: ACT_WIDTH];
            wire [ACT_WIDTH-1:0] br = i_feat     [gi*ACT_WIDTH +: ACT_WIDTH];
            wire [ACT_WIDTH-1:0] mtop = (tl >= tr) ? tl : tr;
            wire [ACT_WIDTH-1:0] mbot = (bl >= br) ? bl : br;
            assign win_max[gi*ACT_WIDTH +: ACT_WIDTH] = (mtop >= mbot) ? mtop : mbot;
        end
    endgenerate

    // Registered output: data and valid aligned.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_pool     <= {DATA_W{1'b0}};
            o_pool_vld <= 1'b0;
        end else begin
            o_pool     <= win_max;
            o_pool_vld <= emit;
        end
    end

    // Column / row phase and neighbour capture.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col        <= 16'd0;
            row_odd    <= 1'b0;
            cur_left   <= {DATA_W{1'b0}};
            above_left <= {DATA_W{1'b0}};
        end else if (i_start) begin
            col        <= 16'd0;
            row_odd    <= 1'b0;
            cur_left   <= {DATA_W{1'b0}};
            above_left <= {DATA_W{1'b0}};
        end else if (i_feat_vld) begin
            line_buf[col[ADDR_W-1:0]] <= i_feat;  // store current row for the next row
            cur_left   <= i_feat;                  // left neighbour for next column
            above_left <= above;                   // top-left neighbour for next column
            if (last_col) begin
                col     <= 16'd0;
                row_odd <= ~row_odd;
            end else begin
                col <= col + 16'd1;
            end
        end
    end

endmodule
