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
    parameter ADDR_W    = 8,                   // log2(MAX_WIDTH)
    parameter NUM_TILES = 4,                   // decision P: per-OC-tile pooler state
    parameter TILE_W    = 2                    // log2(NUM_TILES)
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              i_start,      // pulse: reset pooling phase at op start
    input  wire [DATA_W-1:0] i_feat,
    input  wire              i_feat_vld,
    input  wire [15:0]       i_width,
    input  wire [TILE_W-1:0] i_tile,       // decision P: active OC-tile (0 ⇒ legacy)
    output reg  [DATA_W-1:0] o_pool,
    output reg               o_pool_vld
);

    // Decision P: the 2×2 pooler keeps cross-ROW state (previous-row line buffer +
    // column/row phase + neighbour regs). In oc_single the OC-inner loop interleaves
    // OC tiles between rows, so this state is replicated PER OC-tile and indexed by
    // i_tile. i_tile is constant for the whole of a tile's drain/replay (S_POST is
    // held by rp_pool_done), so state[i_tile] always sees the right tile. i_tile==0
    // (oc_single off) touches only tile 0 ⇒ byte-identical to the single-state pooler.

    // One-row line buffer (previous row) per tile, asynchronous read.
    reg [DATA_W-1:0] line_buf [0:NUM_TILES-1][0:MAX_WIDTH-1];

    reg [15:0] col        [0:NUM_TILES-1];  // current column, 0..width-1
    reg        row_odd    [0:NUM_TILES-1];  // current row index is odd (1,3,5,...)

    // Registered neighbours, captured on the previous valid cycle.
    reg [DATA_W-1:0] cur_left   [0:NUM_TILES-1];  // i_feat at col c-1, same row (bottom-left)
    reg [DATA_W-1:0] above_left [0:NUM_TILES-1];  // line_buf at col c-1, prev row (top-left)

    wire [15:0] col_t = col[i_tile];

    // Previous-row pixel at the current column (top-right), async read (this tile).
    wire [DATA_W-1:0] above    = line_buf[i_tile][col_t[ADDR_W-1:0]];
    wire [DATA_W-1:0] cl_t     = cur_left[i_tile];
    wire [DATA_W-1:0] al_t     = above_left[i_tile];
    wire              col_odd  = col_t[0];
    wire              last_col = (col_t == i_width - 16'd1);
    wire              emit     = i_feat_vld && row_odd[i_tile] && col_odd;

    // Per-channel 2x2 max of {above_left, above, cur_left, i_feat}.
    wire [DATA_W-1:0] win_max;
    genvar gi;
    generate
        for (gi = 0; gi < NUM_CH; gi = gi + 1) begin : gen_max
            wire [ACT_WIDTH-1:0] tl = al_t  [gi*ACT_WIDTH +: ACT_WIDTH];
            wire [ACT_WIDTH-1:0] tr = above [gi*ACT_WIDTH +: ACT_WIDTH];
            wire [ACT_WIDTH-1:0] bl = cl_t  [gi*ACT_WIDTH +: ACT_WIDTH];
            wire [ACT_WIDTH-1:0] br = i_feat[gi*ACT_WIDTH +: ACT_WIDTH];
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

    // Column / row phase and neighbour capture — per OC-tile (decision P).
    integer ti;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (ti = 0; ti < NUM_TILES; ti = ti + 1) begin
                col[ti]        <= 16'd0;
                row_odd[ti]    <= 1'b0;
                cur_left[ti]   <= {DATA_W{1'b0}};
                above_left[ti] <= {DATA_W{1'b0}};
            end
        end else if (i_start) begin
            for (ti = 0; ti < NUM_TILES; ti = ti + 1) begin
                col[ti]        <= 16'd0;
                row_odd[ti]    <= 1'b0;
                cur_left[ti]   <= {DATA_W{1'b0}};
                above_left[ti] <= {DATA_W{1'b0}};
            end
        end else if (i_feat_vld) begin
            line_buf[i_tile][col_t[ADDR_W-1:0]] <= i_feat;  // store this tile's row
            cur_left[i_tile]   <= i_feat;                    // left neighbour, next col
            above_left[i_tile] <= above;                     // top-left neighbour, next col
            if (last_col) begin
                col[i_tile]     <= 16'd0;
                row_odd[i_tile] <= ~row_odd[i_tile];
            end else begin
                col[i_tile] <= col_t + 16'd1;
            end
        end
    end

endmodule
