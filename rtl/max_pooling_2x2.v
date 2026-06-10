// Filename: max_pooling_2x2.v
// -------------------------------------------------------------------
// Max Pooling 2×2 — reduces spatial dimensions by 2 using 2×2 max.
// Stores one row in a line-buffer FIFO; compares the current 2×2
// window each cycle.  Output valid only on even rows & even columns.
// Boundary: odd-width columns are zero-padded on the right.
// -------------------------------------------------------------------

module max_pooling_2x2 #(
    parameter MAX_WIDTH    = 256,
    parameter ACT_WIDTH    = 8,
    parameter NUM_CH       = 16,
    parameter DATA_W       = NUM_CH * ACT_WIDTH,  // 128
    parameter ADDR_W       = 8                    // log2(256)
) (
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire [DATA_W-1:0]    i_feat,
    input  wire                 i_feat_vld,
    input  wire [15:0]          i_width,

    output wire [DATA_W-1:0]    o_pool,
    output wire                 o_pool_vld
);

    // -------------------------------------------------------------------
    // Line buffer for the previous row (simple dual-port RAM style)
    // -------------------------------------------------------------------
    reg [DATA_W-1:0] line_buf [0:MAX_WIDTH-1];

    // Column counter and row tracker
    reg [15:0]        col_cnt;
    reg               even_row;    // Current row is even-numbered (0,2,4...)

    // Previous-row pixel, read from line buffer
    reg [DATA_W-1:0]  prev_row_pixel;

    // Pipeline registers for 2×2 window
    reg [DATA_W-1:0]  cur_pixel_d1;    // Current pixel, delayed 1 cycle (left neighbor)
    reg [DATA_W-1:0]  prev_pixel_d1;   // Above-left pixel, delayed 1 cycle

    // -------------------------------------------------------------------
    // Read previous row data and write current row data
    // -------------------------------------------------------------------
    always @(posedge clk) begin
        if (i_feat_vld) begin
            // Write current pixel to line buffer for use by the next row
            line_buf[col_cnt[ADDR_W-1:0]] <= i_feat;
        end
    end

    always @(posedge clk) begin
        if (i_feat_vld) begin
            // Read previous row's corresponding column pixel (1 cycle read latency)
            prev_row_pixel <= line_buf[col_cnt[ADDR_W-1:0]];
        end
    end

    // -------------------------------------------------------------------
    // Pipeline: delay current pixel and prev_row_pixel by 1 cycle
    // to form the 2×2 window with the left neighbor
    // -------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cur_pixel_d1  <= {DATA_W{1'b0}};
            prev_pixel_d1 <= {DATA_W{1'b0}};
        end else if (i_feat_vld) begin
            cur_pixel_d1  <= i_feat;
            prev_pixel_d1 <= prev_row_pixel;
        end
    end

    // -------------------------------------------------------------------
    // 2×2 max comparison: per-channel INT8 max on 4 values
    //   win_tl = prev_pixel_d1  (top-left,  i.e. above-left)
    //   win_tr = prev_row_pixel  (top-right, i.e. above)
    //   win_bl = cur_pixel_d1   (bot-left,  i.e. left)
    //   win_br = i_feat         (bot-right, i.e. current)
    // -------------------------------------------------------------------
    genvar gi;
    wire [ACT_WIDTH-1:0] max_stage1 [0:NUM_CH-1][0:1];
    wire [ACT_WIDTH-1:0] max_stage2 [0:NUM_CH-1];
    wire [ACT_WIDTH-1:0] max_result  [0:NUM_CH-1];

    generate
        for (gi = 0; gi < NUM_CH; gi = gi + 1) begin : gen_max
            // Stage 1: compare pairs
            assign max_stage1[gi][0] = (prev_pixel_d1[gi*ACT_WIDTH +: ACT_WIDTH] >= i_feat[gi*ACT_WIDTH +: ACT_WIDTH])
                                      ? prev_pixel_d1[gi*ACT_WIDTH +: ACT_WIDTH]
                                      : i_feat[gi*ACT_WIDTH +: ACT_WIDTH];

            assign max_stage1[gi][1] = (cur_pixel_d1[gi*ACT_WIDTH +: ACT_WIDTH] >= prev_row_pixel[gi*ACT_WIDTH +: ACT_WIDTH])
                                      ? cur_pixel_d1[gi*ACT_WIDTH +: ACT_WIDTH]
                                      : prev_row_pixel[gi*ACT_WIDTH +: ACT_WIDTH];

            // Stage 2: max of the two pairwise maxes
            assign max_result[gi] = (max_stage1[gi][0] >= max_stage1[gi][1])
                                   ? max_stage1[gi][0]
                                   : max_stage1[gi][1];

            assign o_pool[gi*ACT_WIDTH +: ACT_WIDTH] = max_result[gi];
        end
    endgenerate

    // -------------------------------------------------------------------
    // Output valid: even row && even column && data is valid
    // -------------------------------------------------------------------
    reg col_cnt_d1;
    reg even_row_d1;
    reg feat_vld_d1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col_cnt_d1  <= 1'b0;
            even_row_d1 <= 1'b0;
            feat_vld_d1 <= 1'b0;
        end else begin
            col_cnt_d1  <= col_cnt[0];   // LSB of column = even/odd
            even_row_d1 <= even_row;
            feat_vld_d1 <= i_feat_vld;
        end
    end

    // Output valid on delayed pipeline (2 cycles after input):
    //   Cycle N:   i_feat_vld, col_cnt=N → data enters
    //   Cycle N+1: line_buf read completes, prev_row_pixel valid
    //   Cycle N+2: all 4 window pixels aligned → output
    reg feat_vld_d2;
    reg col_even_d2;
    reg even_row_d2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            feat_vld_d2  <= 1'b0;
            col_even_d2  <= 1'b0;
            even_row_d2  <= 1'b0;
        end else begin
            feat_vld_d2  <= feat_vld_d1;
            col_even_d2  <= col_cnt_d1;
            even_row_d2  <= even_row_d1;
        end
    end

    assign o_pool_vld = feat_vld_d2 && even_row_d2 && col_even_d2;

    // -------------------------------------------------------------------
    // Column and row tracking
    // -------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col_cnt  <= 16'd0;
            even_row <= 1'b1;   // Row 0 is even
        end else if (i_feat_vld) begin
            if (col_cnt == i_width - 16'd1) begin
                col_cnt  <= 16'd0;
                even_row <= ~even_row;
            end else begin
                col_cnt <= col_cnt + 16'd1;
            end
        end
    end

endmodule
