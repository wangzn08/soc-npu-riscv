// Filename: systolic_16x16.v
// -------------------------------------------------------------------
// Systolic_16x16 — top-level 16x16 systolic array.
// Instantiates a 4x4 grid of gp_4x4 modules, forming 16 rows × 16 cols
// of PE cores (256 PEs total, 4096 MACs/cycle peak).
//
// Activation mapping (Output Stationary):
//   i_act[2047:0] = 16 groups × 128 bits.  Each 128-bit group carries
//   16 INT8 input-channel values for one kernel-offset position.
//   Group g feeds PE row g (all 16 PEs in that row receive the same data).
//
// Weight mapping:
//   i_wgt[2047:0] = 16 groups × 128 bits.  Each 128-bit group carries
//   16 INT8 weights for a distinct output channel.  Group oc feeds
//   PE column oc (all 16 PEs in that column receive the same data).
//
// Cascade: partial sums flow down each column, through 4 PE rows per GP
// and 4 GP rows, accumulating results for the same output channel across
// input-channel tiles.
// -------------------------------------------------------------------

module systolic_16x16 #(
    parameter ARRAY_ROWS       = 16,
    parameter ARRAY_COLS       = 16,
    parameter GP_SIZE          = 4,
    parameter GP_GRID_ROWS     = ARRAY_ROWS / GP_SIZE,   // 4
    parameter GP_GRID_COLS     = ARRAY_COLS / GP_SIZE,   // 4
    parameter ACT_WIDTH        = 8,
    parameter WGT_WIDTH        = 8,
    parameter PSUM_WIDTH       = 32,
    parameter NUM_MUL          = 16,
    parameter ACT_GROUP_W      = NUM_MUL * ACT_WIDTH,     // 128
    parameter WGT_GROUP_W      = NUM_MUL * WGT_WIDTH,     // 128
    parameter ACT_BUS_W        = ARRAY_ROWS * ACT_GROUP_W, // 2048
    parameter WGT_BUS_W        = ARRAY_COLS * WGT_GROUP_W  // 2048
) (
    input  wire                              clk,
    input  wire                              rst_n,

    // Activation bus: 16 groups of 128-bit activation data
    input  wire [ACT_BUS_W-1:0]              i_act,

    // Weight bus: 16 groups of 128-bit weight data
    input  wire [WGT_BUS_W-1:0]              i_wgt,

    // Column partial-sum outputs (bottom of each column)
    output wire [ARRAY_COLS-1:0][PSUM_WIDTH-1:0] o_psum_col,

    // Control — broadcast to the entire array
    input  wire                              i_vld,
    input  wire                              i_k_end,
    input  wire                              i_drain_en
);

    // -------------------------------------------------------------------
    // GP-level cascade wires
    // gp_casc[gp_col][gp_row] carries the 4-column cascade from GP[gp_row][gp_col]
    // gp_casc_link[gp_col][gp_row] is between GP rows gp_row-1 and gp_row
    // -------------------------------------------------------------------
    wire [GP_GRID_COLS-1:0][GP_GRID_ROWS:0][GP_SIZE-1:0][PSUM_WIDTH-1:0] gp_casc_chain;

    // Top-of-array cascade inputs are zero
    genvar gc, gr;
    generate
        for (gc = 0; gc < GP_GRID_COLS; gc = gc + 1) begin : gen_gc_top
            for (gr = 0; gr < GP_SIZE; gr = gr + 1) begin : gen_gc_inner
                assign gp_casc_chain[gc][0][gr] = {PSUM_WIDTH{1'b0}};
            end
        end
    endgenerate

    // -------------------------------------------------------------------
    // Instantiate GP grid: GP_GRID_ROWS × GP_GRID_COLS
    // -------------------------------------------------------------------
    generate
        for (gr = 0; gr < GP_GRID_ROWS; gr = gr + 1) begin : gen_gp_row
            for (gc = 0; gc < GP_GRID_COLS; gc = gc + 1) begin : gen_gp_col

                // Activation slice for this GP row:
                //   GP row gr handles PE rows gr*4 .. gr*4+3
                localparam ACT_HI = (gr*GP_SIZE + GP_SIZE) * ACT_GROUP_W - 1;
                localparam ACT_LO = (gr*GP_SIZE) * ACT_GROUP_W;
                wire [GP_SIZE*ACT_GROUP_W-1:0] gp_act;
                assign gp_act = i_act[ACT_HI:ACT_LO];

                // Weight slice for this GP column:
                //   GP col gc handles PE cols gc*4 .. gc*4+3
                localparam WGT_HI = (gc*GP_SIZE + GP_SIZE) * WGT_GROUP_W - 1;
                localparam WGT_LO = (gc*GP_SIZE) * WGT_GROUP_W;
                wire [GP_SIZE*WGT_GROUP_W-1:0] gp_wgt;
                assign gp_wgt = i_wgt[WGT_HI:WGT_LO];

                gp_4x4 #(
                    .ACT_WIDTH    (ACT_WIDTH),
                    .WGT_WIDTH    (WGT_WIDTH),
                    .PSUM_WIDTH   (PSUM_WIDTH),
                    .NUM_MUL      (NUM_MUL),
                    .GP_ROWS      (GP_SIZE),
                    .GP_COLS      (GP_SIZE),
                    .ACT_GROUP_W  (ACT_GROUP_W),
                    .WGT_GROUP_W  (WGT_GROUP_W)
                ) u_gp (
                    .clk          (clk),
                    .rst_n        (rst_n),
                    .i_act        (gp_act),
                    .i_wgt        (gp_wgt),
                    .i_psum_casc  (gp_casc_chain[gc][gr]),
                    .o_psum_casc  (gp_casc_chain[gc][gr+1]),
                    .i_vld        (i_vld),
                    .i_k_end      (i_k_end),
                    .i_drain_en   (i_drain_en)
                );

            end
        end
    endgenerate

    // -------------------------------------------------------------------
    // Column outputs: bottom GP row outputs drive o_psum_col
    // -------------------------------------------------------------------
    generate
        for (gc = 0; gc < GP_GRID_COLS; gc = gc + 1) begin : gen_col_out
            for (gr = 0; gr < GP_SIZE; gr = gr + 1) begin : gen_col_inner
                assign o_psum_col[gc*GP_SIZE + gr] =
                    gp_casc_chain[gc][GP_GRID_ROWS][gr];
            end
        end
    endgenerate

endmodule
