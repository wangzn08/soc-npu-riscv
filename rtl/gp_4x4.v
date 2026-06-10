// Filename: gp_4x4.v
// -------------------------------------------------------------------
// GP_4x4 — a 4x4 group of PE cores.
// Broadcasts 4 activation groups to 4 rows of PEs and 4 weight groups
// to 4 columns of PEs.  Cascades partial sums vertically per column.
// Pure connectivity — no state machine.
// -------------------------------------------------------------------

module gp_4x4 #(
    parameter ACT_WIDTH      = 8,
    parameter WGT_WIDTH      = 8,
    parameter PSUM_WIDTH     = 32,
    parameter NUM_MUL        = 16,
    parameter GP_ROWS        = 4,
    parameter GP_COLS        = 4,
    parameter ACT_GROUP_W    = NUM_MUL * ACT_WIDTH,   // 128
    parameter WGT_GROUP_W    = NUM_MUL * WGT_WIDTH    // 128
) (
    input  wire                                    clk,
    input  wire                                    rst_n,

    // Activation input: GP_ROWS groups of ACT_GROUP_W bits each
    input  wire [GP_ROWS*ACT_GROUP_W-1:0]          i_act,

    // Weight input: GP_COLS groups of WGT_GROUP_W bits each
    input  wire [GP_COLS*WGT_GROUP_W-1:0]          i_wgt,

    // Cascaded partial-sum inputs (one per column from GP above)
    input  wire [GP_COLS-1:0][PSUM_WIDTH-1:0]      i_psum_casc,

    // Cascaded partial-sum outputs (one per column to GP below)
    output wire [GP_COLS-1:0][PSUM_WIDTH-1:0]      o_psum_casc,

    // Control — broadcast to all PEs
    input  wire                                    i_vld,
    input  wire                                    i_k_end,
    input  wire                                    i_drain_en
);

    // -------------------------------------------------------------------
    // Inter-PE psum cascade wires (vertical within each column)
    // psum_col_wire[c][r] connects PE[r][c].o_psum_casc to PE[r+1][c].i_psum_casc
    // psum_col_wire[c][0] driven by PE[0][c].o_psum_casc
    // psum_col_wire[c][GP_ROWS] would be the output of the last PE
    // -------------------------------------------------------------------
    wire [GP_COLS-1:0][GP_ROWS-1:0][PSUM_WIDTH-1:0] psum_casc_link;

    genvar r, c;
    generate
        for (r = 0; r < GP_ROWS; r = r + 1) begin : gen_row
            for (c = 0; c < GP_COLS; c = c + 1) begin : gen_col

                // Determine cascade input for this PE
                wire [PSUM_WIDTH-1:0] pe_psum_in;
                if (r == 0) begin : gen_top_casc
                    assign pe_psum_in = i_psum_casc[c];
                end else begin : gen_mid_casc
                    assign pe_psum_in = psum_casc_link[c][r-1];
                end

                pe_core #(
                    .ACT_WIDTH  (ACT_WIDTH),
                    .WGT_WIDTH  (WGT_WIDTH),
                    .PSUM_WIDTH (PSUM_WIDTH),
                    .NUM_MUL    (NUM_MUL)
                ) u_pe (
                    .clk         (clk),
                    .rst_n       (rst_n),
                    .i_act       (i_act[r*ACT_GROUP_W +: ACT_GROUP_W]),
                    .i_wgt       (i_wgt[c*WGT_GROUP_W +: WGT_GROUP_W]),
                    .i_psum_casc (pe_psum_in),
                    .o_psum_casc (psum_casc_link[c][r]),
                    .i_vld       (i_vld),
                    .i_k_end     (i_k_end),
                    .i_drain_en  (i_drain_en)
                );

            end
        end
    endgenerate

    // Output cascade: bottom-most PE in each column drives o_psum_casc
    generate
        for (c = 0; c < GP_COLS; c = c + 1) begin : gen_out_casc
            assign o_psum_casc[c] = psum_casc_link[c][GP_ROWS-1];
        end
    endgenerate

endmodule
