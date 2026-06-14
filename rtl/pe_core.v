// Filename: pe_core.v
// -------------------------------------------------------------------
// PE Core — single processing element in the systolic array.
// Computes dot product of 16 INT8 activations x 16 INT8 weights,
// adds to a 32-bit accumulator, and supports double-buffered drain.
// -------------------------------------------------------------------

module pe_core #(
    parameter ACT_WIDTH  = 8,
    parameter WGT_WIDTH  = 8,
    parameter PSUM_WIDTH = 32,
    parameter NUM_MUL    = 16
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // Activation input: NUM_MUL x INT8 packed into 128 bits
    input  wire [NUM_MUL*ACT_WIDTH-1:0] i_act,

    // Weight input: NUM_MUL x INT8 packed into 128 bits
    input  wire [NUM_MUL*WGT_WIDTH-1:0] i_wgt,

    // Cascaded partial sum from the PE above in the same column
    input  wire [PSUM_WIDTH-1:0]    i_psum_casc,

    // Partial sum output to the PE below (or column output)
    output wire [PSUM_WIDTH-1:0]    o_psum_casc,

    // Control signals
    input  wire                     i_vld,       // Data valid for accumulation
    input  wire                     i_k_end,     // End of accumulation window: latch & clear
    input  wire                     i_drain_en,  // Drain shift register serially
    input  wire                     i_reduce     // GEMM spatial-reduce: combinational column sum
);

    // -----------------------------------------------------------------------
    // Product array: 16 parallel INT8 multiplications
    // -----------------------------------------------------------------------
    wire signed [15:0] products [0:NUM_MUL-1];

    genvar gi;
    generate
        for (gi = 0; gi < NUM_MUL; gi = gi + 1) begin : gen_mul
            // Both act and wgt treated as signed INT8
            assign products[gi] = $signed(i_act[gi*ACT_WIDTH +: ACT_WIDTH])
                                * $signed(i_wgt[gi*WGT_WIDTH +: WGT_WIDTH]);
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Adder tree: 16 → 8 → 4 → 2 → 1
    // -----------------------------------------------------------------------
    wire signed [19:0] tree_s1 [0:7];
    wire signed [20:0] tree_s2 [0:3];
    wire signed [21:0] tree_s3 [0:1];
    wire signed [22:0] tree_sum;

    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : gen_s1
            assign tree_s1[gi] = products[2*gi] + products[2*gi+1];
        end
        for (gi = 0; gi < 4; gi = gi + 1) begin : gen_s2
            assign tree_s2[gi] = tree_s1[2*gi] + tree_s1[2*gi+1];
        end
        for (gi = 0; gi < 2; gi = gi + 1) begin : gen_s3
            assign tree_s3[gi] = tree_s2[2*gi] + tree_s2[2*gi+1];
        end
    endgenerate

    assign tree_sum = tree_s3[0] + tree_s3[1];

    // -----------------------------------------------------------------------
    // Accumulator and shift register (double-buffer drain)
    // -----------------------------------------------------------------------
    reg [PSUM_WIDTH-1:0] psum_acc;    // Active accumulator
    reg [PSUM_WIDTH-1:0] psum_shift;  // Drain shift register

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            psum_acc   <= {PSUM_WIDTH{1'b0}};
            psum_shift <= {PSUM_WIDTH{1'b0}};
        end else begin
            // Normal accumulation: add tree result to accumulator
            if (i_vld && !i_drain_en) begin
                psum_acc <= psum_acc + {{(PSUM_WIDTH-23){tree_sum[22]}}, tree_sum};
            end

            // End of accumulation window: latch psum_acc → psum_shift, clear psum_acc
            if (i_k_end) begin
                psum_shift <= psum_acc;
                psum_acc   <= {PSUM_WIDTH{1'b0}};
            end else if (i_drain_en && !i_reduce) begin
                // Legacy drain: shift register loads from above (byte-identical
                // when i_reduce=0). In reduce mode psum_shift HOLDS its k_end
                // value; the column sum is formed combinationally below.
                psum_shift <= i_psum_casc;
            end
        end
    end

    // During drain, output the shift register value; otherwise output 0.
    // Reduce mode: add the incoming cascade so the column forms a combinational
    // adder chain — the bottom PE's output = sum of all rows' latched psum.
    assign o_psum_casc = i_drain_en
                       ? (i_reduce ? (psum_shift + i_psum_casc) : psum_shift)
                       : {PSUM_WIDTH{1'b0}};

endmodule
