// Filename: vector_alu.v
// -------------------------------------------------------------------
// Vector ALU — per-channel element-wise addition for skip connections.
// 16 parallel 8-bit signed adders with saturation to [0, 127]
// (unsigned ReLU range).  Bypasses conv result when eltwise is disabled.
// -------------------------------------------------------------------

module vector_alu #(
    parameter NUM_CH  = 16,
    parameter DATA_W  = NUM_CH * 8   // 128
) (
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire [DATA_W-1:0]    i_conv_res,     // Convolution result (128-bit)
    input  wire [DATA_W-1:0]    i_skip_res,     // Skip-connection result (128-bit)
    input  wire                 i_eltwise_en,   // Element-wise add enable

    input  wire                 i_vld,

    output wire [DATA_W-1:0]    o_res,
    output wire                 o_vld
);

    genvar gi;
    wire [7:0] sum_raw [0:NUM_CH-1];
    wire [7:0] sum_sat [0:NUM_CH-1];

    generate
        for (gi = 0; gi < NUM_CH; gi = gi + 1) begin : gen_alu
            // 8-bit addition with sign extension for overflow detection
            wire [8:0] sum_ext;
            assign sum_ext = {1'b0, i_conv_res[gi*8 +: 8]}
                           + {1'b0, i_skip_res[gi*8 +: 8]};

            // Saturate: if sum > 127, clamp to 127 (unsigned range [0,127])
            assign sum_raw[gi] = sum_ext[7:0];
            assign sum_sat[gi] = (sum_ext > 9'd127) ? 8'd127 : sum_raw[gi];

            assign o_res[gi*8 +: 8] = i_eltwise_en ? sum_sat[gi]
                                                   : i_conv_res[gi*8 +: 8];
        end
    endgenerate

    // Pass-through valid with 1-cycle latency to align
    reg vld_d1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            vld_d1 <= 1'b0;
        else
            vld_d1 <= i_vld;
    end

    assign o_vld = vld_d1;

endmodule
