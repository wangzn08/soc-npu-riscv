// Filename: vector_alu.v
// -------------------------------------------------------------------
// Vector ALU — per-channel element-wise addition for skip connections.
// 16 parallel 8-bit adders.  Two modes (i_signed_mode):
//   0 (default, MNIST legacy): unsigned add saturated to [0,127]
//      (ReLU range), no zero-point.  Byte-identical to the original.
//   1 (YOLO C2f residual): both operands already requantized to the
//      glue scale/zero-point, so out = sat_s8(s8(conv)+s8(skip)-s8(zp))
//      saturated to the signed INT8 range [-128,127].
// Bypasses conv result when eltwise is disabled.
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
    input  wire                 i_signed_mode,  // 1=signed INT8 + zero-point (YOLO)
    input  wire [7:0]           i_elt_zp,       // glue zero-point (signed mode only)

    input  wire                 i_vld,

    output wire [DATA_W-1:0]    o_res,
    output wire                 o_vld
);

    genvar gi;
    wire [7:0] sum_unsigned [0:NUM_CH-1];
    wire [7:0] sum_signed   [0:NUM_CH-1];
    wire [7:0] sum_out       [0:NUM_CH-1];

    generate
        for (gi = 0; gi < NUM_CH; gi = gi + 1) begin : gen_alu
            // --- legacy unsigned path: [0,127] saturation, no zp ---
            wire [8:0] usum;
            assign usum = {1'b0, i_conv_res[gi*8 +: 8]}
                        + {1'b0, i_skip_res[gi*8 +: 8]};
            assign sum_unsigned[gi] = (usum > 9'd127) ? 8'd127 : usum[7:0];

            // --- signed path: s8(conv)+s8(skip)-s8(zp), clamp [-128,127] ---
            wire signed [10:0] ssum;
            assign ssum = $signed({{3{i_conv_res[gi*8+7]}}, i_conv_res[gi*8 +: 8]})
                        + $signed({{3{i_skip_res[gi*8+7]}}, i_skip_res[gi*8 +: 8]})
                        - $signed({{3{i_elt_zp[7]}},        i_elt_zp[7:0]});
            assign sum_signed[gi] = (ssum >  11'sd127) ? 8'h7F :
                                    (ssum < -11'sd128) ? 8'h80 :
                                                         ssum[7:0];

            assign sum_out[gi] = i_signed_mode ? sum_signed[gi] : sum_unsigned[gi];

            assign o_res[gi*8 +: 8] = i_eltwise_en ? sum_out[gi]
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
