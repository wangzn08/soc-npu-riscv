// Filename: vector_alu.v
// -------------------------------------------------------------------
// Vector ALU — per-channel element-wise addition for skip connections.
// 16 parallel 8-bit adders.  Two modes (i_signed_mode):
//   0 (default, MNIST legacy): unsigned add saturated to [0,127]
//      (ReLU range), no zero-point.  Byte-identical to the original.
//   1 (YOLO C2f residual): out = sat_s8(s8(conv) + skip_term), where
//      skip_term depends on i_elt_ratio_en:
//        ratio_en=0 (legacy signed): skip_term = s8(skip) - s8(zp)
//        ratio_en=1 (YOLO C2f true residual): the bottleneck input (skip) is
//          at its OWN scale and must be rescaled to the mcv2 glue scale, so
//          skip_term = ((s8(skip) - s8(zp))*ratio_mul + round) >>> ratio_sh
//          (round-half: + (1<<(ratio_sh-1))).  Bit-identical to the CPU
//          add_word() residual in firmware/yolo_c2f.c.
// Bypasses conv result when eltwise is disabled.
// MNIST byte-identical: it uses i_signed_mode=0 (unsigned path), so the signed
// ratio extension never affects the MLP baseline.
// -------------------------------------------------------------------

module vector_alu #(
    parameter NUM_CH       = 16,
    parameter DATA_W       = NUM_CH * 8,   // 128
    parameter RATIO_MUL_W  = 17            // ratio_mul up to 131071 (YOLO ratios ~0.6..1.1 * 2^16, max ~71211)
) (
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire [DATA_W-1:0]       i_conv_res,     // Convolution result (128-bit)
    input  wire [DATA_W-1:0]       i_skip_res,     // Skip-connection result (128-bit)
    input  wire                    i_eltwise_en,   // Element-wise add enable
    input  wire                    i_signed_mode,  // 1=signed INT8 + zero-point (YOLO)
    input  wire [7:0]              i_elt_zp,       // glue zero-point (signed mode only)
    input  wire                    i_elt_ratio_en, // 1=apply ratio rescale to skip (C2f true residual)
    input  wire [RATIO_MUL_W-1:0]  i_elt_ratio_mul,// skip rescale multiplier (unsigned)
    input  wire [5:0]              i_elt_ratio_shift, // skip rescale right-shift (round-half)

    input  wire                    i_vld,

    output wire [DATA_W-1:0]       o_res,
    output wire                    o_vld
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

            // --- signed path: s8(conv) + skip_term, clamp [-128,127] ---
            // skip_minus_zp = s8(skip) - s8(zp)  (range [-255,255], 10b signed)
            wire signed [9:0] skip_minus_zp =
                  $signed({{2{i_skip_res[gi*8+7]}}, i_skip_res[gi*8 +: 8]})
                - $signed({{2{i_elt_zp[7]}},        i_elt_zp[7:0]});
            // ratio rescale: (skip_minus_zp * ratio_mul + round) >>> ratio_sh
            wire signed [RATIO_MUL_W:0] rmul = $signed({1'b0, i_elt_ratio_mul}); // unsigned -> positive
            wire signed [9+RATIO_MUL_W+1:0] rprod = skip_minus_zp * rmul;
            wire signed [9+RATIO_MUL_W+1:0] rround =
                  (i_elt_ratio_shift == 6'd0) ? {(9+RATIO_MUL_W+2){1'b0}}
                                              : ($signed({{(9+RATIO_MUL_W+1){1'b0}},1'b1}) <<< (i_elt_ratio_shift - 6'd1));
            wire signed [9+RATIO_MUL_W+1:0] rscaled = (rprod + rround) >>> i_elt_ratio_shift;
            // skip term: rescaled (ratio_en) or plain skip - zp (legacy signed)
            wire signed [10:0] skip_term =
                  i_elt_ratio_en ? rscaled[10:0]
                                 : {skip_minus_zp[9], skip_minus_zp};
            wire signed [11:0] ssum =
                  $signed({{4{i_conv_res[gi*8+7]}}, i_conv_res[gi*8 +: 8]})
                + {{1{skip_term[10]}}, skip_term};
            assign sum_signed[gi] = (ssum >  12'sd127) ? 8'h7F :
                                    (ssum < -12'sd128) ? 8'h80 :
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
