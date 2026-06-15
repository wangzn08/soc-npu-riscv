// Filename: depthwise_engine.v
// -------------------------------------------------------------------
// Depthwise convolution engine (CTRL[15] dw_en).
//
// Depthwise conv does NOT reduce across input channels: output channel c
// = sum over kernel taps of act_c(tap) * wgt_c(tap). The 16x16 systolic
// array sums the 16 IC lanes per PE (cross-channel), which is wrong for
// depthwise — so this engine bypasses the array entirely with 16 channel-
// parallel INT32 MACs, fed by the SAME im2col window the array would see.
//
// Weights: a depthwise layer's weights are NUM_CH channels x KO taps. Word
// ko (NUM_CH INT8) holds every channel's weight for tap ko, so the layer is
// KO words at i_wgt_base. The engine prefetches those KO words from Wgt SRAM
// Port A (borrowed during LOAD_ROW while wgt_reader is idle) at op start.
//
// Compute: during the per-position CALC sweep the FSM presents one tap per
// cycle (i_offset = 0..KO-1) on i_act (the tap's NUM_CH activations). The
// engine accumulates act_c * wbuf[tap]_c per channel; on i_k_end it latches
// the NUM_CH INT32 sums to o_psum (held stable through DRAIN for post_process).
//
// dw_en off => engine idle and its outputs unused in npu_top => byte-identical.
// -------------------------------------------------------------------
module depthwise_engine #(
    parameter NUM_CH      = 16,
    parameter ACT_WIDTH   = 8,
    parameter PSUM_WIDTH  = 32,
    parameter KO          = 9,             // 3x3 taps
    parameter DATA_W      = NUM_CH * ACT_WIDTH,  // 128
    parameter SRAM_ADDR_W = 14
) (
    input  wire                              clk,
    input  wire                              rst_n,

    // weight prefetch (Wgt SRAM Port A, muxed in npu_top when o_wgt_active)
    input  wire                              i_start,        // cfg_start: kick KO-word prefetch
    input  wire [SRAM_ADDR_W-1:0]            i_wgt_base,
    output wire [SRAM_ADDR_W-1:0]            o_wgt_rd_addr,
    output wire                              o_wgt_rd_en,
    output wire                              o_wgt_active,   // 1 => engine owns Wgt Port A
    input  wire [DATA_W-1:0]                 i_wgt_rd_data,  // wgt_sram_doa (NUM_CH INT8)

    // compute (driven by the conv CALC sweep)
    input  wire                              i_calc_vld,     // CALC cycle (one tap presented)
    input  wire [3:0]                        i_offset,       // tap index 0..KO-1
    input  wire [DATA_W-1:0]                 i_act,          // NUM_CH activations at this tap
    input  wire                              i_k_end,        // taps done -> latch result

    output reg  [NUM_CH-1:0][PSUM_WIDTH-1:0] o_psum,         // per-channel INT32 sums (held)
    output reg                               o_vld           // pulse on latch
);

    // -------------------- weight buffer + prefetch --------------------
    reg [DATA_W-1:0] wbuf [0:KO-1];

    localparam DW_IDLE = 2'd0, DW_READ = 2'd1, DW_DONE = 2'd2;
    reg [1:0] pf_state;
    reg [3:0] rcnt, rcnt_d;
    reg       rd_en_d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pf_state <= DW_IDLE; rcnt <= 4'd0; rcnt_d <= 4'd0; rd_en_d <= 1'b0;
        end else begin
            rcnt_d  <= rcnt;
            rd_en_d <= (pf_state == DW_READ) && (rcnt < KO[3:0]);
            if (rd_en_d) wbuf[rcnt_d] <= i_wgt_rd_data;   // capture word read 1 cycle ago

            if (i_start) begin
                pf_state <= DW_READ; rcnt <= 4'd0;        // (re)start prefetch on op start
            end else case (pf_state)
                DW_READ: if (rcnt == KO[3:0]) pf_state <= DW_DONE;
                         else                 rcnt     <= rcnt + 4'd1;
                DW_DONE: ;                                // weights resident across positions
                default: pf_state <= DW_IDLE;
            endcase
        end
    end

    assign o_wgt_rd_addr = i_wgt_base + {{(SRAM_ADDR_W-4){1'b0}}, rcnt};
    assign o_wgt_rd_en   = (pf_state == DW_READ) && (rcnt < KO[3:0]);
    assign o_wgt_active  = (pf_state == DW_READ);

    // -------------------- channel-parallel accumulate --------------------
    reg [NUM_CH-1:0][PSUM_WIDTH-1:0] acc;

    wire [DATA_W-1:0] wsel = wbuf[i_offset];   // this tap's NUM_CH weights

    integer c;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (c = 0; c < NUM_CH; c = c + 1) begin
                acc[c]    <= {PSUM_WIDTH{1'b0}};
                o_psum[c] <= {PSUM_WIDTH{1'b0}};
            end
            o_vld <= 1'b0;
        end else begin
            o_vld <= 1'b0;
            if (i_calc_vld) begin
                for (c = 0; c < NUM_CH; c = c + 1) begin : gen_mac
                    reg signed [ACT_WIDTH-1:0]   a_c, w_c;
                    reg signed [2*ACT_WIDTH-1:0] prod;
                    a_c  = i_act[c*ACT_WIDTH +: ACT_WIDTH];
                    w_c  = wsel [c*ACT_WIDTH +: ACT_WIDTH];
                    prod = a_c * w_c;
                    // tap 0 starts a fresh sum for this position; later taps accumulate
                    acc[c] <= (i_offset == 4'd0)
                            ? {{(PSUM_WIDTH-2*ACT_WIDTH){prod[2*ACT_WIDTH-1]}}, prod}
                            : acc[c] + {{(PSUM_WIDTH-2*ACT_WIDTH){prod[2*ACT_WIDTH-1]}}, prod};
                end
            end
            if (i_k_end) begin
                o_psum <= acc;     // full per-channel sum (taps 0..KO-1), held through DRAIN
                o_vld  <= 1'b1;
            end
        end
    end

endmodule
