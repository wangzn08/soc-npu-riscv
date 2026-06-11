// Filename: wgt_reader.v
// -------------------------------------------------------------------
// Weight Reader — pre-fetches INT8 weights from Wgt SRAM into a local
// buffer, then outputs 2048-bit weight vectors synchronized with the
// Im2Col kernel-offset sequence.
//
// Since the SRAM has one read port but the array needs 16×128 bits per
// cycle, the reader pre-fetches 9 kernel-offset × 16 output-channel
// weight groups during the line-load phase and holds them in registers.
// During CALC_MAC the correct offset group is selected combinationally.
//
// Weight SRAM organization (NHWC mapping, innermost→outermost):
//   KW → KH → IC_group → OC
// Address = oc * IC_GROUPS * KH * KW + ic_group * KH * KW + kh * KW + kw
// -------------------------------------------------------------------

module wgt_reader #(
    parameter NUM_OC         = 16,
    parameter NUM_IC_GROUP   = 16,    // IC per group
    parameter KH             = 3,
    parameter KW             = 3,
    parameter KERNEL_OFFSETS = KH * KW,
    parameter ACT_WIDTH      = 8,
    parameter WGT_GROUP_W    = NUM_IC_GROUP * ACT_WIDTH,  // 128
    parameter WGT_BUS_W      = NUM_OC * WGT_GROUP_W,      // 2048
    parameter SRAM_ADDR_W    = 14,
    parameter IC_GROUPS_MAX  = 64    // max IC groups (1024 ch / 16)
) (
    input  wire                         clk,
    input  wire                         rst_n,

    // === SRAM read port (port B of Wgt SRAM) ===
    output wire [SRAM_ADDR_W-1:0]       o_sram_addr,
    output wire                         o_sram_en,
    input  wire [WGT_GROUP_W-1:0]       i_sram_data,

    // === Control from FSM ===
    input  wire                         i_start_prefetch,  // Pulse to begin pre-fetch
    input  wire [9:0]                   i_oc_base,         // Output channel base
    input  wire [9:0]                   i_ic_group,        // Input channel group index
    input  wire [15:0]                  i_ic_groups_total,  // IC/16 for the layer
    input  wire [SRAM_ADDR_W-1:0]       i_wgt_base,         // Wgt SRAM base (resident weights)
    output wire                         o_prefetch_done,   // Pre-fetch complete

    // === Kernel offset selection (during CALC_MAC) ===
    input  wire [3:0]                   i_wgt_offset,      // 0..8 kernel offset selector

    // === Weight output to systolic array ===
    output wire [WGT_BUS_W-1:0]         o_wgt,
    output wire                         o_wgt_vld
);

    // -------------------------------------------------------------------
    // Pre-fetch buffer: 9 offsets × 16 OC groups × 128 bits
    // wgt_buf[ko][oc] = 128-bit weight group
    // -------------------------------------------------------------------
    reg [WGT_GROUP_W-1:0] wgt_buf [0:KERNEL_OFFSETS-1][0:NUM_OC-1];

    // -------------------------------------------------------------------
    // Pre-fetch state machine
    // -------------------------------------------------------------------
    localparam PF_IDLE      = 3'd0;
    localparam PF_READING   = 3'd1;
    localparam PF_WAIT_LAST = 3'd2;
    localparam PF_DONE      = 3'd3;

    reg [2:0] pf_state;
    reg [3:0] pf_ko;       // Current kernel offset being pre-fetched (0..8)
    reg [4:0] pf_oc;       // Current OC being pre-fetched (0..15)
    reg [3:0] pf_ko_d;     // Delayed ko (matches SRAM 1-cycle read latency)
    reg [4:0] pf_oc_d;     // Delayed oc
    reg       pf_done_r;

    // Address stride between consecutive OCs for the same (ic_group, kh, kw)
    wire [SRAM_ADDR_W-1:0] oc_stride;
    assign oc_stride = {{(SRAM_ADDR_W-10){1'b0}}, i_ic_groups_total} * KH * KW;

    // Base address for current pre-fetch:
    //   oc_base * IC_GROUPS * KH * KW + ic_group * KH * KW + kh * KW + kw
    wire [SRAM_ADDR_W-1:0] addr_oc_component;
    wire [SRAM_ADDR_W-1:0] addr_ic_component;
    wire [SRAM_ADDR_W-1:0] addr_kh;
    wire [SRAM_ADDR_W-1:0] addr_kw;

    assign addr_oc_component = i_oc_base * i_ic_groups_total * KH * KW;
    assign addr_ic_component = i_ic_group * KH * KW;
    assign addr_kh = (pf_ko / KW) * KW;  // pf_ko / 3 * 3
    assign addr_kw = pf_ko % KW;          // pf_ko % 3

    wire [SRAM_ADDR_W-1:0] sram_rd_addr;
    assign sram_rd_addr = i_wgt_base
                        + addr_oc_component + addr_ic_component + addr_kh + addr_kw
                        + pf_oc * oc_stride;

    assign o_sram_addr = sram_rd_addr;
    assign o_sram_en   = (pf_state == PF_READING);

    // -------------------------------------------------------------------
    // Pre-fetch FSM
    // -------------------------------------------------------------------
    reg pf_reading_d;  // Delayed SRAM-enable gate for valid write

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pf_state     <= PF_IDLE;
            pf_ko        <= 4'd0;
            pf_oc        <= 5'd0;
            pf_ko_d      <= 4'd0;
            pf_oc_d      <= 5'd0;
            pf_done_r    <= 1'b0;
            pf_reading_d <= 1'b0;
        end else begin
            // Delayed versions to match SRAM 1-cycle read latency
            pf_ko_d <= pf_ko;
            pf_oc_d <= pf_oc;
            pf_reading_d <= (pf_state == PF_READING);

            case (pf_state)
                PF_IDLE: begin
                    if (i_start_prefetch) begin
                        pf_state <= PF_READING;
                        pf_ko    <= 4'd0;
                        pf_oc    <= 5'd0;
                    end
                    pf_done_r <= 1'b0;
                end

                PF_READING: begin
                    // Write SRAM data from PREVIOUS cycle's address (1-cycle SRAM latency)
                    // Only write when pf_reading_d is asserted, which gates out the
                    // stale SRAM data present on the first cycle after PF_DONE→PF_READING.
                    if (pf_reading_d)
                        wgt_buf[pf_ko_d][pf_oc_d] <= i_sram_data;

                    // After last address presented, transition to PF_WAIT_LAST
                    // to capture the final SRAM read data (1-cycle latency).
                    if (pf_ko == (KERNEL_OFFSETS - 1) && pf_oc == 5'd15) begin
                        pf_state <= PF_WAIT_LAST;
                    end else begin
                        // Advance OC, then kernel offset
                        if (pf_oc == 5'd15) begin
                            pf_oc <= 5'd0;
                            pf_ko <= pf_ko + 4'd1;
                        end else begin
                            pf_oc <= pf_oc + 5'd1;
                        end
                    end
                end

                PF_WAIT_LAST: begin
                    // Capture the final SRAM read data (1 cycle after last address)
                    wgt_buf[pf_ko_d][pf_oc_d] <= i_sram_data;
                    pf_state  <= PF_DONE;
                    pf_done_r <= 1'b1;
                end

                PF_DONE: begin
                    pf_done_r <= 1'b0;
                    if (i_start_prefetch) begin
                        // Restart pre-fetch for new IC group
                        pf_state <= PF_READING;
                        pf_ko    <= 4'd0;
                        pf_oc    <= 5'd0;
                    end else begin
                        pf_state <= PF_IDLE;
                    end
                end

                default: pf_state <= PF_IDLE;
            endcase
        end
    end

    assign o_prefetch_done = pf_done_r;

    // -------------------------------------------------------------------
    // Output selection: mux the correct 16 OC groups from wgt_buf[ko]
    // -------------------------------------------------------------------
    wire [3:0] ko_sel;
    assign ko_sel = i_wgt_offset;

    genvar gi;
    generate
        for (gi = 0; gi < NUM_OC; gi = gi + 1) begin : gen_wgt_out
            assign o_wgt[gi*WGT_GROUP_W +: WGT_GROUP_W] = wgt_buf[ko_sel][gi];
        end
    endgenerate

    // Weight valid when pre-fetch is done and offset is within range
    assign o_wgt_vld = (pf_state == PF_DONE) && (i_wgt_offset < KERNEL_OFFSETS);

endmodule
