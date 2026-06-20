// Filename: dfl_unit.v
// -------------------------------------------------------------------
// Streaming YOLOv8n DFL (Distribution Focal Loss) expectation engine.
//
// Follows the shared Act-SRAM Port-B engine convention (see upsample2x.v):
// triggered by a pulse, reads source words from Act SRAM Port B (COMB_B=1),
// writes results back to Act SRAM Port B, raises o_done when finished.
//
// Each input 128-bit word = 16 INT8 logits z[0..15] = the 16 DFL bins of ONE
// coordinate at one anchor (NPU tile-major: ch_group == coord). Per word:
//   zmax  = max_k z[k]
//   idx_k = zmax - z[k]                  (0..255)
//   e_k   = EXP_LUT[idx_k]               (Q1.15 unsigned, exp(0)=32768)
//   Sden  = sum_k e_k                    (unsigned)
//   Snum  = sum_k e_k * W_k              (W_k Q8.8 signed, loadable)
//   distq  = Snum / Sden                  (Q8.8; the Q1.15 factors cancel)
//
// 4 consecutive input words (one anchor's 4 coords) are packed into ONE output
// word: lanes 0..3 = 4 signed Q8.8 distances. i_cnt must be a multiple of 4.
// -------------------------------------------------------------------
module dfl_unit #(
    parameter ADDR_W = 14,
    parameter DATA_W = 128
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               i_trig,
    input  wire [ADDR_W-1:0]  i_src_base,
    input  wire [ADDR_W-1:0]  i_dst_base,
    input  wire [15:0]        i_cnt,        // input word count (anchors*4)
    // load ports (W_k constants, EXP_LUT); active high, one entry/cycle
    input  wire               i_wload_en,
    input  wire [3:0]         i_wload_idx,
    input  wire [15:0]        i_wload_val,
    input  wire               i_eload_en,
    input  wire [7:0]         i_eload_idx,
    input  wire [15:0]        i_eload_val,
    // Act SRAM Port B
    output wire [ADDR_W-1:0]  o_addr,
    output wire               o_en,
    output wire               o_we,
    output wire [DATA_W-1:0]  o_wdata,
    input  wire [DATA_W-1:0]  i_rdata,
    output wire               o_busy,
    output reg                o_done
);
    reg signed [15:0] wk   [0:15];   // Q8.8
    reg        [15:0] elut [0:255];  // Q1.15 unsigned

    always @(posedge clk) begin
        if (i_wload_en) wk[i_wload_idx]   <= i_wload_val;
        if (i_eload_en) elut[i_eload_idx] <= i_eload_val;
    end

    localparam S_IDLE  = 3'd0;
    localparam S_READ  = 3'd1;
    localparam S_ACC   = 3'd2;
    localparam S_DIV   = 3'd3;
    localparam S_PLACE = 3'd4;
    localparam S_WRITE = 3'd5;

    reg [2:0]         state;
    reg [15:0]        widx, cnt_q;
    reg [ADDR_W-1:0]  src_q, dst_q;
    reg signed [7:0]  z [0:15];
    reg signed [7:0]  zmax;
    reg [4:0]         k;
    reg [31:0]        sden;
    reg signed [47:0] snum;
    reg signed [15:0] distq;
    reg [DATA_W-1:0]  outw;
    reg [1:0]         coord;

    // sequential divider state (|snum| / sden -> Q8.8)
    reg [5:0]         dstep;
    reg [47:0]        rem, quo, snum_abs;
    reg               neg;

    wire [7:0] eidx = zmax - z[k];   // 0..255 (zmax >= z[k])

    assign o_busy  = (state != S_IDLE);
    assign o_en    = (state == S_READ) || (state == S_WRITE);
    assign o_we    = (state == S_WRITE);
    assign o_addr  = (state == S_WRITE) ? (dst_q + (widx >> 2))
                                        : (src_q + widx[ADDR_W-1:0]);
    assign o_wdata = outw;

    integer m;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; o_done <= 1'b0; widx <= 16'd0; coord <= 2'd0;
            outw <= {DATA_W{1'b0}};
        end else begin
            case (state)
            S_IDLE: begin
                o_done <= 1'b0;
                if (i_trig) begin
                    src_q <= i_src_base; dst_q <= i_dst_base; cnt_q <= i_cnt;
                    widx <= 16'd0; coord <= 2'd0; outw <= {DATA_W{1'b0}};
                    state <= S_READ;
                end
            end

            S_READ: begin
                // COMB_B: i_rdata valid this cycle for o_addr = src + widx
                for (m = 0; m < 16; m = m + 1)
                    z[m] <= $signed(i_rdata[8*m +: 8]);
                // combinational max over the freshly-read word
                begin : maxcalc
                    reg signed [7:0] mx; integer j;
                    mx = $signed(i_rdata[7:0]);
                    for (j = 1; j < 16; j = j + 1)
                        if ($signed(i_rdata[8*j +: 8]) > mx)
                            mx = $signed(i_rdata[8*j +: 8]);
                    zmax <= mx;
                end
                k <= 5'd0; sden <= 32'd0; snum <= 48'sd0;
                state <= S_ACC;
            end

            S_ACC: begin
                if (k < 5'd16) begin
                    sden <= sden + {16'd0, elut[eidx]};
                    snum <= snum + $signed({1'b0, elut[eidx]}) * wk[k];
                    k <= k + 5'd1;
                end else begin
                    neg      <= (snum < 0);
                    snum_abs <= (snum < 0) ? -snum : snum;
                    rem <= 48'd0; quo <= 48'd0; dstep <= 6'd0;
                    state <= S_DIV;
                end
            end

            S_DIV: begin
                if (dstep < 6'd48) begin
                    // rem = rem<<1 | bit; if rem>=sden subtract & set quo bit
                    if (((rem << 1) | ((snum_abs >> (47 - dstep)) & 48'd1)) >= {28'd0, sden}) begin
                        rem <= ((rem << 1) | ((snum_abs >> (47 - dstep)) & 48'd1)) - {28'd0, sden};
                        quo <= (quo << 1) | 48'd1;
                    end else begin
                        rem <= (rem << 1) | ((snum_abs >> (47 - dstep)) & 48'd1);
                        quo <= quo << 1;
                    end
                    dstep <= dstep + 6'd1;
                end else begin
                    distq <= neg ? -$signed(quo[15:0]) : $signed(quo[15:0]);
                    state <= S_PLACE;
                end
            end

            S_PLACE: begin
                outw[16*coord +: 16] <= distq;
                if (coord == 2'd3 || (widx + 16'd1 == cnt_q))
                    state <= S_WRITE;
                else begin
                    coord <= coord + 2'd1;
                    widx  <= widx + 16'd1;
                    state <= S_READ;
                end
            end

            S_WRITE: begin
                if (widx + 16'd1 == cnt_q) begin
                    o_done <= 1'b1;
                    state  <= S_IDLE;
                end else begin
                    coord <= 2'd0;
                    widx  <= widx + 16'd1;
                    outw  <= {DATA_W{1'b0}};
                    state <= S_READ;
                end
            end

            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
