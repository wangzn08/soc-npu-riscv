// Filename: transpose_engine.v
// -------------------------------------------------------------------
// On-chip transpose engine (Conv6 -> FC1 hardware reorder, decision L).
// Transposes ONE OC-pass (16 channels x n_pos positions) of a conv layer's
// tile-major Out SRAM output into the channel-major FC-input layout, writing
// it to Act SRAM.  Mirrors sram_copy (decision J) but adds a byte transpose.
//
//   Out SRAM (tile-major): word p (p=0..n_pos-1) = the 16 channels at position
//     p (16 bytes, byte ch_in = channel ch_in).  src base = i_src_base.
//   Act SRAM (channel-major): the flattened vector for this pass, element
//     s = ch_in*n_pos + pos, packed 16 bytes/word -> n_pos words at i_dst_base.
//
// Two phases, n_pos cycles each (one Out word read / one Act word write per
// cycle, Out Port B read is COMBINATIONAL like sram_copy):
//   LOAD  : read Out[src+p], scatter its 16 bytes into M[ch_in*n_pos + p].
//   DRAIN : assemble Act word w from M[w*16 +: 16 bytes], write Act[dst+w].
//
// MAX_NPOS is a hardware capacity knob (like ICG_BUF): the transpose buffer M
// holds 16*MAX_NPOS bytes, so n_pos <= MAX_NPOS is handled in one pass
// (multi-Act-word channels, n_pos>16, are handled by the sequential drain).
// n_pos > MAX_NPOS falls back to CPU (firmware) — not exercised by MNIST.
//
// 片上转置引擎（Conv6->FC1 硬件 reorder，决策 L）：把一个 OC-pass 的 tile-major
// 输出转置成 channel-major 的 FC 输入，写 Act SRAM。与 sram_copy 同构，多了字节转置。
// -------------------------------------------------------------------
module transpose_engine #(
    parameter ADDR_W   = 14,
    parameter DATA_W   = 128,
    parameter MAX_NPOS = 16            // transpose-buffer capacity (positions)
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                i_trig,        // 1-cycle pulse: start one pass
    input  wire [ADDR_W-1:0]   i_src_base,    // Out SRAM word base (this pass)
    input  wire [ADDR_W-1:0]   i_dst_base,    // Act SRAM word base (= pass*n_pos)
    input  wire [15:0]         i_npos,        // positions this pass (<= MAX_NPOS)
    // Out SRAM Port B read (combinational)
    output wire [ADDR_W-1:0]   o_out_rd_addr,
    output wire                o_out_rd_en,
    input  wire [DATA_W-1:0]   i_out_rd_data,
    // Act SRAM Port B write
    output wire [ADDR_W-1:0]   o_act_wr_addr,
    output wire                o_act_wr_en,
    output wire [DATA_W-1:0]   o_act_wr_data,
    output wire                o_busy,
    output reg                 o_done         // set on completion, cleared on i_trig
);
    localparam BYTES = 16 * MAX_NPOS;       // transpose-buffer byte count

    // Transpose buffer: M[s], s = ch_in*n_pos + pos
    reg [7:0] M [0:BYTES-1];

    localparam [1:0] S_IDLE = 2'd0, S_LOAD = 2'd1, S_DRAIN = 2'd2;
    reg [1:0]        state;
    reg [15:0]       cnt;          // word index within the current phase
    reg [15:0]       npos_q;
    reg [ADDR_W-1:0] src_q, dst_q;

    integer ci;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state  <= S_IDLE;
            cnt    <= 16'd0;
            npos_q <= 16'd0;
            src_q  <= {ADDR_W{1'b0}};
            dst_q  <= {ADDR_W{1'b0}};
            o_done <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (i_trig) begin
                        state  <= S_LOAD;
                        cnt    <= 16'd0;
                        npos_q <= i_npos;
                        src_q  <= i_src_base;
                        dst_q  <= i_dst_base;
                        o_done <= 1'b0;
                    end
                end
                // LOAD: read Out[src+cnt] (= position cnt), scatter into M.
                S_LOAD: begin
                    // M[ch_in*npos + cnt] <= OutWord byte ch_in
                    for (ci = 0; ci < 16; ci = ci + 1)
                        M[ci*npos_q + cnt] <= i_out_rd_data[ci*8 +: 8];
                    if (cnt + 16'd1 == npos_q) begin
                        cnt   <= 16'd0;
                        state <= S_DRAIN;
                    end else begin
                        cnt <= cnt + 16'd1;
                    end
                end
                // DRAIN: write Act[dst+cnt] = M[cnt*16 +: 16 bytes].
                S_DRAIN: begin
                    if (cnt + 16'd1 == npos_q) begin
                        cnt    <= 16'd0;
                        state  <= S_IDLE;
                        o_done <= 1'b1;
                    end else begin
                        cnt <= cnt + 16'd1;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    // ---- Out Port B read (LOAD phase) ----
    assign o_out_rd_addr = src_q + cnt[ADDR_W-1:0];
    assign o_out_rd_en   = (state == S_LOAD);

    // ---- Act Port B write (DRAIN phase): assemble word cnt from M ----
    wire [15:0] dbase = cnt << 4;   // cnt*16
    genvar b;
    generate
        for (b = 0; b < 16; b = b + 1) begin : gen_drain
            assign o_act_wr_data[b*8 +: 8] = M[dbase + b];
        end
    endgenerate
    assign o_act_wr_addr = dst_q + cnt[ADDR_W-1:0];
    assign o_act_wr_en   = (state == S_DRAIN);

    assign o_busy = (state != S_IDLE);
endmodule
