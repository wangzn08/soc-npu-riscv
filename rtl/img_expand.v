// Filename: img_expand.v
// -------------------------------------------------------------------
// Image expand engine: reads packed bytes from Act SRAM (Port B) and writes
// one zero-extended 128-bit word per byte back to Act SRAM (Port B). Used for
// Conv1 input: 16 packed pixels/word -> 16 tile-major words (pixel in ch0,
// ch1..15 = 0). Act Port B is a single port, so per source word it spends 1
// READ cycle (o_we=0, latch the packed word) then 16 WRITE cycles.
//   i_n_out = number of OUTPUT words (= pixels); src words = ceil(n_out/16).
// Outputs are COMBINATIONAL on the registered state; act Port B read is
// combinational (COMB_B=1) so i_rdata in the READ cycle = Act[src+sw]; the
// SRAM latches writes synchronously on the edge.
//
// 图像扩展引擎：Act 打包字(16像素) → 16 个 tile-major 词(ch0=像素, ch1..15=0)。
// 单口 Port B：每源字 读1拍(锁存) + 写16拍。组合输出。
// -------------------------------------------------------------------
module img_expand #(
    parameter ADDR_W = 14,
    parameter DATA_W = 128
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                i_trig,       // 1-cycle pulse: start
    input  wire [ADDR_W-1:0]   i_src_base,   // Act scratch word base (packed)
    input  wire [ADDR_W-1:0]   i_dst_base,   // Act output word base (expanded)
    input  wire [15:0]         i_n_out,      // output word count (= pixel count)
    // Act SRAM Port B (shared read/write)
    output wire [ADDR_W-1:0]   o_addr,
    output wire                o_en,
    output wire                o_we,
    output wire [DATA_W-1:0]   o_wdata,
    input  wire [DATA_W-1:0]   i_rdata,      // act_sram_dob (combinational)
    output wire                o_busy,
    output reg                 o_done
);
    localparam S_IDLE = 2'd0, S_READ = 2'd1, S_WRITE = 2'd2;
    reg [1:0]        state;
    reg [ADDR_W-1:0] src_q, dst_q;
    reg [15:0]       n_q;          // output words target
    reg [15:0]       out_cnt;      // output words written so far
    reg [15:0]       sw;           // current source word index
    reg [3:0]        b;            // byte within the latched word (0..15)
    reg [DATA_W-1:0] word_q;       // latched packed source word

    assign o_busy = (state != S_IDLE);

    // Combinational Port-B drive
    wire [7:0] sel_byte = word_q[{b, 3'b000} +: 8];
    assign o_en    = (state == S_READ) || (state == S_WRITE);
    assign o_we    = (state == S_WRITE);
    assign o_addr  = (state == S_WRITE)
                   ? dst_q + (sw[ADDR_W-1:0] << 4) + {{(ADDR_W-4){1'b0}}, b}
                   : src_q + sw[ADDR_W-1:0];
    assign o_wdata = {{(DATA_W-8){1'b0}}, sel_byte};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; src_q <= 0; dst_q <= 0; n_q <= 0;
            out_cnt <= 0; sw <= 0; b <= 0; word_q <= 0; o_done <= 0;
        end else begin
            o_done <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (i_trig) begin
                        src_q <= i_src_base; dst_q <= i_dst_base; n_q <= i_n_out;
                        out_cnt <= 16'd0; sw <= 16'd0; b <= 4'd0;
                        state <= S_READ;
                    end
                end
                S_READ: begin
                    // o_addr = src+sw this cycle (o_we=0); latch the combinational read.
                    word_q <= i_rdata;
                    b      <= 4'd0;
                    state  <= S_WRITE;
                end
                S_WRITE: begin
                    // This cycle the SRAM commits a write of word_q[b] to dst+sw*16+b.
                    out_cnt <= out_cnt + 16'd1;
                    if (out_cnt + 16'd1 == n_q) begin
                        state <= S_IDLE; o_done <= 1'b1;
                    end else if (b == 4'd15) begin
                        sw    <= sw + 16'd1;
                        state <= S_READ;
                    end else begin
                        b <= b + 4'd1;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
