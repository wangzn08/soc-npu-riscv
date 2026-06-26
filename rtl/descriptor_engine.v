// -------------------------------------------------------------------
// Hardware descriptor queue engine, v1 skeleton.
//
// V1 command format is fixed at 16 x 32-bit words. The engine fetches one
// descriptor as four 128-bit AXI beats, executes it, then advances linearly.
// This skeleton supports NOP and STOP_IRQ; later tasks add DMA/NPU ops.
// -------------------------------------------------------------------

module descriptor_engine #(
    parameter ADDR_W      = 32,
    parameter SRAM_ADDR_W = 14
) (
    input  wire             clk,
    input  wire             rst_n,

    input  wire [31:0]      i_desc_base_lo,
    input  wire [31:0]      i_desc_base_hi,
    input  wire [15:0]      i_desc_count,
    input  wire             i_desc_start,
    input  wire             i_desc_abort,
    input  wire             i_desc_irq_en,
    input  wire             i_desc_clear_done,
    input  wire             i_global_idle,

    output reg              o_busy,
    output reg              o_done,
    output reg              o_err,
    output reg              o_aborted,
    output reg [15:0]       o_pc,
    output reg [7:0]        o_err_code,
    output wire             o_irq,

    output reg              m_axi_arvalid,
    input  wire             m_axi_arready,
    output reg [ADDR_W-1:0] m_axi_araddr,
    output reg [7:0]        m_axi_arlen,
    output wire [2:0]       m_axi_arsize,
    output wire [1:0]       m_axi_arburst,
    input  wire             m_axi_rvalid,
    output reg              m_axi_rready,
    input  wire [127:0]     m_axi_rdata,
    input  wire [1:0]       m_axi_rresp,
    input  wire             m_axi_rlast
);

    localparam OP_NOP      = 8'h00;
    localparam OP_STOP_IRQ = 8'h08;
    localparam VERSION     = 8'h01;

    localparam ERR_NONE          = 8'd0;
    localparam ERR_BAD_VERSION   = 8'd1;
    localparam ERR_BAD_OPCODE    = 8'd2;
    localparam ERR_BAD_COUNT     = 8'd4;
    localparam ERR_BUSY_AT_START = 8'd7;
    localparam ERR_AXI_DESC_READ = 8'd8;

    localparam S_IDLE     = 4'd0;
    localparam S_FETCH_AR = 4'd1;
    localparam S_FETCH_R  = 4'd2;
    localparam S_DECODE   = 4'd3;
    localparam S_ADVANCE  = 4'd4;
    localparam S_DONE     = 4'd5;
    localparam S_ERROR    = 4'd6;
    localparam S_ABORT    = 4'd7;

    reg [3:0] state;
    reg [1:0] beat_idx;
    reg [31:0] desc_w [0:15];
    integer i;

    wire [7:0] desc_op      = desc_w[0][7:0];
    wire [7:0] desc_version = desc_w[0][15:8];
    wire [31:0] desc_addr   = i_desc_base_lo + {14'd0, o_pc, 6'b0};

    assign m_axi_arsize  = 3'd4; // 16 bytes per beat
    assign m_axi_arburst = 2'b01; // INCR
    assign o_irq = o_done & i_desc_irq_en;

    task set_error;
        input [7:0] code;
        begin
            o_err      <= 1'b1;
            o_err_code <= code;
            o_busy     <= 1'b0;
            state      <= S_ERROR;
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            o_busy <= 1'b0;
            o_done <= 1'b0;
            o_err <= 1'b0;
            o_aborted <= 1'b0;
            o_pc <= 16'd0;
            o_err_code <= ERR_NONE;
            m_axi_arvalid <= 1'b0;
            m_axi_araddr <= {ADDR_W{1'b0}};
            m_axi_arlen <= 8'd0;
            m_axi_rready <= 1'b0;
            beat_idx <= 2'd0;
            for (i = 0; i < 16; i = i + 1)
                desc_w[i] <= 32'd0;
        end else begin
            if (i_desc_clear_done) begin
                o_done <= 1'b0;
                if (!o_busy) begin
                    o_err <= 1'b0;
                    o_aborted <= 1'b0;
                    o_err_code <= ERR_NONE;
                end
            end

            if (i_desc_abort && o_busy) begin
                m_axi_arvalid <= 1'b0;
                m_axi_rready <= 1'b0;
                o_busy <= 1'b0;
                o_aborted <= 1'b1;
                state <= S_ABORT;
            end else begin
                case (state)
                S_IDLE: begin
                    m_axi_arvalid <= 1'b0;
                    m_axi_rready <= 1'b0;
                    if (i_desc_start) begin
                        o_done <= 1'b0;
                        o_err <= 1'b0;
                        o_aborted <= 1'b0;
                        o_err_code <= ERR_NONE;
                        o_pc <= 16'd0;
                        if (i_desc_count == 16'd0) begin
                            set_error(ERR_BAD_COUNT);
                        end else if (!i_global_idle) begin
                            set_error(ERR_BUSY_AT_START);
                        end else begin
                            o_busy <= 1'b1;
                            state <= S_FETCH_AR;
                        end
                    end
                end

                S_FETCH_AR: begin
                    m_axi_arvalid <= 1'b1;
                    m_axi_araddr <= desc_addr[ADDR_W-1:0];
                    m_axi_arlen <= 8'd3;
                    if (m_axi_arvalid && m_axi_arready) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready <= 1'b1;
                        beat_idx <= 2'd0;
                        state <= S_FETCH_R;
                    end
                end

                S_FETCH_R: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        if (m_axi_rresp != 2'b00) begin
                            m_axi_rready <= 1'b0;
                            set_error(ERR_AXI_DESC_READ);
                        end else begin
                            desc_w[{beat_idx, 2'b00} + 0] <= m_axi_rdata[31:0];
                            desc_w[{beat_idx, 2'b00} + 1] <= m_axi_rdata[63:32];
                            desc_w[{beat_idx, 2'b00} + 2] <= m_axi_rdata[95:64];
                            desc_w[{beat_idx, 2'b00} + 3] <= m_axi_rdata[127:96];
                            if (m_axi_rlast) begin
                                m_axi_rready <= 1'b0;
                                state <= S_DECODE;
                            end else begin
                                beat_idx <= beat_idx + 2'd1;
                            end
                        end
                    end
                end

                S_DECODE: begin
                    if (desc_version != VERSION) begin
                        set_error(ERR_BAD_VERSION);
                    end else if (desc_op == OP_NOP) begin
                        state <= S_ADVANCE;
                    end else if (desc_op == OP_STOP_IRQ) begin
                        o_busy <= 1'b0;
                        o_done <= 1'b1;
                        state <= S_DONE;
                    end else begin
                        set_error(ERR_BAD_OPCODE);
                    end
                end

                S_ADVANCE: begin
                    if (o_pc + 16'd1 >= i_desc_count) begin
                        o_busy <= 1'b0;
                        o_done <= 1'b1;
                        state <= S_DONE;
                    end else begin
                        o_pc <= o_pc + 16'd1;
                        state <= S_FETCH_AR;
                    end
                end

                S_DONE: begin
                    if (i_desc_start) begin
                        o_done <= 1'b0;
                        o_err <= 1'b0;
                        o_aborted <= 1'b0;
                        o_err_code <= ERR_NONE;
                        o_pc <= 16'd0;
                        if (i_desc_count == 16'd0) begin
                            set_error(ERR_BAD_COUNT);
                        end else if (!i_global_idle) begin
                            set_error(ERR_BUSY_AT_START);
                        end else begin
                            o_busy <= 1'b1;
                            state <= S_FETCH_AR;
                        end
                    end
                end

                S_ERROR: begin
                    if (i_desc_start) begin
                        o_done <= 1'b0;
                        o_err <= 1'b0;
                        o_aborted <= 1'b0;
                        o_err_code <= ERR_NONE;
                        o_pc <= 16'd0;
                        if (i_desc_count == 16'd0) begin
                            set_error(ERR_BAD_COUNT);
                        end else if (!i_global_idle) begin
                            set_error(ERR_BUSY_AT_START);
                        end else begin
                            o_busy <= 1'b1;
                            state <= S_FETCH_AR;
                        end
                    end
                end

                S_ABORT: begin
                    if (i_desc_start) begin
                        o_done <= 1'b0;
                        o_err <= 1'b0;
                        o_aborted <= 1'b0;
                        o_err_code <= ERR_NONE;
                        o_pc <= 16'd0;
                        if (i_desc_count == 16'd0) begin
                            set_error(ERR_BAD_COUNT);
                        end else if (!i_global_idle) begin
                            set_error(ERR_BUSY_AT_START);
                        end else begin
                            o_busy <= 1'b1;
                            state <= S_FETCH_AR;
                        end
                    end
                end

                default: state <= S_IDLE;
                endcase
            end
        end
    end

endmodule
