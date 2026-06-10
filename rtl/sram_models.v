// Filename: sram_models.v
// -------------------------------------------------------------------
// SRAM Models — simple dual-port synchronous BRAM macros.
// Synthesizable; infers block RAM on FPGA, SRAM macro on ASIC.
//
// Instances:
//   Act SRAM:  128b × 16384  (128 KB ping + 128 KB pong = 256 KB total)
//   Wgt SRAM:  128b × 16384  (same as Act)
//   Out SRAM:  128b ×  8192  ( 64 KB ping +  64 KB pong = 128 KB total)
//
// Port A: used for NPU read/write (FSM / DMA)
// Port B: used for DMA read/write (concurrent access)
// -------------------------------------------------------------------

// ===================================================================
// Generic simple-dual-port BRAM
// ===================================================================
module sdp_bram #(
    parameter DATA_W = 128,
    parameter ADDR_W = 14,
    parameter DEPTH  = 16384,
    parameter COMB_B = 0           // 1 = combinational read on Port B (for DMA)
) (
    input  wire                     clk,

    // Port A: read/write
    input  wire                     ena,
    input  wire                     wea,
    input  wire [ADDR_W-1:0]        addra,
    input  wire [DATA_W-1:0]        dia,
    output wire [DATA_W-1:0]        doa,

    // Port B: read/write
    input  wire                     enb,
    input  wire                     web,
    input  wire [ADDR_W-1:0]        addrb,
    input  wire [DATA_W-1:0]        dib,
    output wire [DATA_W-1:0]        dob
);

    // Memory array (zero-initialized for simulation)
    reg [DATA_W-1:0] mem [0:DEPTH-1];
    integer init_i;
    initial begin
        for (init_i = 0; init_i < DEPTH; init_i = init_i + 1)
            mem[init_i] = {DATA_W{1'b0}};
    end

    // Port A read data register (always registered)
    reg [DATA_W-1:0] doa_r;
    always @(posedge clk) begin
        if (ena) begin
            if (wea)
                mem[addra] <= dia;
            doa_r <= mem[addra];
        end
    end
    assign doa = doa_r;

    // Port B read: registered or combinational
    generate
        if (COMB_B) begin : gen_comb_b
            // Combinational read — DMA reads data in same cycle as address
            // Write still synchronous (non-blocking)
            always @(posedge clk) begin
                if (enb && web)
                    mem[addrb] <= dib;
            end
            assign dob = mem[addrb];
        end else begin : gen_reg_b
            // Registered read — standard BRAM inference
            reg [DATA_W-1:0] dob_r;
            always @(posedge clk) begin
                if (enb) begin
                    if (web)
                        mem[addrb] <= dib;
                    dob_r <= mem[addrb];
                end
            end
            assign dob = dob_r;
        end
    endgenerate

endmodule


// ===================================================================
// Act SRAM wrapper: Ping-Pong double buffer
//   Ping: address range [0      .. 8191]
//   Pong: address range [8192   .. 16383]
// ===================================================================
module act_sram_wrapper #(
    parameter DATA_W  = 128,
    parameter ADDR_W  = 14,
    parameter PING_BASE = 0,
    parameter PONG_BASE = 8192,
    parameter DEPTH   = 16384,
    parameter COMB_B  = 1            // combinational read on Port B (DMA)
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // Port A: NPU read (FSM / im2col) — read-only
    input  wire                     ena,
    input  wire [ADDR_W-1:0]        addra,  // Logical address within selected buffer
    output wire [DATA_W-1:0]        doa,

    // Port B: DMA access — read/write
    input  wire                     enb,
    input  wire                     web,
    input  wire [ADDR_W-1:0]        addrb,  // Logical address within selected buffer
    input  wire [DATA_W-1:0]        dib,
    output wire [DATA_W-1:0]        dob,

    // Buffer selection
    input  wire                     npu_ping_sel,  // 0 = Ping, 1 = Pong (for Port A)
    input  wire                     dma_ping_sel   // 0 = Ping, 1 = Pong (for Port B)
);

    // Map logical address → physical address with buffer offset
    wire [ADDR_W-1:0] phys_addra;
    wire [ADDR_W-1:0] phys_addrb;

    assign phys_addra = addra + (npu_ping_sel ? PONG_BASE : PING_BASE);
    assign phys_addrb = addrb + (dma_ping_sel ? PONG_BASE : PING_BASE);

    sdp_bram #(
        .DATA_W (DATA_W),
        .ADDR_W (ADDR_W),
        .DEPTH  (DEPTH),
        .COMB_B (COMB_B)
    ) u_bram (
        .clk   (clk),
        .ena   (ena),
        .wea   (1'b0),           // NPU reads only
        .addra (phys_addra),
        .dia   ({DATA_W{1'b0}}),
        .doa   (doa),
        .enb   (enb),
        .web   (web),
        .addrb (phys_addrb),
        .dib   (dib),
        .dob   (dob)
    );

endmodule


// ===================================================================
// Wgt SRAM wrapper: Ping-Pong double buffer (structurally identical to Act)
//   Ping: [0..8191], Pong: [8192..16383]
// ===================================================================
module wgt_sram_wrapper #(
    parameter DATA_W  = 128,
    parameter ADDR_W  = 14,
    parameter PING_BASE = 0,
    parameter PONG_BASE = 8192,
    parameter DEPTH   = 16384,
    parameter COMB_B  = 1
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     ena,
    input  wire [ADDR_W-1:0]        addra,
    output wire [DATA_W-1:0]        doa,

    input  wire                     enb,
    input  wire                     web,
    input  wire [ADDR_W-1:0]        addrb,
    input  wire [DATA_W-1:0]        dib,
    output wire [DATA_W-1:0]        dob,

    input  wire                     npu_ping_sel,
    input  wire                     dma_ping_sel
);

    wire [ADDR_W-1:0] phys_addra = addra + (npu_ping_sel ? PONG_BASE : PING_BASE);
    wire [ADDR_W-1:0] phys_addrb = addrb + (dma_ping_sel ? PONG_BASE : PING_BASE);

    sdp_bram #(
        .DATA_W (DATA_W),
        .ADDR_W (ADDR_W),
        .DEPTH  (DEPTH),
        .COMB_B (COMB_B)
    ) u_bram (
        .clk   (clk),
        .ena   (ena),
        .wea   (1'b0),
        .addra (phys_addra),
        .dia   ({DATA_W{1'b0}}),
        .doa   (doa),
        .enb   (enb),
        .web   (web),
        .addrb (phys_addrb),
        .dib   (dib),
        .dob   (dob)
    );

endmodule


// ===================================================================
// Out / Psum SRAM wrapper: Ping-Pong double buffer
//   Each buffer: 64 KB = 65536 bytes → 128-bit width → 4096 deep
//   Ping: [0..4095], Pong: [4096..8191]
// ===================================================================
module out_sram_wrapper #(
    parameter DATA_W  = 128,
    parameter ADDR_W  = 13,
    parameter PING_BASE = 0,
    parameter PONG_BASE = 4096,
    parameter DEPTH   = 8192,
    parameter COMB_B  = 1
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // Port A: NPU write (post-processing output)
    input  wire                     ena,
    input  wire                     wea,
    input  wire [ADDR_W-1:0]        addra,
    input  wire [DATA_W-1:0]        dia,
    output wire [DATA_W-1:0]        doa,

    // Port B: DMA read/write
    input  wire                     enb,
    input  wire                     web,
    input  wire [ADDR_W-1:0]        addrb,
    input  wire [DATA_W-1:0]        dib,
    output wire [DATA_W-1:0]        dob,

    input  wire                     npu_ping_sel,
    input  wire                     dma_ping_sel
);

    wire [ADDR_W-1:0] phys_addra = addra + (npu_ping_sel ? PONG_BASE : PING_BASE);
    wire [ADDR_W-1:0] phys_addrb = addrb + (dma_ping_sel ? PONG_BASE : PING_BASE);

    sdp_bram #(
        .DATA_W (DATA_W),
        .ADDR_W (ADDR_W),
        .DEPTH  (DEPTH),
        .COMB_B (COMB_B)
    ) u_bram (
        .clk   (clk),
        .ena   (ena),
        .wea   (wea),
        .addra (phys_addra),
        .dia   (dia),
        .doa   (doa),
        .enb   (enb),
        .web   (web),
        .addrb (phys_addrb),
        .dib   (dib),
        .dob   (dob)
    );

endmodule
