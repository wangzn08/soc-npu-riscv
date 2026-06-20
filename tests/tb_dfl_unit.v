// tests/tb_dfl_unit.v
// Self-checking directed TB for dfl_unit.v.
// Parses tests/dfl_vectors.mem: NWORDS, 256 EXP(Q1.15), 16 Wk(Q8.8),
// NWORDS input words(128-bit), NWORDS golden(Q8.8). Loads LUT/Wk into the DUT
// via its load ports, DMAs nothing (drives Act SRAM model directly), triggers,
// then compares packed output words (4 distances each) to golden.
`timescale 1ns/1ps
module tb_dfl_unit;
    localparam ADDR_W = 14, DATA_W = 128;

    reg clk = 0, rst_n = 0, trig = 0;
    reg  [DATA_W-1:0] mem [0:1023];           // Act SRAM model (Port B comb read)
    wire [ADDR_W-1:0] addr;
    wire en, we;
    wire [DATA_W-1:0] wdata;
    reg  [DATA_W-1:0] rdata;
    wire busy, done;

    reg        wload_en, eload_en;
    reg [3:0]  wload_idx;
    reg [7:0]  eload_idx;
    reg [15:0] wload_val, eload_val;

    integer nwords, i, kk, errors;
    reg [15:0] exp_lut [0:255];
    reg [15:0] wk      [0:15];
    reg [15:0] golden  [0:255];

    dfl_unit #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) dut (
        .clk(clk), .rst_n(rst_n), .i_trig(trig),
        .i_src_base(14'd0), .i_dst_base(14'd512), .i_cnt(nwords[15:0]),
        .i_wload_en(wload_en), .i_wload_idx(wload_idx), .i_wload_val(wload_val),
        .i_eload_en(eload_en), .i_eload_idx(eload_idx), .i_eload_val(eload_val),
        .o_addr(addr), .o_en(en), .o_we(we), .o_wdata(wdata), .i_rdata(rdata),
        .o_busy(busy), .o_done(done));

    always #5 clk = ~clk;
    always @(*) rdata = mem[addr];            // COMB_B=1
    always @(posedge clk) if (en && we) mem[addr] <= wdata;

    integer fd, code;
    reg [127:0] win;

    initial begin
        wload_en = 0; eload_en = 0; wload_idx = 0; eload_idx = 0;
        wload_val = 0; eload_val = 0;

        fd = $fopen("tests/dfl_vectors.mem", "r");
        if (fd == 0) begin $display("TB_DFL_UNIT FAIL: cannot open mem"); $finish; end
        code = $fscanf(fd, "%d\n", nwords);
        for (i = 0; i < 256; i = i + 1) code = $fscanf(fd, "%h\n", exp_lut[i]);
        for (i = 0; i < 16;  i = i + 1) code = $fscanf(fd, "%h\n", wk[i]);
        for (i = 0; i < nwords; i = i + 1) begin
            code = $fscanf(fd, "%h\n", win);
            mem[i] = win;
        end
        for (i = 0; i < nwords; i = i + 1) code = $fscanf(fd, "%h\n", golden[i]);
        $fclose(fd);

        rst_n = 0; #23 rst_n = 1;
        @(posedge clk);
        // load EXP_LUT
        for (i = 0; i < 256; i = i + 1) begin
            eload_en <= 1; eload_idx <= i[7:0]; eload_val <= exp_lut[i]; @(posedge clk);
        end
        eload_en <= 0;
        // load W_k
        for (i = 0; i < 16; i = i + 1) begin
            wload_en <= 1; wload_idx <= i[3:0]; wload_val <= wk[i]; @(posedge clk);
        end
        wload_en <= 0;
        @(posedge clk);

        trig <= 1; @(posedge clk); trig <= 0;
        wait (done);
        @(posedge clk);

        errors = 0;
        for (i = 0; i < nwords; i = i + 4) begin
            for (kk = 0; kk < 4; kk = kk + 1) begin
                if (mem[512 + (i/4)][16*kk +: 16] !== golden[i + kk]) begin
                    errors = errors + 1;
                    $display("MISMATCH word=%0d coord=%0d got=%h exp=%h",
                             i + kk, kk, mem[512 + (i/4)][16*kk +: 16], golden[i + kk]);
                end
            end
        end
        if (errors == 0) $display("TB_DFL_UNIT PASS");
        else             $display("TB_DFL_UNIT FAIL errors=%0d", errors);
        $finish;
    end

    initial begin #200000 $display("TB_DFL_UNIT TIMEOUT"); $finish; end
endmodule
