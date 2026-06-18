`timescale 1ns/1ps
// Directed test for optional SiLU post-requant:
//   silu_q4.4 -> ((signed_silu * mul) >>> shift) + zp -> signed INT8 clamp.
module tb_silu_requant;
    localparam NUM_OC = 16, PSUM_WIDTH = 32;

    reg clk = 1, rst_n = 0;
    reg [NUM_OC-1:0][PSUM_WIDTH-1:0] i_psum, i_bias, i_scale_mul;
    reg [NUM_OC-1:0][5:0]            i_scale_shift;
    reg        i_psum_vld, i_relu_en, i_pool_en, i_pool_avg, i_silu_en, i_start;
    reg        i_silu_requant_en;
    reg [15:0] i_silu_requant_mul;
    reg [5:0]  i_silu_requant_shift;
    reg [7:0]  i_silu_requant_zp;
    reg        i_in_drain, i_in_post, i_row_par_en;
    reg [15:0] i_width, i_group_size;
    reg [3:0]  i_rows_per_grp;
    reg [1:0]  i_oc_tile;
    reg [7:0]  i_clip_max;

    wire [127:0]                 o_feat;
    wire                         o_feat_vld;
    wire [NUM_OC*PSUM_WIDTH-1:0] o_feat32;
    wire [1:0]                   o_pool_tile;
    wire                         o_rp_pool_done;

    integer i, errors = 0, cap_idx = 0;
    reg [7:0] cap [0:3];

    post_process_top #(.NUM_OC(NUM_OC), .MAX_WIDTH(256)) dut (
        .clk(clk), .rst_n(rst_n),
        .i_psum(i_psum), .i_psum_vld(i_psum_vld),
        .i_bias(i_bias), .i_scale_mul(i_scale_mul), .i_scale_shift(i_scale_shift),
        .i_width(i_width), .i_pool_en(i_pool_en), .i_pool_avg(i_pool_avg),
        .i_relu_en(i_relu_en), .i_silu_en(i_silu_en),
        .i_silu_requant_en(i_silu_requant_en),
        .i_silu_requant_mul(i_silu_requant_mul),
        .i_silu_requant_shift(i_silu_requant_shift),
        .i_silu_requant_zp(i_silu_requant_zp),
        .i_clip_max(i_clip_max),
        .i_start(i_start), .i_in_drain(i_in_drain), .i_in_post(i_in_post),
        .i_row_par_en(i_row_par_en), .i_group_size(i_group_size),
        .i_rows_per_grp(i_rows_per_grp), .i_oc_tile(i_oc_tile),
        .o_pool_tile(o_pool_tile), .o_rp_pool_done(o_rp_pool_done),
        .o_feat(o_feat), .o_feat_vld(o_feat_vld), .o_feat32(o_feat32)
    );

    always #5 clk = ~clk;

    always @(posedge clk) if (rst_n && o_feat_vld) begin
        cap[cap_idx] = o_feat[7:0];
        cap_idx = cap_idx + 1;
    end

    task drive_psum(input integer val);
        integer c;
        begin
            for (c = 0; c < NUM_OC; c = c + 1) i_psum[c] = val;
            i_psum_vld = 1; @(posedge clk); i_psum_vld = 0;
        end
    endtask

    initial begin
        i_psum_vld = 0; i_pool_en = 0; i_pool_avg = 0; i_relu_en = 0; i_silu_en = 1; i_start = 0;
        i_silu_requant_en = 1; i_silu_requant_mul = 16'd64; i_silu_requant_shift = 6'd8; i_silu_requant_zp = 8'h80;
        i_in_drain = 0; i_in_post = 0; i_row_par_en = 0;
        i_width = 16; i_group_size = 1; i_rows_per_grp = 1; i_oc_tile = 0; i_clip_max = 8'd127;
        for (i = 0; i < NUM_OC; i = i + 1) begin
            i_bias[i] = 0; i_scale_mul[i] = 1; i_scale_shift[i] = 0;
        end

        #20 rst_n = 1; @(posedge clk);

        drive_psum(16);   // LUT = 0x0c -> 12; (12*64>>8)-128 = -125 = 0x83
        drive_psum(32);   // LUT = 0x1c -> 28; (28*64>>8)-128 = -121 = 0x87
        drive_psum(64);   // LUT = 0x3f -> 63; (63*64>>8)-128 = -113 = 0x8f
        drive_psum(127);  // LUT = 0x7f ->127; (127*64>>8)-128 = -97  = 0x9f
        repeat (12) @(posedge clk);

        if (cap_idx != 4) begin
            $display("FAIL cap_idx=%0d", cap_idx);
            errors = errors + 1;
        end
        if (cap[0] !== 8'h83) begin $display("FAIL cap0 got=%02h", cap[0]); errors = errors + 1; end
        if (cap[1] !== 8'h87) begin $display("FAIL cap1 got=%02h", cap[1]); errors = errors + 1; end
        if (cap[2] !== 8'h8f) begin $display("FAIL cap2 got=%02h", cap[2]); errors = errors + 1; end
        if (cap[3] !== 8'h9f) begin $display("FAIL cap3 got=%02h", cap[3]); errors = errors + 1; end

        if (errors == 0) $display("TB_SILU_REQUANT PASS");
        else $display("TB_SILU_REQUANT FAIL errors=%0d", errors);
        $finish;
    end
endmodule
