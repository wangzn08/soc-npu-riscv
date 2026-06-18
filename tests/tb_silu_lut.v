`timescale 1ns/1ps
// ===================================================================
// Directed test for the YOLO SiLU LUT path in post_process_top.
//
// With bias=0, scale_mul=1, scale_shift=0, stage-2 equals i_psum.
// i_silu_en=1 should bypass ReLU/clip and map signed INT8 Q4.4 input
// through rtl/silu_lut_q4_4.hex:
//   x     -64  -16    0   16   32   64  127
//   silu   ff   fc   00   0c   1c   3f   7f
// ===================================================================
module tb_silu_lut;
    localparam NUM_OC = 16, PSUM_WIDTH = 32;

    reg clk = 1, rst_n = 0;
    reg [NUM_OC-1:0][PSUM_WIDTH-1:0] i_psum, i_bias, i_scale_mul;
    reg [NUM_OC-1:0][5:0]            i_scale_shift;
    reg        i_psum_vld, i_relu_en, i_pool_en, i_pool_avg, i_silu_en, i_start;
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
    reg [7:0] cap  [0:15];
    reg [7:0] gold [0:6];

    post_process_top #(.NUM_OC(NUM_OC), .MAX_WIDTH(256)) dut (
        .clk(clk), .rst_n(rst_n),
        .i_psum(i_psum), .i_psum_vld(i_psum_vld),
        .i_bias(i_bias), .i_scale_mul(i_scale_mul), .i_scale_shift(i_scale_shift),
        .i_width(i_width), .i_pool_en(i_pool_en), .i_pool_avg(i_pool_avg),
        .i_relu_en(i_relu_en), .i_silu_en(i_silu_en),
        .i_silu_requant_en(1'b0), .i_silu_requant_mul(16'd0),
        .i_silu_requant_shift(6'd0), .i_silu_requant_zp(8'd0),
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
        i_in_drain = 0; i_in_post = 0; i_row_par_en = 0;
        i_width = 16; i_group_size = 1; i_rows_per_grp = 1; i_oc_tile = 0; i_clip_max = 8'd127;
        for (i = 0; i < NUM_OC; i = i + 1) begin
            i_bias[i] = 0; i_scale_mul[i] = 1; i_scale_shift[i] = 0;
        end
        gold[0]=8'hff; gold[1]=8'hfc; gold[2]=8'h00; gold[3]=8'h0c;
        gold[4]=8'h1c; gold[5]=8'h3f; gold[6]=8'h7f;

        #20 rst_n = 1; @(posedge clk);

        drive_psum(-64);
        drive_psum(-16);
        drive_psum(0);
        drive_psum(16);
        drive_psum(32);
        drive_psum(64);
        drive_psum(127);
        repeat (12) @(posedge clk);

        for (i = 0; i < 7; i = i + 1)
            if (cap[i] !== gold[i]) begin
                errors = errors + 1;
                $display("FAIL idx=%0d exp=%02h got=%02h", i, gold[i], cap[i]);
            end
        if (errors == 0 && cap_idx == 7) $display("TB_SILU_LUT PASS");
        else $display("TB_SILU_LUT FAIL errors=%0d cap_idx=%0d", errors, cap_idx);
        $finish;
    end
endmodule
