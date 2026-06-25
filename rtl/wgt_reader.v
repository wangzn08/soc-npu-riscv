// Filename: wgt_reader.v
// -------------------------------------------------------------------
// Weight Reader — pre-fetches INT8 weights from Wgt SRAM into a local
// buffer, then outputs 2048-bit weight vectors synchronized with the
// Im2Col kernel-offset sequence.
//
// Since the SRAM has one read port but the array needs 16×128 bits per
// cycle, the reader pre-fetches i_kernel_offsets (= KH·KW: 9 for 3×3 conv,
// 1 for 1×1 GEMM/FC) × 16 output-channel weight groups during the line-load
// phase and holds them in registers.  During CALC_MAC the correct offset
// group is selected combinationally.
//
// KERNEL_OFFSETS (param) = compile-time max buffer depth; i_kernel_offsets
// (port) = runtime active offset count (≤ KERNEL_OFFSETS).
//
// Weight SRAM organization (NHWC mapping, innermost→outermost):
//   KW → KH → IC_group → OC
// Address = oc * IC_GROUPS * KO + ic_group * KO + ko   (KO = i_kernel_offsets,
//           ko = kh*KW+kw collapsed to the linear offset index)
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
    parameter IC_GROUPS_MAX  = 64,   // max IC groups (1024 ch / 16)
    parameter ICG_BUF        = 16,   // IC-tiles held on-chip for per-OC-tile reuse
                                     // (8->16: matches im2col ICG_MAX so icg<=16 3x3 is resident)
    parameter MAX_OC_RESIDENT = 64,  // decision O: all-OC-tiles held on-chip (OC tiles × 16)
    parameter OC_TILES_W     = 3     // width of OC-tile counter (1..MAX_OC_RESIDENT/16)
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
    input  wire [7:0]                   i_kernel_offsets,  // runtime kh*kw (conv 9, GEMM 1)
    input  wire                         i_prefetch_all,    // 1=prefetch ALL ic_groups (reuse), 0=single i_ic_group
    input  wire [3:0]                   i_wgt_ic_sel,      // during CALC: which ic_group's weights to present (reuse)
    output wire                         o_prefetch_done,   // Pre-fetch complete

    // === Decision O: all-OC-resident (oc_single) ===
    input  wire                         i_oc_single,       // 1=prefetch ALL OC-tiles once, hold resident
    input  wire [OC_TILES_W-1:0]        i_oc_tiles_total,  // number of 16-OC tiles for the layer (1..4)
    input  wire [OC_TILES_W-1:0]        i_oc_tile_sel,     // during CALC: which OC-tile's weights to present

    // === Kernel offset selection (during CALC_MAC) ===
    input  wire [3:0]                   i_wgt_offset,      // 0..8 kernel offset selector

    // === Weight output to systolic array ===
    output wire [WGT_BUS_W-1:0]         o_wgt,
    output wire                         o_wgt_vld,

    // === GEMM-reduce 16×16 weight plane (decision M) ===
    input  wire                         i_gemm_reduce,     // 1 = reduce mode (plane prefetch owns SRAM port)
    input  wire                         i_plane_trig,      // pulse: prefetch one super-step's 256-word plane
    input  wire [3:0]                   i_superstep,       // super-step index s (IC-tile base = s*16)
    output wire [NUM_OC*NUM_OC*WGT_GROUP_W-1:0] o_wgt_plane, // plane[row r][col c] = word(oc=c, icg=s*16+r)
    output wire                         o_plane_done
);

    // -------------------------------------------------------------------
    // Pre-fetch buffer: 9 offsets × 16 OC groups × 128 bits
    // wgt_buf[ko][oc] = 128-bit weight group
    // -------------------------------------------------------------------
    // Decision O: 2nd dim sized to MAX_OC_RESIDENT so all OC-tiles can be held
    // resident in oc_single mode; legacy modes only ever touch indices 0..NUM_OC-1.
    reg [WGT_GROUP_W-1:0] wgt_buf [0:KERNEL_OFFSETS-1][0:MAX_OC_RESIDENT-1][0:ICG_BUF-1];

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
    reg [4:0] pf_icg;      // Current IC group being pre-fetched (reuse mode)
    reg [4:0] pf_icg_d;    // Delayed (matches SRAM 1-cycle latency)
    reg [OC_TILES_W-1:0] pf_oct;    // Current OC-tile being pre-fetched (oc_single)
    reg [OC_TILES_W-1:0] pf_oct_d;  // Delayed (matches SRAM 1-cycle latency)
    reg       pf_done_r;

    // oc_single behaves like prefetch_all for IC groups (hold every group resident).
    wire pf_all = i_prefetch_all || i_oc_single;

    // Address stride between consecutive OCs for the same (ic_group, ko)
    // KO = i_kernel_offsets (runtime kh*kw): conv 3x3 -> 9 (identical to old
    // compile-time KH*KW), GEMM 1x1 -> 1.
    wire [SRAM_ADDR_W-1:0] oc_stride;
    assign oc_stride = i_ic_groups_total[SRAM_ADDR_W-1:0]
                     * {{(SRAM_ADDR_W-8){1'b0}}, i_kernel_offsets};

    // Base address: oc_base*icg*KO + ic_group*KO + ko   (kh*KW+kw == pf_ko)
    wire [SRAM_ADDR_W-1:0] addr_oc_component;
    wire [SRAM_ADDR_W-1:0] addr_ic_component;

    // oc_single sweeps all OC-tiles: the effective OC base is pf_oct*16.
    wire [9:0] eff_oc_base = i_oc_single ? {{(6-OC_TILES_W){1'b0}}, pf_oct, 4'd0} : i_oc_base;
    assign addr_oc_component = {{(SRAM_ADDR_W-10){1'b0}}, eff_oc_base} * oc_stride;
    // Reuse / oc_single prefetch every ic_group (pf_icg); legacy uses single i_ic_group.
    wire [9:0] eff_icg = pf_all ? {5'd0, pf_icg} : i_ic_group;
    assign addr_ic_component = {{(SRAM_ADDR_W-10){1'b0}}, eff_icg}
                             * {{(SRAM_ADDR_W-8){1'b0}}, i_kernel_offsets};

    // Global OC store / select index = oc_tile*16 + oc_in_tile (oc_single), else oc.
    wire [6:0] pf_oc_store_idx = i_oc_single ? {pf_oct_d, pf_oc_d[3:0]} : {3'b0, pf_oc_d[3:0]};
    wire [3:0] pf_icg_store    = pf_all ? pf_icg_d[3:0] : 4'd0;

    wire [SRAM_ADDR_W-1:0] sram_rd_addr;
    assign sram_rd_addr = i_wgt_base
                        + addr_oc_component + addr_ic_component
                        + {{(SRAM_ADDR_W-4){1'b0}}, pf_ko}
                        + pf_oc * oc_stride;

    // -------------------------------------------------------------------
    // GEMM-reduce plane prefetch (decision M): read 256 words/super-step into
    // gemm_plane[r*16+c] = Wgt[i_wgt_base + c*icg + s*16 + r]. Owns the SRAM
    // port when i_gemm_reduce (legacy pf_state stays idle in reduce mode).
    // -------------------------------------------------------------------
    localparam PLANE_N = NUM_OC * NUM_OC;   // 256
    reg [WGT_GROUP_W-1:0] gemm_plane [0:PLANE_N-1];

    localparam PL_IDLE = 2'd0, PL_READ = 2'd1, PL_DONE = 2'd2;
    reg [1:0] pl_state;
    reg [8:0] pl_cnt;       // 0..256
    reg [8:0] pl_cnt_d;     // delayed (SRAM 1-cycle latency)
    reg       pl_rd_d;      // delayed read-enable gate
    reg       pl_done_r;

    wire [3:0] pl_r = pl_cnt[3:0];
    wire [3:0] pl_c = pl_cnt[7:4];
    wire [SRAM_ADDR_W-1:0] super_off =
        {{(SRAM_ADDR_W-8){1'b0}}, i_superstep, 4'd0};   // s*16
    wire [SRAM_ADDR_W-1:0] plane_addr =
          i_wgt_base
        + {{(SRAM_ADDR_W-4){1'b0}}, pl_c} * i_ic_groups_total[SRAM_ADDR_W-1:0]
        + super_off
        + {{(SRAM_ADDR_W-4){1'b0}}, pl_r};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pl_state  <= PL_IDLE;
            pl_cnt    <= 9'd0;
            pl_cnt_d  <= 9'd0;
            pl_rd_d   <= 1'b0;
            pl_done_r <= 1'b0;
        end else begin
            pl_cnt_d <= pl_cnt;
            pl_rd_d  <= (pl_state == PL_READ) && (pl_cnt < 9'd256);
            pl_done_r <= 1'b0;

            // Capture the word read 1 cycle ago
            if (pl_rd_d)
                gemm_plane[{pl_cnt_d[3:0], 4'd0} + pl_cnt_d[7:4]] <= i_sram_data;
                // index = r_d*16 + c_d  (r_d=pl_cnt_d[3:0], c_d=pl_cnt_d[7:4])

            case (pl_state)
                PL_IDLE: if (i_gemm_reduce && i_plane_trig) begin
                    pl_state <= PL_READ;
                    pl_cnt   <= 9'd0;
                end
                PL_READ: begin
                    if (pl_cnt == 9'd256) begin
                        pl_state  <= PL_DONE;
                        pl_done_r <= 1'b1;
                    end else begin
                        pl_cnt <= pl_cnt + 9'd1;
                    end
                end
                PL_DONE: pl_state <= PL_IDLE;
                default: pl_state <= PL_IDLE;
            endcase
        end
    end

    assign o_plane_done = pl_done_r;

    genvar pgi;
    generate
        for (pgi = 0; pgi < PLANE_N; pgi = pgi + 1) begin : gen_plane_out
            assign o_wgt_plane[pgi*WGT_GROUP_W +: WGT_GROUP_W] = gemm_plane[pgi];
        end
    endgenerate

    // SRAM port: plane prefetch owns it in reduce mode; legacy prefetch otherwise.
    assign o_sram_addr = i_gemm_reduce ? plane_addr : sram_rd_addr;
    assign o_sram_en   = i_gemm_reduce ? ((pl_state == PL_READ) && (pl_cnt < 9'd256))
                                       : (pf_state == PF_READING);

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
            pf_icg       <= 5'd0;
            pf_icg_d     <= 5'd0;
            pf_oct       <= {OC_TILES_W{1'b0}};
            pf_oct_d     <= {OC_TILES_W{1'b0}};
            pf_done_r    <= 1'b0;
            pf_reading_d <= 1'b0;
        end else begin
            // Delayed versions to match SRAM 1-cycle read latency
            pf_ko_d <= pf_ko;
            pf_oc_d <= pf_oc;
            pf_icg_d <= pf_icg;
            pf_oct_d <= pf_oct;
            pf_reading_d <= (pf_state == PF_READING);

            case (pf_state)
                PF_IDLE: begin
                    if (i_start_prefetch) begin
                        pf_state <= PF_READING;
                        pf_ko    <= 4'd0;
                        pf_oc    <= 5'd0;
                        pf_icg   <= 5'd0;
                        pf_oct   <= {OC_TILES_W{1'b0}};
                    end
                    pf_done_r <= 1'b0;
                end

                PF_READING: begin
                    // Write SRAM data from PREVIOUS cycle's address (1-cycle SRAM latency)
                    // Only write when pf_reading_d is asserted, which gates out the
                    // stale SRAM data present on the first cycle after PF_DONE→PF_READING.
                    if (pf_reading_d)
                        wgt_buf[pf_ko_d][pf_oc_store_idx][pf_icg_store] <= i_sram_data;

                    // Done after last OC of last offset; reuse/oc_single also after the
                    // last IC group; oc_single also after the last OC tile.
                    if (pf_oc == 5'd15 && pf_ko == (i_kernel_offsets[3:0] - 4'd1) &&
                        (!pf_all || pf_icg == (i_ic_groups_total[4:0] - 5'd1)) &&
                        (!i_oc_single || pf_oct == (i_oc_tiles_total - {{(OC_TILES_W-1){1'b0}},1'b1}))) begin
                        pf_state <= PF_WAIT_LAST;
                    end else begin
                        // Advance OC, then kernel offset, then IC group, then OC tile
                        if (pf_oc == 5'd15) begin
                            pf_oc <= 5'd0;
                            if (pf_ko == (i_kernel_offsets[3:0] - 4'd1)) begin
                                pf_ko  <= 4'd0;
                                if (!pf_all || pf_icg == (i_ic_groups_total[4:0] - 5'd1)) begin
                                    pf_icg <= 5'd0;
                                    pf_oct <= pf_oct + {{(OC_TILES_W-1){1'b0}},1'b1}; // only in oc_single
                                end else begin
                                    pf_icg <= pf_icg + 5'd1;   // advances when pf_all
                                end
                            end else begin
                                pf_ko <= pf_ko + 4'd1;
                            end
                        end else begin
                            pf_oc <= pf_oc + 5'd1;
                        end
                    end
                end

                PF_WAIT_LAST: begin
                    // Capture the final SRAM read data (1 cycle after last address)
                    wgt_buf[pf_ko_d][pf_oc_store_idx][pf_icg_store] <= i_sram_data;
                    pf_state  <= PF_DONE;
                    pf_done_r <= 1'b1;
                end

                PF_DONE: begin
                    pf_done_r <= 1'b0;
                    if (i_start_prefetch) begin
                        // Restart pre-fetch for new IC group / OC tile
                        pf_state <= PF_READING;
                        pf_ko    <= 4'd0;
                        pf_oc    <= 5'd0;
                        pf_icg   <= 5'd0;
                        pf_oct   <= {OC_TILES_W{1'b0}};
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

    // Reuse / oc_single select the current ic_group from the buffer; legacy uses slot 0.
    wire [3:0] ic_sel = pf_all ? i_wgt_ic_sel : 4'd0;
    // oc_single selects the active OC-tile's 16 channels (base = tile*16); else tile 0.
    wire [6:0] oc_out_base = i_oc_single ? {i_oc_tile_sel, 4'd0} : 7'd0;

    genvar gi;
    generate
        for (gi = 0; gi < NUM_OC; gi = gi + 1) begin : gen_wgt_out
            assign o_wgt[gi*WGT_GROUP_W +: WGT_GROUP_W] =
                wgt_buf[ko_sel][oc_out_base + gi[6:0]][ic_sel];
        end
    endgenerate

    // Weight valid when pre-fetch is done and offset is within range
    assign o_wgt_vld = (pf_state == PF_DONE) && (i_wgt_offset < i_kernel_offsets[3:0]);

endmodule
