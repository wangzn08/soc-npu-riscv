// deepnet_deploy.c — MNIST DeepConvNet inference on SoC
// Runs 15-layer CNN (Conv1-Conv6 + Pool1-3 + Affine1-2) on 10 test images.

#include "firmware.h"
#include "deepnet.h"
#include "deepnet_weights.h"
#include "mnist_test_images.h"
#include <stdint.h>

// ---- OC-pass overlap switch (measurement) ----
// 1 = NPU computes pass P while DMA drains the PREVIOUS pass (compute/drain
//     overlap via independent Out SRAM banks).
// 0 = fully serial baseline: start NPU, wait done, drain, repeat. No overlap.
// Override at build time with -DNPU_OC_OVERLAP=0 to measure the overlap gain.
#ifndef NPU_OC_OVERLAP
#define NPU_OC_OVERLAP 1
#endif

// ---- Cycle profiling switch (measurement) ----
// 1 = instrument the inference path with rdcycle and print a per-phase /
//     per-conv-layer cycle breakdown at the end.  0 = no probes (zero overhead).
#ifndef NPU_PROFILE
#define NPU_PROFILE 0
#endif

#if NPU_PROFILE
// Read the low 32 bits of PicoRV32's cycle counter (ENABLE_COUNTERS=1).
// 32 bits suffice: the whole run is < 2^32 cycles.
static inline uint32_t rdcycle32(void) {
    uint32_t c;
    __asm__ volatile ("rdcycle %0" : "=r"(c));
    return c;
}
// Per-phase accumulators (summed over all 10 images).
static uint32_t prof_pad, prof_load, prof_npu, prof_reorder,
                prof_affine, prof_argmax, prof_infer;
static uint32_t prof_npu_layer[6];   // per-conv NPU time
static int      prof_conv_idx;       // reset to 0 at each image
#define PROF_T0()       uint32_t _pt = rdcycle32()
#define PROF_ADD(acc)   do { (acc) += rdcycle32() - _pt; } while (0)
#else
#define PROF_T0()       do {} while (0)
#define PROF_ADD(acc)   do {} while (0)
#endif

// ---- GEMM parity switch (validation) ----
// 1 = run BOTH CPU affine and NPU GEMM, compare (CPU is the deploy result).
// 0 = deploy on NPU GEMM (CPU path compiled out / used only as needed).
#ifndef NPU_GEMM_PARITY
#define NPU_GEMM_PARITY 0
#endif

// ---- NPU register helpers ----
static inline void npu_wr(uint32_t addr, uint32_t data) {
    *(volatile uint32_t *)addr = data;
}
static inline uint32_t npu_rd(uint32_t addr) {
    return *(volatile uint32_t *)addr;
}

// ---- DDR buffer layout ----
#define ACT_BUF_A    0x40001000   // 12288 int32 (48KB)
#define ACT_BUF_B    0x40004000   // 12288 int32 (48KB)
#define WGT_BUF      0x40007000   // 16384 int32 (64KB)
#define NPU_OUT_BUF  0x40010000   // 4096 int32 (16KB)
#define PAD_BUF      0x40012000   // 3072 int32 (12KB)
#define SCORES       0x40015000   // 10 int32
#define AFFINE_SCR   0x40016000   // 4096 int32 (16KB)

// ---- DMA timeouts ----
// DMA_IRQ_TIMEOUT: spins on RAM flag (much cheaper than MMIO); generous budget.
#define DMA_TIMEOUT      50000
#define DMA_IRQ_TIMEOUT  500000
#define NPU_TIMEOUT      200000
// IRQ-wait spins on a RAM flag (much faster than an MMIO poll), so it needs a
// larger iteration budget to cover the same wall-clock as NPU_TIMEOUT. Normal
// completion exits the moment the interrupt fires; this is only a fallback.
#define NPU_IRQ_TIMEOUT  1000000

// ---- Conv layer Act SRAM word counts ----
// SRAM word = 128 bits = 16 bytes (one spatial position, 16 channels)
// (Packed Wgt SRAM sizes are computed at runtime as tile_words; per-layer
//  resident bases are the CONV*_WGT_BASE values below.)
#define CONV1_ACT_SRAM  900   // padded 30x30, 1 word/pos
#define CONV2_ACT_SRAM  900   // padded 30x30, 1 word/pos
#define CONV3_ACT_SRAM  256   // padded 16x16, 1 word/pos
#define CONV4_ACT_SRAM  648   // padded 18x18, 2 words/pos (IC=32)
#define CONV5_ACT_SRAM  200   // padded 10x10, 2 words/pos (IC=32)
#define CONV6_ACT_SRAM  400   // padded 10x10, 4 words/pos (IC=64)

// ---- Weight SRAM base addresses ----
#define CONV1_WGT_BASE  0
#define CONV2_WGT_BASE  144
#define CONV3_WGT_BASE  288
#define CONV4_WGT_BASE  576
#define CONV5_WGT_BASE  1152
#define CONV6_WGT_BASE  2304

// ---- Quantization scale factors (round(scale_f * 2^20)) ----
#define SCALE_CONV1   8389
#define SCALE_CONV2   5381
#define SCALE_CONV3   5587
#define SCALE_CONV4   4959
#define SCALE_CONV5   5185
#define SCALE_CONV6   4119
#define SCALE_AFFINE1 3907
#define SCALE_AFFINE2 7349
#define SCALE_SHIFT   20

// ================================================================
// DMA helper: DDR → Act SRAM (read path, auto-splits)
// ================================================================
static void dma_ddr_to_act(uint32_t ddr_addr, uint32_t sram_base, int nbeats)
{
    PROF_T0();
    npu_wr(NPU_DMA_SRAM_SEL, 0);        // write target = Act SRAM
    npu_wr(NPU_DMA_PATH_CTL, 0x2);      // read source = Act SRAM

    // DMA length register is 8-bit (max 256 beats per request)
    int sent = 0;
    while (sent < nbeats) {
        int chunk = nbeats - sent;
        if (chunk > 256) chunk = 256;

        npu_wr(NPU_DMA_RD_DDR_ADDR, ddr_addr + sent * 16);
        npu_wr(NPU_DMA_RD_LEN, chunk - 1);
        npu_wr(NPU_DMA_RD_SRAM_BASE, sram_base + sent);
        // Clear IRQ flag before trigger (trigger also clears the HW latch)
        dma_rd_irq_flag = 0;
        npu_wr(NPU_DMA_RD_TRIG, 1);

        int t = DMA_IRQ_TIMEOUT;
        while (t-- > 0)
            if (dma_rd_irq_flag) break;
        if (t <= 0) print_str("  DMA rd timeout!\n");

        sent += chunk;
    }
    PROF_ADD(prof_load);
}

// ================================================================
// DMA helper: DDR → Wgt SRAM (read path, auto-splits)
// ================================================================
static void dma_ddr_to_wgt(uint32_t ddr_addr, uint32_t sram_base, int nbeats)
{
    npu_wr(NPU_DMA_SRAM_SEL, 1);        // write target = Wgt SRAM
    npu_wr(NPU_DMA_PATH_CTL, 0x4);      // read source = Wgt SRAM
    npu_wr(NPU_DMA_RD_DDR_ADDR, ddr_addr);
    npu_wr(NPU_DMA_RD_LEN, nbeats - 1);
    npu_wr(NPU_DMA_RD_SRAM_BASE, sram_base);
    // Clear IRQ flag before trigger (trigger also clears the HW latch)
    dma_rd_irq_flag = 0;
    npu_wr(NPU_DMA_RD_TRIG, 1);

    int t = DMA_IRQ_TIMEOUT;
    while (t-- > 0)
        if (dma_rd_irq_flag) break;
    if (t <= 0) print_str("  DMA wgt rd timeout!\n");
}

// ================================================================
// DMA helper: Out SRAM → DDR (write path, max 64 beats/req)
// ================================================================
static void dma_out_to_ddr(uint32_t ddr_addr, uint32_t sram_base, int nbeats,
                           int out_bank)
{
    npu_wr(NPU_DMA_SRAM_SEL, 0);
    npu_wr(NPU_DMA_PATH_CTL, 0x1);      // read source = Out SRAM
    // Select which Out SRAM bank DMA reads (bit2); keep Act/Wgt DMA banks at 0.
    npu_wr(NPU_DMA_PING_SEL, out_bank ? 0x4 : 0x0);

    int sent = 0;
    while (sent < nbeats) {
        int chunk = nbeats - sent;
        if (chunk > 64) chunk = 64;

        npu_wr(NPU_DMA_WR_DDR_ADDR, ddr_addr + sent * 16);
        npu_wr(NPU_DMA_WR_LEN, chunk - 1);
        npu_wr(NPU_DMA_WR_SRAM_BASE, sram_base + sent);
        // Clear IRQ flag before trigger (trigger also clears the HW latch)
        dma_wr_irq_flag = 0;
        npu_wr(NPU_DMA_WR_TRIG, 1);

        int t = DMA_IRQ_TIMEOUT;
        while (t-- > 0)
            if (dma_wr_irq_flag) break;
        if (t <= 0) print_str("  DMA out wr timeout!\n");

        sent += chunk;
    }
}

// ================================================================
// Pack one 16-output-channel tile [oc_start, oc_start+16) of conv weights
// into the layout wgt_reader expects, then DMA it to Wgt SRAM word
// `wgt_sram_base`.
//
// wgt_reader reads SRAM word at:
//   addr = NPU_WGT_ADDR_A + oc_local*ic_groups*KHKW + ic_group*KHKW + ko
// and treats the 128-bit word as 16 INT8 weights for input channels
// {ic_group*16 .. ic_group*16+15} of output channel (oc_start+oc_local),
// kernel offset ko.  wgt_reader now adds NPU_WGT_ADDR_A, so each tile can
// stay resident at its own base across all images (see preload_conv_weights).
//
// ROM weight layout: W[oc][ic*KHKW + ko]  (row stride = ic*kh*kw).
// For ic<16 the unused lanes of the single group are zero-filled.
// ================================================================
static void load_conv_weights(const int8_t *W, int oc_start,
                              int ic, int kh, int kw, int wgt_sram_base)
{
    volatile int8_t *wbuf8 = (volatile int8_t *)WGT_BUF;
    int khkw      = kh * kw;
    int ic_groups = (ic + 15) / 16;
    int word      = 0;

    for (int ocl = 0; ocl < 16; ocl++) {
        int oc = oc_start + ocl;
        for (int g = 0; g < ic_groups; g++) {
            for (int ko = 0; ko < khkw; ko++) {
                int boff = word * 16;
                for (int lane = 0; lane < 16; lane++) {
                    int ich = g * 16 + lane;
                    wbuf8[boff + lane] = (ich < ic)
                        ? W[oc * ic * khkw + ich * khkw + ko] : 0;
                }
                word++;
            }
        }
    }

    // DMA: DDR WgtBuf → Wgt SRAM base 0 (max 256 beats/req)
    int n_words = 16 * ic_groups * khkw;
    int sent = 0;
    while (sent < n_words) {
        int chunk = n_words - sent;
        if (chunk > 256) chunk = 256;
        dma_ddr_to_wgt(WGT_BUF + sent * 16, wgt_sram_base + sent, chunk);
        sent += chunk;
    }
}

// ================================================================
// Preload ALL conv weights into Wgt SRAM once (resident for every image).
// Each (layer,pass) 16-OC tile lands at a distinct Wgt SRAM word base so the
// NPU selects it via NPU_WGT_ADDR_A instead of re-packing + re-DMA'ing the
// same weights on every pass of every image.
// Total resident = 4608 words <= 8192 (one Wgt ping bank).
// ================================================================
static void preload_one_conv(const int8_t *W, int rom_ic, int kh, int kw,
                             int oc, int wgt_base)
{
    int passes     = oc / 16;
    int tile_words = 16 * ((rom_ic + 15) / 16) * kh * kw;
    for (int pass = 0; pass < passes; pass++)
        load_conv_weights(W, pass * 16, rom_ic, kh, kw,
                          wgt_base + pass * tile_words);
}

static void preload_conv_weights(void)
{
    preload_one_conv(&conv1_W[0][0],  1, 3, 3, 16, CONV1_WGT_BASE);
    preload_one_conv(&conv2_W[0][0], 16, 3, 3, 16, CONV2_WGT_BASE);
    preload_one_conv(&conv3_W[0][0], 16, 3, 3, 32, CONV3_WGT_BASE);
    preload_one_conv(&conv4_W[0][0], 32, 3, 3, 32, CONV4_WGT_BASE);
    preload_one_conv(&conv5_W[0][0], 32, 3, 3, 64, CONV5_WGT_BASE);
    preload_one_conv(&conv6_W[0][0], 64, 3, 3, 64, CONV6_WGT_BASE);
}

// ================================================================
// Pad activation in DDR
// For IC=16: 1 word/position (contiguous spatial layout)
// For IC=32: 2 words/position (IC-group-per-tile layout)
// For IC=64: 4 words/position (IC-group-per-tile layout)
// ================================================================
static void __attribute__((unused)) pad_activation(
    uint32_t src_ddr, int in_w, int in_h, int in_ch,
    int pad_w, int pad_h)
{
    PROF_T0();
    volatile int32_t *dbuf = (volatile int32_t *)PAD_BUF;
    volatile int32_t *abuf = (volatile int32_t *)src_ddr;

    int tiles      = in_ch / 16;       // 16-channel SRAM words per position
    int in_spatial = in_w * in_h;
    // Each SRAM word = 128 bit = 16 bytes = 4 int32.
    int total_i32  = pad_w * pad_h * tiles * 4;

    // Zero-fill the whole padded buffer
    for (int i = 0; i < total_i32; i++)
        dbuf[i] = 0;

    int pad_left = (pad_w - in_w) / 2;
    int pad_top  = (pad_h - in_h) / 2;

    // Source is IC-tile-major (conv/pool output: word = tile*spatial + pos).
    // Dest is position-major with ic_tile innermost (what the FSM reads):
    //   word = (padded_y*pad_w + padded_x)*tiles + ic_tile
    // Copy the full 16-byte (4×int32) word for each (position, ic_tile).
    for (int t = 0; t < tiles; t++) {
        for (int y = 0; y < in_h; y++) {
            for (int x = 0; x < in_w; x++) {
                int dst_word = ((y + pad_top) * pad_w + (x + pad_left)) * tiles + t;
                int src_word = t * in_spatial + (y * in_w + x);
                for (int q = 0; q < 4; q++)
                    dbuf[dst_word * 4 + q] = abuf[src_word * 4 + q];
            }
        }
    }
    PROF_ADD(prof_pad);
}

// ================================================================
// Run one conv layer with NPU (supports OC tiling)
// ================================================================
static void npu_conv_pass(
    int in_w, int in_h, int ic, int oc,
    int kh, int kw, int sx, int sy,
    int rom_ic,        // input channels present in ROM (conv1: 1, else == ic)
    int bias_scale,  // scale_mul for this layer
    const int32_t *biases,
    int out_ddr_addr,
    int out_spatial,   // out_w * out_h (conv output, before pooling)
    int wgt_base,      // Wgt SRAM word base of this layer's resident weights
    int pool_en,       // 1 = NPU does 2x2 maxpool on the conv output
    int pad)           // hardware-pad amount each side (0 = caller pre-padded the input)
{
    PROF_T0();
    int oc_passes  = oc / 16;
    // 2x2 pooling quarters the spatial size; only pooled points are written.
    int out_words  = pool_en ? (out_spatial >> 2) : out_spatial;
    int out_nbeats = out_words;    // SRAM words written per pass
    int tile_words = 16 * ((rom_ic + 15) / 16) * kh * kw;  // per-pass Wgt stride

    // OC-pass ping-pong overlap (Issue C): the NPU writes pass P into Out SRAM
    // bank (P&1) via CTRL[6]; while it computes pass P, the CPU DMA-drains the
    // PREVIOUS pass (P-1) from the other bank.  NPU write bank (CTRL[6]) and DMA
    // read bank (NPU_DMA_PING_SEL[2]) are independent signals, so compute and
    // drain truly overlap with no Port/bank conflict.  Act/Wgt read banks stay
    // fixed (global ping_pong unchanged) since their data is constant per layer.
    int prev_pass = -1;   // previous pass whose output still sits in Out SRAM
    int prev_bank = 0;
    (void)prev_pass; (void)prev_bank;  // unused when overlap is disabled

    for (int pass = 0; pass < oc_passes; pass++) {
        int oc_base  = pass * 16;
        int out_bank = pass & 1;   // NPU writes this pass into this Out bank

        // Configure NPU dimensions
        npu_wr(NPU_IN_W, in_w);
        npu_wr(NPU_IN_H, in_h);
        npu_wr(NPU_IC, ic);
        npu_wr(NPU_OC, 16);  // always 16 per pass
        npu_wr(NPU_KERNEL, (kh << 8) | kw);
        npu_wr(NPU_STRIDE, (sx << 8) | sy);
        npu_wr(NPU_PAD, (pad << 8) | pad);   // hardware padding (0 = none)

        // SRAM addresses
        npu_wr(NPU_ACT_ADDR_A, 0);   // padded data at SRAM base 0
        // Weights resident in Wgt SRAM (preloaded once); select this pass's
        // 16-OC tile by base address instead of reloading every pass.
        npu_wr(NPU_WGT_ADDR_A, wgt_base + pass * tile_words);
        npu_wr(NPU_OUT_ADDR_A, 0);

        // Per-channel bias, scale, shift
        for (int ch = 0; ch < 16; ch++) {
            npu_wr(NPU_BIAS(ch),  (uint32_t)biases[oc_base + ch]);
            npu_wr(NPU_SCALE(ch), bias_scale);
            npu_wr(NPU_SHIFT(ch), SCALE_SHIFT);
        }

        // Start NPU: relu_en=1, Out write bank = out_bank (CTRL[6]).
        // The CTRL write also clears any stale NPU-done latch from a prior pass.
        npu_irq_flag = 0;
        npu_wr(NPU_CTRL, NPU_CTRL_START | NPU_CTRL_RELU_EN |
                         (pool_en ? NPU_CTRL_POOL_EN : 0) |
                         (out_bank ? NPU_CTRL_OUT_PING : 0) |
                         (pad ? NPU_CTRL_HW_PAD : 0));

#if NPU_OC_OVERLAP
        // OVERLAP: while the NPU computes this pass, drain the PREVIOUS pass's
        // output (in prev_bank) to DDR.  The two banks differ, so there is no
        // conflict between NPU Port-A writes and DMA Port-B reads.
        if (prev_pass >= 0)
            dma_out_to_ddr(out_ddr_addr + prev_pass * out_words * 16, 0,
                           out_nbeats, prev_bank);
#endif

        // Wait for this pass's NPU compute to finish (irq.c sets npu_irq_flag).
        int t = NPU_IRQ_TIMEOUT;
        while (t-- > 0)
            if (npu_irq_flag) break;
        if (t <= 0) print_str("  NPU IRQ timeout!\n");
        if (npu_rd(NPU_STATUS) & NPU_STATUS_DMA_RD_ERR) print_str("  NPU DMA rd err!\n");
        if (npu_rd(NPU_STATUS) & NPU_STATUS_DMA_WR_ERR) print_str("  NPU DMA wr err!\n");

#if NPU_OC_OVERLAP
        prev_pass = pass;
        prev_bank = out_bank;
#else
        // SERIAL baseline: no overlap — drain THIS pass right after it finishes,
        // before starting the next pass.  Compute and drain never run together.
        dma_out_to_ddr(out_ddr_addr + pass * out_words * 16, 0,
                       out_nbeats, out_bank);
#endif
    }

#if NPU_OC_OVERLAP
    // Drain the final pass's output (no further compute to overlap with).
    dma_out_to_ddr(out_ddr_addr + prev_pass * out_words * 16, 0,
                   out_nbeats, prev_bank);
#endif
#if NPU_PROFILE
    {
        uint32_t _d = rdcycle32() - _pt;
        prof_npu += _d;
        if (prof_conv_idx < 6) prof_npu_layer[prof_conv_idx++] += _d;
    }
#endif
}

// ================================================================
// CPU 2×2 max pool (stride=2, no padding)
// Input: spatial-first-per-tile layout in DDR
// Output: spatial-first-per-tile layout in DDR
// ================================================================
static void __attribute__((unused)) cpu_max_pool_2x2(
    uint32_t src_ddr, int in_w, int in_h, int ch,
    uint32_t dst_ddr)
{
    // Per-channel (per-byte) 2×2 max on 16-byte words (16 int8 channels).
    // Layout is IC-tile-major: word = tile*spatial + position.
    volatile int8_t *src = (volatile int8_t *)src_ddr;
    volatile int8_t *dst = (volatile int8_t *)dst_ddr;

    int out_w = in_w / 2;
    int out_h = in_h / 2;
    int tiles = ch / 16;

    for (int t = 0; t < tiles; t++) {
        int sbase = t * in_w * in_h;        // source word base
        int dbase = t * out_w * out_h;      // dest word base
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {
                int iy = oy * 2, ix = ox * 2;
                int o00 = (sbase + (iy    ) * in_w + (ix    )) * 16;
                int o01 = (sbase + (iy    ) * in_w + (ix + 1)) * 16;
                int o10 = (sbase + (iy + 1) * in_w + (ix    )) * 16;
                int o11 = (sbase + (iy + 1) * in_w + (ix + 1)) * 16;
                int doff = (dbase + oy * out_w + ox) * 16;

                for (int c = 0; c < 16; c++) {
                    int8_t mx = src[o00 + c];
                    if (src[o01 + c] > mx) mx = src[o01 + c];
                    if (src[o10 + c] > mx) mx = src[o10 + c];
                    if (src[o11 + c] > mx) mx = src[o11 + c];
                    dst[doff + c] = mx;
                }
            }
        }
    }
}

// ================================================================
// GEMM / 全连接 (FC) on NPU — general vector×matrix via the systolic array.
// Wgt SRAM PONG-bank word bases for resident FC weights (conv weights occupy
// the PING bank). FC1 = 4 OC-tiles × 64 IC-groups = 4096 words; FC2 = 64 words.
// ================================================================
#define FC1_WGT_BASE 0
#define FC2_WGT_BASE 4096
#define FC1_OUT_DDR  NPU_OUT_BUF            // FC1 int8 output staging (4 words)
#define FC2_OUT_DDR  (NPU_OUT_BUF + 0x100)  // FC2 int8 scores (1 word)

// Pack one 16-OC tile of FC weights into the GEMM layout (KH=KW=1):
//   word(o,g) = o*icg + g  (o = 0..15 OC-in-tile, g = 0..icg-1 IC group of 16)
// Out-of-range oc/ic are zero-filled; padded OCs then compute exact 0 in HW.
static void pack_fc_tile(const int8_t *W, int in_dim, int out_dim,
                         int oc_base, uint32_t stage_ddr)
{
    int icg = (in_dim + 15) / 16;
    volatile int8_t *dst = (volatile int8_t *)stage_ddr;
    for (int o = 0; o < 16; o++) {
        int oc = oc_base + o;
        for (int g = 0; g < icg; g++)
            for (int b = 0; b < 16; b++) {
                int ic = g * 16 + b;
                dst[(o * icg + g) * 16 + b] =
                    (oc < out_dim && ic < in_dim) ? W[oc * in_dim + ic] : 0;
            }
    }
}

// Preload all FC weights into the Wgt SRAM PONG bank once (resident).
static void preload_fc_weights(void)
{
    int icg1 = (AFFINE1_IN + 15) / 16;          // 64
    npu_wr(NPU_DMA_PING_SEL, 0x2);              // DMA wgt writes -> PONG bank
    for (int t = 0; t < (AFFINE1_OUT + 15) / 16; t++) {   // 4 OC tiles
        pack_fc_tile(&affine1_W[0][0], AFFINE1_IN, AFFINE1_OUT, t * 16, WGT_BUF);
        int base   = FC1_WGT_BASE + t * 16 * icg1;
        int nwords = 16 * icg1;
        int sent   = 0;
        while (sent < nwords) {                 // DMA len reg is 8-bit: 256/req
            int chunk = nwords - sent;
            if (chunk > 256) chunk = 256;
            dma_ddr_to_wgt(WGT_BUF + sent * 16, (uint32_t)(base + sent), chunk);
            sent += chunk;
        }
    }
    {
        int icg2 = (AFFINE2_IN + 15) / 16;      // 4
        pack_fc_tile(&affine2_W[0][0], AFFINE2_IN, AFFINE2_OUT, 0, WGT_BUF);
        dma_ddr_to_wgt(WGT_BUF, (uint32_t)FC2_WGT_BASE, 16 * icg2);
    }
    npu_wr(NPU_DMA_PING_SEL, 0x0);              // restore: conv DMAs use PING
}

// 通用 NPU GEMM/全连接: out[0..out_dim) = quant(act · W + bias), 可选 ReLU.
// Runs with global ping_pong=1 so the FSM latches the *_B (pong) base regs:
// input vector lives in Act PONG, weights resident in Wgt PONG (pack_fc_tile
// layout at wgt_base), output INT8 channel-major to out_ddr. OC tiled by 16
// (decision D). in_dim <= 1024 (HW ic_group counter width).
static void npu_gemm_pass(int in_dim, int out_dim, int scale_mul_val,
                          const int32_t *biases, int relu_en,
                          uint32_t in_ddr, uint32_t out_ddr, int wgt_base)
{
    int icg        = (in_dim + 15) / 16;
    int oc_passes  = (out_dim + 15) / 16;
    int tile_words = 16 * icg;

    npu_wr(NPU_DMA_PING_SEL, 0x1);             // input vector DMA -> Act PONG
    dma_ddr_to_act(in_ddr, 0, icg);
    npu_wr(NPU_DMA_PING_SEL, 0x0);

    for (int pass = 0; pass < oc_passes; pass++) {
        npu_wr(NPU_IN_W, 1);
        npu_wr(NPU_IN_H, 1);
        npu_wr(NPU_IC, (uint32_t)in_dim);
        npu_wr(NPU_OC, 16);                    // decision D: 16 OC per start
        npu_wr(NPU_KERNEL, (1 << 8) | 1);      // KH=KW=1
        npu_wr(NPU_STRIDE, (1 << 8) | 1);
        // ping_pong=1 -> FSM latches the *_B (pong) base registers
        npu_wr(NPU_ACT_ADDR_B, 0);
        npu_wr(NPU_WGT_ADDR_B, (uint32_t)(wgt_base + pass * tile_words));
        npu_wr(NPU_OUT_ADDR_B, (uint32_t)pass);   // pass p -> Out word p

        for (int ch = 0; ch < 16; ch++) {
            int oc = pass * 16 + ch;
            npu_wr(NPU_BIAS(ch),  (oc < out_dim) ? (uint32_t)biases[oc] : 0u);
            npu_wr(NPU_SCALE(ch), (oc < out_dim) ? (uint32_t)scale_mul_val : 0u);
            npu_wr(NPU_SHIFT(ch), SCALE_SHIFT);
        }

        npu_irq_flag = 0;
        npu_wr(NPU_CTRL, NPU_CTRL_START | NPU_CTRL_PING_PONG | NPU_CTRL_GEMM_EN |
                         (relu_en ? NPU_CTRL_RELU_EN : 0));
        int t = NPU_IRQ_TIMEOUT;
        while (t-- > 0)
            if (npu_irq_flag) break;
        if (t <= 0) print_str("  GEMM IRQ timeout!\n");
    }
    // All passes wrote Out PING (CTRL[6]=0) words 0..oc_passes-1.
    dma_out_to_ddr(out_ddr, 0, oc_passes, 0);
}

// ================================================================
// CPU affine layer with INT8 quantization
// ================================================================
static void __attribute__((unused)) cpu_affine_layer(
    const int8_t *input, const int8_t *W, const int32_t *bias,
    int in_dim, int out_dim, int scale_mul, int do_relu,
    int8_t *output)
{
    for (int oc = 0; oc < out_dim; oc++) {
        int32_t acc = bias[oc];
        for (int ic = 0; ic < in_dim; ic++)
            acc += (int32_t)input[ic] * (int32_t)W[oc * in_dim + ic];

        int32_t val = (int32_t)(((int64_t)acc * scale_mul) >> SCALE_SHIFT);
        if (do_relu && val < 0) val = 0;
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        output[oc] = (int8_t)val;
    }
}

#ifdef DEBUG_VERBOSE
// Debug: scan a layer's int8 DDR output, print max-abs and nonzero count
static void dbg_layer(const char *name, uint32_t ddr, int nbytes)
{
    volatile int8_t *p = (volatile int8_t *)ddr;
    int mx = 0, nz = 0;
    for (int i = 0; i < nbytes; i++) {
        int v = p[i];
        if (v < 0) v = -v;
        if (v > mx) mx = v;
        if (p[i]) nz++;
    }
    print_str("  ["); print_str(name);
    print_str("] max="); print_dec((uint32_t)mx);
    print_str(" nz="); print_dec((uint32_t)nz);
    print_str("/"); print_dec((uint32_t)nbytes);
    print_chr('\n');
}
#endif

// ================================================================
// DeepNet inference: 15-layer MNIST CNN
// ================================================================
void deepnet_inference(const int8_t *input, int32_t *scores)
{
#if NPU_GEMM_PARITY
    // CPU affine oracle operands (parity build only). Kept in fast private RAM.
    static int8_t affine_in[1024];   // reorder output / Affine1 input
    static int8_t affine_mid[64];    // Affine1 output / Affine2 input (50 used)
#endif
#if NPU_PROFILE
    prof_conv_idx = 0;   // per-image: conv layers index 0..5 in call order
#endif

#ifdef DEBUG_VERBOSE
    // Debug: print first 8 input pixels
    print_str("  Input[0..7]:");
    for (int i = 0; i < 8; i++) { print_chr(' '); print_hex((uint32_t)(uint8_t)input[i], 2); }
    print_chr('\n');
#endif

    // ---- Conv1: image(28×28×1) → HW-pad(30×30) → Conv → 28×28×16 → ActBuf_B ----
    {
        PROF_T0();
        // Format image into tile-major 16-byte words (ch0 = pixel, ch1..15 = 0).
        // The 1-pixel border is added by hardware padding, not stored.
        volatile int32_t *abuf = (volatile int32_t *)ACT_BUF_A;
        for (int i = 0; i < 28 * 28 * 16 / 4; i++)
            abuf[i] = 0;
        for (int i = 0; i < 28 * 28; i++)
            ((volatile int8_t *)abuf)[i * 16] = input[i];
        PROF_ADD(prof_pad);   // Conv1 image formatting (no padded scatter)
    }
#ifdef DEBUG_VERBOSE
    {
        int imx = 0;
        for (int i = 0; i < 784; i++) { int v = input[i]; if (v < 0) v = -v; if (v > imx) imx = v; }
        print_str("  input max="); print_dec((uint32_t)imx); print_chr('\n');
        dbg_layer("PadAct", ACT_BUF_A, 900 * 16);
    }
#endif
    dma_ddr_to_act(ACT_BUF_A, 0, 28 * 28 * 1);   // 28x28x16, tiles=1 → 784 words
    npu_conv_pass(30, 30, 16, 16, 3, 3, 1, 1,
                  1, SCALE_CONV1, conv1_b,
                  ACT_BUF_B, 784, CONV1_WGT_BASE, 0, 1);   // pad=1 (HW)
#ifdef DEBUG_VERBOSE
    dbg_layer("Conv1", ACT_BUF_B, 784 * 16);
    {
        volatile int8_t *cb = (volatile int8_t *)ACT_BUF_B;
        int pys[4] = {0, 14, 7, 14}, pxs[4] = {0, 14, 10, 0};
        for (int p = 0; p < 4; p++) {
            int base = (pys[p] * 28 + pxs[p]) * 16;
            print_str("    conv1("); print_dec((uint32_t)pys[p]);
            print_chr(','); print_dec((uint32_t)pxs[p]); print_str(") ch0-15:");
            for (int i = 0; i < 16; i++) { print_chr(' '); print_hex((uint32_t)(uint8_t)cb[base + i], 2); }
            print_chr('\n');
        }
    }
#endif

    // ---- Conv2: ActBuf_B(28×28×16) → HW-pad(30×30) → Conv → 28×28 → 2x2 pool → ActBuf_B ----
    // Hardware padding (pad=1): DMA the unpadded Conv1 output straight into Act
    // SRAM (tile-major); the FSM injects border zeros. No CPU pad_activation.
    dma_ddr_to_act(ACT_BUF_B, 0, 28 * 28 * 1);   // 28x28x16, tiles=1 → 784 words
    npu_conv_pass(30, 30, 16, 16, 3, 3, 1, 1,
                  16, SCALE_CONV2, conv2_b,
                  ACT_BUF_B, 784, CONV2_WGT_BASE, 1, 1);   // pool_en=1, pad=1 (HW)
#ifdef DEBUG_VERBOSE
    dbg_layer("Pool1", ACT_BUF_B, 196 * 16);
#endif

    // ---- Conv3: ActBuf_B(14×14×16) → HW-pad(16×16) → Conv → 14×14×32 → ActBuf_A ----
    dma_ddr_to_act(ACT_BUF_B, 0, 14 * 14 * 1);   // 14x14x16, tiles=1 → 196 words
    npu_conv_pass(16, 16, 16, 32, 3, 3, 1, 1,
                  16, SCALE_CONV3, conv3_b,
                  ACT_BUF_A, 196, CONV3_WGT_BASE, 0, 1);   // pad=1 (HW)
#ifdef DEBUG_VERBOSE
    dbg_layer("Conv3", ACT_BUF_A, 196 * 32);
    {
        volatile int8_t *b = (volatile int8_t *)ACT_BUF_A;
        print_str("    conv3(7,7) ch0-15:");
        for (int c = 0; c < 16; c++) { print_chr(' '); print_hex((uint32_t)(uint8_t)b[(0*196 + 7*14+7)*16 + c], 2); }
        print_str("\n    conv3(7,7) ch16-31:");
        for (int c = 0; c < 16; c++) { print_chr(' '); print_hex((uint32_t)(uint8_t)b[(1*196 + 7*14+7)*16 + c], 2); }
        print_chr('\n');
    }
#endif

    // ---- Conv4: ActBuf_A(14×14×32) → HW-pad(18×18, pad=2) → Conv → 16×16×32 → 2x2 pool → ActBuf_A ----
    dma_ddr_to_act(ACT_BUF_A, 0, 14 * 14 * 2);   // 14x14x32, tiles=2 → 392 words
    // NPU does Conv4 + Pool2: 16x16 conv -> 2x2 maxpool -> 8x8.
    npu_conv_pass(18, 18, 32, 32, 3, 3, 1, 1,
                  32, SCALE_CONV4, conv4_b,
                  ACT_BUF_A, 256, CONV4_WGT_BASE, 1, 2);   // pool_en=1, pad=2 (HW)
#ifdef DEBUG_VERBOSE
    dbg_layer("Pool2", ACT_BUF_A, 64 * 32);
#endif

    // ---- Conv5: ActBuf_A(8×8×32) → HW-pad(10×10) → Conv → 8×8×64 → ActBuf_B ----
    dma_ddr_to_act(ACT_BUF_A, 0, 8 * 8 * 2);   // 8x8x32, tiles=2 → 128 words
    npu_conv_pass(10, 10, 32, 64, 3, 3, 1, 1,
                  32, SCALE_CONV5, conv5_b,
                  ACT_BUF_B, 64, CONV5_WGT_BASE, 0, 1);   // pad=1 (HW)
#ifdef DEBUG_VERBOSE
    dbg_layer("Conv5", ACT_BUF_B, 64 * 64);
#endif

    // ---- Conv6: ActBuf_B(8×8×64) → HW-pad(10×10) → Conv → 8×8×64 → 2x2 pool → ActBuf_B ----
    dma_ddr_to_act(ACT_BUF_B, 0, 8 * 8 * 4);   // 8x8x64, tiles=4 → 256 words
    // NPU does Conv6 + Pool3: 8x8 conv -> 2x2 maxpool -> 4x4.
    npu_conv_pass(10, 10, 64, 64, 3, 3, 1, 1,
                  64, SCALE_CONV6, conv6_b,
                  ACT_BUF_B, 64, CONV6_WGT_BASE, 1, 1);   // pool_en=1, pad=1 (HW)

    // ---- Reorder Pool3 output: position-first → channels-first for affine ----
    // Pool3 output (ActBuf_B): spatial-first-per-tile layout
    //   pass 0: pos(0,0)ch0..15, pos(0,1)ch0..15, ..., pos(3,3)ch0..15
    //   pass 1: pos(0,0)ch16..31, ..., pos(3,3)ch16..31
    //   pass 2: pos(0,0)ch32..47, ..., pos(3,3)ch32..47
    //   pass 3: pos(0,0)ch48..63, ..., pos(3,3)ch48..63
    // Affine expects: ch0: all 16 positions, ch1: all 16 positions, ...
    {
        PROF_T0();
        int8_t *pool_out = (int8_t *)ACT_BUF_B;
        int n_pos = 16;  // 4×4
        int n_ch  = 64;
        for (int ch = 0; ch < n_ch; ch++) {
            int pass  = ch / 16;
            int ch_in = ch % 16;
            for (int pos = 0; pos < n_pos; pos++) {
                int8_t v = pool_out[(pass * n_pos + pos) * 16 + ch_in];
                ((volatile int8_t *)AFFINE_SCR)[ch * n_pos + pos] = v; // NPU GEMM input (DDR)
#if NPU_GEMM_PARITY
                affine_in[ch * n_pos + pos] = v;                       // CPU oracle (RAM)
#endif
            }
        }
        PROF_ADD(prof_reorder);
    }

#if NPU_GEMM_PARITY
    {
        PROF_T0();
        // ---- CPU oracle: Affine1 (1024→50, ReLU) + Affine2 (50→10) ----
        cpu_affine_layer(affine_in, &affine1_W[0][0], affine1_b,
                         AFFINE1_IN, AFFINE1_OUT, SCALE_AFFINE1, 1, affine_mid);
        const int8_t *w2 = &affine2_W[0][0];
        int8_t *act1 = affine_mid;
        for (int oc = 0; oc < AFFINE2_OUT; oc++) {
            int32_t acc = affine2_b[oc];
            for (int ic = 0; ic < AFFINE2_IN; ic++)
                acc += (int32_t)act1[ic] * (int32_t)w2[oc * AFFINE2_IN + ic];
            scores[oc] = (int32_t)(((int64_t)acc * SCALE_AFFINE2) >> SCALE_SHIFT);
        }
        PROF_ADD(prof_affine);
    }
#else
    {
        PROF_T0();
        // ---- Deploy: NPU FC1 (1024→50, ReLU) → FC1_OUT_DDR ----
        npu_gemm_pass(AFFINE1_IN, AFFINE1_OUT, SCALE_AFFINE1, affine1_b, 1,
                      AFFINE_SCR, FC1_OUT_DDR, FC1_WGT_BASE);
        // ---- CPU FC2 (50→10, raw int32): reads NPU FC1 output (channel-major,
        //      OCs 50..63 are exact 0 from padded weights). FC2 stays on CPU
        //      because the NPU's INT8 output saturates the final logits. ----
        volatile int8_t *f1 = (volatile int8_t *)FC1_OUT_DDR;
        const int8_t *w2 = &affine2_W[0][0];
        for (int oc = 0; oc < AFFINE2_OUT; oc++) {
            int32_t acc = affine2_b[oc];
            for (int ic = 0; ic < AFFINE2_IN; ic++)
                acc += (int32_t)f1[ic] * (int32_t)w2[oc * AFFINE2_IN + ic];
            scores[oc] = (int32_t)(((int64_t)acc * SCALE_AFFINE2) >> SCALE_SHIFT);
        }
        PROF_ADD(prof_affine);
    }
#endif

#if NPU_GEMM_PARITY
    {
        // NPU FC1 vs CPU affine1 — must be bit-identical (same quant path).
        npu_gemm_pass(AFFINE1_IN, AFFINE1_OUT, SCALE_AFFINE1, affine1_b, 1,
                      AFFINE_SCR, FC1_OUT_DDR, FC1_WGT_BASE);
        volatile int8_t *nf = (volatile int8_t *)FC1_OUT_DDR;
        int mism = 0;
        for (int i = 0; i < AFFINE1_OUT; i++)
            if (nf[i] != affine_mid[i]) mism++;
        print_str("  FC1 parity: ");
        if (mism == 0) print_str("OK\n");
        else { print_str("FAIL mism="); print_dec((uint32_t)mism); print_chr('\n'); }

        // NPU FC2 (no ReLU): input = NPU FC1 output (channel-major in DDR, OCs
        // 50..63 are exact 0 from padded weights). Scores are INT8-clamped; the
        // ship criterion is argmax(NPU int8) == argmax(CPU int32) for every image.
        npu_gemm_pass(AFFINE2_IN, AFFINE2_OUT, SCALE_AFFINE2, affine2_b, 0,
                      FC1_OUT_DDR, FC2_OUT_DDR, FC2_WGT_BASE);
        volatile int8_t *ns = (volatile int8_t *)FC2_OUT_DDR;
        int nbest = 0, cbest = 0;
        for (int i = 1; i < AFFINE2_OUT; i++) {
            if (ns[i] > ns[nbest]) nbest = i;
            if (scores[i] > scores[cbest]) cbest = i;
        }
        print_str("  FC2 argmax: npu="); print_dec((uint32_t)nbest);
        print_str(" cpu="); print_dec((uint32_t)cbest);
        print_str(nbest == cbest ? " OK\n" : " MISMATCH\n");
    }
#endif
#ifdef DEBUG_VERBOSE
    print_str("  Aff1 out:");
    for (int i = 0; i < 10; i++) { print_chr(' '); print_hex((uint32_t)(uint8_t)((volatile int8_t *)ACT_BUF_A)[i], 2); }
    print_chr('\n');
    print_str("  Scores:");
    for (int i = 0; i < 10; i++) { print_chr(' '); print_hex((uint32_t)scores[i], 8); }
    print_chr('\n');
#endif
}

// ================================================================
// usercode7: run inference on 2 MNIST test images
// ================================================================
void usercode7(void)
{
    print_str("=== MNIST DeepConvNet Deploy ===\n");

    // Preload all conv weights into Wgt SRAM once; they stay resident for
    // every image instead of being re-packed and re-DMA'd on every pass.
    preload_conv_weights();
    // Preload FC (affine) weights into the Wgt SRAM PONG bank (resident).
    preload_fc_weights();

    volatile int32_t *scr = (volatile int32_t *)SCORES;

    int correct = 0;
    for (int d = 0; d < 10; d++) {
        print_str("Digit "); print_dec(d); print_str(": ");

#if NPU_PROFILE
        uint32_t _ti = rdcycle32();
#endif
        deepnet_inference(mnist_images[d], (int32_t *)SCORES);
#if NPU_PROFILE
        prof_infer += rdcycle32() - _ti;
#endif

        // Argmax
        PROF_T0();
        int best = 0;
        for (int i = 1; i < 10; i++)
            if (scr[i] > scr[best]) best = i;
        PROF_ADD(prof_argmax);

        print_str("pred="); print_dec(best);
        if (best == d) {
            print_str(" OK\n");
            correct++;
        } else {
            print_str(" FAIL (scores:");
            for (int i = 0; i < 10; i++) {
                print_chr(' ');
                print_hex((uint32_t)scr[i], 8);
            }
            print_str(")\n");
        }
    }

    print_str("\n=== Result: "); print_dec(correct);
    print_str("/10 correct ===\n");

    if (correct >= 10)
        print_str("DEPLOY SUCCESS.\n");
    else
        print_str("DEPLOY FAILED.\n");

#if NPU_PROFILE
    // Per-phase cycle breakdown, summed over all 10 images.
    print_str("\n=== CYCLE PROFILE (10 images total) ===\n");
    print_str("infer_total: "); print_dec(prof_infer);   print_chr('\n');
    print_str("  npu      : "); print_dec(prof_npu);     print_chr('\n');
    print_str("  pad      : "); print_dec(prof_pad);     print_chr('\n');
    print_str("  load     : "); print_dec(prof_load);    print_chr('\n');
    print_str("  reorder  : "); print_dec(prof_reorder); print_chr('\n');
    print_str("  affine   : "); print_dec(prof_affine);  print_chr('\n');
    print_str("argmax     : "); print_dec(prof_argmax);  print_chr('\n');
    print_str("npu_per_layer (Conv1..Conv6):\n");
    for (int i = 0; i < 6; i++) {
        print_str("  Conv"); print_dec((uint32_t)(i + 1));
        print_str(": "); print_dec(prof_npu_layer[i]); print_chr('\n');
    }
    print_str("=======================================\n");
#endif
}
