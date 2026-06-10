// deepnet_deploy.c — MNIST DeepConvNet inference on SoC
// Runs 15-layer CNN (Conv1-Conv6 + Pool1-3 + Affine1-2) on 10 test images.

#include "firmware.h"
#include "deepnet.h"
#include "deepnet_weights.h"
#include "mnist_test_images.h"
#include <stdint.h>

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
#define DMA_TIMEOUT  50000
#define NPU_TIMEOUT  200000

// ---- Conv layer SRAM word counts ----
// SRAM word = 128 bits = 16 bytes (one spatial position, 16 channels)
#define CONV1_WGT_SRAM  9     // IC=1, 9 kernel offsets
#define CONV2_WGT_SRAM  144   // IC=16, 144 kernel offsets
#define CONV3_WGT_SRAM  288   // IC=32, 288 kernel offsets (per OC pass)
#define CONV4_WGT_SRAM  288   // IC=32, 288 kernel offsets
#define CONV5_WGT_SRAM  288   // IC=32, 288 kernel offsets
#define CONV6_WGT_SRAM  576   // IC=64, 576 kernel offsets

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
        npu_wr(NPU_DMA_RD_TRIG, 1);

        int t = DMA_TIMEOUT;
        while (t-- > 0)
            if (npu_rd(NPU_DMA_STATUS) & 0x1) break;
        if (t <= 0) print_str("  DMA rd timeout!\n");

        sent += chunk;
    }
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
    npu_wr(NPU_DMA_RD_TRIG, 1);

    int t = DMA_TIMEOUT;
    while (t-- > 0)
        if (npu_rd(NPU_DMA_STATUS) & 0x1) break;
    if (t <= 0) print_str("  DMA wgt rd timeout!\n");
}

// ================================================================
// DMA helper: Out SRAM → DDR (write path, max 64 beats/req)
// ================================================================
static void dma_out_to_ddr(uint32_t ddr_addr, uint32_t sram_base, int nbeats)
{
    npu_wr(NPU_DMA_SRAM_SEL, 0);
    npu_wr(NPU_DMA_PATH_CTL, 0x1);      // read source = Out SRAM

    int sent = 0;
    while (sent < nbeats) {
        int chunk = nbeats - sent;
        if (chunk > 64) chunk = 64;

        npu_wr(NPU_DMA_WR_DDR_ADDR, ddr_addr + sent * 16);
        npu_wr(NPU_DMA_WR_LEN, chunk - 1);
        npu_wr(NPU_DMA_WR_SRAM_BASE, sram_base + sent);
        npu_wr(NPU_DMA_WR_TRIG, 1);

        int t = DMA_TIMEOUT;
        while (t-- > 0)
            if (npu_rd(NPU_DMA_STATUS) & 0x2) break;
        if (t <= 0) print_str("  DMA out wr timeout!\n");

        sent += chunk;
    }
}

// ================================================================
// Load one 16-output-channel tile [oc_start, oc_start+16) of conv weights
// into Wgt SRAM base 0, in the layout wgt_reader expects.
//
// wgt_reader reads SRAM word at:
//   addr = oc_local*ic_groups*KHKW + ic_group*KHKW + ko
// and treats the 128-bit word as 16 INT8 weights for input channels
// {ic_group*16 .. ic_group*16+15} of output channel (oc_start+oc_local),
// kernel offset ko.  (NOTE: wgt_reader ignores NPU_WGT_ADDR_A, so every
// pass must reload its 16 OCs to Wgt SRAM base 0.)
//
// ROM weight layout: W[oc][ic*KHKW + ko]  (row stride = ic*kh*kw).
// For ic<16 the unused lanes of the single group are zero-filled.
// ================================================================
static void load_conv_weights(const int8_t *W, int oc_start,
                              int ic, int kh, int kw)
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
        dma_ddr_to_wgt(WGT_BUF + sent * 16, sent, chunk);
        sent += chunk;
    }
}

// ================================================================
// Pad activation in DDR
// For IC=16: 1 word/position (contiguous spatial layout)
// For IC=32: 2 words/position (IC-group-per-tile layout)
// For IC=64: 4 words/position (IC-group-per-tile layout)
// ================================================================
static void pad_activation(
    uint32_t src_ddr, int in_w, int in_h, int in_ch,
    int pad_w, int pad_h)
{
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
}

// ================================================================
// Run one conv layer with NPU (supports OC tiling)
// ================================================================
static void npu_conv_pass(
    int in_w, int in_h, int ic, int oc,
    int kh, int kw, int sx, int sy,
    const int8_t *W,   // weight ROM pointer (W[oc][ic*kh*kw + ko])
    int bias_scale,  // scale_mul for this layer
    const int32_t *biases,
    int out_ddr_addr,
    int out_spatial)  // out_w * out_h
{
    int oc_passes = oc / 16;
    int out_nbeats = out_spatial;  // SRAM words per pass

    for (int pass = 0; pass < oc_passes; pass++) {
        int oc_base = pass * 16;

        // Load this pass's 16 OCs into Wgt SRAM base 0 (wgt_reader reads base 0)
        load_conv_weights(W, oc_base, ic, kh, kw);

        // Configure NPU dimensions
        npu_wr(NPU_IN_W, in_w);
        npu_wr(NPU_IN_H, in_h);
        npu_wr(NPU_IC, ic);
        npu_wr(NPU_OC, 16);  // always 16 per pass
        npu_wr(NPU_KERNEL, (kh << 8) | kw);
        npu_wr(NPU_STRIDE, (sx << 8) | sy);

        // SRAM addresses
        npu_wr(NPU_ACT_ADDR_A, 0);   // padded data at SRAM base 0
        npu_wr(NPU_WGT_ADDR_A, 0);
        npu_wr(NPU_OUT_ADDR_A, 0);

        // Per-channel bias, scale, shift
        for (int ch = 0; ch < 16; ch++) {
            npu_wr(NPU_BIAS(ch),  (uint32_t)biases[oc_base + ch]);
            npu_wr(NPU_SCALE(ch), bias_scale);
            npu_wr(NPU_SHIFT(ch), SCALE_SHIFT);
        }

        // Start NPU: relu_en=1, pool_en=0
        npu_wr(NPU_CTRL, NPU_CTRL_START | NPU_CTRL_RELU_EN);

        // Wait for done
        int t = NPU_TIMEOUT;
        while (t-- > 0)
            if (npu_rd(NPU_STATUS) & NPU_STATUS_DONE_IRQ) break;
        if (t <= 0) print_str("  NPU timeout!\n");
        if (npu_rd(NPU_STATUS) & NPU_STATUS_DMA_RD_ERR) print_str("  NPU DMA rd err!\n");
        if (npu_rd(NPU_STATUS) & NPU_STATUS_DMA_WR_ERR) print_str("  NPU DMA wr err!\n");

        // DMA Out SRAM → DDR. Each output position = 16 bytes (16 ch),
        // so pass p (OCs 16p..16p+15) lands at byte offset p*out_spatial*16
        // → buffer is IC-tile-major: word = pass*out_spatial + pos.
        int ddr_off = pass * out_spatial * 16;  // byte offset
        dma_out_to_ddr(out_ddr_addr + ddr_off, 0, out_nbeats);
    }
}

// ================================================================
// CPU 2×2 max pool (stride=2, no padding)
// Input: spatial-first-per-tile layout in DDR
// Output: spatial-first-per-tile layout in DDR
// ================================================================
static void cpu_max_pool_2x2(
    uint32_t src_ddr, int in_w, int in_h, int ch,
    uint32_t dst_ddr)
{
    volatile int32_t *src = (volatile int32_t *)src_ddr;
    volatile int32_t *dst = (volatile int32_t *)dst_ddr;

    int out_w = in_w / 2;
    int out_h = in_h / 2;
    int passes = ch / 16;

    for (int oy = 0; oy < out_h; oy++) {
        for (int ox = 0; ox < out_w; ox++) {
            int iy = oy * 2;
            int ix = ox * 2;

            for (int pass = 0; pass < passes; pass++) {
                int src_base = pass * (in_w * in_h) * 4;
                int dst_base = pass * (out_w * out_h) * 4;
                int dst_off  = dst_base + (oy * out_w + ox) * 4;

                for (int sub = 0; sub < 4; sub++) {
                    int32_t v00 = src[src_base + ((iy    ) * in_w + (ix    )) * 4 + sub];
                    int32_t v01 = src[src_base + ((iy    ) * in_w + (ix + 1)) * 4 + sub];
                    int32_t v10 = src[src_base + ((iy + 1) * in_w + (ix    )) * 4 + sub];
                    int32_t v11 = src[src_base + ((iy + 1) * in_w + (ix + 1)) * 4 + sub];

                    int32_t mx = v00;
                    if (v01 > mx) mx = v01;
                    if (v10 > mx) mx = v10;
                    if (v11 > mx) mx = v11;

                    dst[dst_off + sub] = mx;
                }
            }
        }
    }
}

// ================================================================
// CPU affine layer with INT8 quantization
// ================================================================
static void cpu_affine_layer(
    const int8_t *input, const int8_t *W, const int32_t *bias,
    int in_dim, int out_dim, int scale_mul, int do_relu,
    int8_t *output)
{
    for (int oc = 0; oc < out_dim; oc++) {
        int32_t acc = bias[oc];
        for (int ic = 0; ic < in_dim; ic++)
            acc += (int32_t)input[ic] * (int32_t)W[oc * in_dim + ic];

        int32_t val = (acc * scale_mul) >> SCALE_SHIFT;
        if (do_relu && val < 0) val = 0;
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        output[oc] = (int8_t)val;
    }
}

// ================================================================
// Debug: scan a layer's int8 DDR output, print max-abs and nonzero count
// ================================================================
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

// ================================================================
// DeepNet inference: 15-layer MNIST CNN
// ================================================================
void deepnet_inference(const int8_t *input, int32_t *scores)
{
    int8_t *aff_scr = (int8_t *)AFFINE_SCR;

    // Debug: print first 8 input pixels
    print_str("  Input[0..7]:");
    for (int i = 0; i < 8; i++) {
        print_chr(' ');
        print_hex((uint32_t)(uint8_t)input[i], 2);
    }
    print_chr('\n');

    // ---- Conv1: input(28×28×1) → padded(30×30×16) → Conv → 28×28×16 → ActBuf_B ----
    // CPU: zero-fill ActBuf_A with padded 30×30×16 layout, copy input
    {
        volatile int32_t *abuf = (volatile int32_t *)ACT_BUF_A;
        // Zero entire padded buffer: 30*30*16 = 14400 bytes = 3600 int32
        for (int i = 0; i < 30 * 30 * 16 / 4; i++)
            abuf[i] = 0;
        for (int y = 0; y < 28; y++)
            for (int x = 0; x < 28; x++)
                ((volatile int8_t *)abuf)[(y + 1) * 30 * 16 + (x + 1) * 16] = input[y * 28 + x];
    }
    {
        int imx = 0;
        for (int i = 0; i < 784; i++) { int v = input[i]; if (v < 0) v = -v; if (v > imx) imx = v; }
        print_str("  input max="); print_dec((uint32_t)imx); print_chr('\n');
        dbg_layer("PadAct", ACT_BUF_A, 900 * 16);
    }
    dma_ddr_to_act(ACT_BUF_A, 0, CONV1_ACT_SRAM);
    npu_conv_pass(30, 30, 16, 16, 3, 3, 1, 1,
                  &conv1_W[0][0], SCALE_CONV1, conv1_b,
                  ACT_BUF_B, 784);
    dbg_layer("Conv1", ACT_BUF_B, 784 * 16);

    // Debug: print Conv1 output (first 16 bytes = pos(0,0) ch0..15)
    print_str("  Conv1 out[0]:");
    for (int i = 0; i < 16; i++) {
        print_chr(' ');
        print_hex((uint32_t)(uint8_t)((volatile int8_t *)ACT_BUF_B)[i], 2);
    }
    print_chr('\n');

    // ---- Conv2: ActBuf_B(28×28×16) → padded(30×30×16) → Conv → 28×28×16 → ActBuf_A ----
    pad_activation(ACT_BUF_B, 28, 28, 16, 30, 30);
    dma_ddr_to_act(PAD_BUF, 0, CONV2_ACT_SRAM);
    npu_conv_pass(30, 30, 16, 16, 3, 3, 1, 1,
                  &conv2_W[0][0], SCALE_CONV2, conv2_b,
                  ACT_BUF_A, 784);
    dbg_layer("Conv2", ACT_BUF_A, 784 * 16);

    // ---- Pool1: ActBuf_A(28×28×16) → 14×14×16 → ActBuf_B ----
    cpu_max_pool_2x2(ACT_BUF_A, 28, 28, 16, ACT_BUF_B);
    dbg_layer("Pool1", ACT_BUF_B, 196 * 16);

    // ---- Conv3: ActBuf_B(14×14×16) → padded(16×16×16) → Conv → 14×14×32 → ActBuf_A ----
    pad_activation(ACT_BUF_B, 14, 14, 16, 16, 16);
    dma_ddr_to_act(PAD_BUF, 0, CONV3_ACT_SRAM);
    npu_conv_pass(16, 16, 16, 32, 3, 3, 1, 1,
                  &conv3_W[0][0], SCALE_CONV3, conv3_b,
                  ACT_BUF_A, 196);
    dbg_layer("Conv3", ACT_BUF_A, 196 * 32);

    // ---- Conv4: ActBuf_A(14×14×32) → padded(18×18×32) → Conv → 16×16×32 → ActBuf_B ----
    pad_activation(ACT_BUF_A, 14, 14, 32, 18, 18);
    dma_ddr_to_act(PAD_BUF, 0, CONV4_ACT_SRAM);
    npu_conv_pass(18, 18, 32, 32, 3, 3, 1, 1,
                  &conv4_W[0][0], SCALE_CONV4, conv4_b,
                  ACT_BUF_B, 256);
    dbg_layer("Conv4", ACT_BUF_B, 256 * 32);

    // ---- Pool2: ActBuf_B(16×16×32) → 8×8×32 → ActBuf_A ----
    cpu_max_pool_2x2(ACT_BUF_B, 16, 16, 32, ACT_BUF_A);
    dbg_layer("Pool2", ACT_BUF_A, 64 * 32);

    // ---- Conv5: ActBuf_A(8×8×32) → padded(10×10×32) → Conv → 8×8×64 → ActBuf_B ----
    pad_activation(ACT_BUF_A, 8, 8, 32, 10, 10);
    dma_ddr_to_act(PAD_BUF, 0, CONV5_ACT_SRAM);
    npu_conv_pass(10, 10, 32, 64, 3, 3, 1, 1,
                  &conv5_W[0][0], SCALE_CONV5, conv5_b,
                  ACT_BUF_B, 64);
    dbg_layer("Conv5", ACT_BUF_B, 64 * 64);

    // ---- Conv6: ActBuf_B(8×8×64) → padded(10×10×64) → Conv → 8×8×64 → ActBuf_A ----
    pad_activation(ACT_BUF_B, 8, 8, 64, 10, 10);
    dma_ddr_to_act(PAD_BUF, 0, CONV6_ACT_SRAM);
    npu_conv_pass(10, 10, 64, 64, 3, 3, 1, 1,
                  &conv6_W[0][0], SCALE_CONV6, conv6_b,
                  ACT_BUF_A, 64);
    dbg_layer("Conv6", ACT_BUF_A, 64 * 64);

    // ---- Pool3: ActBuf_A(8×8×64) → 4×4×64 → ActBuf_B ----
    cpu_max_pool_2x2(ACT_BUF_A, 8, 8, 64, ACT_BUF_B);

    // Debug: print Pool3 output (first 16 bytes = pos(0,0) ch0..15)
    print_str("  Pool3 out[0]:");
    for (int i = 0; i < 16; i++) {
        print_chr(' ');
        print_hex((uint32_t)(uint8_t)((volatile int8_t *)ACT_BUF_B)[i], 2);
    }
    print_chr('\n');

    // ---- Reorder Pool3 output: position-first → channels-first for affine ----
    // Pool3 output (ActBuf_B): spatial-first-per-tile layout
    //   pass 0: pos(0,0)ch0..15, pos(0,1)ch0..15, ..., pos(3,3)ch0..15
    //   pass 1: pos(0,0)ch16..31, ..., pos(3,3)ch16..31
    //   pass 2: pos(0,0)ch32..47, ..., pos(3,3)ch32..47
    //   pass 3: pos(0,0)ch48..63, ..., pos(3,3)ch48..63
    // Affine expects: ch0: all 16 positions, ch1: all 16 positions, ...
    {
        int8_t *pool_out = (int8_t *)ACT_BUF_B;
        int n_pos = 16;  // 4×4
        int n_ch  = 64;
        for (int ch = 0; ch < n_ch; ch++) {
            int pass  = ch / 16;
            int ch_in = ch % 16;
            for (int pos = 0; pos < n_pos; pos++)
                aff_scr[ch * n_pos + pos] = pool_out[(pass * n_pos + pos) * 16 + ch_in];
        }
    }

    // ---- Affine1: 1024 → 50 (ReLU) ----
    cpu_affine_layer(aff_scr, &affine1_W[0][0], affine1_b,
                     AFFINE1_IN, AFFINE1_OUT, SCALE_AFFINE1, 1, (int8_t *)ACT_BUF_A);

    // Debug: print Affine1 output (first 10 bytes)
    print_str("  Aff1 out:");
    for (int i = 0; i < 10; i++) {
        print_chr(' ');
        print_hex((uint32_t)(uint8_t)((volatile int8_t *)ACT_BUF_A)[i], 2);
    }
    print_chr('\n');

    // ---- Affine2: 50 → 10 (no ReLU, int32 output) ----
    {
        const int8_t *w2 = &affine2_W[0][0];
        int8_t *act1 = (int8_t *)ACT_BUF_A;
        for (int oc = 0; oc < 10; oc++) {
            int32_t acc = affine2_b[oc];
            for (int ic = 0; ic < 50; ic++)
                acc += (int32_t)act1[ic] * (int32_t)w2[oc * 50 + ic];
            scores[oc] = (acc * SCALE_AFFINE2) >> SCALE_SHIFT;
        }
        // Debug: print raw scores
        print_str("  Scores:");
        for (int i = 0; i < 10; i++) {
            print_chr(' ');
            print_hex((uint32_t)scores[i], 8);
        }
        print_chr('\n');
    }
}

// ================================================================
// usercode7: run inference on 2 MNIST test images
// ================================================================
void usercode7(void)
{
    print_str("=== MNIST DeepConvNet Deploy ===\n");

    volatile int32_t *scr = (volatile int32_t *)SCORES;

    int correct = 0;
    for (int d = 0; d < 2; d++) {
        print_str("Digit "); print_dec(d); print_str(": ");

        deepnet_inference(mnist_images[d], (int32_t *)SCORES);

        // Argmax
        int best = 0;
        for (int i = 1; i < 10; i++)
            if (scr[i] > scr[best]) best = i;

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
    print_str("/2 correct ===\n");

    if (correct >= 2)
        print_str("DEPLOY SUCCESS.\n");
    else
        print_str("DEPLOY FAILED.\n");
}
