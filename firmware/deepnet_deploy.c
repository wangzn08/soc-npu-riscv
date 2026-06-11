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
    int rom_ic,        // input channels present in ROM (conv1: 1, else == ic)
    int bias_scale,  // scale_mul for this layer
    const int32_t *biases,
    int out_ddr_addr,
    int out_spatial,   // out_w * out_h (conv output, before pooling)
    int wgt_base,      // Wgt SRAM word base of this layer's resident weights
    int pool_en)       // 1 = NPU does 2x2 maxpool on the conv output
{
    int oc_passes  = oc / 16;
    // 2x2 pooling quarters the spatial size; only pooled points are written.
    int out_words  = pool_en ? (out_spatial >> 2) : out_spatial;
    int out_nbeats = out_words;    // SRAM words written per pass
    int tile_words = 16 * ((rom_ic + 15) / 16) * kh * kw;  // per-pass Wgt stride

    for (int pass = 0; pass < oc_passes; pass++) {
        int oc_base = pass * 16;

        // Configure NPU dimensions
        npu_wr(NPU_IN_W, in_w);
        npu_wr(NPU_IN_H, in_h);
        npu_wr(NPU_IC, ic);
        npu_wr(NPU_OC, 16);  // always 16 per pass
        npu_wr(NPU_KERNEL, (kh << 8) | kw);
        npu_wr(NPU_STRIDE, (sx << 8) | sy);

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

        // Start NPU: relu_en=1, pool_en=0. Clear the done flag first; the CTRL
        // write also clears any stale NPU-done latch from a previous pass.
        npu_irq_flag = 0;
        npu_wr(NPU_CTRL, NPU_CTRL_START | NPU_CTRL_RELU_EN |
                         (pool_en ? NPU_CTRL_POOL_EN : 0));

        // Wait for the NPU-done interrupt (irq.c sets npu_irq_flag on bit 3).
        int t = NPU_IRQ_TIMEOUT;
        while (t-- > 0)
            if (npu_irq_flag) break;
        if (t <= 0) print_str("  NPU IRQ timeout!\n");
        if (npu_rd(NPU_STATUS) & NPU_STATUS_DMA_RD_ERR) print_str("  NPU DMA rd err!\n");
        if (npu_rd(NPU_STATUS) & NPU_STATUS_DMA_WR_ERR) print_str("  NPU DMA wr err!\n");

        // DMA Out SRAM → DDR. Each output position = 16 bytes (16 ch),
        // so pass p (OCs 16p..16p+15) lands at byte offset p*out_spatial*16
        // → buffer is IC-tile-major: word = pass*out_spatial + pos.
        int ddr_off = pass * out_words * 16;  // byte offset (pooled size if pool_en)
        dma_out_to_ddr(out_ddr_addr + ddr_off, 0, out_nbeats);
    }
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

        int32_t val = (int32_t)(((int64_t)acc * scale_mul) >> SCALE_SHIFT);
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
                  1, SCALE_CONV1, conv1_b,
                  ACT_BUF_B, 784, CONV1_WGT_BASE, 0);
    dbg_layer("Conv1", ACT_BUF_B, 784 * 16);
    // PROBE: read Wgt SRAM back (64-beat chunks), print lane0 of word oc*9
    {
        uint32_t scr = AFFINE_SCR;
        npu_wr(NPU_DMA_PATH_CTL, 0x4);  // rd src = Wgt SRAM
        int sent = 0, n = 144;
        while (sent < n) {
            int chunk = n - sent; if (chunk > 64) chunk = 64;
            npu_wr(NPU_DMA_WR_DDR_ADDR, scr + sent * 16);
            npu_wr(NPU_DMA_WR_LEN, chunk - 1);
            npu_wr(NPU_DMA_WR_SRAM_BASE, sent);
            npu_wr(NPU_DMA_WR_TRIG, 1);
            int t = DMA_TIMEOUT; while (t-- > 0) if (npu_rd(NPU_DMA_STATUS) & 0x2) break;
            sent += chunk;
        }
        volatile int8_t *wb = (volatile int8_t *)scr;
        print_str("    wgtSRAM lane0[oc*9]:");
        for (int oc = 0; oc < 16; oc++) { print_chr(' '); print_hex((uint32_t)(uint8_t)wb[(oc * 9) * 16], 2); }
        print_chr('\n');
        // CPU-direct read of WGT_BUF (DDR) to check formatting (no DMA)
        volatile int8_t *fb = (volatile int8_t *)WGT_BUF;
        print_str("    WgtBuf lane0[oc*9]:");
        for (int oc = 0; oc < 16; oc++) { print_chr(' '); print_hex((uint32_t)(uint8_t)fb[(oc * 9) * 16], 2); }
        print_chr('\n');
    }

    // Debug: dump Conv1 output at the same positions as golden.py
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

    // ---- Conv2: ActBuf_B(28×28×16) → padded(30×30×16) → Conv → 28×28×16 → ActBuf_A ----
    pad_activation(ACT_BUF_B, 28, 28, 16, 30, 30);
    dma_ddr_to_act(PAD_BUF, 0, CONV2_ACT_SRAM);
    // NPU does Conv2 + Pool1 in one pass: 28x28 conv -> 2x2 maxpool -> 14x14.
    npu_conv_pass(30, 30, 16, 16, 3, 3, 1, 1,
                  16, SCALE_CONV2, conv2_b,
                  ACT_BUF_B, 784, CONV2_WGT_BASE, 1);   // pool_en=1 -> 14x14 to ActBuf_B
    dbg_layer("Pool1", ACT_BUF_B, 196 * 16);

    // ---- Conv3: ActBuf_B(14×14×16) → padded(16×16×16) → Conv → 14×14×32 → ActBuf_A ----
    pad_activation(ACT_BUF_B, 14, 14, 16, 16, 16);
    dma_ddr_to_act(PAD_BUF, 0, CONV3_ACT_SRAM);
    npu_conv_pass(16, 16, 16, 32, 3, 3, 1, 1,
                  16, SCALE_CONV3, conv3_b,
                  ACT_BUF_A, 196, CONV3_WGT_BASE, 0);
    dbg_layer("Conv3", ACT_BUF_A, 196 * 32);
    {
        volatile int8_t *b = (volatile int8_t *)ACT_BUF_A;
        print_str("    conv3(7,7) ch0-15:");
        for (int c = 0; c < 16; c++) { print_chr(' '); print_hex((uint32_t)(uint8_t)b[(0*196 + 7*14+7)*16 + c], 2); }
        print_str("\n    conv3(7,7) ch16-31:");
        for (int c = 0; c < 16; c++) { print_chr(' '); print_hex((uint32_t)(uint8_t)b[(1*196 + 7*14+7)*16 + c], 2); }
        print_chr('\n');
    }

    // ---- Conv4: ActBuf_A(14×14×32) → padded(18×18×32) → Conv → 16×16×32 → ActBuf_B ----
    pad_activation(ACT_BUF_A, 14, 14, 32, 18, 18);
    {   // verify pad: conv3(7,7) → padded(9,9); word=(9*18+9)*2+tile
        volatile int8_t *p = (volatile int8_t *)PAD_BUF;
        print_str("    pad4(9,9) t0:");
        for (int c = 0; c < 16; c++) { print_chr(' '); print_hex((uint32_t)(uint8_t)p[((9*18+9)*2+0)*16 + c], 2); }
        print_str("\n    pad4(9,9) t1:");
        for (int c = 0; c < 16; c++) { print_chr(' '); print_hex((uint32_t)(uint8_t)p[((9*18+9)*2+1)*16 + c], 2); }
        print_chr('\n');
        // 3x3 block rows 8..10 cols 8..10, lo32 (t0 ch0-3, t1 ch16-19)
        for (int ry = 8; ry <= 10; ry++)
            for (int rx = 8; rx <= 10; rx++) {
                int w0 = ((ry*18+rx)*2+0)*16, w1 = ((ry*18+rx)*2+1)*16;
                print_str("    pad4("); print_dec((uint32_t)ry); print_chr(',');
                print_dec((uint32_t)rx); print_str(") t0lo:");
                for (int c = 0; c < 4; c++) { print_chr(' '); print_hex((uint32_t)(uint8_t)p[w0+c], 2); }
                print_str(" t1lo:");
                for (int c = 0; c < 4; c++) { print_chr(' '); print_hex((uint32_t)(uint8_t)p[w1+c], 2); }
                print_chr('\n');
            }
    }
    dma_ddr_to_act(PAD_BUF, 0, CONV4_ACT_SRAM);
    // NPU does Conv4 + Pool2: 16x16 conv -> 2x2 maxpool -> 8x8.
    npu_conv_pass(18, 18, 32, 32, 3, 3, 1, 1,
                  32, SCALE_CONV4, conv4_b,
                  ACT_BUF_A, 256, CONV4_WGT_BASE, 1);   // pool_en=1 -> 8x8 to ActBuf_A
    dbg_layer("Pool2", ACT_BUF_A, 64 * 32);

    // ---- Conv5: ActBuf_A(8×8×32) → padded(10×10×32) → Conv → 8×8×64 → ActBuf_B ----
    pad_activation(ACT_BUF_A, 8, 8, 32, 10, 10);
    dma_ddr_to_act(PAD_BUF, 0, CONV5_ACT_SRAM);
    npu_conv_pass(10, 10, 32, 64, 3, 3, 1, 1,
                  32, SCALE_CONV5, conv5_b,
                  ACT_BUF_B, 64, CONV5_WGT_BASE, 0);
    dbg_layer("Conv5", ACT_BUF_B, 64 * 64);

    // ---- Conv6: ActBuf_B(8×8×64) → padded(10×10×64) → Conv → 8×8×64 → ActBuf_A ----
    pad_activation(ACT_BUF_B, 8, 8, 64, 10, 10);
    dma_ddr_to_act(PAD_BUF, 0, CONV6_ACT_SRAM);
    // NPU does Conv6 + Pool3: 8x8 conv -> 2x2 maxpool -> 4x4.
    npu_conv_pass(10, 10, 64, 64, 3, 3, 1, 1,
                  64, SCALE_CONV6, conv6_b,
                  ACT_BUF_B, 64, CONV6_WGT_BASE, 1);   // pool_en=1 -> 4x4 to ActBuf_B

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
            scores[oc] = (int32_t)(((int64_t)acc * SCALE_AFFINE2) >> SCALE_SHIFT);
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

    // Preload all conv weights into Wgt SRAM once; they stay resident for
    // every image instead of being re-packed and re-DMA'd on every pass.
    preload_conv_weights();

    volatile int32_t *scr = (volatile int32_t *)SCORES;

    int correct = 0;
    for (int d = 0; d < 10; d++) {
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
    print_str("/10 correct ===\n");

    if (correct >= 10)
        print_str("DEPLOY SUCCESS.\n");
    else
        print_str("DEPLOY FAILED.\n");
}
