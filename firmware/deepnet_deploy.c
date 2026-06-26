// deepnet_deploy.c — MNIST DeepConvNet inference on SoC
// Runs 15-layer CNN (Conv1-Conv6 + Pool1-3 + Affine1-2) on 10 test images.

#include "firmware.h"
#include "deepnet.h"
#include "deepnet_weights.h"
#include "mnist_test_images.h"
#include "npu_desc.h"
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

// ---- Hardware descriptor-queue deploy switch ----
// 1 = drive the whole MNIST inference through the on-chip descriptor engine
//     (descriptors + qparams resident in DDR, NPU executes them autonomously).
// 0 = legacy per-layer CPU MMIO scheduling.
#ifndef NPU_HW_DESC
#define NPU_HW_DESC 0
#endif

#define WGT_DDR_BASE 0x40100000   // DDR resident weight image (8768 128-bit words)

#define ACT_DDR_BASE 0x40140000   // DDR resident RAW images (camera model: 10*49 packed words)
#define IMG_RAW_WORDS 49          // ceil(784/16): byte-packed pixels per image in DDR
#define ACT_SCRATCH   2048        // Act SRAM word base for the raw image (img_expand source)

// ---- Inter-layer SRAM residency (sram_copy) ----
// Conv activations stay on-chip in two Act-SRAM PING regions that ping-pong:
//   R0 = word base 0    (also the img_expand output / Conv1 input)
//   R1 = word base 1024 (ACT_RES_B)
// Each conv reads the previous layer's region and copies its output to the
// other, so no layer round-trips through DDR. Every layer's activation is
// <= 1024 words, so R0 (0..1023) and R1 (1024..2047) never overlap, and the
// img_expand scratch (2048+) sits above both.
#define ACT_RES_R0    0
#define ACT_RES_B     1024

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
static uint32_t prof_pre_conv, prof_pre_fc, prof_pre_dma;  // one-time weight preload
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
#define NPU_OUT_BUF  0x40010000   // 4096 int32 (16KB)
#define SCORES       0x40015000   // 10 int32

// ---- DMA timeouts ----
// DMA_IRQ_TIMEOUT: spins on RAM flag (much cheaper than MMIO); generous budget.
#define DMA_IRQ_TIMEOUT  500000
// IRQ-wait spins on a RAM flag instead of MMIO polling. Normal completion exits
// the moment the interrupt fires; this is only a fallback.
#define NPU_IRQ_TIMEOUT  1000000

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
    PROF_T0();
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
    PROF_ADD(prof_pre_dma);
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
// HW img_expand engine: Act scratch (raw packed bytes, 16 pixels/word) ->
// Act dst (zero-extended 16-channel tile-major words, pixel in ch0). The HW
// expansion is bit-identical to the old offline pad. n_out = output word count
// (= pixel count). Reads+writes Act PING; polls NPU_DMA_STATUS bit3.
// ================================================================
static void img_expand(uint32_t act_dst_word, uint32_t act_src_word, int n_out)
{
    npu_wr(NPU_DMA_PING_SEL, 0x0);                // Act read+write bank = PING
    npu_wr(NPU_DMA_RD_SRAM_BASE, act_src_word);   // src = Act scratch (packed)
    npu_wr(NPU_DMA_WR_SRAM_BASE, act_dst_word);   // dst = Act output region
    npu_wr(NPU_DMA_RD_LEN, (uint32_t)n_out);      // output word count
    npu_wr(NPU_DMA_EXPAND_TRIG, 1);

    int t = DMA_IRQ_TIMEOUT;
    while (t-- > 0)
        if (npu_rd(NPU_DMA_STATUS) & NPU_DMA_STATUS_EXPAND_DONE) break;
    if (t <= 0) print_str("  EXPAND timeout!\n");
}

// ================================================================
// HW sram_copy engine: Out SRAM -> Act SRAM (on-chip residency, no DDR round
// trip). Reads Out bank `out_bank` (word base 0), writes Act `act_dst_word`.
// `act_pong` selects the Act write bank: 0 = PING (next conv's input region),
// 1 = PONG (FC1 GEMM input). nwords = FULL word count. Polls NPU_DMA_STATUS
// bit2. Time-shares SRAM Port B with axi_dma (mutually exclusive — the NPU is
// idle here, so the copy never races the compute).
// ================================================================
static void dma_out_to_act(uint32_t act_dst_word, int nwords, int out_bank,
                           int act_pong)
{
    PROF_T0();
    npu_wr(NPU_DMA_PING_SEL, (out_bank ? 0x4u : 0x0u) | (act_pong ? 0x1u : 0x0u));
    npu_wr(NPU_DMA_RD_SRAM_BASE, 0);              // src = Out SRAM word base 0
    npu_wr(NPU_DMA_WR_SRAM_BASE, act_dst_word);   // dst = Act SRAM word base
    npu_wr(NPU_DMA_RD_LEN, (uint32_t)nwords);     // FULL word count (copy convention)
    npu_wr(NPU_DMA_COPY_TRIG, 1);                 // start; clears STATUS copy_done

    int t = DMA_IRQ_TIMEOUT;
    while (t-- > 0)
        if (npu_rd(NPU_DMA_STATUS) & NPU_DMA_STATUS_COPY_DONE) break;
    if (t <= 0) print_str("  COPY timeout!\n");

    npu_wr(NPU_DMA_PING_SEL, 0x0);                // restore default banks
    PROF_ADD(prof_load);
}

#if NPU_HW_DESC
// ================================================================
// Hardware descriptor-queue runtime (deploy path).
// Build a list of 16-word descriptors + a resident qparam table in DDR, then
// hand the whole inference to the on-chip descriptor engine. The CPU only
// programs the queue base/count and waits for the done status.
// ================================================================
#define HW_DESC_DDR_BASE   0x4003C000u
#define HW_QPARAM_DDR_BASE 0x4003E000u
#define HW_DESC_MAX        32u
#define HW_QPARAM_MAX      512u

static uint32_t hw_qparam_cursor;

static volatile uint32_t *hw_desc_word(uint32_t desc_idx)
{
    return (volatile uint32_t *)(HW_DESC_DDR_BASE + desc_idx * NPU_HW_DESC_WORDS * 4u);
}

static void hw_desc_clear_words(volatile uint32_t *d)
{
    for (uint32_t i = 0; i < NPU_HW_DESC_WORDS; i++)
        d[i] = 0u;
}

static void hw_desc_set_op_words(volatile uint32_t *d, uint32_t op, uint32_t flags)
{
    d[0] = (op & 0xFFu) | ((NPU_HW_DESC_VERSION & 0xFFu) << 8) |
           ((flags & 0xFFFFu) << 16);
    d[1] = flags >> 16;
}

// Write `count` resident qparam entries (bias/scale/0/shift, 4 words each) to
// DDR and return their base address. Channels >= valid_count are zero-padded.
static uint32_t hw_qparams_write(const int32_t *biases, int count,
                                 uint32_t scale_mul, uint32_t shift,
                                 int valid_count)
{
    uint32_t base = HW_QPARAM_DDR_BASE + hw_qparam_cursor * 16u;
    volatile uint32_t *q = (volatile uint32_t *)base;
    for (int i = 0; i < count; i++) {
        int valid = (i < valid_count);
        q[i * 4 + 0] = valid ? (uint32_t)biases[i] : 0u;
        q[i * 4 + 1] = valid ? scale_mul : 0u;
        q[i * 4 + 2] = 0u;
        q[i * 4 + 3] = valid ? shift : 0u;
    }
    hw_qparam_cursor += (uint32_t)count;
    if (hw_qparam_cursor > HW_QPARAM_MAX)
        print_str("  HW qparam overflow!\n");
    return base;
}

// Same as above but the bias array is indexed from `offset` (used by FC1's
// 16-OC passes that slice a 50-wide bias vector).
static uint32_t hw_qparams_write_offset(const int32_t *biases, int offset,
                                        int count, uint32_t scale_mul,
                                        uint32_t shift, int total_valid)
{
    uint32_t base = HW_QPARAM_DDR_BASE + hw_qparam_cursor * 16u;
    volatile uint32_t *q = (volatile uint32_t *)base;
    for (int i = 0; i < count; i++) {
        int oc = offset + i;
        int valid = (oc < total_valid);
        q[i * 4 + 0] = valid ? (uint32_t)biases[oc] : 0u;
        q[i * 4 + 1] = valid ? scale_mul : 0u;
        q[i * 4 + 2] = 0u;
        q[i * 4 + 3] = valid ? shift : 0u;
    }
    hw_qparam_cursor += (uint32_t)count;
    if (hw_qparam_cursor > HW_QPARAM_MAX)
        print_str("  HW qparam overflow!\n");
    return base;
}

static void hw_desc_dma_ddr_to_act(uint32_t *idx, uint32_t src, uint32_t act_base,
                                   uint32_t words, uint32_t act_pong)
{
    volatile uint32_t *d = hw_desc_word((*idx)++);
    hw_desc_clear_words(d);
    hw_desc_set_op_words(d, NPU_HW_DESC_OP_DMA_DDR_TO_ACT, act_pong ? (1u << 16) : 0u);
    d[2] = src;
    d[4] = act_base;
    d[7] = words;
}

static void hw_desc_dma_out_to_ddr(uint32_t *idx, uint32_t out_base,
                                   uint32_t dst, uint32_t words, uint32_t out_pong)
{
    volatile uint32_t *d = hw_desc_word((*idx)++);
    hw_desc_clear_words(d);
    hw_desc_set_op_words(d, NPU_HW_DESC_OP_DMA_OUT_TO_DDR, out_pong ? (4u << 16) : 0u);
    d[2] = out_base;
    d[4] = dst;
    d[7] = words;
}

static void hw_desc_img_expand(uint32_t *idx, uint32_t src_act,
                               uint32_t dst_act, uint32_t words)
{
    volatile uint32_t *d = hw_desc_word((*idx)++);
    hw_desc_clear_words(d);
    hw_desc_set_op_words(d, NPU_HW_DESC_OP_IMG_EXPAND, 0u);
    d[2] = src_act;
    d[4] = dst_act;
    d[7] = words;
}

static void hw_desc_copy_out_to_act(uint32_t *idx, uint32_t out_base,
                                    uint32_t act_base, uint32_t words,
                                    uint32_t act_pong, uint32_t out_pong)
{
    volatile uint32_t *d = hw_desc_word((*idx)++);
    uint32_t flags = (act_pong ? (1u << 16) : 0u) | (out_pong ? (4u << 16) : 0u);
    hw_desc_clear_words(d);
    hw_desc_set_op_words(d, NPU_HW_DESC_OP_SRAM_COPY_OUT_TO_ACT, flags);
    d[2] = out_base;
    d[4] = act_base;
    d[7] = words;
}

static void hw_desc_conv(uint32_t *idx, uint32_t act, uint32_t wgt, uint32_t out,
                         uint32_t in_w, uint32_t in_h, uint32_t ic, uint32_t oc,
                         uint32_t kh, uint32_t kw, uint32_t stride, uint32_t pad,
                         uint32_t ctrl_flags, uint32_t qparam_base,
                         uint32_t qparam_count)
{
    volatile uint32_t *d = hw_desc_word((*idx)++);
    hw_desc_clear_words(d);
    hw_desc_set_op_words(d, NPU_HW_DESC_OP_CONV2D, ctrl_flags);
    d[2] = act;
    d[3] = wgt;
    d[4] = out;
    d[8] = (in_h << 16) | in_w;
    d[9] = (oc << 16) | ic;
    // pad byte [31:28]=pad_h, [27:24]=pad_w; symmetric pad -> same in both nibbles.
    d[10] = ((((pad & 0xFu) << 4) | (pad & 0xFu)) << 24) | (stride << 16) | (kh << 8) | kw;
    d[11] = qparam_base;
    d[12] = qparam_count;
}

static void hw_desc_gemm(uint32_t *idx, uint32_t act, uint32_t wgt, uint32_t out,
                         uint32_t ic, uint32_t oc, uint32_t ctrl_flags,
                         uint32_t qparam_base, uint32_t qparam_count)
{
    volatile uint32_t *d = hw_desc_word((*idx)++);
    hw_desc_clear_words(d);
    hw_desc_set_op_words(d, NPU_HW_DESC_OP_GEMM, ctrl_flags);
    d[2] = act;
    d[3] = wgt;
    d[4] = out;
    d[8] = (1u << 16) | 1u;
    d[9] = (oc << 16) | ic;
    d[10] = (0u << 24) | (1u << 16) | (1u << 8) | 1u;
    d[11] = qparam_base;
    d[12] = qparam_count;
}

static void hw_desc_stop(uint32_t *idx)
{
    volatile uint32_t *d = hw_desc_word((*idx)++);
    hw_desc_clear_words(d);
    hw_desc_set_op_words(d, NPU_HW_DESC_OP_STOP_IRQ, 0u);
}

static int hw_desc_submit(uint32_t count)
{
    if (count == 0u || count > HW_DESC_MAX)
        return 0;
    npu_wr(NPU_DESC_BASE_LO, HW_DESC_DDR_BASE);
    npu_wr(NPU_DESC_BASE_HI, 0u);
    npu_wr(NPU_DESC_COUNT, count);
    npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_CLEAR_DONE);
    npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_START);

    int t = NPU_IRQ_TIMEOUT * 4;
    while (t-- > 0) {
        uint32_t st = npu_rd(NPU_DESC_STATUS);
        if (st & NPU_DESC_STATUS_ERR) {
            print_str("  DESC err pc=");
            print_dec(npu_rd(NPU_DESC_PC));
            print_str(" code=");
            print_dec(npu_rd(NPU_DESC_ERR));
            print_chr('\n');
            return 0;
        }
        if (st & NPU_DESC_STATUS_DONE)
            break;
    }
    if (t <= 0) {
        print_str("  DESC timeout pc=");
        print_dec(npu_rd(NPU_DESC_PC));
        print_chr('\n');
        return 0;
    }
    npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_CLEAR_DONE);
    return 1;
}
#endif

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
    int pad,           // hardware-pad amount each side (0 = caller pre-padded the input)
    int act_in,        // Act SRAM word base the NPU reads input from (residency ping-pong)
    int act_dst,       // >=0: copy each pass's output to Act act_dst (resident); <0: DMA to out_ddr_addr
    int act_dst_pong,  // 1: residency copy targets Act PONG (FC1 GEMM input) instead of PING
    int row_par,       // 1: 16-row spatial parallelism (task E) + auto row-block (#4)
    int oc_single)     // 1: compute ALL OC tiles in ONE NPU start (decision O)
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

    if (oc_single) {
        // ---- Decision O: ONE NPU start computes ALL OC tiles (HW OC-inner loop) ----
        // wgt_reader walks every 16-OC tile from wgt_base; the 64-entry regfile
        // holds bias/scale/shift for all `oc` channels.  im2col + activation rows
        // are loaded once and reused across OC tiles (the main win).  row_par
        // (task E) packs 16 output pixels/group; row_block (#4) auto-engages on
        // narrow (out_w==8) layers.  All tiles write Out bank 0 consecutively
        // (tile p at word p*out_words), so one drain/copy moves the whole layer.
        int rb_out_w  = (in_w - kw) / sx + 1;
        int row_block = row_par && (rb_out_w == 8);

        npu_wr(NPU_IN_W, in_w);
        npu_wr(NPU_IN_H, in_h);
        npu_wr(NPU_IC, ic);
        npu_wr(NPU_OC, oc);                  // full OC: FSM tiles internally
        npu_wr(NPU_KERNEL, (kh << 8) | kw);
        npu_wr(NPU_STRIDE, (sx << 8) | sy);
        npu_wr(NPU_PAD, (pad << 8) | pad);
        npu_wr(NPU_ACT_ADDR_A, act_in);
        npu_wr(NPU_WGT_ADDR_A, wgt_base);    // wgt_reader addresses all tiles from base
        npu_wr(NPU_OUT_ADDR_A, 0);
        for (int ch = 0; ch < oc; ch++) {
            npu_wr(NPU_BIAS(ch),  (uint32_t)biases[ch]);
            npu_wr(NPU_SCALE(ch), bias_scale);
            npu_wr(NPU_SHIFT(ch), SCALE_SHIFT);
        }
        npu_irq_flag = 0;
        npu_wr(NPU_CTRL, NPU_CTRL_START | NPU_CTRL_RELU_EN | NPU_CTRL_OC_SINGLE |
                         (pool_en   ? NPU_CTRL_POOL_EN   : 0) |
                         (pad       ? NPU_CTRL_HW_PAD    : 0) |
                         (row_par   ? NPU_CTRL_ROW_PAR   : 0) |
                         (row_block ? NPU_CTRL_ROW_BLOCK : 0));
        int t = NPU_IRQ_TIMEOUT;
        while (t-- > 0) if (npu_irq_flag) break;
        if (t <= 0) print_str("  NPU IRQ timeout!\n");
        if (npu_rd(NPU_STATUS) & NPU_STATUS_DMA_RD_ERR) print_str("  NPU DMA rd err!\n");
        if (npu_rd(NPU_STATUS) & NPU_STATUS_DMA_WR_ERR) print_str("  NPU DMA wr err!\n");

        if (act_dst >= 0)
            dma_out_to_act((uint32_t)act_dst, out_words * oc_passes, 0, act_dst_pong);
        else
            dma_out_to_ddr(out_ddr_addr, 0, out_words * oc_passes, 0);
#if NPU_PROFILE
        {
            uint32_t _d = rdcycle32() - _pt;
            prof_npu += _d;
            if (prof_conv_idx < 6) prof_npu_layer[prof_conv_idx++] += _d;
        }
#endif
        return;
    }

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
        npu_wr(NPU_ACT_ADDR_A, act_in);   // input region base (residency ping-pong)
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
        // #4 row-block packing auto-engages on narrow (out_w==8) layers.
        int rb_out_w  = (in_w - kw) / sx + 1;
        int row_block = row_par && (rb_out_w == 8);
        npu_wr(NPU_CTRL, NPU_CTRL_START | NPU_CTRL_RELU_EN |
                         (pool_en ? NPU_CTRL_POOL_EN : 0) |
                         (out_bank ? NPU_CTRL_OUT_PING : 0) |
                         (pad ? NPU_CTRL_HW_PAD : 0) |
                         (row_par ? NPU_CTRL_ROW_PAR : 0) |
                         (row_block ? NPU_CTRL_ROW_BLOCK : 0));

#if NPU_OC_OVERLAP
        // OVERLAP (DDR path only): while the NPU computes this pass, drain the
        // PREVIOUS pass's output (in prev_bank) to DDR.  Resident layers copy
        // serially after compute (below) and do not use the overlap drain.
        if (act_dst < 0 && prev_pass >= 0)
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

        if (act_dst >= 0) {
            // RESIDENT: copy this pass's output (Out bank out_bank, word base 0)
            // to Act act_dst + pass*out_words.  Serial (NPU idle) — frees the
            // per-pass Out bank before the next pass reuses it.  No DDR traffic.
            dma_out_to_act((uint32_t)act_dst + pass * out_words,
                           out_words, out_bank, act_dst_pong);
        } else {
#if NPU_OC_OVERLAP
            prev_pass = pass;
            prev_bank = out_bank;
#else
            // SERIAL baseline: no overlap — drain THIS pass right after it
            // finishes, before starting the next pass.
            dma_out_to_ddr(out_ddr_addr + pass * out_words * 16, 0,
                           out_nbeats, out_bank);
#endif
        }
    }

#if NPU_OC_OVERLAP
    // Drain the final pass's output (DDR path only; no further compute to overlap
    // with).  Resident layers already copied every pass on-chip via sram_copy.
    if (act_dst < 0)
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

// ================================================================
// Weights resident in DDR (flash/boot image): CPU streams them into Wgt SRAM
// via DMA. Layout: conv 4608 words (PING), FC 4160 words (PONG).
// ================================================================
static void dma_ddr_to_wgt_chunks(uint32_t ddr_addr, int sram_base, int nwords)
{
    int sent = 0;
    while (sent < nwords) {
        int chunk = nwords - sent;
        if (chunk > 256) chunk = 256;
        dma_ddr_to_wgt(ddr_addr + sent * 16, sram_base + sent, chunk);
        sent += chunk;
    }
}

static void preload_weights_dma(void)
{
    npu_wr(NPU_DMA_PING_SEL, 0x0);                             // Wgt PING (conv)
    dma_ddr_to_wgt_chunks(WGT_DDR_BASE, 0, 4608);             // conv1..conv6
    npu_wr(NPU_DMA_PING_SEL, 0x2);                             // Wgt PONG (FC)
    dma_ddr_to_wgt_chunks(WGT_DDR_BASE + 4608 * 16, 0, 4160); // FC1+FC2
    npu_wr(NPU_DMA_PING_SEL, 0x0);                             // restore
}

// 通用 NPU GEMM/全连接: out[0..out_dim) = quant(act · W + bias), 可选 ReLU.
// Runs with global ping_pong=1 so the FSM latches the *_B (pong) base regs:
// input vector lives in Act PONG, weights resident in Wgt PONG (pack_fc_tile
// layout at wgt_base), output INT8 channel-major to out_ddr. OC tiled by 16
// (decision D). in_dim <= 1024 (HW ic_group counter width).
static void npu_gemm_pass(int in_dim, int out_dim, int scale_mul_val,
                          const int32_t *biases, int relu_en,
                          uint32_t in_ddr, uint32_t out_ddr, int wgt_base,
                          int in_resident,  // 1: input already in Act PONG, skip DMA
                          int reduce,       // 1: GEMM 16-row IC-reduction (decision M)
                          int int32_out)    // 1: raw INT32 output (decision Q) — 4 words/pass
{
    int icg        = (in_dim + 15) / 16;
    int oc_passes  = (out_dim + 15) / 16;
    int tile_words = 16 * icg;

    if (!in_resident) {
        npu_wr(NPU_DMA_PING_SEL, 0x1);         // input vector DMA -> Act PONG
        dma_ddr_to_act(in_ddr, 0, icg);
        npu_wr(NPU_DMA_PING_SEL, 0x0);
    }
    // else: input already resident in Act PONG base 0 (e.g., Conv6 -> FC1).

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
        // INT32 raw output uses 4 Out words/pass (16xINT32); INT8 uses 1.
        npu_wr(NPU_OUT_ADDR_B, (uint32_t)(int32_out ? pass * 4 : pass));

        for (int ch = 0; ch < 16; ch++) {
            int oc = pass * 16 + ch;
            npu_wr(NPU_BIAS(ch),  (oc < out_dim) ? (uint32_t)biases[oc] : 0u);
            npu_wr(NPU_SCALE(ch), (oc < out_dim) ? (uint32_t)scale_mul_val : 0u);
            npu_wr(NPU_SHIFT(ch), SCALE_SHIFT);
        }

        npu_irq_flag = 0;
        npu_wr(NPU_CTRL, NPU_CTRL_START | NPU_CTRL_PING_PONG | NPU_CTRL_GEMM_EN |
                         (relu_en   ? NPU_CTRL_RELU_EN     : 0) |
                         (reduce    ? NPU_CTRL_GEMM_REDUCE : 0) |
                         (int32_out ? NPU_CTRL_INT32_OUT   : 0));
        int t = NPU_IRQ_TIMEOUT;
        while (t-- > 0)
            if (npu_irq_flag) break;
        if (t <= 0) print_str("  GEMM IRQ timeout!\n");
    }
    // INT8: oc_passes words (1/pass). INT32 (decision Q): 4 words/pass.
    dma_out_to_ddr(out_ddr, 0, int32_out ? oc_passes * 4 : oc_passes, 0);
}

#if NPU_GEMM_PARITY
// ================================================================
// CPU affine layer with INT8 quantization (parity oracle only)
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
#endif

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
void deepnet_inference(const int8_t *input, int32_t *scores, int img_idx)
{
#if NPU_GEMM_PARITY
    // CPU affine oracle operands (parity build only). Kept in fast private RAM.
    static int8_t affine_in[1024];   // reorder output / Affine1 input
    static int8_t affine_mid[64];    // Affine1 output / Affine2 input (50 used)
#endif
#if NPU_PROFILE
    prof_conv_idx = 0;   // per-image: conv layers index 0..5 in call order
#endif
    (void)input;     // image comes pre-formatted from DDR; img_idx selects it

#ifdef DEBUG_VERBOSE
    // Debug: print first 8 input pixels
    print_str("  Input[0..7]:");
    for (int i = 0; i < 8; i++) { print_chr(' '); print_hex((uint32_t)(uint8_t)input[i], 2); }
    print_chr('\n');
#endif

    // ---- Conv1: image(28×28×1) → HW-pad(30×30) → Conv → 28×28×16 → ActBuf_B ----
    // Raw image resident in DDR (camera/sensor writes raw bytes to SDRAM). DMA the
    // 49 byte-packed words into an Act-SRAM scratch region, then the HW img_expand
    // engine expands them in SRAM into the tile-major 16-channel input the NPU
    // reads -- no offline pad, no CPU pixel scatter.
    dma_ddr_to_act(ACT_DDR_BASE + img_idx * IMG_RAW_WORDS * 16, ACT_SCRATCH, IMG_RAW_WORDS);
    img_expand(0, ACT_SCRATCH, 28 * 28);
    npu_conv_pass(30, 30, 16, 16, 3, 3, 1, 1,
                  1, SCALE_CONV1, conv1_b,
                  ACT_BUF_B, 784, CONV1_WGT_BASE, 0, 1,
                  ACT_RES_R0, ACT_RES_B, 0, 1, 0);   // in=R0 (img_expand), out->R1; row_par, 1 tile
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

    // ---- Conv2: R1(28×28×16) → HW-pad(30×30) → Conv → 28×28 → 2x2 pool → R0 ----
    // Conv1 output is resident in R1 (no DDR load); the FSM HW-pads on read.
    npu_conv_pass(30, 30, 16, 16, 3, 3, 1, 1,
                  16, SCALE_CONV2, conv2_b,
                  ACT_BUF_B, 784, CONV2_WGT_BASE, 1, 1,
                  ACT_RES_B, ACT_RES_R0, 0, 1, 0);   // in=R1, out->R0; row_par, 1 tile
#ifdef DEBUG_VERBOSE
    dbg_layer("Pool1", ACT_BUF_B, 196 * 16);
#endif

    // ---- Conv3: R0(14×14×16) → HW-pad(16×16) → Conv → 14×14×32 → R1 ----
    npu_conv_pass(16, 16, 16, 32, 3, 3, 1, 1,
                  16, SCALE_CONV3, conv3_b,
                  ACT_BUF_A, 196, CONV3_WGT_BASE, 0, 1,
                  ACT_RES_R0, ACT_RES_B, 0, 1, 1);   // in=R0, out->R1; row_par + oc_single
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

    // ---- Conv4: R1(14×14×32) → HW-pad(18×18, pad=2) → Conv → 16×16×32 → 2x2 pool → R0 ----
    // NPU does Conv4 + Pool2: 16x16 conv -> 2x2 maxpool -> 8x8.
    npu_conv_pass(18, 18, 32, 32, 3, 3, 1, 1,
                  32, SCALE_CONV4, conv4_b,
                  ACT_BUF_A, 256, CONV4_WGT_BASE, 1, 2,
                  ACT_RES_B, ACT_RES_R0, 0, 1, 1);   // in=R1, out->R0; row_par + oc_single (pooled)
#ifdef DEBUG_VERBOSE
    dbg_layer("Pool2", ACT_BUF_A, 64 * 32);
#endif

    // ---- Conv5: R0(8×8×32) → HW-pad(10×10) → Conv → 8×8×64 → R1 ----
    npu_conv_pass(10, 10, 32, 64, 3, 3, 1, 1,
                  32, SCALE_CONV5, conv5_b,
                  ACT_BUF_B, 64, CONV5_WGT_BASE, 0, 1,
                  ACT_RES_R0, ACT_RES_B, 0, 1, 1);   // in=R0, out->R1; row_par + oc_single
#ifdef DEBUG_VERBOSE
    dbg_layer("Conv5", ACT_BUF_B, 64 * 64);
#endif

    // ---- Conv6: R1(8×8×64) → HW-pad(10×10) → Conv → 8×8×64 → 2x2 pool → Act PONG ----
    // NPU does Conv6 + Pool3: 8x8 conv -> 2x2 maxpool -> 4x4.  The pooled output
    // (Pool3, tile-major) is copied straight into Act PONG base 0 — exactly where
    // the FC1 GEMM reads its input — so FC1 needs no DDR load (act_dst_pong=1).
    npu_conv_pass(10, 10, 64, 64, 3, 3, 1, 1,
                  64, SCALE_CONV6, conv6_b,
                  ACT_BUF_B, 64, CONV6_WGT_BASE, 1, 1,
                  ACT_RES_B, 0, 1, 1, 1);   // in=R1, out->Act PONG[0] (FC1 input); row_par + oc_single (pooled)

    // ---- Pool3 -> FC1: no runtime transpose, no DDR ----
    // FC1 weights are pre-packed in CONV-OUTPUT order (pack_fc1_convorder), so the
    // NPU GEMM reads the Pool3 output directly. Conv6 left Pool3 resident in Act
    // PONG base 0 (the GEMM input bank), so FC1 reads it on-chip with no DDR load.
    // The dot product is order-independent; aligning the weight column order to the
    // activation order removes the old CPU Pool3->FC1 reorder from the deploy path.
#if NPU_GEMM_PARITY
    // The CPU affine oracle still wants the channel-major input vector.
    {
        int8_t *pool_out = (int8_t *)ACT_BUF_B;
        for (int ch = 0; ch < 64; ch++) {
            int pass = ch / 16, ch_in = ch % 16;
            for (int pos = 0; pos < 16; pos++)
                affine_in[ch * 16 + pos] = pool_out[(pass * 16 + pos) * 16 + ch_in];
        }
    }
#endif

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
        // ---- Deploy: NPU FC1 (1024→50, ReLU) → FC1_OUT_DDR (INT8) ----
        // Input (Pool3) is already resident in Act PONG base 0 (Conv6 residency),
        // so in_resident=1 skips the DDR->Act load.
        npu_gemm_pass(AFFINE1_IN, AFFINE1_OUT, SCALE_AFFINE1, affine1_b, 1,
                      ACT_BUF_B, FC1_OUT_DDR, FC1_WGT_BASE, 1, 1, 0);
        // ---- Deploy: NPU FC2 (50→10, raw INT32, decision Q) → FC2_OUT_DDR ----
        // Reads the NPU FC1 INT8 output (channel-major, OCs 50..63 are exact 0
        // from padded weights) and emits scaled un-clamped INT32 logits — no INT8
        // saturation of the final logits, so argmax matches the old CPU FC2.
        // int32_out=1 -> 4 Out words (16xINT32, OCs 10..15 are padding).
        npu_gemm_pass(AFFINE2_IN, AFFINE2_OUT, SCALE_AFFINE2, affine2_b, 0,
                      FC1_OUT_DDR, FC2_OUT_DDR, FC2_WGT_BASE, 0, 1, 1);
        volatile int32_t *f2 = (volatile int32_t *)FC2_OUT_DDR;
        for (int oc = 0; oc < AFFINE2_OUT; oc++)
            scores[oc] = f2[oc];
        PROF_ADD(prof_affine);
    }
#endif

#if NPU_GEMM_PARITY
    {
        // NPU FC1 vs CPU affine1 — must be bit-identical (same quant path).
        // NOTE: with SRAM residency the CPU oracle's Pool3 input (read from DDR
        // ACT_BUF_B above) is stale, so this parity check is only meaningful in a
        // build that drains Conv6 to DDR. Input here is resident in Act PONG.
        npu_gemm_pass(AFFINE1_IN, AFFINE1_OUT, SCALE_AFFINE1, affine1_b, 1,
                      ACT_BUF_B, FC1_OUT_DDR, FC1_WGT_BASE, 1, 0, 0);
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
                      FC1_OUT_DDR, FC2_OUT_DDR, FC2_WGT_BASE, 0, 0, 0);
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
// Print the RTL hardware performance counters (array util, AXI bandwidth).
// ================================================================
static void print_perf(void)
{
    uint32_t tot    = npu_rd(NPU_PERF_CYC_TOTAL);
    uint32_t busy   = npu_rd(NPU_PERF_CYC_BUSY);
    uint32_t arr    = npu_rd(NPU_PERF_CYC_ARR);
    uint32_t rdb    = npu_rd(NPU_PERF_RD_BEATS);
    uint32_t wrb    = npu_rd(NPU_PERF_WR_BEATS);
    uint32_t rdbusy = npu_rd(NPU_PERF_RD_BUSY);
    uint32_t wrbusy = npu_rd(NPU_PERF_WR_BUSY);

    print_str("\n=== HW Perf (RTL counters) ===\n");
    print_str("  cyc_total=");  print_dec(tot);
    print_str(" npu_busy=");    print_dec(busy);
    print_str(" arr_active=");  print_dec(arr); print_chr('\n');
    print_str("  rd_beats=");   print_dec(rdb);
    print_str(" wr_beats=");    print_dec(wrb);
    print_str(" rd_busy=");     print_dec(rdbusy);
    print_str(" wr_busy=");     print_dec(wrbusy); print_chr('\n');

    if (tot)    { print_str("  array_util(arr/total) = "); print_dec((uint32_t)((uint64_t)arr*100/tot));     print_str("%\n"); }
    if (busy)   { print_str("  array_eff (arr/busy)  = "); print_dec((uint32_t)((uint64_t)arr*100/busy));    print_str("%\n"); }
    if (tot)    { print_str("  npu_active(busy/total)= "); print_dec((uint32_t)((uint64_t)busy*100/tot));    print_str("%\n"); }
    if (rdbusy) { print_str("  rd_bw_util(beats/busy)= "); print_dec((uint32_t)((uint64_t)rdb*100/rdbusy));  print_str("%\n"); }
    if (wrbusy) { print_str("  wr_bw_util(beats/busy)= "); print_dec((uint32_t)((uint64_t)wrb*100/wrbusy));  print_str("%\n"); }
    print_str("  peak=1.64 TOPS@200MHz; effective = array_util(arr/total) x peak\n");
}

#if NPU_HW_DESC
// Append one conv layer (conv + on-chip residency copy) to the descriptor list.
static void hw_desc_add_conv_layer(uint32_t *di,
                                   int in_w, int in_h, int ic, int oc,
                                   int kh, int kw, int stride, int pad,
                                   int scale, const int32_t *biases,
                                   int wgt_base, int pool_en,
                                   int act_in, int act_dst, int act_dst_pong,
                                   int row_par, int oc_single)
{
    int out_spatial = ((in_w - kw) / stride + 1) * ((in_h - kh) / stride + 1);
    int out_words = pool_en ? (out_spatial >> 2) : out_spatial;
    int oc_passes = (oc + 15) >> 4;
    int row_block = row_par && (((in_w - kw) / stride + 1) == 8);
    uint32_t flags = NPU_CTRL_RELU_EN |
                     (pool_en ? NPU_CTRL_POOL_EN : 0u) |
                     (pad ? NPU_CTRL_HW_PAD : 0u) |
                     (row_par ? NPU_CTRL_ROW_PAR : 0u) |
                     (row_block ? NPU_CTRL_ROW_BLOCK : 0u) |
                     (oc_single ? NPU_CTRL_OC_SINGLE : 0u);
    uint32_t qcnt = oc_single ? (uint32_t)oc : 16u;
    uint32_t qbase = hw_qparams_write(biases, (int)qcnt,
                                      (uint32_t)scale, SCALE_SHIFT, (int)qcnt);
    hw_desc_conv(di, (uint32_t)act_in, (uint32_t)wgt_base, 0u,
                 (uint32_t)in_w, (uint32_t)in_h, (uint32_t)ic, (uint32_t)oc,
                 (uint32_t)kh, (uint32_t)kw, (uint32_t)stride, (uint32_t)pad,
                 flags, qbase, qcnt);
    hw_desc_copy_out_to_act(di, 0u, (uint32_t)act_dst,
                            (uint32_t)(out_words * oc_passes),
                            (uint32_t)act_dst_pong, 0u);
}

// Whole MNIST CNN+MLP described as a single descriptor program.
static int deepnet_inference_hw_desc(int img_idx, int32_t *scores)
{
    uint32_t di = 0;
    hw_qparam_cursor = 0;

    hw_desc_dma_ddr_to_act(&di, ACT_DDR_BASE + (uint32_t)img_idx * IMG_RAW_WORDS * 16u,
                           ACT_SCRATCH, IMG_RAW_WORDS, 0u);
    hw_desc_img_expand(&di, ACT_SCRATCH, ACT_RES_R0, 28u * 28u);

    hw_desc_add_conv_layer(&di, 30, 30, 16, 16, 3, 3, 1, 1,
                           SCALE_CONV1, conv1_b, CONV1_WGT_BASE, 0,
                           ACT_RES_R0, ACT_RES_B, 0, 1, 0);
    hw_desc_add_conv_layer(&di, 30, 30, 16, 16, 3, 3, 1, 1,
                           SCALE_CONV2, conv2_b, CONV2_WGT_BASE, 1,
                           ACT_RES_B, ACT_RES_R0, 0, 1, 0);
    hw_desc_add_conv_layer(&di, 16, 16, 16, 32, 3, 3, 1, 1,
                           SCALE_CONV3, conv3_b, CONV3_WGT_BASE, 0,
                           ACT_RES_R0, ACT_RES_B, 0, 1, 1);
    hw_desc_add_conv_layer(&di, 18, 18, 32, 32, 3, 3, 1, 2,
                           SCALE_CONV4, conv4_b, CONV4_WGT_BASE, 1,
                           ACT_RES_B, ACT_RES_R0, 0, 1, 1);
    hw_desc_add_conv_layer(&di, 10, 10, 32, 64, 3, 3, 1, 1,
                           SCALE_CONV5, conv5_b, CONV5_WGT_BASE, 0,
                           ACT_RES_R0, ACT_RES_B, 0, 1, 1);
    hw_desc_add_conv_layer(&di, 10, 10, 64, 64, 3, 3, 1, 1,
                           SCALE_CONV6, conv6_b, CONV6_WGT_BASE, 1,
                           ACT_RES_B, 0, 1, 1, 1);

    int fc1_tile_words = ((AFFINE1_IN + 15) >> 4) * 16;
    int fc1_passes = (AFFINE1_OUT + 15) >> 4;
    for (int pass = 0; pass < fc1_passes; pass++) {
        uint32_t qbase = hw_qparams_write_offset(affine1_b, pass * 16, 16,
                                                 SCALE_AFFINE1, SCALE_SHIFT,
                                                 AFFINE1_OUT);
        hw_desc_gemm(&di, 0u, (uint32_t)(FC1_WGT_BASE + pass * fc1_tile_words),
                     (uint32_t)pass, AFFINE1_IN, 16u,
                     NPU_CTRL_PING_PONG | NPU_CTRL_RELU_EN | NPU_CTRL_GEMM_REDUCE,
                     qbase, 16u);
    }
    hw_desc_dma_out_to_ddr(&di, 0u, FC1_OUT_DDR, (uint32_t)fc1_passes, 0u);

    hw_desc_dma_ddr_to_act(&di, FC1_OUT_DDR, 0u,
                           (uint32_t)((AFFINE2_IN + 15) >> 4), 1u);
    {
        uint32_t qbase = hw_qparams_write_offset(affine2_b, 0, 16,
                                                 SCALE_AFFINE2, SCALE_SHIFT,
                                                 AFFINE2_OUT);
        hw_desc_gemm(&di, 0u, FC2_WGT_BASE, 0u, AFFINE2_IN, 16u,
                     NPU_CTRL_PING_PONG | NPU_CTRL_GEMM_REDUCE | NPU_CTRL_INT32_OUT,
                     qbase, 16u);
    }
    hw_desc_dma_out_to_ddr(&di, 0u, FC2_OUT_DDR, 4u, 0u);
    hw_desc_stop(&di);

    if (!hw_desc_submit(di))
        return 0;

    volatile int32_t *f2 = (volatile int32_t *)FC2_OUT_DDR;
    for (int oc = 0; oc < AFFINE2_OUT; oc++)
        scores[oc] = f2[oc];
    return 1;
}
#endif

// ================================================================
// usercode7: run inference on 2 MNIST test images
// ================================================================
void usercode7(void)
{
    print_str("=== MNIST DeepConvNet Deploy ===\n");

    {
        const npu_desc_t smoke[] = {
            { .op = NPU_DESC_OP_NOP },
            { .op = NPU_DESC_OP_NOP }
        };
        if (!npu_desc_run_many(smoke, 2u)) {
            print_str("DESC SMOKE FAIL\n");
            return;
        }
    }

    // Preload all conv weights into Wgt SRAM once; they stay resident for
    // every image instead of being re-packed and re-DMA'd on every pass.
#if NPU_PROFILE
    uint32_t _pc = rdcycle32();
#endif
    // Weights resident in DDR (flash/boot image); CPU streams them via DMA.
    preload_weights_dma();
#if NPU_PROFILE
    prof_pre_conv = rdcycle32() - _pc;   // whole preload (DMA from resident DDR)
#endif

    volatile int32_t *scr = (volatile int32_t *)SCORES;

    int correct = 0;
#if NPU_HW_DESC
    (void)mnist_images;   // descriptor path selects images from resident DDR
#endif
    npu_wr(NPU_PERF_CLR, 1);   // clear RTL performance counters (HW)
    for (int d = 0; d < 10; d++) {
        print_str("Digit "); print_dec(d); print_str(": ");

#if NPU_PROFILE
        uint32_t _ti = rdcycle32();
#endif
#if NPU_HW_DESC
        if (!deepnet_inference_hw_desc(d, (int32_t *)SCORES)) {
            print_str("DESC INFER FAIL\n");
            return;
        }
#else
        deepnet_inference(mnist_images[d], (int32_t *)SCORES, d);
#endif
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

    print_perf();   // RTL hardware counters: array util + AXI bandwidth

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
    print_str("---- weight preload (one-time, before counter clear) ----\n");
    print_str("  pre_conv : "); print_dec(prof_pre_conv); print_chr('\n');
    print_str("  pre_fc   : "); print_dec(prof_pre_fc);   print_chr('\n');
    print_str("  pre_dma  : "); print_dec(prof_pre_dma);  print_chr('\n');
    print_str("  pre_cpu  : "); print_dec(prof_pre_conv + prof_pre_fc - prof_pre_dma); print_chr('\n');
    print_str("=======================================\n");
#endif
}
