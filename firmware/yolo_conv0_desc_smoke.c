// conv0 (model.0, 3->16 3x3 s2 pad1) @320 driven ENTIRELY by the hardware
// descriptor queue. Mirrors yolo_run_conv2d_tiled's row-strip tiling, but each
// strip is emitted as one descriptor program (DMA-in rows + CONV2D + drain) and
// handed to the descriptor engine in one submit. Proves a large, on-chip-too-big
// YOLO conv can run via descriptors. Validated against the conv0 golden checksum.
//
// Config that the descriptor conv needs (exact-SiLU LUT, requant=0, pad_value)
// is carried in-band: the LUT is CPU-preloaded once (it persists in the NPU),
// and OP_ACTIVATION_CFG sets silu_exact + pad_value for the descriptor convs.

#include "firmware.h"
#include "yolo_ops.h"
#include "npu_desc.h"
#include "yolo_conv0_320_exact_data.h"
#include <stdint.h>

#define ACT_DDR     0x40400000u
#define WGT_DDR     0x405C0000u
#define OUT_DDR     0x40600000u
#define PAD_ROW_DDR 0x40780000u
#define DESC_DDR    0x40000000u
#define QPARAM_DDR  0x40080000u
#define WGT_BASE    0u
#define STRIP_OUT_ROWS 16u
#define DMA_RD_MAX  256u   // DDR->Act beats/descriptor (axi_dma read len 8-bit)
#define DMA_WR_MAX  64u    // Out->DDR beats/descriptor

static inline void npu_wr(uint32_t a, uint32_t d){ *(volatile uint32_t*)a = d; }
static inline uint32_t npu_rd(uint32_t a){ return *(volatile uint32_t*)a; }

static void wrw(uint32_t a, uint32_t w, const uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); p[0]=l[0];p[1]=l[1];p[2]=l[2];p[3]=l[3]; }
static int32_t rs8(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); uint32_t v=p[b>>2]>>((b&3u)*8u)&0xFFu; return (v&0x80u)?((int32_t)v-256):(int32_t)v; }

// ---- descriptor builders (write 16-word records into DESC_DDR) ----
static volatile uint32_t *dword(uint32_t idx)
{ return (volatile uint32_t *)(DESC_DDR + idx * NPU_HW_DESC_WORDS * 4u); }
static void dclr(volatile uint32_t *d)
{ uint32_t i; for (i=0u;i<NPU_HW_DESC_WORDS;i++) d[i]=0u; }
static void dop(volatile uint32_t *d, uint32_t op, uint32_t flags)
{ d[0]=(op&0xFFu)|((NPU_HW_DESC_VERSION&0xFFu)<<8)|((flags&0xFFFFu)<<16); d[1]=flags>>16; }

static void d_act_cfg(uint32_t *idx, int32_t pad_value)
{
    volatile uint32_t *d = dword((*idx)++);
    dclr(d);
    dop(d, NPU_HW_DESC_OP_ACTIVATION_CFG, 0u);
    d[2] = (uint32_t)pad_value & 0xFFu;     // pad_value (input zp)
    d[3] = 0u;                              // requant = 0 (exact mode bakes it in)
    d[4] = 0x1u;                            // flags: silu_exact_en
    d[5] = 0u;                              // clip_max -> default 127
}

static void d_dma_in(uint32_t *idx, uint32_t src_ddr, uint32_t act_base, uint32_t words)
{
    uint32_t off = 0u;
    while (off < words) {
        uint32_t ch = words - off;
        volatile uint32_t *d;
        if (ch > DMA_RD_MAX) ch = DMA_RD_MAX;
        d = dword((*idx)++); dclr(d);
        dop(d, NPU_HW_DESC_OP_DMA_DDR_TO_ACT, 0u);
        d[2] = src_ddr + off * 16u;
        d[4] = act_base + off;
        d[7] = ch;
        off += ch;
    }
}

static void d_drain(uint32_t *idx, uint32_t out_base, uint32_t dst_ddr, uint32_t words)
{
    uint32_t off = 0u;
    while (off < words) {
        uint32_t ch = words - off;
        volatile uint32_t *d;
        if (ch > DMA_WR_MAX) ch = DMA_WR_MAX;
        d = dword((*idx)++); dclr(d);
        dop(d, NPU_HW_DESC_OP_DMA_OUT_TO_DDR, 0u);
        d[2] = out_base + off;              // Out SRAM word base
        d[4] = dst_ddr + off * 16u;         // DDR dst
        d[7] = ch;
        off += ch;
    }
}

static void d_conv(uint32_t *idx, uint32_t act, uint32_t in_w, uint32_t in_h,
                   uint32_t in_c, uint32_t out_c, uint32_t stride,
                   uint32_t pad_w, uint32_t pad_h, uint32_t qbase, uint32_t qcnt)
{
    volatile uint32_t *d = dword((*idx)++);
    dclr(d);
    dop(d, NPU_HW_DESC_OP_CONV2D,
        NPU_CTRL_OC_SINGLE | NPU_CTRL_HW_PAD);
    d[2] = act;
    d[3] = WGT_BASE;
    d[4] = 0u;                              // Out SRAM base 0
    d[8] = (in_h << 16) | in_w;
    d[9] = (out_c << 16) | in_c;
    d[10] = (((pad_h & 0xFu) << 28) | ((pad_w & 0xFu) << 24)) |
            (stride << 16) | (3u << 8) | 3u;
    d[11] = qbase;
    d[12] = qcnt;
}

static void d_stop(uint32_t *idx)
{ volatile uint32_t *d = dword((*idx)++); dclr(d); dop(d, NPU_HW_DESC_OP_STOP_IRQ, 0u); }

static int d_submit(uint32_t count)
{
    int t;
    npu_wr(NPU_DESC_BASE_LO, DESC_DDR);
    npu_wr(NPU_DESC_BASE_HI, 0u);
    npu_wr(NPU_DESC_COUNT, count);
    npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_CLEAR_DONE);
    npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_START);
    t = 4000000;
    while (t-- > 0) {
        uint32_t st = npu_rd(NPU_DESC_STATUS);
        if (st & NPU_DESC_STATUS_ERR) {
            print_str("  DESC err pc="); print_dec(npu_rd(NPU_DESC_PC));
            print_str(" code="); print_dec(npu_rd(NPU_DESC_ERR)); print_str("\n");
            return 0;
        }
        if (st & NPU_DESC_STATUS_DONE) break;
    }
    if (t <= 0) { print_str("  DESC timeout pc="); print_dec(npu_rd(NPU_DESC_PC)); print_str("\n"); return 0; }
    npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_CLEAR_DONE);
    return 1;
}

void usercode7(void)
{
    uint32_t i, pos, oc, errors = 0u;
    const uint32_t in_w = C0E_IN_W, in_h = C0E_IN_H, in_c = C0E_IC, out_c = C0E_OC;
    const uint32_t stride = 2u, kh = 3u, pad = 1u;
    uint32_t out_w = (in_w + 2u*pad - kh)/stride + 1u;
    uint32_t out_h = (in_h + 2u*pad - kh)/stride + 1u;
    uint32_t oy0;

    print_str("YOLO CONV0 @320 DESCRIPTOR-TILED SMOKE\n");

    // Stage image + weights into DDR (camera/flash model).
    for (i = 0u; i < C0E_ACT_WORDS; i++) wrw(ACT_DDR, i, yolo_conv0_320e_act_words[i]);
    for (i = 0u; i < C0E_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_conv0_320e_wgt_words[i]);

    // One pad row (in_w words, every lane = pad_value) for vertical halo DMA.
    {
        uint32_t lane = (uint32_t)C0E_PAD_VALUE & 0xFFu;
        uint32_t word = lane | (lane<<8) | (lane<<16) | (lane<<24);
        volatile uint32_t *p = (volatile uint32_t *)PAD_ROW_DDR;
        for (i = 0u; i < in_w * 4u; i++) p[i] = word;
    }

    // CPU preload: weights resident in Wgt SRAM, exact-SiLU LUT in the NPU.
    if (!yolo_dma_ddr_to_wgt(WGT_DDR, WGT_BASE, C0E_WGT_WORDS)) { print_str("wgt dma fail\n"); errors++; }
    yolo_load_silu_lut(yolo_conv0_320e_silu_lut);

    // Resident qparam table (bias/scale/0/shift) for the 16 OCs.
    {
        volatile uint32_t *q = (volatile uint32_t *)QPARAM_DDR;
        for (oc = 0u; oc < out_c; oc++) {
            q[oc*4+0] = (uint32_t)yolo_conv0_320e_bias_q[oc];
            q[oc*4+1] = yolo_conv0_320e_scale_mul[oc];
            q[oc*4+2] = 0u;
            q[oc*4+3] = yolo_conv0_320e_scale_shift[oc];
        }
    }

    *(volatile uint32_t *)NPU_PERF_CLR = 1u;

    // Row-strip loop: one descriptor program per strip.
    for (oy0 = 0u; oy0 < out_h; oy0 += STRIP_OUT_ROWS) {
        uint32_t so = out_h - oy0;
        uint32_t strip_in_h, ri, di = 0u;
        int32_t ir0;
        if (so > STRIP_OUT_ROWS) so = STRIP_OUT_ROWS;
        strip_in_h = (so - 1u) * stride + kh;   // input rows for 'so' out rows, pad_h=0
        ir0 = (int32_t)(oy0 * stride) - (int32_t)pad;

        d_act_cfg(&di, C0E_PAD_VALUE);

        // Stage strip input rows (icg=1), DMA'ing pad rows where out of bounds.
        for (ri = 0u; ri < strip_in_h; ri++) {
            int32_t r = ir0 + (int32_t)ri;
            uint32_t dst = ri * in_w;
            uint32_t src = (r >= 0 && r < (int32_t)in_h)
                ? ACT_DDR + (uint32_t)r * in_w * 16u
                : PAD_ROW_DDR;
            d_dma_in(&di, src, dst, in_w);
        }

        // One conv: dims are the PADDED size (HW pad reads the unpadded tile and
        // injects borders to reach in_w/in_h). pad_w=1 horizontal, pad_h=0 (the
        // vertical halo/pad rows are DMA'd into the strip input).
        d_conv(&di, 0u, in_w + 2u*pad, strip_in_h, in_c, out_c, stride, pad, 0u,
               QPARAM_DDR, out_c);

        // Drain 'so' rows into the full-height output at row oy0.
        d_drain(&di, 0u, OUT_DDR + oy0 * out_w * 16u, so * out_w);
        d_stop(&di);

        if (!d_submit(di)) { print_str("  strip submit fail oy0="); print_dec(oy0); print_str("\n"); errors++; break; }
    }

    print_str("PERF cyc_total="); print_dec(npu_rd(NPU_PERF_CYC_TOTAL));
    print_str(" npu_busy=");      print_dec(npu_rd(NPU_PERF_CYC_BUSY)); print_str("\n");

    // Position-weighted checksum over the whole output (matches the smoke golden).
    {
        uint32_t chk = 0u, idx = 0u;
        for (pos = 0u; pos < C0E_OUT_SPATIAL; pos++)
            for (oc = 0u; oc < out_c; oc++) {
                int32_t got = rs8(OUT_DDR, (oc>>4)*C0E_OUT_SPATIAL + pos, oc&15u);
                chk += ((uint32_t)(got + 128) * (idx + 1u));
                idx++;
            }
        if (chk != C0E_GOLDEN_CHK) {
            errors++;
            print_str("  checksum mismatch got=0x"); print_hex(chk, 8);
            print_str(" exp=0x"); print_hex(C0E_GOLDEN_CHK, 8); print_str("\n");
        }
    }

    if (errors == 0u) { print_str("YOLO CONV0 DESC-TILED SMOKE PASS\n"); return; }
    print_str("YOLO CONV0 DESC-TILED SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
