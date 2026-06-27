// Descriptor-driven YOLO runtime. See yolo_desc.h.
#include "firmware.h"
#include "npu_desc.h"
#include "yolo_desc.h"
#include "yolo_ops.h"   // yolo_dma_ddr_to_wgt
#include <stdint.h>

#define DMA_RD_MAX  256u   // DDR->Act beats/descriptor (axi_dma read len 8-bit)
#define DMA_WR_MAX  64u    // Out->DDR beats/descriptor

static inline void d_npu_wr(uint32_t a, uint32_t d){ *(volatile uint32_t*)a = d; }
static inline uint32_t d_npu_rd(uint32_t a){ return *(volatile uint32_t*)a; }

// ---- descriptor record builders (write 16-word records into YOLO_DESC_DDR) ----
static volatile uint32_t *dword(uint32_t idx)
{ return (volatile uint32_t *)(YOLO_DESC_DDR + idx * NPU_HW_DESC_WORDS * 4u); }
static void dclr(volatile uint32_t *d)
{ uint32_t i; for (i=0u;i<NPU_HW_DESC_WORDS;i++) d[i]=0u; }
static void dop(volatile uint32_t *d, uint32_t op, uint32_t flags)
{ d[0]=(op&0xFFu)|((NPU_HW_DESC_VERSION&0xFFu)<<8)|((flags&0xFFFFu)<<16); d[1]=flags>>16; }

static void d_act_cfg(uint32_t *idx, int32_t pad_value, uint32_t silu_exact)
{
    volatile uint32_t *d = dword((*idx)++);
    dclr(d);
    dop(d, NPU_HW_DESC_OP_ACTIVATION_CFG, 0u);
    d[2] = (uint32_t)pad_value & 0xFFu;
    d[3] = 0u;                          // requant 0 (exact mode bakes it into LUT)
    d[4] = silu_exact ? 0x1u : 0u;      // flags: silu_exact_en
    d[5] = 0u;                          // clip_max -> default 127
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

static void d_dma_wgt(uint32_t *idx, uint32_t src_ddr, uint32_t wgt_base, uint32_t words)
{
    uint32_t off = 0u;
    while (off < words) {
        uint32_t ch = words - off;
        volatile uint32_t *d;
        if (ch > DMA_RD_MAX) ch = DMA_RD_MAX;
        d = dword((*idx)++); dclr(d);
        dop(d, NPU_HW_DESC_OP_DMA_DDR_TO_WGT, 0u);
        d[2] = src_ddr + off * 16u;
        d[4] = wgt_base + off;
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
        d[2] = out_base + off;
        d[4] = dst_ddr + off * 16u;
        d[7] = ch;
        off += ch;
    }
}

static void d_conv(uint32_t *idx, uint32_t act, uint32_t wgt_base,
                   uint32_t cfg_w, uint32_t cfg_h, uint32_t in_c, uint32_t out_c,
                   uint32_t kh, uint32_t kw, uint32_t stride,
                   uint32_t pad_w, uint32_t pad_h, uint32_t ctrl_flags,
                   uint32_t qbase, uint32_t qcnt)
{
    volatile uint32_t *d = dword((*idx)++);
    dclr(d);
    dop(d, NPU_HW_DESC_OP_CONV2D, ctrl_flags);
    d[2] = act;
    d[3] = wgt_base;
    d[4] = 0u;                          // Out SRAM base 0
    d[8] = (cfg_h << 16) | cfg_w;       // PADDED dims (HW pad reads unpadded tile)
    d[9] = (out_c << 16) | in_c;
    d[10] = (((pad_h & 0xFu) << 28) | ((pad_w & 0xFu) << 24)) |
            (stride << 16) | (kh << 8) | kw;
    d[11] = qbase;
    d[12] = qcnt;
}

static void d_stop(uint32_t *idx)
{ volatile uint32_t *d = dword((*idx)++); dclr(d); dop(d, NPU_HW_DESC_OP_STOP_IRQ, 0u); }

static int d_submit(uint32_t count)
{
    int t;
    d_npu_wr(NPU_DESC_BASE_LO, YOLO_DESC_DDR);
    d_npu_wr(NPU_DESC_BASE_HI, 0u);
    d_npu_wr(NPU_DESC_COUNT, count);
    d_npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_CLEAR_DONE);
    d_npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_START);
    t = 8000000;
    while (t-- > 0) {
        uint32_t st = d_npu_rd(NPU_DESC_STATUS);
        if (st & NPU_DESC_STATUS_ERR) return 0;
        if (st & NPU_DESC_STATUS_DONE) break;
    }
    if (t <= 0) return 0;
    d_npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_CLEAR_DONE);
    return 1;
}

int yolo_run_conv2d_tiled_desc(uint32_t in_ddr, uint32_t wgt_all_ddr,
                               uint32_t wgt_base, uint32_t out_ddr,
                               uint32_t pad_row_ddr,
                               uint32_t in_w, uint32_t in_h,
                               uint32_t in_c, uint32_t out_c,
                               uint32_t kernel_h, uint32_t kernel_w,
                               uint32_t stride, uint32_t pad,
                               const int32_t *bias, const uint32_t *scale_mul,
                               const uint32_t *scale_shift, uint32_t ctrl_flags,
                               uint32_t wgt_words_per_oc, uint32_t strip_out_rows,
                               int32_t pad_value)
{
    uint32_t icg = in_c / 16u;
    uint32_t out_w, out_h, oy0;
    uint32_t silu_exact = (ctrl_flags & NPU_CTRL_SILU_EXACT_EN) ? 1u : 0u;
    // CONV2D descriptor flags: HW pad + single-start OC loop (+ row_par for
    // stride-1). silu_exact/pad live in ACTIVATION_CFG, not the conv flags.
    uint32_t conv_flags = NPU_CTRL_OC_SINGLE | NPU_CTRL_HW_PAD;

    // Weights fit Wgt SRAM (16384 words)? If so preload all OCs once (resident,
    // conv reads chunk c at base done*wgt_words_per_oc). Otherwise reload each
    // <=64-OC chunk into Wgt SRAM via descriptor right before its conv.
    uint32_t preload_all = (out_c * wgt_words_per_oc <= 16384u);

    if (icg == 0u || in_w == 0u || in_h == 0u || out_c == 0u ||
        kernel_h == 0u || kernel_w == 0u || stride == 0u || strip_out_rows == 0u)
        return 0;

    if (strip_out_rows >= 16u && stride == 1u)
        conv_flags |= NPU_CTRL_ROW_PAR;

    out_w = (in_w + 2u * pad - kernel_w) / stride + 1u;
    out_h = (in_h + 2u * pad - kernel_h) / stride + 1u;

    // One pad row in DDR (in_w words, every lane = pad_value) for vertical halo.
    {
        uint32_t lane = (uint32_t)pad_value & 0xFFu;
        uint32_t word = lane | (lane << 8) | (lane << 16) | (lane << 24);
        volatile uint32_t *p = (volatile uint32_t *)pad_row_ddr;
        uint32_t i;
        for (i = 0u; i < in_w * 4u; i++) p[i] = word;
    }

    // Weights resident in Wgt SRAM when they fit; else loaded per chunk below.
    if (preload_all) {
        if (!yolo_dma_ddr_to_wgt(wgt_all_ddr, wgt_base, out_c * wgt_words_per_oc))
            return 0;
    }

    // Resident qparam table for the out_c OCs.
    {
        volatile uint32_t *q = (volatile uint32_t *)YOLO_QPARAM_DDR;
        uint32_t oc;
        for (oc = 0u; oc < out_c; oc++) {
            q[oc*4+0] = (uint32_t)(bias ? bias[oc] : 0);
            q[oc*4+1] = scale_mul[oc];
            q[oc*4+2] = 0u;
            q[oc*4+3] = scale_shift[oc];
        }
    }

    for (oy0 = 0u; oy0 < out_h; oy0 += strip_out_rows) {
        uint32_t so = out_h - oy0;
        uint32_t strip_in_h, g, ri, di = 0u;
        int32_t ir0;
        if (so > strip_out_rows) so = strip_out_rows;
        strip_in_h = (so - 1u) * stride + kernel_h;  /* pad_h=0 */
        ir0 = (int32_t)(oy0 * stride) - (int32_t)pad;

        d_act_cfg(&di, pad_value, silu_exact);

        for (g = 0u; g < icg; g++) {
            for (ri = 0u; ri < strip_in_h; ri++) {
                int32_t r = ir0 + (int32_t)ri;
                uint32_t dst = (g * strip_in_h + ri) * in_w;
                uint32_t src = (r >= 0 && r < (int32_t)in_h)
                    ? in_ddr + (g * in_h + (uint32_t)r) * in_w * 16u
                    : pad_row_ddr;
                d_dma_in(&di, src, dst, in_w);
            }
        }

        // OC chunks: one CONV2D start computes up to 64 OCs (4 groups), then the
        // chunk's OC groups are drained before the next chunk overwrites Out SRAM.
        {
            uint32_t done = 0u;
            while (done < out_c) {
                uint32_t chunk = out_c - done;
                uint32_t sg, cgroups, cwgt;
                if (chunk > 64u) chunk = 64u;
                cgroups = (chunk + 15u) / 16u;
                if (preload_all) {
                    cwgt = wgt_base + done * wgt_words_per_oc;   // resident slice
                } else {
                    // Reload this chunk's weights into Wgt SRAM base.
                    d_dma_wgt(&di, wgt_all_ddr + done * wgt_words_per_oc * 16u,
                              wgt_base, chunk * wgt_words_per_oc);
                    cwgt = wgt_base;
                }
                // PADDED dims; pad_w horizontal HW pad, pad_h=0 (rows DMA'd).
                d_conv(&di, 0u, cwgt,
                       in_w + 2u * pad, strip_in_h, in_c, chunk,
                       kernel_h, kernel_w, stride, pad, 0u, conv_flags,
                       YOLO_QPARAM_DDR + done * 16u, chunk);
                for (sg = 0u; sg < cgroups; sg++) {
                    uint32_t oc_grp = done / 16u + sg;
                    d_drain(&di, sg * so * out_w,
                            out_ddr + (oc_grp * out_h + oy0) * out_w * 16u, so * out_w);
                }
                done += chunk;
            }
        }
        d_stop(&di);

        if (!d_submit(di))
            return 0;
    }
    return 1;
}
