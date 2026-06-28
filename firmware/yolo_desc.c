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

// ---- Phase-0 probe: split conv-desc time into CPU descriptor BUILD vs engine
// RUN (submit+DMA+wait). Free-running NPU_PERF_CYC_TOTAL; diffs only. ----
static uint32_t g_desc_build = 0u;   // CPU building records + qparam + pad row
static uint32_t g_desc_run   = 0u;   // d_submit (engine: submit + DMA + wait)
static uint32_t g_desc_wgt   = 0u;   // preload weight DMA (real transfer)
static uint32_t g_desc_recs  = 0u;   // total descriptor records summed over all submits
static uint32_t g_desc_calls = 0u;   // number of layer submits (for resident region sizing)
static uint32_t g_desc_maxrec = 0u;  // largest single program (records) -> peak region
static inline uint32_t d_cyc(void){ return *(volatile uint32_t *)NPU_PERF_CYC_TOTAL; }
void yolo_desc_prof_print(void)
{
    print_str("[DESCPROF] build="); print_dec(g_desc_build);
    print_str(" run(submit+dma+wait)="); print_dec(g_desc_run);
    print_str(" wgt_preload="); print_dec(g_desc_wgt); print_str("\n");
    print_str("[DESCSIZE] total_records="); print_dec(g_desc_recs);
    print_str(" submits="); print_dec(g_desc_calls);
    print_str(" max_prog_records="); print_dec(g_desc_maxrec);
    print_str(" total_words="); print_dec(g_desc_recs * NPU_HW_DESC_WORDS);
    print_str(" (bytes="); print_dec(g_desc_recs * NPU_HW_DESC_WORDS * 4u); print_str(")\n");
}

// ---- Pre-compiled descriptor image: record (build into resident slots + catalog)
// vs replay (submit pre-loaded programs by call order). ----
typedef struct { uint32_t off_words; uint32_t count; } desc_cat_t;
static volatile desc_cat_t *const g_catalog =
    (volatile desc_cat_t *)DESC_CATALOG_BASE;
static uint32_t g_prog_idx  = 0u;            // call-order cursor
static uint32_t g_img_top   = 0u;            // descriptor-image bump (words), record
static uint32_t g_qp_top    = 0u;            // qparam bump (words), record
static uint32_t g_prog_base = YOLO_DESC_DDR; // current program's DDR byte base

void yolo_desc_reset(void)
{ g_prog_idx = 0u; g_img_top = 0u; g_qp_top = 0u; g_prog_base = YOLO_DESC_DDR; }

// Point the record builders at this program's slot (record) or the throwaway
// scratch (replay rebuilds nothing the engine uses).
static void desc_prog_begin(void)
{
#ifdef DESC_RECORD
    g_prog_base = DESC_IMAGE_BASE + g_img_top * NPU_HW_DESC_WORDS * 4u;
#else
    g_prog_base = YOLO_DESC_DDR;
#endif
}

// Record: append {slot offset, di} to the catalog, submit, bump. Replay: ignore
// the just-built records, submit the pre-loaded program at catalog[g_prog_idx].
static int desc_submit_cataloged(uint32_t di)
{
    g_desc_recs += di; g_desc_calls += 1u;
    if (di > g_desc_maxrec) g_desc_maxrec = di;
#ifdef DESC_RECORD
    g_catalog[g_prog_idx].off_words = (g_prog_base - DESC_IMAGE_BASE) / 4u;
    g_catalog[g_prog_idx].count = di;
    g_prog_idx++; g_img_top += di * NPU_HW_DESC_WORDS;
    { int ok = d_submit(di); g_prog_base += di * NPU_HW_DESC_WORDS * 4u; return ok; }
#else
    {
        desc_cat_t c;
        int t;
        c.off_words = g_catalog[g_prog_idx].off_words;
        c.count     = g_catalog[g_prog_idx].count;
        g_prog_idx++; (void)di;
        d_npu_wr(NPU_DESC_BASE_LO, DESC_IMAGE_BASE + c.off_words * 4u);
        d_npu_wr(NPU_DESC_BASE_HI, 0u);
        d_npu_wr(NPU_DESC_COUNT, c.count);
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
#endif
}

// ---- descriptor record builders (write 16-word records into g_prog_base) ----
static volatile uint32_t *dword(uint32_t idx)
{ return (volatile uint32_t *)(g_prog_base + idx * NPU_HW_DESC_WORDS * 4u); }
static void dclr(volatile uint32_t *d)
{ uint32_t i; for (i=0u;i<NPU_HW_DESC_WORDS;i++) d[i]=0u; }
static void dop(volatile uint32_t *d, uint32_t op, uint32_t flags)
{ d[0]=(op&0xFFu)|((NPU_HW_DESC_VERSION&0xFFu)<<8)|((flags&0xFFFFu)<<16); d[1]=flags>>16; }

// flags: bit0 silu_exact_en, bit1 silu_en, bit2 silu_requant_en.
static void d_act_cfg(uint32_t *idx, int32_t pad_value, uint32_t flags,
                      uint32_t rq_mul, uint32_t rq_shift, int32_t rq_zp)
{
    volatile uint32_t *d = dword((*idx)++);
    dclr(d);
    dop(d, NPU_HW_DESC_OP_ACTIVATION_CFG, 0u);
    d[2] = (uint32_t)pad_value & 0xFFu;
    d[3] = (rq_mul & 0xFFFFu) | ((rq_shift & 0x3Fu) << 8) |
           (((uint32_t)rq_zp & 0xFFu) << 24);   // packed like NPU_SILU_REQUANT_CFG
    d[4] = flags & 0x7u;
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

// Act SRAM -> DDR drain (OP_DMA_ACT_TO_DDR), used by the maxpool/eltwise programs
// whose result lands in Act SRAM (not Out SRAM).
static void d_dma_out_act(uint32_t *idx, uint32_t act_base, uint32_t dst_ddr, uint32_t words)
{
    uint32_t off = 0u;
    while (off < words) {
        uint32_t ch = words - off;
        volatile uint32_t *d;
        if (ch > DMA_WR_MAX) ch = DMA_WR_MAX;
        d = dword((*idx)++); dclr(d);
        dop(d, NPU_HW_DESC_OP_DMA_ACT_TO_DDR, 0u);
        d[2] = act_base + off;
        d[4] = dst_ddr + off * 16u;
        d[7] = ch;
        off += ch;
    }
}

// 5x5 stride-1 signed maxpool over Act SRAM (in_c = channels = ic_groups*16).
static void d_maxpool(uint32_t *idx, uint32_t src_act, uint32_t dst_act,
                      uint32_t in_w, uint32_t in_h, uint32_t in_c)
{
    volatile uint32_t *d = dword((*idx)++);
    dclr(d);
    dop(d, NPU_HW_DESC_OP_MAXPOOL5X5, 0u);
    d[2] = src_act;
    d[4] = dst_act;
    d[8] = (in_h << 16) | in_w;
    d[9] = in_c & 0xFFFFu;
}

// Signed eltwise add over Act SRAM: dst = src0 + rescaled(src1). w[6] packs the
// zp/ratio exactly like NPU_ELTWISE_ZP (NPU_ELT_PACK). len is raw words.
static void d_eltwise(uint32_t *idx, uint32_t src0_act, uint32_t dst_act,
                      uint32_t src1_act, uint32_t packed_zp, uint32_t words)
{
    volatile uint32_t *d = dword((*idx)++);
    dclr(d);
    dop(d, NPU_HW_DESC_OP_ELTWISE_ADD, 0u);
    d[2] = src0_act;
    d[4] = dst_act;
    d[5] = src1_act;
    d[6] = packed_zp;
    d[7] = words;
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

static int d_submit(uint32_t count) __attribute__((unused));
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
                               int32_t pad_value,
                               uint32_t silu_requant_mul,
                               uint32_t silu_requant_shift,
                               int32_t silu_requant_zp)
{
    uint32_t icg = in_c / 16u;
    uint32_t out_w, out_h, oy0;
    // Activation flags forwarded to ACTIVATION_CFG (silu_exact / silu_en+requant).
    uint32_t act_flags = ((ctrl_flags & NPU_CTRL_SILU_EXACT_EN) ? 0x1u : 0u) |
                         ((ctrl_flags & NPU_CTRL_SILU_EN) ? 0x2u : 0u) |
                         ((ctrl_flags & NPU_CTRL_SILU_REQUANT_EN) ? 0x4u : 0u);
    // CONV2D descriptor flags: single-start OC loop. 1x1 uses the pointwise
    // engine (CTRL[14], reads pixels directly, no im2col/halo); 3x3 uses the
    // HW-pad im2col path. row_par only for stride-1 3x3 (not 1x1, not stride2).
    uint32_t is_pw = (kernel_h == 1u && kernel_w == 1u);
    // Large-IC 1x1 PW (icg > ICG_BUF) cannot use oc_single: oc_single forces
    // pf_all in wgt_reader (pf_all = prefetch_all || oc_single), which holds every
    // IC group resident, but pf_icg_store is [3:0] (16 groups) so IC tiles 16+
    // alias 0-15 -> wrong output. Mirror yolo_run_conv2d_tiled: stream IC per
    // group (oc_single off, FSM reuse_mode=0 since icg>ICG_BUF) and tile OC <=16.
    uint32_t pw_stream = is_pw && (icg > YOLO_ICG_BUF);
    uint32_t chunk_cap = pw_stream ? 16u : 64u;
    uint32_t conv_flags = (is_pw ? NPU_CTRL_PW_EN : NPU_CTRL_HW_PAD);
    if (!pw_stream) conv_flags |= NPU_CTRL_OC_SINGLE;

    // Weights fit Wgt SRAM (16384 words)? If so preload all OCs once (resident,
    // conv reads chunk c at base done*wgt_words_per_oc). Otherwise reload each
    // <=64-OC chunk into Wgt SRAM via descriptor right before its conv.
    uint32_t preload_all = (out_c * wgt_words_per_oc <= 16384u);

    if (icg == 0u || in_w == 0u || in_h == 0u || out_c == 0u ||
        kernel_h == 0u || kernel_w == 0u || stride == 0u || strip_out_rows == 0u)
        return 0;

    if (strip_out_rows >= 16u && stride == 1u && kernel_h > 1u)
        conv_flags |= NPU_CTRL_ROW_PAR;

    out_w = (in_w + 2u * pad - kernel_w) / stride + 1u;
    out_h = (in_h + 2u * pad - kernel_h) / stride + 1u;

    // One pad row in DDR (in_w words, every lane = pad_value) for vertical halo.
    {
        uint32_t _t = d_cyc();
        uint32_t lane = (uint32_t)pad_value & 0xFFu;
        uint32_t word = lane | (lane << 8) | (lane << 16) | (lane << 24);
        volatile uint32_t *p = (volatile uint32_t *)pad_row_ddr;
        uint32_t i;
        for (i = 0u; i < in_w * 4u; i++) p[i] = word;
        g_desc_build += d_cyc() - _t;
    }

    // Weights resident in Wgt SRAM when they fit; else loaded per chunk below.
    if (preload_all) {
        uint32_t _t = d_cyc();
        int ok = yolo_dma_ddr_to_wgt(wgt_all_ddr, wgt_base, out_c * wgt_words_per_oc);
        g_desc_wgt += d_cyc() - _t;
        if (!ok)
            return 0;
    }

    // Resident qparam table for the out_c OCs. Record: per-layer slot in the
    // qparam image (replay reads it pre-loaded, so no CPU write). Replay: the
    // pre-loaded records already point at the baked qparam addresses.
    uint32_t qbase_ddr = YOLO_QPARAM_DDR;
#ifdef DESC_RECORD
    qbase_ddr = DESC_QPARAM_BASE + g_qp_top * 4u;
    g_qp_top += out_c * 4u;
    {
        uint32_t _t = d_cyc();
        volatile uint32_t *q = (volatile uint32_t *)qbase_ddr;
        uint32_t oc;
        for (oc = 0u; oc < out_c; oc++) {
            q[oc*4+0] = (uint32_t)(bias ? bias[oc] : 0);
            q[oc*4+1] = scale_mul[oc];
            q[oc*4+2] = 0u;
            q[oc*4+3] = scale_shift[oc];
        }
        g_desc_build += d_cyc() - _t;
    }
#else
    (void)bias; (void)scale_mul; (void)scale_shift;  /* qparams pre-loaded in DDR */
#endif

    for (oy0 = 0u; oy0 < out_h; oy0 += strip_out_rows) {
        uint32_t so = out_h - oy0;
        uint32_t strip_in_h, g, ri, di = 0u;
        int32_t ir0;
        if (so > strip_out_rows) so = strip_out_rows;
        uint32_t _tb = d_cyc();
        strip_in_h = (so - 1u) * stride + kernel_h;  /* pad_h=0 */
        ir0 = (int32_t)(oy0 * stride) - (int32_t)pad;

        desc_prog_begin();
        d_act_cfg(&di, pad_value, act_flags, silu_requant_mul,
                  silu_requant_shift, silu_requant_zp);

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
                if (chunk > chunk_cap) chunk = chunk_cap;
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
                       qbase_ddr + done * 16u, chunk);
                for (sg = 0u; sg < cgroups; sg++) {
                    uint32_t oc_grp = done / 16u + sg;
                    d_drain(&di, sg * so * out_w,
                            out_ddr + (oc_grp * out_h + oy0) * out_w * 16u, so * out_w);
                }
                done += chunk;
            }
        }
        d_stop(&di);
        g_desc_build += d_cyc() - _tb;

        {
            uint32_t _tr = d_cyc();
            int ok = desc_submit_cataloged(di);
            g_desc_run += d_cyc() - _tr;
            if (!ok)
                return 0;
        }
    }
    return 1;
}

// 5x5 maxpool (DDR->DDR) as a single descriptor program: DMA src into Act SRAM,
// run the HW 5x5 maxpool (Act->Act), drain the result back to DDR. Mirrors
// yolo_run_maxpool5x5 (yolo_ops.c) but chains the three ops in one submit.
int yolo_run_maxpool5x5_desc(uint32_t src_ddr, uint32_t dst_ddr,
                             uint32_t scratch_act_base, uint32_t in_w,
                             uint32_t in_h, uint32_t ic_groups)
{
    uint32_t words = in_w * in_h * ic_groups;
    uint32_t dst_act = scratch_act_base + words;
    uint32_t di = 0u;

    if (in_w == 0u || in_h == 0u || ic_groups == 0u)
        return 1;

    desc_prog_begin();
    d_dma_in(&di, src_ddr, scratch_act_base, words);
    d_maxpool(&di, scratch_act_base, dst_act, in_w, in_h, ic_groups * 16u);
    d_dma_out_act(&di, dst_act, dst_ddr, words);
    d_stop(&di);
    return desc_submit_cataloged(di);
}

// Signed eltwise residual add (DDR->DDR) as one descriptor program. Mirrors
// yolo_run_eltwise_add_ddr (yolo_ops.c): chunk so src0/src1/dst all fit Act SRAM,
// and for each chunk chain DMA-in(src0) + DMA-in(src1) + ELTWISE + DMA-out.
int yolo_run_eltwise_add_desc(uint32_t src0_ddr, uint32_t src1_ddr,
                              uint32_t dst_ddr, uint32_t scratch_act_base,
                              uint32_t words, int32_t zp, uint32_t ratio_en,
                              uint32_t ratio_mul, uint32_t ratio_shift)
{
    uint32_t done = 0u, di = 0u;
    const uint32_t max_chunk = 4096u;
    uint32_t packed = NPU_ELT_PACK((uint32_t)zp, ratio_en, ratio_shift, ratio_mul);

    if (words == 0u)
        return 1;

    desc_prog_begin();
    while (done < words) {
        uint32_t chunk = words - done;
        uint32_t src1_act, dst_act;
        if (chunk > max_chunk) chunk = max_chunk;
        src1_act = scratch_act_base + chunk;
        dst_act  = src1_act + chunk;
        d_dma_in(&di, src0_ddr + done * 16u, scratch_act_base, chunk);
        d_dma_in(&di, src1_ddr + done * 16u, src1_act, chunk);
        d_eltwise(&di, scratch_act_base, dst_act, src1_act, packed, chunk);
        d_dma_out_act(&di, dst_act, dst_ddr + done * 16u, chunk);
        done += chunk;
    }
    d_stop(&di);
    return desc_submit_cataloged(di);
}
