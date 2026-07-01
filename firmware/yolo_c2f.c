// Generic C2f block runner. Convs on the shared NPU; residual add and concat
// requant on the CPU (uniform, faithful to the C reference). See yolo_c2f.h.

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_desc.h"
#include "yolo_c2f.h"

#define WGT_BASE 0u
#define C2F_ELTWISE_ACT 0u

// Route a tiled conv either through the CPU MMIO helper (use_desc=0) or the
// hardware descriptor queue (use_desc=1). For exact-SiLU the LUT is preloaded by
// silu_setup and the desc requant triple is (0,0,0); legacy requant passes its
// mul/shift/zp through. Keeps the C2f conv compute migratable without changing
// the CPU fallback used by the standalone c2f smokes.
static int c2f_tiled(int use_desc, uint32_t in, uint32_t wd, uint32_t out,
                     uint32_t pad_row, uint32_t inw, uint32_t inh,
                     uint32_t ic, uint32_t oc, uint32_t kh, uint32_t kw,
                     uint32_t stride, uint32_t pad, const int32_t *bias,
                     const uint32_t *mul, const uint32_t *shift, uint32_t f,
                     uint32_t wpo, uint32_t strip, int32_t padv,
                     uint32_t rq_mul, uint32_t rq_shift, int32_t rq_zp)
{
    if (use_desc)
        return yolo_run_conv2d_tiled_desc(in, wd, WGT_BASE, out, pad_row, inw, inh,
                   ic, oc, kh, kw, stride, pad, bias, mul, shift, f, wpo, strip,
                   padv, rq_mul, rq_shift, rq_zp);
    return yolo_run_conv2d_tiled(in, wd, WGT_BASE, out, pad_row, inw, inh, ic, oc,
               kh, kw, stride, pad, bias, mul, shift, f, wpo, strip, padv);
}

static void wr_word(uint32_t a, uint32_t w, const uint32_t l[4])
{
    volatile uint32_t *p = (volatile uint32_t *)(a + w * 16u);
    p[0]=l[0]; p[1]=l[1]; p[2]=l[2]; p[3]=l[3];
}
static void rd_word(uint32_t a, uint32_t w, uint32_t l[4])
{
    volatile uint32_t *p = (volatile uint32_t *)(a + w * 16u);
    l[0]=p[0]; l[1]=p[1]; l[2]=p[2]; l[3]=p[3];
}
static void push_wgt(uint32_t ddr, const uint32_t (*words)[4], uint32_t n)
{
    uint32_t i; for (i = 0u; i < n; i++) wr_word(ddr, i, words[i]);
}
// Program the per-conv SiLU mode and return the ctrl flags. Legacy: fixed Q4.4
// LUT + Q4.4->out requant. Exact: load the per-layer out-grid LUT, output zp
// only, NPU_CTRL_SILU_EXACT_EN (the *_mul/*_shift carry out-grid qparams).
static uint32_t silu_setup(const yolo_c2f_cfg_t *cfg, const uint8_t *lut,
                           uint32_t rq_mul, uint32_t rq_shift, int32_t zp)
{
    if (cfg->silu_exact) {
        /* preact-scale exact SiLU: the LUT index is preact-centered, so the index
         * zero-point is ALWAYS 0 (the output zp is baked into the LUT content). The
         * per-conv zp arg is then only the output zp used by the CPU add/concat. */
        (void)zp;
        yolo_load_silu_lut(lut);
        yolo_set_silu_requant(0u, 0u, 0);
        return NPU_CTRL_SILU_EXACT_EN;
    }
    yolo_set_silu_requant(rq_mul, rq_shift, zp);
    return NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN;
}

static int32_t s8c(int32_t v) { return v > 127 ? 127 : (v < -128 ? -128 : v); }
static int32_t lane_s8(uint32_t w, uint32_t k) {
    uint32_t b = (w >> ((k & 3u) * 8u)) & 0xFFu;
    return (b & 0x80u) ? ((int32_t)b - 256) : (int32_t)b;
}

// CPU residual add over one 128-bit word: add = clamp(round((prev-prev_zp)*ratio>>sh) + mcv2)
static void add_word(uint32_t prev_ddr, uint32_t mcv2_ddr, uint32_t widx,
                     uint32_t ratio_mul, uint32_t ratio_sh, int32_t prev_zp,
                     uint32_t dst, uint32_t dw) __attribute__((unused));
static void add_word(uint32_t prev_ddr, uint32_t mcv2_ddr, uint32_t widx,
                     uint32_t ratio_mul, uint32_t ratio_sh, int32_t prev_zp,
                     uint32_t dst, uint32_t dw)
{
    uint32_t pv[4], cv[4], out[4] = {0u,0u,0u,0u}; uint32_t k;
    rd_word(prev_ddr, widx, pv); rd_word(mcv2_ddr, widx, cv);
    for (k = 0u; k < 16u; k++) {
        int32_t prev = lane_s8(pv[k>>2], k);
        int32_t conv = lane_s8(cv[k>>2], k);
        int32_t v = (((prev - prev_zp) * (int32_t)ratio_mul + (1 << (ratio_sh-1u))) >> ratio_sh) + conv;
        out[k>>2] |= ((uint32_t)(s8c(v) & 0xFF)) << ((k&3u)*8u);
    }
    wr_word(dst, dw, out);
}

// CPU requant of one word to the concat scale.
static void requant_word(uint32_t src, uint32_t sw, uint32_t mul, int32_t in_zp,
                         uint32_t shift, int32_t cat_zp, uint32_t dst, uint32_t dw)
{
    uint32_t in[4], out[4] = {0u,0u,0u,0u}; uint32_t k;
    rd_word(src, sw, in);
    for (k = 0u; k < 16u; k++) {
        int32_t q = lane_s8(in[k>>2], k);
        int32_t v = (((q - in_zp) * (int32_t)mul + (1 << (shift-1u))) >> shift) + cat_zp;
        out[k>>2] |= ((uint32_t)(s8c(v) & 0xFF)) << ((k&3u)*8u);
    }
    wr_word(dst, dw, out);
}

// Resolve a conv's weight DDR base: blob address (no copy) or push the C array
// into the wgt_ddr scratch and return that.
static uint32_t wgt_src(const yolo_c2f_cfg_t *cfg, uint32_t blob_addr,
                        const uint32_t (*arr)[4], uint32_t words)
{
    if (cfg->wgt_in_blob) return blob_addr;
    push_wgt(cfg->wgt_ddr, arr, words);
    return cfg->wgt_ddr;
}

// i-th bottleneck's half_c output ("add") base. Folded: a slot inside the
// contiguous concat buffer [s0|s1|add0|add1|...]; legacy: the separate add_ddr[i].
static uint32_t add_slot(const yolo_c2f_cfg_t *cfg, uint32_t i,
                         uint32_t full_groups, uint32_t half_groups, uint32_t sp)
{
    if (cfg->cv2_folded)
        return cfg->concat_ddr + (full_groups + i * half_groups) * sp * 16u;
    return cfg->add_ddr[i];
}

static int c2f_impl(const yolo_c2f_cfg_t *cfg, int use_desc)
{
    uint32_t half_groups = (cfg->full_c / 2u) / 16u;
    uint32_t full_groups = cfg->full_c / 16u;
    uint32_t sp = cfg->spatial;
    uint32_t strip = (cfg->strip != 0u) ? cfg->strip : 16u;
    uint32_t i, pos, g;

    // Concat-folded mode: the per-source requant is baked into cv2's weights, so
    // cv1 writes s0|s1 straight into the contiguous concat buffer and each residual
    // add_i is written into the concat add slot (no cat_req copy). add_dst() resolves
    // the i-th half_c block's base inside concat_ddr ([s0|s1|add0|add1|...]).
    uint32_t fold    = cfg->cv2_folded;
    uint32_t cv1_out = fold ? cfg->concat_ddr : cfg->cv1_out_ddr;

    if (cfg->n_bottleneck == 0u || cfg->n_bottleneck > YOLO_C2F_MAX_BN)
        return 0;

    // ---------- cv1 (1x1): block input -> s0|s1 (tiled DDR->DDR, supports >64) ----------
    {
        uint32_t wd = wgt_src(cfg, cfg->cv1_wgt_ddr, cfg->cv1_wgt, cfg->cv1_wgt_words);
        uint32_t f = silu_setup(cfg, cfg->cv1_silu_lut, cfg->cv1_rq_mul,
                                cfg->cv1_rq_shift, cfg->cv1_rq_zp);
        if (!c2f_tiled(use_desc, cfg->in_ddr, wd, cv1_out,
                       cfg->pad_row_ddr, cfg->in_w, cfg->in_h, cfg->cv1_ic, cfg->full_c,
                       1u, 1u, 1u, 0u,
                       cfg->cv1_bias, cfg->cv1_mul, cfg->cv1_shift,
                       f, cfg->cv1_wgt_words / cfg->full_c, strip, 0,
                       cfg->silu_exact ? 0u : cfg->cv1_rq_mul,
                       cfg->silu_exact ? 0u : cfg->cv1_rq_shift,
                       cfg->silu_exact ? 0  : cfg->cv1_rq_zp))
            return 0;
    }
    (void)full_groups;

    // ---------- bottleneck chain ----------
    for (i = 0u; i < cfg->n_bottleneck; i++) {
        // bottleneck input (half_c) lives in DDR: i==0 => s1 slice, i>0 => add_{i-1}.
        uint32_t prev_ddr;
        if (i == 0u)
            prev_ddr = cv1_out + (half_groups * sp) * 16u; // s1 slice base
        else
            prev_ddr = add_slot(cfg, i-1u, full_groups, half_groups, sp);

        uint32_t add_dst = add_slot(cfg, i, full_groups, half_groups, sp);
        uint32_t local_pair = use_desc && cfg->silu_exact && cfg->cv2_folded &&
                              (cfg->shortcut == 0u) && (cfg->n_bottleneck == 1u) &&
                              (half_groups * sp <= 4096u);

        if (local_pair) {
            uint32_t half_c = cfg->full_c / 2u;
            uint32_t wd;
            uint32_t f;

            yolo_set_pad_value(cfg->mcv1_pad_value[i]);
            wd = wgt_src(cfg, cfg->mcv1_wgt_ddr[i], cfg->mcv1_wgt[i], cfg->mcv1_wgt_words);
            f = silu_setup(cfg, cfg->mcv1_silu_lut[i], cfg->mcv1_rq_mul[i],
                           cfg->mcv1_rq_shift[i], cfg->mcv1_rq_zp[i]);
            if (!yolo_run_conv2d_resident_desc(prev_ddr, wd, WGT_BASE, 0u,
                                   cfg->in_w, cfg->in_h, half_c, half_c,
                                   3u, 3u, 1u, 1u,
                                   cfg->mcv1_bias[i], cfg->mcv1_mul[i], cfg->mcv1_shift[i],
                                   f, cfg->mcv1_wgt_words / half_c,
                                   cfg->mcv1_pad_value[i], 0u, 0u))
                return 0;

            yolo_set_pad_value(cfg->mcv2_pad_value[i]);
            wd = wgt_src(cfg, cfg->mcv2_wgt_ddr[i], cfg->mcv2_wgt[i], cfg->mcv2_wgt_words);
            f = silu_setup(cfg, cfg->mcv2_silu_lut[i], cfg->glue_rq_mul[i],
                           cfg->glue_rq_shift[i], cfg->glue_zp[i]);
            if (!yolo_run_conv2d_resident_desc(0u, wd, WGT_BASE, add_dst,
                                   cfg->in_w, cfg->in_h, half_c, half_c,
                                   3u, 3u, 1u, 1u,
                                   cfg->mcv2_bias[i], cfg->mcv2_mul[i], cfg->mcv2_shift[i],
                                   f, cfg->mcv2_wgt_words / half_c,
                                   cfg->mcv2_pad_value[i], 1u, 1u))
                return 0;
            continue;
        }

        // Large-IC 3x3 (exact mode, half_c/16 > ICG_BUF) can't fit the im2col
        // window: stream IC via INT32 psum accumulate (CPU final SiLU). Otherwise
        // the resident tiled path.
        uint32_t bn_stream = cfg->silu_exact &&
                             (((cfg->full_c/2u) / 16u) > YOLO_ICG_BUF);

        // m_cv1 (3x3) -> bn_out (tiled DDR->DDR; half_c may be >64)
        yolo_set_pad_value(cfg->mcv1_pad_value[i]);
        {
            uint32_t wd = wgt_src(cfg, cfg->mcv1_wgt_ddr[i], cfg->mcv1_wgt[i], cfg->mcv1_wgt_words);
            if (bn_stream) {
                if (!yolo_run_conv2d_ic_stream(prev_ddr, wd, WGT_BASE, cfg->bn_out_ddr,
                                       cfg->psum_ddr, cfg->pad_row_ddr, cfg->in_w, cfg->in_h,
                                       cfg->full_c/2u, cfg->full_c/2u, 3u, 3u, 1u, 1u,
                                       cfg->mcv1_bias[i], cfg->mcv1_mul[i], cfg->mcv1_shift[i],
                                       cfg->mcv1_silu_lut[i], cfg->mcv1_pad_value[i]))
                    return 0;
            } else {
                uint32_t f = silu_setup(cfg, cfg->mcv1_silu_lut[i], cfg->mcv1_rq_mul[i],
                                        cfg->mcv1_rq_shift[i], cfg->mcv1_rq_zp[i]);
                if (!c2f_tiled(use_desc, prev_ddr, wd, cfg->bn_out_ddr,
                               cfg->pad_row_ddr, cfg->in_w, cfg->in_h,
                               cfg->full_c/2u, cfg->full_c/2u,
                               3u, 3u, 1u, 1u,
                               cfg->mcv1_bias[i], cfg->mcv1_mul[i], cfg->mcv1_shift[i],
                               f, cfg->mcv1_wgt_words / (cfg->full_c/2u), strip,
                               cfg->mcv1_pad_value[i],
                               cfg->silu_exact ? 0u : cfg->mcv1_rq_mul[i],
                               cfg->silu_exact ? 0u : cfg->mcv1_rq_shift[i],
                               cfg->silu_exact ? 0  : cfg->mcv1_rq_zp[i]))
                    return 0;
            }
        }

        // m_cv2 (3x3) requant to glue[i] -> mcv2_ddr (tiled DDR->DDR)
        yolo_set_pad_value(cfg->mcv2_pad_value[i]);
        {
            uint32_t wd = wgt_src(cfg, cfg->mcv2_wgt_ddr[i], cfg->mcv2_wgt[i], cfg->mcv2_wgt_words);
            if (bn_stream) {
                if (!yolo_run_conv2d_ic_stream(cfg->bn_out_ddr, wd, WGT_BASE, cfg->mcv2_ddr,
                                       cfg->psum_ddr, cfg->pad_row_ddr, cfg->in_w, cfg->in_h,
                                       cfg->full_c/2u, cfg->full_c/2u, 3u, 3u, 1u, 1u,
                                       cfg->mcv2_bias[i], cfg->mcv2_mul[i], cfg->mcv2_shift[i],
                                       cfg->mcv2_silu_lut[i], cfg->mcv2_pad_value[i]))
                    return 0;
            } else {
                uint32_t f = silu_setup(cfg, cfg->mcv2_silu_lut[i], cfg->glue_rq_mul[i],
                                        cfg->glue_rq_shift[i], cfg->glue_zp[i]);
                if (!c2f_tiled(use_desc, cfg->bn_out_ddr, wd, cfg->mcv2_ddr,
                               cfg->pad_row_ddr, cfg->in_w, cfg->in_h,
                               cfg->full_c/2u, cfg->full_c/2u,
                               3u, 3u, 1u, 1u,
                               cfg->mcv2_bias[i], cfg->mcv2_mul[i], cfg->mcv2_shift[i],
                               f, cfg->mcv2_wgt_words / (cfg->full_c/2u), strip,
                               cfg->mcv2_pad_value[i],
                               cfg->silu_exact ? 0u : cfg->glue_rq_mul[i],
                               cfg->silu_exact ? 0u : cfg->glue_rq_shift[i],
                               cfg->silu_exact ? 0  : cfg->glue_zp[i]))
                    return 0;
            }
        }

        // residual add -> add slot; shortcut uses the generic Act-SRAM eltwise
        // engine in chunks so large early C2f tensors do not require more SRAM.
        if (cfg->shortcut) {
            int ok = use_desc
                ? yolo_run_eltwise_add_desc(prev_ddr, cfg->mcv2_ddr, add_dst,
                                            C2F_ELTWISE_ACT, half_groups * sp,
                                            cfg->add_prev_zp[i], 1u,
                                            cfg->add_ratio_mul[i], cfg->add_ratio_shift)
                : yolo_run_eltwise_add_ddr(prev_ddr, cfg->mcv2_ddr, add_dst,
                                           C2F_ELTWISE_ACT, half_groups * sp,
                                           cfg->add_prev_zp[i], 1u,
                                           cfg->add_ratio_mul[i], cfg->add_ratio_shift);
            if (!ok)
                return 0;
        } else {
            for (g = 0u; g < half_groups; g++)
                for (pos = 0u; pos < sp; pos++) {
                    uint32_t widx = g * sp + pos;
                    uint32_t cw[4]; rd_word(cfg->mcv2_ddr, widx, cw);
                    wr_word(add_dst, widx, cw);
                }
        }
    }

    // ---------- concat(s0, s1, add_0..add_{n-1}) requant to cv2 in-scale ----------
    // Folded: cv2 weights already absorb the per-source requant and the concat
    // buffer is filled in place, so this whole CPU loop is skipped.
    if (!fold)
    for (pos = 0u; pos < sp; pos++) {
        uint32_t dstg = 0u;
        for (g = 0u; g < half_groups; g++)
            requant_word(cfg->cv1_out_ddr, g*sp+pos, cfg->cat_mul_s0s1, cfg->cat_inzp_s0s1,
                         cfg->cat_req_shift, cfg->cat_zp, cfg->concat_ddr, (dstg++)*sp+pos);
        for (g = 0u; g < half_groups; g++)
            requant_word(cfg->cv1_out_ddr, (half_groups+g)*sp+pos, cfg->cat_mul_s0s1, cfg->cat_inzp_s0s1,
                         cfg->cat_req_shift, cfg->cat_zp, cfg->concat_ddr, (dstg++)*sp+pos);
        for (i = 0u; i < cfg->n_bottleneck; i++)
            for (g = 0u; g < half_groups; g++)
                requant_word(cfg->add_ddr[i], g*sp+pos, cfg->cat_mul_add[i], cfg->cat_inzp_add[i],
                             cfg->cat_req_shift, cfg->cat_zp, cfg->concat_ddr, (dstg++)*sp+pos);
    }

    // ---------- cv2 (1x1, tiled DDR->DDR, supports >64) ----------
    {
        uint32_t wd = wgt_src(cfg, cfg->cv2_wgt_ddr, cfg->cv2_wgt, cfg->cv2_wgt_words);
        uint32_t f = silu_setup(cfg, cfg->cv2_silu_lut, cfg->cv2_rq_mul,
                                cfg->cv2_rq_shift, cfg->cv2_rq_zp);
        if (!c2f_tiled(use_desc, cfg->concat_ddr, wd, cfg->out_ddr,
                       cfg->pad_row_ddr, cfg->in_w, cfg->in_h, cfg->cv2_ic, cfg->cv2_oc,
                       1u, 1u, 1u, 0u,
                       cfg->cv2_bias, cfg->cv2_mul, cfg->cv2_shift,
                       f, cfg->cv2_wgt_words / cfg->cv2_oc, strip, 0,
                       cfg->silu_exact ? 0u : cfg->cv2_rq_mul,
                       cfg->silu_exact ? 0u : cfg->cv2_rq_shift,
                       cfg->silu_exact ? 0  : cfg->cv2_rq_zp))
            return 0;
    }

    return 1;
}

// CPU-helper path (default; used by the standalone c2f smokes).
int yolo_run_c2f_block(const yolo_c2f_cfg_t *cfg)
{ return c2f_impl(cfg, 0); }

// Descriptor-queue path for the convs (cv1/mcv/cv2); residual/concat glue stays
// on the CPU. Used by the full-net deploy (yolo_full_stem.c).
int yolo_run_c2f_block_desc(const yolo_c2f_cfg_t *cfg)
{ return c2f_impl(cfg, 1); }
