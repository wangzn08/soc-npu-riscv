// Generic C2f block runner. Convs on the shared NPU; residual add and concat
// requant on the CPU (uniform, faithful to the C reference). See yolo_c2f.h.

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_c2f.h"

#define ACT_BASE 0u
#define WGT_BASE 0u
#define OUT_BASE 0u

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
static int32_t s8c(int32_t v) { return v > 127 ? 127 : (v < -128 ? -128 : v); }
static int32_t lane_s8(uint32_t w, uint32_t k) {
    uint32_t b = (w >> ((k & 3u) * 8u)) & 0xFFu;
    return (b & 0x80u) ? ((int32_t)b - 256) : (int32_t)b;
}

// CPU residual add over one 128-bit word: add = clamp(round((prev-prev_zp)*ratio>>sh) + mcv2)
static void add_word(uint32_t prev_ddr, uint32_t mcv2_ddr, uint32_t widx,
                     uint32_t ratio_mul, uint32_t ratio_sh, int32_t prev_zp,
                     uint32_t dst, uint32_t dw)
{
    uint32_t pv[4], cv[4], out[4] = {0u,0u,0u,0u}; uint32_t k;
    rd_word(prev_ddr, widx, pv); rd_word(mcv2_ddr, widx, cv);
    for (k = 0u; k < 16u; k++) {
        int32_t prev = lane_s8(pv[k>>2], k);
        int32_t conv = lane_s8(cv[k>>2], k);
        int32_t v = (((prev - prev_zp) * (int32_t)ratio_mul) >> ratio_sh) + conv;
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
        int32_t v = (((q - in_zp) * (int32_t)mul) >> shift) + cat_zp;
        out[k>>2] |= ((uint32_t)(s8c(v) & 0xFF)) << ((k&3u)*8u);
    }
    wr_word(dst, dw, out);
}

int yolo_run_c2f_block(const yolo_c2f_cfg_t *cfg)
{
    uint32_t half_groups = (cfg->full_c / 2u) / 16u;
    uint32_t full_groups = cfg->full_c / 16u;
    uint32_t sp = cfg->spatial;
    uint32_t i, pos, g;
    uint32_t bn_oc = (cfg->full_c / 2u > 16u) ? NPU_CTRL_OC_SINGLE : 0u;  // half_c>16 needs oc_single

    if (cfg->n_bottleneck == 0u || cfg->n_bottleneck > YOLO_C2F_MAX_BN)
        return 0;

    // ---------- cv1 (1x1): block input -> s0|s1 (OC-chunked, supports >64) ----------
    push_wgt(cfg->wgt_ddr, cfg->cv1_wgt, cfg->cv1_wgt_words);
    yolo_set_silu_requant(cfg->cv1_rq_mul, cfg->cv1_rq_shift, cfg->cv1_rq_zp);
    if (!yolo_dma_ddr_to_act(cfg->in_ddr, ACT_BASE, sp * (cfg->cv1_ic / 16u)) ||
        !yolo_run_pw_conv1x1_oc_chunks(ACT_BASE, cfg->wgt_ddr, WGT_BASE, cfg->cv1_out_ddr,
                                       cfg->in_w, cfg->in_h, cfg->cv1_ic, cfg->full_c,
                                       cfg->cv1_bias, cfg->cv1_mul, cfg->cv1_shift,
                                       NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN, sp))
        return 0;
    (void)full_groups;

    // ---------- bottleneck chain ----------
    for (i = 0u; i < cfg->n_bottleneck; i++) {
        // bottleneck input -> Act SRAM (half_c): i==0 => s1, i>0 => add_{i-1}.
        uint32_t prev_ddr;
        if (i == 0u) {
            if (!yolo_slice_ddr_to_act(cfg->cv1_out_ddr, ACT_BASE, sp, half_groups, half_groups))
                return 0;
            prev_ddr = cfg->cv1_out_ddr + (half_groups * sp) * 16u; // s1 slice base
        } else {
            if (!yolo_dma_ddr_to_act(cfg->add_ddr[i-1u], ACT_BASE, sp * half_groups))
                return 0;
            prev_ddr = cfg->add_ddr[i-1u];
        }

        // m_cv1 (3x3) -> bn_out
        push_wgt(cfg->wgt_ddr, cfg->mcv1_wgt[i], cfg->mcv1_wgt_words);
        yolo_set_pad_value(cfg->mcv1_pad_value[i]);
        yolo_set_silu_requant(cfg->mcv1_rq_mul[i], cfg->mcv1_rq_shift[i], cfg->mcv1_rq_zp[i]);
        if (!yolo_dma_ddr_to_wgt(cfg->wgt_ddr, WGT_BASE, cfg->mcv1_wgt_words) ||
            !yolo_run_conv2d_qparams(ACT_BASE, WGT_BASE, OUT_BASE,
                                     cfg->in_w, cfg->in_h, cfg->full_c/2u, cfg->full_c/2u,
                                     3u, 3u, 1u, 1u,
                                     cfg->mcv1_bias[i], cfg->mcv1_mul[i], cfg->mcv1_shift[i],
                                     NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN | bn_oc) ||
            !yolo_dma_out_to_ddr(cfg->bn_out_ddr, OUT_BASE, sp * half_groups, 0u))
            return 0;

        // m_cv2 (3x3) requant to glue[i] -> mcv2_ddr
        push_wgt(cfg->wgt_ddr, cfg->mcv2_wgt[i], cfg->mcv2_wgt_words);
        yolo_set_pad_value(cfg->mcv2_pad_value[i]);
        yolo_set_silu_requant(cfg->glue_rq_mul[i], cfg->glue_rq_shift[i], cfg->glue_zp[i]);
        if (!yolo_dma_ddr_to_act(cfg->bn_out_ddr, ACT_BASE, sp * half_groups) ||
            !yolo_dma_ddr_to_wgt(cfg->wgt_ddr, WGT_BASE, cfg->mcv2_wgt_words) ||
            !yolo_run_conv2d_qparams(ACT_BASE, WGT_BASE, OUT_BASE,
                                     cfg->in_w, cfg->in_h, cfg->full_c/2u, cfg->full_c/2u,
                                     3u, 3u, 1u, 1u,
                                     cfg->mcv2_bias[i], cfg->mcv2_mul[i], cfg->mcv2_shift[i],
                                     NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN | bn_oc) ||
            !yolo_dma_out_to_ddr(cfg->mcv2_ddr, OUT_BASE, sp * half_groups, 0u))
            return 0;

        // residual add (CPU) -> add_ddr[i]; or pass-through when !shortcut.
        for (g = 0u; g < half_groups; g++)
            for (pos = 0u; pos < sp; pos++) {
                uint32_t widx = g * sp + pos;
                if (cfg->shortcut)
                    add_word(prev_ddr, cfg->mcv2_ddr, widx, cfg->add_ratio_mul[i],
                             cfg->add_ratio_shift, cfg->add_prev_zp[i], cfg->add_ddr[i], widx);
                else {
                    uint32_t cw[4]; rd_word(cfg->mcv2_ddr, widx, cw);
                    wr_word(cfg->add_ddr[i], widx, cw);
                }
            }
    }

    // ---------- concat(s0, s1, add_0..add_{n-1}) requant to cv2 in-scale ----------
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

    // ---------- cv2 (1x1, OC-chunked, supports >64) ----------
    push_wgt(cfg->wgt_ddr, cfg->cv2_wgt, cfg->cv2_wgt_words);
    yolo_set_silu_requant(cfg->cv2_rq_mul, cfg->cv2_rq_shift, cfg->cv2_rq_zp);
    if (!yolo_dma_ddr_to_act(cfg->concat_ddr, ACT_BASE, sp * (cfg->cv2_ic / 16u)) ||
        !yolo_run_pw_conv1x1_oc_chunks(ACT_BASE, cfg->wgt_ddr, WGT_BASE, cfg->out_ddr,
                                       cfg->in_w, cfg->in_h, cfg->cv2_ic, cfg->cv2_oc,
                                       cfg->cv2_bias, cfg->cv2_mul, cfg->cv2_shift,
                                       NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN, sp))
        return 0;

    return 1;
}
