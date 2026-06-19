// Generic C2f block runner (route 2). Orchestrates the shared NPU conv/eltwise
// path for one C2f block. See yolo_c2f.h and the M5u/M5v bring-up for the proven
// per-stage mechanics this generalizes.

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_c2f.h"

#define ACT_BASE 0u
#define WGT_BASE 0u
#define OUT_BASE 0u

static void wr_word(uint32_t byte_addr, uint32_t widx, const uint32_t lanes[4])
{
    volatile uint32_t *p = (volatile uint32_t *)(byte_addr + widx * 16u);
    p[0] = lanes[0]; p[1] = lanes[1]; p[2] = lanes[2]; p[3] = lanes[3];
}

// Push a firmware weight-word array into DDR so the DMA can load it to Wgt SRAM.
static void push_wgt(uint32_t ddr, const uint32_t (*words)[4], uint32_t n)
{
    uint32_t i;
    for (i = 0u; i < n; i++)
        wr_word(ddr, i, words[i]);
}

static int32_t s8c(int32_t v) { return v > 127 ? 127 : (v < -128 ? -128 : v); }

// CPU integer requant of one 128-bit DDR word (16 lanes) to the concat scale.
static void requant_word(uint32_t src, uint32_t sw, uint32_t mul, int32_t in_zp,
                         uint32_t shift, int32_t cat_zp, uint32_t dst, uint32_t dw)
{
    volatile uint32_t *sp = (volatile uint32_t *)(src + sw * 16u);
    uint32_t in[4]; uint32_t out[4] = {0u,0u,0u,0u}; uint32_t k;
    in[0]=sp[0]; in[1]=sp[1]; in[2]=sp[2]; in[3]=sp[3];
    for (k = 0u; k < 16u; k++) {
        uint32_t b = (in[k>>2] >> ((k&3u)*8u)) & 0xFFu;
        int32_t q = (b & 0x80u) ? ((int32_t)b - 256) : (int32_t)b;
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

    if (cfg->n_bottleneck == 0u || cfg->n_bottleneck > YOLO_C2F_MAX_BN)
        return 0;

    // ---------- cv1 (1x1) : block input -> s0|s1 ----------
    push_wgt(cfg->wgt_ddr, cfg->cv1_wgt, cfg->cv1_wgt_words);
    yolo_set_silu_requant(cfg->cv1_rq_mul, cfg->cv1_rq_shift, cfg->cv1_rq_zp);
    if (!yolo_dma_ddr_to_act(cfg->in_ddr, ACT_BASE, sp * (cfg->cv1_ic / 16u)) ||
        !yolo_dma_ddr_to_wgt(cfg->wgt_ddr, WGT_BASE, cfg->cv1_wgt_words) ||
        !yolo_run_pw_conv1x1_qparams(ACT_BASE, WGT_BASE, OUT_BASE,
                                     cfg->in_w, cfg->in_h, cfg->cv1_ic, cfg->full_c,
                                     cfg->cv1_bias, cfg->cv1_mul, cfg->cv1_shift,
                                     NPU_CTRL_OC_SINGLE | NPU_CTRL_SILU_EN |
                                     NPU_CTRL_SILU_REQUANT_EN) ||
        !yolo_dma_out_to_ddr(cfg->cv1_out_ddr, OUT_BASE, sp * full_groups, 0u))
        return 0;

    // ---------- bottleneck chain ----------
    for (i = 0u; i < cfg->n_bottleneck; i++) {
        // bottleneck input -> Act SRAM (half_c). i==0: s1 (cv1 group half..);
        // i>0: previous add output.
        if (i == 0u) {
            if (!yolo_slice_ddr_to_act(cfg->cv1_out_ddr, ACT_BASE, sp,
                                       half_groups, half_groups))
                return 0;
        } else {
            if (!yolo_dma_ddr_to_act(cfg->add_ddr[i-1u], ACT_BASE, sp * half_groups))
                return 0;
        }

        // m_cv1 (3x3) -> bn_out_ddr
        push_wgt(cfg->wgt_ddr, cfg->mcv1_wgt[i], cfg->mcv1_wgt_words);
        yolo_set_pad_value(cfg->mcv1_pad_value[i]);
        yolo_set_silu_requant(cfg->mcv1_rq_mul[i], cfg->mcv1_rq_shift[i], cfg->mcv1_rq_zp[i]);
        if (!yolo_dma_ddr_to_wgt(cfg->wgt_ddr, WGT_BASE, cfg->mcv1_wgt_words) ||
            !yolo_run_conv2d_qparams(ACT_BASE, WGT_BASE, OUT_BASE,
                                     cfg->in_w, cfg->in_h, cfg->full_c / 2u, cfg->full_c / 2u,
                                     3u, 3u, 1u, 1u,
                                     cfg->mcv1_bias[i], cfg->mcv1_mul[i], cfg->mcv1_shift[i],
                                     NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN) ||
            !yolo_dma_out_to_ddr(cfg->bn_out_ddr, OUT_BASE, sp * half_groups, 0u))
            return 0;

        if (cfg->shortcut && i == 0u) {
            // Stage prev=s1 to the glue scale into Out SRAM (re-run cv1 group-1).
            push_wgt(cfg->wgt_ddr, cfg->stage_wgt[i], cfg->stage_wgt_words);
            yolo_set_silu_requant(cfg->glue_rq_mul[i], cfg->glue_rq_shift[i], cfg->glue_zp[i]);
            if (!yolo_dma_ddr_to_act(cfg->in_ddr, ACT_BASE, sp * (cfg->cv1_ic / 16u)) ||
                !yolo_dma_ddr_to_wgt(cfg->wgt_ddr, WGT_BASE, cfg->stage_wgt_words) ||
                !yolo_run_pw_conv1x1_qparams(ACT_BASE, WGT_BASE, cfg->skip_out_base,
                                             cfg->in_w, cfg->in_h, cfg->cv1_ic, cfg->full_c / 2u,
                                             cfg->stage_bias[i], cfg->stage_mul[i], cfg->stage_shift[i],
                                             NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN))
                return 0;
        } else if (cfg->shortcut) {
            // i>0 residual staging not yet implemented (m6a-2).
            return 0;
        }

        // m_cv2 (3x3) -> requant to glue; residual add via HW signed eltwise.
        push_wgt(cfg->wgt_ddr, cfg->mcv2_wgt[i], cfg->mcv2_wgt_words);
        yolo_set_pad_value(cfg->mcv2_pad_value[i]);
        yolo_set_silu_requant(cfg->glue_rq_mul[i], cfg->glue_rq_shift[i], cfg->glue_zp[i]);
        if (cfg->shortcut)
            yolo_set_eltwise(cfg->glue_zp[i], cfg->skip_out_base);
        if (!yolo_dma_ddr_to_act(cfg->bn_out_ddr, ACT_BASE, sp * half_groups) ||
            !yolo_dma_ddr_to_wgt(cfg->wgt_ddr, WGT_BASE, cfg->mcv2_wgt_words) ||
            !yolo_run_conv2d_qparams(ACT_BASE, WGT_BASE, OUT_BASE,
                                     cfg->in_w, cfg->in_h, cfg->full_c / 2u, cfg->full_c / 2u,
                                     3u, 3u, 1u, 1u,
                                     cfg->mcv2_bias[i], cfg->mcv2_mul[i], cfg->mcv2_shift[i],
                                     NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN |
                                     (cfg->shortcut ? (NPU_CTRL_ELTWISE_EN | NPU_CTRL_ELT_SIGNED) : 0u)) ||
            !yolo_dma_out_to_ddr(cfg->add_ddr[i], OUT_BASE, sp * half_groups, 0u))
            return 0;
    }

    // ---------- concat(s0, s1, add_0..add_{n-1}) requant to cv2 in-scale ----------
    // piece order: s0 (cv1 g0..half), s1 (cv1 half..full), then each add.
    for (pos = 0u; pos < sp; pos++) {
        uint32_t dstg = 0u;
        for (g = 0u; g < half_groups; g++)   // s0
            requant_word(cfg->cv1_out_ddr, g * sp + pos, cfg->cat_mul_s0s1,
                         cfg->cat_inzp_s0s1, cfg->cat_req_shift, cfg->cat_zp,
                         cfg->concat_ddr, (dstg++) * sp + pos);
        for (g = 0u; g < half_groups; g++)   // s1
            requant_word(cfg->cv1_out_ddr, (half_groups + g) * sp + pos, cfg->cat_mul_s0s1,
                         cfg->cat_inzp_s0s1, cfg->cat_req_shift, cfg->cat_zp,
                         cfg->concat_ddr, (dstg++) * sp + pos);
        for (i = 0u; i < cfg->n_bottleneck; i++)   // add_i
            for (g = 0u; g < half_groups; g++)
                requant_word(cfg->add_ddr[i], g * sp + pos, cfg->cat_mul_add,
                             cfg->cat_inzp_add, cfg->cat_req_shift, cfg->cat_zp,
                             cfg->concat_ddr, (dstg++) * sp + pos);
    }

    // ---------- cv2 (1x1) ----------
    push_wgt(cfg->wgt_ddr, cfg->cv2_wgt, cfg->cv2_wgt_words);
    yolo_set_silu_requant(cfg->cv2_rq_mul, cfg->cv2_rq_shift, cfg->cv2_rq_zp);
    if (!yolo_dma_ddr_to_act(cfg->concat_ddr, ACT_BASE, sp * (cfg->cv2_ic / 16u)) ||
        !yolo_dma_ddr_to_wgt(cfg->wgt_ddr, WGT_BASE, cfg->cv2_wgt_words) ||
        !yolo_run_pw_conv1x1_qparams(ACT_BASE, WGT_BASE, OUT_BASE,
                                     cfg->in_w, cfg->in_h, cfg->cv2_ic, cfg->cv2_oc,
                                     cfg->cv2_bias, cfg->cv2_mul, cfg->cv2_shift,
                                     NPU_CTRL_OC_SINGLE | NPU_CTRL_SILU_EN |
                                     NPU_CTRL_SILU_REQUANT_EN) ||
        !yolo_dma_out_to_ddr(cfg->out_ddr, OUT_BASE, sp * (cfg->cv2_oc / 16u), 0u))
        return 0;

    return 1;
}
