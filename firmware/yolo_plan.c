#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_plan.h"

#define YOLO_CONV0_IN_W 640u
#define YOLO_CONV0_IN_C 3u
#define YOLO_CONV0_OUT_C 16u
#define YOLO_CONV0_K 3u
#define YOLO_CONV0_STRIDE 2u
#define YOLO_CONV0_PAD_VALUE (-128)

uint32_t yolo_ctrl_from_plan_flags(uint32_t plan_flags)
{
    uint32_t ctrl = 0u;

    if ((plan_flags & YOLO_PLAN_FLAG_PW_EN) != 0u)
        ctrl |= NPU_CTRL_PW_EN;
    if ((plan_flags & YOLO_PLAN_FLAG_HW_PAD) != 0u)
        ctrl |= NPU_CTRL_HW_PAD;
    if ((plan_flags & YOLO_PLAN_FLAG_OC_SINGLE) != 0u)
        ctrl |= NPU_CTRL_OC_SINGLE;
    if ((plan_flags & YOLO_PLAN_FLAG_SILU) != 0u)
        ctrl |= NPU_CTRL_SILU_EN;
    if ((plan_flags & YOLO_PLAN_FLAG_SILU_REQUANT) != 0u)
        ctrl |= NPU_CTRL_SILU_REQUANT_EN;

    return ctrl;
}

int yolo_run_conv_desc(const yolo_conv_desc_t *desc)
{
    if (desc == (const yolo_conv_desc_t *)0)
        return 0;
    if (desc->bias == (const int32_t *)0 ||
        desc->scale_mul == (const uint32_t *)0 ||
        desc->scale_shift == (const uint32_t *)0)
        return 0;
    if (desc->act_words == 0u || desc->wgt_words == 0u || desc->out_words == 0u)
        return 0;

    if (!yolo_dma_ddr_to_act(desc->act_ddr, desc->act_base, desc->act_words))
        return 0;
    if (!yolo_dma_ddr_to_wgt(desc->wgt_ddr, desc->wgt_base, desc->wgt_words))
        return 0;

    yolo_set_pad_value((int8_t)desc->pad_value);
    yolo_set_silu_requant(desc->silu_requant_mul,
                          desc->silu_requant_shift,
                          desc->silu_requant_zp);

    if (desc->kernel_h == 1u && desc->kernel_w == 1u) {
        if (!yolo_run_pw_conv1x1_qparams(desc->act_base,
                                         desc->wgt_base,
                                         desc->out_base,
                                         desc->in_w,
                                         desc->in_h,
                                         desc->in_c,
                                         desc->out_c,
                                         desc->bias,
                                         desc->scale_mul,
                                         desc->scale_shift,
                                         desc->ctrl_flags))
            return 0;
    } else {
        if (!yolo_run_conv2d_qparams(desc->act_base,
                                     desc->wgt_base,
                                     desc->out_base,
                                     desc->in_w,
                                     desc->in_h,
                                     desc->in_c,
                                     desc->out_c,
                                     desc->kernel_h,
                                     desc->kernel_w,
                                     desc->stride,
                                     desc->pad,
                                     desc->bias,
                                     desc->scale_mul,
                                     desc->scale_shift,
                                     desc->ctrl_flags))
            return 0;
    }

    return yolo_dma_out_to_ddr(desc->out_ddr,
                               desc->out_base,
                               desc->out_words,
                               desc->out_pong);
}

int yolo_run_conv0_strip_from_plan(const yolo_strip_plan_entry_t *strip,
                                   uint32_t act_ddr,
                                   uint32_t wgt_ddr,
                                   uint32_t out_ddr,
                                   uint32_t act_base,
                                   uint32_t wgt_base,
                                   uint32_t out_base,
                                   const int32_t *bias,
                                   const uint32_t *scale_mul,
                                   const uint32_t *scale_shift,
                                   uint32_t silu_requant_mul,
                                   uint32_t silu_requant_shift,
                                   int32_t silu_requant_zp)
{
    uint32_t act_words;
    uint32_t out_words;
    uint32_t pad_h;

    if (strip == (const yolo_strip_plan_entry_t *)0)
        return 0;
    if (bias == (const int32_t *)0 ||
        scale_mul == (const uint32_t *)0 ||
        scale_shift == (const uint32_t *)0)
        return 0;

    act_words = YOLO_CONV0_IN_W * (uint32_t)strip->in_rows;
    out_words = (YOLO_CONV0_IN_W / YOLO_CONV0_STRIDE) * (uint32_t)strip->out_rows;
    pad_h = (uint32_t)strip->top_pad_rows;
    if ((uint32_t)strip->bottom_pad_rows > pad_h)
        pad_h = (uint32_t)strip->bottom_pad_rows;

    if (!yolo_dma_ddr_to_act(act_ddr + (uint32_t)strip->in_y * YOLO_CONV0_IN_W * 16u,
                             act_base,
                             act_words))
        return 0;
    if (!yolo_dma_ddr_to_wgt(wgt_ddr, wgt_base, YOLO_CONV0_OUT_C * YOLO_CONV0_K * YOLO_CONV0_K))
        return 0;

    yolo_set_pad_value(YOLO_CONV0_PAD_VALUE);
    yolo_set_silu_requant(silu_requant_mul, silu_requant_shift, silu_requant_zp);

    if (!yolo_run_conv2d_qparams_pads(act_base,
                                      wgt_base,
                                      out_base,
                                      YOLO_CONV0_IN_W,
                                      (uint32_t)strip->in_rows,
                                      YOLO_CONV0_IN_C,
                                      YOLO_CONV0_OUT_C,
                                      YOLO_CONV0_K,
                                      YOLO_CONV0_K,
                                      YOLO_CONV0_STRIDE,
                                      pad_h,
                                      1u,
                                      bias,
                                      scale_mul,
                                      scale_shift,
                                      NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN))
        return 0;

    return yolo_dma_out_to_ddr(out_ddr, out_base, out_words, 0u);
}
