#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_plan.h"

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
