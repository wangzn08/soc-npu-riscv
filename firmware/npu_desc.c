#include "npu_desc.h"

extern int yolo_dma_ddr_to_act(uint32_t ddr_addr,
                               uint32_t act_base,
                               uint32_t words) __attribute__((weak));
extern int yolo_dma_act_to_ddr(uint32_t ddr_addr,
                               uint32_t act_base,
                               uint32_t words) __attribute__((weak));
extern int yolo_run_upsample2x_ddr(uint32_t src_ddr,
                                   uint32_t dst_ddr,
                                   uint32_t scratch_act_base,
                                   uint32_t in_w,
                                   uint32_t in_h,
                                   uint32_t ic_groups) __attribute__((weak));
extern int yolo_copy_ddr_to_ddr_via_act(uint32_t src_ddr,
                                        uint32_t dst_ddr,
                                        uint32_t scratch_act_base,
                                        uint32_t words) __attribute__((weak));
extern int yolo_run_conv2d_tiled(uint32_t in_ddr,
                                 uint32_t wgt_all_ddr,
                                 uint32_t wgt_base,
                                 uint32_t out_ddr,
                                 uint32_t pad_row_ddr,
                                 uint32_t in_w,
                                 uint32_t in_h,
                                 uint32_t in_c,
                                 uint32_t out_c,
                                 uint32_t kernel_h,
                                 uint32_t kernel_w,
                                 uint32_t stride,
                                 uint32_t pad,
                                 const int32_t *bias,
                                 const uint32_t *scale_mul,
                                 const uint32_t *scale_shift,
                                 uint32_t ctrl_flags,
                                 uint32_t wgt_words_per_oc,
                                 uint32_t strip_out_rows,
                                 int32_t pad_value) __attribute__((weak));

int npu_desc_validate(const npu_desc_t *d)
{
    if (d == (const npu_desc_t *)0)
        return 0;
    switch ((npu_desc_op_t)d->op) {
    case NPU_DESC_OP_NOP:
        return 1;
    case NPU_DESC_OP_DMA_DDR_TO_ACT:
    case NPU_DESC_OP_DMA_ACT_TO_DDR:
    case NPU_DESC_OP_COPY_DDR_TO_DDR:
        return d->words != 0u;
    case NPU_DESC_OP_UPSAMPLE2X_DDR:
        return d->in_w != 0u && d->in_h != 0u && d->in_c != 0u &&
               (d->in_c & 15u) == 0u;
    case NPU_DESC_OP_CONV2D_TILED:
        return d->in_w != 0u && d->in_h != 0u && d->in_c != 0u &&
               d->out_c != 0u && d->kh != 0u && d->kw != 0u &&
               d->stride != 0u && d->bias != (const int32_t *)0 &&
               d->scale_mul != (const uint32_t *)0 &&
               d->scale_shift != (const uint32_t *)0;
    default:
        return 0;
    }
}

int npu_desc_run(const npu_desc_t *d)
{
    if (!npu_desc_validate(d))
        return 0;

    switch ((npu_desc_op_t)d->op) {
    case NPU_DESC_OP_NOP:
        return 1;
    case NPU_DESC_OP_DMA_DDR_TO_ACT:
        if (yolo_dma_ddr_to_act == (void *)0)
            return 0;
        return yolo_dma_ddr_to_act(d->src0, d->dst, d->words);
    case NPU_DESC_OP_DMA_ACT_TO_DDR:
        if (yolo_dma_act_to_ddr == (void *)0)
            return 0;
        return yolo_dma_act_to_ddr(d->dst, d->src0, d->words);
    case NPU_DESC_OP_UPSAMPLE2X_DDR:
        if (yolo_run_upsample2x_ddr == (void *)0)
            return 0;
        return yolo_run_upsample2x_ddr(d->src0, d->dst, d->scratch0,
                                       d->in_w, d->in_h, d->in_c / 16u);
    case NPU_DESC_OP_COPY_DDR_TO_DDR:
        if (yolo_copy_ddr_to_ddr_via_act == (void *)0)
            return 0;
        return yolo_copy_ddr_to_ddr_via_act(d->src0, d->dst, d->scratch0, d->words);
    case NPU_DESC_OP_CONV2D_TILED:
        if (yolo_run_conv2d_tiled == (void *)0)
            return 0;
        return yolo_run_conv2d_tiled(d->src0, d->wgt, d->scratch0, d->dst, d->scratch1,
                                     d->in_w, d->in_h, d->in_c, d->out_c,
                                     d->kh, d->kw, d->stride, d->pad,
                                     d->bias, d->scale_mul, d->scale_shift,
                                     d->flags, d->wgt_words_per_oc,
                                     d->strip_out_rows, d->pad_value);
    default:
        return 0;
    }
}

int npu_desc_run_many(const npu_desc_t *list, uint32_t count)
{
    uint32_t i;
    if (list == (const npu_desc_t *)0 && count != 0u)
        return 0;
    for (i = 0u; i < count; i++) {
        if (!npu_desc_run(&list[i]))
            return 0;
    }
    return 1;
}
