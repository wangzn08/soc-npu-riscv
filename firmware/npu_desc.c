#include "npu_desc.h"

int npu_desc_validate(const npu_desc_t *d)
{
    if (d == (const npu_desc_t *)0)
        return 0;
    switch ((npu_desc_op_t)d->op) {
    case NPU_DESC_OP_NOP:
        return 1;
    case NPU_DESC_OP_DMA_DDR_TO_ACT:
    case NPU_DESC_OP_DMA_ACT_TO_DDR:
    case NPU_DESC_OP_UPSAMPLE2X_DDR:
    case NPU_DESC_OP_COPY_DDR_TO_DDR:
    case NPU_DESC_OP_CONV2D_TILED:
        return 1;
    default:
        return 0;
    }
}

int npu_desc_run(const npu_desc_t *d)
{
    if (!npu_desc_validate(d))
        return 0;
    if (d->op == NPU_DESC_OP_NOP)
        return 1;
    return 0;
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
