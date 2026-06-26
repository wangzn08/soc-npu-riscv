#ifndef NPU_DESC_H
#define NPU_DESC_H

#include <stdint.h>

typedef enum {
    NPU_DESC_OP_NOP = 0,
    NPU_DESC_OP_DMA_DDR_TO_ACT = 1,
    NPU_DESC_OP_DMA_ACT_TO_DDR = 2,
    NPU_DESC_OP_UPSAMPLE2X_DDR = 3,
    NPU_DESC_OP_COPY_DDR_TO_DDR = 4,
    NPU_DESC_OP_CONV2D_TILED = 5
} npu_desc_op_t;

typedef struct {
    uint32_t op;
    uint32_t src0;
    uint32_t src1;
    uint32_t dst;
    uint32_t wgt;
    uint32_t scratch0;
    uint32_t scratch1;
    uint16_t in_w;
    uint16_t in_h;
    uint16_t in_c;
    uint16_t out_c;
    uint8_t  kh;
    uint8_t  kw;
    uint8_t  stride;
    uint8_t  pad;
    uint32_t words;
    uint32_t flags;
    const int32_t  *bias;
    const uint32_t *scale_mul;
    const uint32_t *scale_shift;
    const uint8_t  *lut;
    int32_t pad_value;
    uint32_t wgt_words_per_oc;
    uint32_t strip_out_rows;
} npu_desc_t;

int npu_desc_validate(const npu_desc_t *d);
int npu_desc_run(const npu_desc_t *d);
int npu_desc_run_many(const npu_desc_t *list, uint32_t count);

#endif
