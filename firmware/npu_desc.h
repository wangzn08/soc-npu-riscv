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

#define NPU_HW_DESC_VERSION 1u
#define NPU_HW_DESC_WORDS   16u

#define NPU_HW_DESC_OP_NOP                  0x00u
#define NPU_HW_DESC_OP_DMA_DDR_TO_ACT       0x01u
#define NPU_HW_DESC_OP_DMA_ACT_TO_DDR       0x02u
#define NPU_HW_DESC_OP_DMA_OUT_TO_DDR       0x03u
#define NPU_HW_DESC_OP_IMG_EXPAND           0x04u
#define NPU_HW_DESC_OP_SRAM_COPY_OUT_TO_ACT 0x05u
#define NPU_HW_DESC_OP_CONV2D               0x06u
#define NPU_HW_DESC_OP_GEMM                 0x07u
#define NPU_HW_DESC_OP_STOP_IRQ             0x08u

#define NPU_HW_DESC_OP_UPSAMPLE2X           0x20u
#define NPU_HW_DESC_OP_MAXPOOL5X5           0x21u
#define NPU_HW_DESC_OP_ELTWISE_ADD          0x22u
#define NPU_HW_DESC_OP_DFL                  0x23u
/* OP_LUT_LOAD (implemented in descriptor_engine.v): stream a 256-entry exact-
 * SiLU LUT into the NPU before the YOLO convs that index it.
 *   w[2] = LUT DDR base (256 bytes = 16 contiguous 128-bit beats, one INT8
 *          entry per byte; entry index = beat*16 + byte). */
#define NPU_HW_DESC_OP_LUT_LOAD             0x24u
#define NPU_HW_DESC_OP_ACTIVATION_CFG       0x25u

#define NPU_HW_DESC_ERR_NONE                0u
#define NPU_HW_DESC_ERR_BAD_VERSION         1u
#define NPU_HW_DESC_ERR_BAD_OPCODE          2u
#define NPU_HW_DESC_ERR_UNSUPPORTED_OP      3u
#define NPU_HW_DESC_ERR_BAD_COUNT           4u
#define NPU_HW_DESC_ERR_BAD_ALIGNMENT       5u
#define NPU_HW_DESC_ERR_BAD_SHAPE           6u
#define NPU_HW_DESC_ERR_BUSY_AT_START       7u
#define NPU_HW_DESC_ERR_AXI_DESC_READ       8u
#define NPU_HW_DESC_ERR_AXI_QPARAM_READ     9u
#define NPU_HW_DESC_ERR_ENGINE_TIMEOUT      10u

typedef struct {
    uint32_t w[NPU_HW_DESC_WORDS];
} npu_hw_desc_t;

static inline void npu_hw_desc_clear(npu_hw_desc_t *d)
{
    uint32_t i;
    for (i = 0u; i < NPU_HW_DESC_WORDS; i++)
        d->w[i] = 0u;
}

static inline void npu_hw_desc_set_op(npu_hw_desc_t *d, uint32_t op, uint32_t flags)
{
    d->w[0] = (op & 0xFFu) | ((NPU_HW_DESC_VERSION & 0xFFu) << 8) |
              ((flags & 0xFFFFu) << 16);
    d->w[1] = flags >> 16;
}

#endif
