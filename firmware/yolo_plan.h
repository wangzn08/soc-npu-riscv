#ifndef YOLO_PLAN_H
#define YOLO_PLAN_H

#include <stdint.h>
#include "yolo_block_plan.h"

typedef struct {
    uint32_t act_ddr;
    uint32_t wgt_ddr;
    uint32_t out_ddr;
    uint32_t act_base;
    uint32_t wgt_base;
    uint32_t out_base;
    uint32_t out_pong;
    uint32_t in_w;
    uint32_t in_h;
    uint32_t in_c;
    uint32_t out_c;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride;
    uint32_t pad;
    uint32_t pad_value;
    uint32_t act_words;
    uint32_t wgt_words;
    uint32_t out_words;
    const int32_t *bias;
    const uint32_t *scale_mul;
    const uint32_t *scale_shift;
    uint32_t silu_requant_mul;
    uint32_t silu_requant_shift;
    int32_t silu_requant_zp;
    uint32_t ctrl_flags;
} yolo_conv_desc_t;

int yolo_run_conv_desc(const yolo_conv_desc_t *desc);
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
                                   int32_t silu_requant_zp);

#endif
