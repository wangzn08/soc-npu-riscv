// CPU/NPU cooperative smoke for a tiny real YOLO-style subgraph against a
// C-reference quantization contract. The subgraph is:
//   pointwise conv + SiLU + requant -> upsample2x -> concat -> pointwise conv + SiLU + requant

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_subgraph_real_data.h"
#include <stdint.h>

#define IN_DDR         0x40044000u
#define WGT0_DDR       0x40046000u
#define CONV0_DDR      0x40048000u
#define UP_DDR         0x4004A000u
#define SKIP_DDR       0x4004C000u
#define WGT1_DDR       0x4004E000u
#define OUT_DDR        0x40050000u

#define ACT_IN_BASE    0u
#define ACT_UP_SRC     128u
#define ACT_UP_DST     256u
#define ACT_CONCAT     512u
#define WGT_BASE       0u
#define OUT_BASE       0u

static void write_ddr_word128(uint32_t byte_addr, uint32_t word_idx, const uint32_t lanes[4])
{
    volatile uint32_t *ptr = (volatile uint32_t *)(byte_addr + word_idx * 16u);
    ptr[0] = lanes[0];
    ptr[1] = lanes[1];
    ptr[2] = lanes[2];
    ptr[3] = lanes[3];
}

static int32_t read_ddr_s8(uint32_t byte_addr, uint32_t word_idx, uint32_t byte_idx)
{
    volatile uint32_t *ptr = (volatile uint32_t *)(byte_addr + word_idx * 16u);
    uint32_t lane = ptr[byte_idx >> 2];
    uint32_t byte = (lane >> ((byte_idx & 3u) * 8u)) & 0xFFu;
    return (byte & 0x80u) ? ((int32_t)byte - 256) : (int32_t)byte;
}

static int32_t s8(uint32_t byte)
{
    byte &= 0xFFu;
    return (byte & 0x80u) ? ((int32_t)byte - 256) : (int32_t)byte;
}

static uint32_t abs_diff(int32_t a, int32_t b)
{
    int32_t d = a - b;
    return (uint32_t)(d < 0 ? -d : d);
}

void usercode7(void)
{
    uint32_t i;
    uint32_t pos;
    uint32_t oc;
    uint32_t errors = 0u;

    print_str("YOLO REAL SUBGRAPH CREF CPU SMOKE\n");

    for (i = 0u; i < YOLO_REAL_SUB_ACT_WORDS; i++)
        write_ddr_word128(IN_DDR, i, yolo_real_sub_act_words[i]);
    for (i = 0u; i < YOLO_REAL_SUB_SKIP_WORDS; i++)
        write_ddr_word128(SKIP_DDR, i, yolo_real_sub_skip_words[i]);
    for (i = 0u; i < YOLO_REAL_SUB_WGT0_WORDS; i++)
        write_ddr_word128(WGT0_DDR, i, yolo_real_sub_wgt0_words[i]);
    for (i = 0u; i < YOLO_REAL_SUB_WGT1_WORDS; i++)
        write_ddr_word128(WGT1_DDR, i, yolo_real_sub_wgt1_words[i]);
    for (i = 0u; i < YOLO_REAL_SUB_OUT_WORDS; i++) {
        const uint32_t zero[4] = {0u, 0u, 0u, 0u};
        write_ddr_word128(OUT_DDR, i, zero);
    }

    yolo_set_silu_requant(YOLO_REAL_SUB_REQUANT_MUL,
                          YOLO_REAL_SUB_REQUANT_SHIFT,
                          YOLO_REAL_SUB_REQUANT_ZP);

    if (!yolo_dma_ddr_to_act(IN_DDR, ACT_IN_BASE, YOLO_REAL_SUB_ACT_WORDS) ||
        !yolo_dma_ddr_to_wgt(WGT0_DDR, WGT_BASE, YOLO_REAL_SUB_WGT0_WORDS) ||
        !yolo_run_pw_conv1x1_qparams(ACT_IN_BASE, WGT_BASE, OUT_BASE,
                                     YOLO_REAL_SUB_IN_W, YOLO_REAL_SUB_IN_H,
                                     YOLO_REAL_SUB_IC0, YOLO_REAL_SUB_OC0,
                                     yolo_real_sub_bias0_q,
                                     yolo_real_sub_scale_mul0,
                                     yolo_real_sub_scale_shift0,
                                     NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN) ||
        !yolo_dma_out_to_ddr(CONV0_DDR, OUT_BASE, YOLO_REAL_SUB_CONV0_WORDS, 0u)) {
        print_str("  real cref conv0 path failed\n");
        errors++;
    }

    if (errors == 0u &&
        (!yolo_dma_ddr_to_act(CONV0_DDR, ACT_UP_SRC, YOLO_REAL_SUB_CONV0_WORDS) ||
         !yolo_run_upsample2x(ACT_UP_SRC, ACT_UP_DST,
                              YOLO_REAL_SUB_IN_W, YOLO_REAL_SUB_IN_H,
                              YOLO_REAL_SUB_OC0_GROUPS) ||
         !yolo_dma_act_to_ddr(UP_DDR, ACT_UP_DST, YOLO_REAL_SUB_UP_WORDS))) {
        print_str("  real cref upsample path failed\n");
        errors++;
    }

    if (errors == 0u &&
        (!yolo_concat2_ddr_to_act(UP_DDR, SKIP_DDR, ACT_CONCAT,
                                  YOLO_REAL_SUB_UP_SPATIAL,
                                  YOLO_REAL_SUB_OC0_GROUPS,
                                  YOLO_REAL_SUB_SKIP_GROUPS) ||
         !yolo_dma_ddr_to_wgt(WGT1_DDR, WGT_BASE, YOLO_REAL_SUB_WGT1_WORDS) ||
         !yolo_run_pw_conv1x1_qparams(ACT_CONCAT, WGT_BASE, OUT_BASE,
                                      YOLO_REAL_SUB_UP_W, YOLO_REAL_SUB_UP_H,
                                      YOLO_REAL_SUB_IC1, YOLO_REAL_SUB_OC1,
                                      yolo_real_sub_bias1_q,
                                      yolo_real_sub_scale_mul1,
                                      yolo_real_sub_scale_shift1,
                                      NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN) ||
         !yolo_dma_out_to_ddr(OUT_DDR, OUT_BASE, YOLO_REAL_SUB_OUT_WORDS, 0u))) {
        print_str("  real cref concat/final conv path failed\n");
        errors++;
    }

    for (pos = 0u; pos < YOLO_REAL_SUB_OUT_WORDS; pos++) {
        for (oc = 0u; oc < YOLO_REAL_SUB_OC1; oc++) {
            int32_t got = read_ddr_s8(OUT_DDR, pos, oc);
            int32_t expect = s8(yolo_real_sub_expected_cref[pos][oc]);
            uint32_t diff = abs_diff(got, expect);
            if (diff > YOLO_REAL_SUB_CREF_TOL) {
                errors++;
                print_str("  mismatch pos=");
                print_dec(pos);
                print_str(" oc=");
                print_dec(oc);
                print_str(" got=");
                print_dec(got);
                print_str(" exp=");
                print_dec(expect);
                print_str(" diff=");
                print_dec(diff);
                print_str("\n");
            }
        }
    }

    if (errors == 0u) {
        print_str("YOLO REAL SUBGRAPH CREF CPU SMOKE PASS\n");
        return;
    }

    print_str("YOLO REAL SUBGRAPH CREF CPU SMOKE FAIL errors=");
    print_dec(errors);
    print_str("\n");
    __asm__ volatile ("ebreak");
}
