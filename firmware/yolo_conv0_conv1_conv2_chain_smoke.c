// CPU/NPU smoke for a real conv0 -> conv1 -> conv2 strip chain.

#include "firmware.h"
#include "yolo_block_plan.h"
#include "yolo_ops.h"
#include "yolo_plan.h"
#include "yolo_conv0_strip_real_data.h"
#include "yolo_conv1_from_conv0_chain_data.h"
#include "yolo_conv2_from_conv1_chain_data.h"
#include <stdint.h>

#define C0_ACT_DDR 0x40090000u
#define C0_WGT_DDR 0x400C0000u
#define C0_OUT_DDR 0x400C4000u
#define C1_WGT_DDR 0x401A0000u
#define C1_OUT_DDR 0x401A8000u
#define C2_WGT_DDR 0x40240000u
#define C2_OUT_DDR 0x40248000u

#define ACT_BASE 0u
#define WGT_BASE 0u
#define OUT_BASE 0u

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
    uint32_t s;
    uint32_t pos;
    uint32_t oc;
    uint32_t errors = 0u;
    const yolo_block_plan_entry_t *conv0 = &yolo_block_plan[0];

    print_str("YOLO CONV0->CONV1->CONV2 CHAIN CPU SMOKE\n");

    for (i = 0u; i < YOLO_CONV0_STRIP_ACT_WORDS; i++)
        write_ddr_word128(C0_ACT_DDR, i, yolo_conv0_strip_act_words[i]);
    for (i = 0u; i < YOLO_CONV0_STRIP_WGT_WORDS; i++)
        write_ddr_word128(C0_WGT_DDR, i, yolo_conv0_strip_wgt_words[i]);
    for (i = 0u; i < YOLO_CONV1_CHAIN_WGT_WORDS; i++)
        write_ddr_word128(C1_WGT_DDR, i, yolo_conv1_chain_wgt_words[i]);
    for (i = 0u; i < YOLO_CONV2_CHAIN_WGT_WORDS; i++)
        write_ddr_word128(C2_WGT_DDR, i, yolo_conv2_chain_wgt_words[i]);

    for (i = 0u; i < YOLO_CONV0_STRIP_OUT_WORDS_PER_STRIP * YOLO_CONV0_STRIP_TEST_COUNT; i++) {
        const uint32_t zero[4] = {0u, 0u, 0u, 0u};
        write_ddr_word128(C0_OUT_DDR, i, zero);
    }
    for (i = 0u; i < YOLO_CONV1_CHAIN_OUT_WORDS; i++) {
        const uint32_t zero[4] = {0u, 0u, 0u, 0u};
        write_ddr_word128(C1_OUT_DDR, i, zero);
    }
    for (i = 0u; i < YOLO_CONV2_CHAIN_OUT_WORDS; i++) {
        const uint32_t zero[4] = {0u, 0u, 0u, 0u};
        write_ddr_word128(C2_OUT_DDR, i, zero);
    }

    for (s = 0u; s < YOLO_CONV0_STRIP_TEST_COUNT; s++) {
        const yolo_strip_plan_entry_t *strip = &yolo_strip_plan[conv0->strip_offset + s];
        if (!yolo_run_conv0_strip_from_plan(strip,
                                            C0_ACT_DDR,
                                            C0_WGT_DDR,
                                            C0_OUT_DDR + s * YOLO_CONV0_STRIP_OUT_WORDS_PER_STRIP * 16u,
                                            ACT_BASE,
                                            WGT_BASE,
                                            OUT_BASE,
                                            yolo_conv0_strip_bias_q,
                                            yolo_conv0_strip_scale_mul,
                                            yolo_conv0_strip_scale_shift,
                                            YOLO_CONV0_STRIP_REQUANT_MUL,
                                            YOLO_CONV0_STRIP_REQUANT_SHIFT,
                                            YOLO_CONV0_STRIP_REQUANT_ZP)) {
            print_str("  conv0 strip failed s=");
            print_dec(s);
            print_str("\n");
            errors++;
            break;
        }
    }

    if (errors == 0u) {
        yolo_set_pad_value(-127);
        yolo_set_silu_requant(YOLO_CONV1_CHAIN_REQUANT_MUL,
                              YOLO_CONV1_CHAIN_REQUANT_SHIFT,
                              YOLO_CONV1_CHAIN_REQUANT_ZP);
        if (!yolo_dma_ddr_to_act(C0_OUT_DDR, ACT_BASE, YOLO_CONV1_CHAIN_ACT_WORDS) ||
            !yolo_dma_ddr_to_wgt(C1_WGT_DDR, WGT_BASE, YOLO_CONV1_CHAIN_WGT_WORDS) ||
            !yolo_run_conv2d_qparams_pads(ACT_BASE,
                                          WGT_BASE,
                                          OUT_BASE,
                                          YOLO_CONV1_CHAIN_IN_W,
                                          YOLO_CONV1_CHAIN_IN_H,
                                          YOLO_CONV1_CHAIN_IC,
                                          YOLO_CONV1_CHAIN_OC,
                                          YOLO_CONV1_CHAIN_KH,
                                          YOLO_CONV1_CHAIN_KW,
                                          YOLO_CONV1_CHAIN_STRIDE,
                                          YOLO_CONV1_CHAIN_PAD,
                                          YOLO_CONV1_CHAIN_PAD,
                                          yolo_conv1_chain_bias_q,
                                          yolo_conv1_chain_scale_mul,
                                          yolo_conv1_chain_scale_shift,
                                          NPU_CTRL_OC_SINGLE |
                                          NPU_CTRL_SILU_EN |
                                          NPU_CTRL_SILU_REQUANT_EN) ||
            !yolo_dma_out_to_ddr(C1_OUT_DDR, OUT_BASE, YOLO_CONV1_CHAIN_OUT_WORDS, 0u)) {
            print_str("  conv1 chain run failed\n");
            errors++;
        }
    }

    if (errors == 0u) {
        yolo_set_pad_value(YOLO_CONV2_CHAIN_PAD_VALUE);
        yolo_set_silu_requant(YOLO_CONV2_CHAIN_REQUANT_MUL,
                              YOLO_CONV2_CHAIN_REQUANT_SHIFT,
                              YOLO_CONV2_CHAIN_REQUANT_ZP);
        if (!yolo_dma_ddr_to_act(C1_OUT_DDR, ACT_BASE, YOLO_CONV2_CHAIN_ACT_WORDS) ||
            !yolo_dma_ddr_to_wgt(C2_WGT_DDR, WGT_BASE, YOLO_CONV2_CHAIN_WGT_WORDS) ||
            !yolo_run_pw_conv1x1_qparams(ACT_BASE,
                                         WGT_BASE,
                                         OUT_BASE,
                                         YOLO_CONV2_CHAIN_IN_W,
                                         YOLO_CONV2_CHAIN_IN_H,
                                         YOLO_CONV2_CHAIN_IC,
                                         YOLO_CONV2_CHAIN_OC,
                                         yolo_conv2_chain_bias_q,
                                         yolo_conv2_chain_scale_mul,
                                         yolo_conv2_chain_scale_shift,
                                         NPU_CTRL_OC_SINGLE |
                                         NPU_CTRL_SILU_EN |
                                         NPU_CTRL_SILU_REQUANT_EN) ||
            !yolo_dma_out_to_ddr(C2_OUT_DDR, OUT_BASE, YOLO_CONV2_CHAIN_OUT_WORDS, 0u)) {
            print_str("  conv2 chain run failed\n");
            errors++;
        }
    }

    for (pos = 0u; pos < YOLO_CONV2_CHAIN_OUT_SPATIAL && errors <= 16u; pos++) {
        for (oc = 0u; oc < YOLO_CONV2_CHAIN_OC; oc++) {
            uint32_t group = oc >> 4;
            uint32_t lane = oc & 15u;
            uint32_t word_idx = group * YOLO_CONV2_CHAIN_OUT_SPATIAL + pos;
            int32_t got = read_ddr_s8(C2_OUT_DDR, word_idx, lane);
            int32_t expect = s8(yolo_conv2_chain_expected_rtl[pos][oc]);
            uint32_t diff = abs_diff(got, expect);
            if (diff > YOLO_CONV2_CHAIN_RTL_TOL) {
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
                if (errors > 16u)
                    break;
            }
        }
    }

    if (errors == 0u) {
        print_str("YOLO CONV0->CONV1->CONV2 CHAIN CPU SMOKE PASS\n");
        return;
    }

    print_str("YOLO CONV0->CONV1->CONV2 CHAIN CPU SMOKE FAIL errors=");
    print_dec(errors);
    print_str("\n");
    __asm__ volatile ("ebreak");
}
