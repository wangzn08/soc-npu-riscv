// CPU/NPU smoke for the first C2f bottleneck residual add (/model.2/m.0/Add):
//   conv0 -> conv1 -> conv2 -> slice(s1) -> conv3 -> conv4,
//   then add_out = s1 + conv4_out via the shared HW signed eltwise adder.
//
// Both operands are requantized to the glue scale/zp:
//   * conv4 conv -> SiLU -> requant-to-glue, written to Out SRAM base 0 with
//     eltwise enabled, adding the staged s1.
//   * s1(glue): an extra conv2 group-1 (cv1, SiLU) pass requantized to glue,
//     left resident in Out SRAM at R_SKIP as the eltwise skip source.

#include "firmware.h"
#include "yolo_block_plan.h"
#include "yolo_ops.h"
#include "yolo_plan.h"
#include "yolo_conv0_strip_real_data.h"
#include "yolo_conv1_from_conv0_chain_data.h"
#include "yolo_conv2_from_conv1_chain_data.h"
#include "yolo_conv3_from_conv2_chain_data.h"
#include "yolo_conv4_from_conv3_chain_data.h"
#include "yolo_c2f_add_m5u_data.h"
#include <stdint.h>

#define C0_ACT_DDR 0x40090000u
#define C0_WGT_DDR 0x400C0000u
#define C0_OUT_DDR 0x400C4000u
#define C1_WGT_DDR 0x401A0000u
#define C1_OUT_DDR 0x401A8000u
#define C2_WGT_DDR 0x40240000u
#define C2_OUT_DDR 0x40248000u
#define C3_WGT_DDR 0x40300000u
#define C3_OUT_DDR 0x40308000u
#define C4_WGT_DDR 0x403C0000u
#define ADD_OUT_DDR 0x40440000u
#define STAGE_WGT_DDR 0x40480000u

#define ACT_BASE 0u
#define WGT_BASE 0u
#define OUT_BASE 0u
#define R_SKIP   4096u   // Out-SRAM word base for the staged s1(glue) skip source

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

// Run conv0->conv1->conv2->slice(s1)->conv3, leaving:
//   C1_OUT_DDR = conv1 output (32ch), C3_OUT_DDR = conv3 output (16ch).
static int run_conv0_to_conv3(void)
{
    uint32_t i;
    uint32_t s;
    const yolo_block_plan_entry_t *conv0 = &yolo_block_plan[0];

    for (i = 0u; i < YOLO_CONV0_STRIP_ACT_WORDS; i++)
        write_ddr_word128(C0_ACT_DDR, i, yolo_conv0_strip_act_words[i]);
    for (i = 0u; i < YOLO_CONV0_STRIP_WGT_WORDS; i++)
        write_ddr_word128(C0_WGT_DDR, i, yolo_conv0_strip_wgt_words[i]);
    for (i = 0u; i < YOLO_CONV1_CHAIN_WGT_WORDS; i++)
        write_ddr_word128(C1_WGT_DDR, i, yolo_conv1_chain_wgt_words[i]);
    for (i = 0u; i < YOLO_CONV2_CHAIN_WGT_WORDS; i++)
        write_ddr_word128(C2_WGT_DDR, i, yolo_conv2_chain_wgt_words[i]);
    for (i = 0u; i < YOLO_CONV3_CHAIN_WGT_WORDS; i++)
        write_ddr_word128(C3_WGT_DDR, i, yolo_conv3_chain_wgt_words[i]);

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
                                            YOLO_CONV0_STRIP_REQUANT_ZP))
            return 0;
    }

    yolo_set_pad_value(-127);
    yolo_set_silu_requant(YOLO_CONV1_CHAIN_REQUANT_MUL,
                          YOLO_CONV1_CHAIN_REQUANT_SHIFT,
                          YOLO_CONV1_CHAIN_REQUANT_ZP);
    if (!yolo_dma_ddr_to_act(C0_OUT_DDR, ACT_BASE, YOLO_CONV1_CHAIN_ACT_WORDS) ||
        !yolo_dma_ddr_to_wgt(C1_WGT_DDR, WGT_BASE, YOLO_CONV1_CHAIN_WGT_WORDS) ||
        !yolo_run_conv2d_qparams_pads(ACT_BASE, WGT_BASE, OUT_BASE,
                                      YOLO_CONV1_CHAIN_IN_W, YOLO_CONV1_CHAIN_IN_H,
                                      YOLO_CONV1_CHAIN_IC, YOLO_CONV1_CHAIN_OC,
                                      YOLO_CONV1_CHAIN_KH, YOLO_CONV1_CHAIN_KW,
                                      YOLO_CONV1_CHAIN_STRIDE,
                                      YOLO_CONV1_CHAIN_PAD, YOLO_CONV1_CHAIN_PAD,
                                      yolo_conv1_chain_bias_q,
                                      yolo_conv1_chain_scale_mul,
                                      yolo_conv1_chain_scale_shift,
                                      NPU_CTRL_OC_SINGLE | NPU_CTRL_SILU_EN |
                                      NPU_CTRL_SILU_REQUANT_EN) ||
        !yolo_dma_out_to_ddr(C1_OUT_DDR, OUT_BASE, YOLO_CONV1_CHAIN_OUT_WORDS, 0u))
        return 0;

    yolo_set_pad_value(YOLO_CONV2_CHAIN_PAD_VALUE);
    yolo_set_silu_requant(YOLO_CONV2_CHAIN_REQUANT_MUL,
                          YOLO_CONV2_CHAIN_REQUANT_SHIFT,
                          YOLO_CONV2_CHAIN_REQUANT_ZP);
    if (!yolo_dma_ddr_to_act(C1_OUT_DDR, ACT_BASE, YOLO_CONV2_CHAIN_ACT_WORDS) ||
        !yolo_dma_ddr_to_wgt(C2_WGT_DDR, WGT_BASE, YOLO_CONV2_CHAIN_WGT_WORDS) ||
        !yolo_run_pw_conv1x1_qparams(ACT_BASE, WGT_BASE, OUT_BASE,
                                     YOLO_CONV2_CHAIN_IN_W, YOLO_CONV2_CHAIN_IN_H,
                                     YOLO_CONV2_CHAIN_IC, YOLO_CONV2_CHAIN_OC,
                                     yolo_conv2_chain_bias_q,
                                     yolo_conv2_chain_scale_mul,
                                     yolo_conv2_chain_scale_shift,
                                     NPU_CTRL_OC_SINGLE | NPU_CTRL_SILU_EN |
                                     NPU_CTRL_SILU_REQUANT_EN) ||
        !yolo_dma_out_to_ddr(C2_OUT_DDR, OUT_BASE, YOLO_CONV2_CHAIN_OUT_WORDS, 0u))
        return 0;

    yolo_set_pad_value(YOLO_CONV3_CHAIN_PAD_VALUE);
    yolo_set_silu_requant(YOLO_CONV3_CHAIN_REQUANT_MUL,
                          YOLO_CONV3_CHAIN_REQUANT_SHIFT,
                          YOLO_CONV3_CHAIN_REQUANT_ZP);
    return yolo_slice_ddr_to_act(C2_OUT_DDR, ACT_BASE,
                                 YOLO_CONV3_CHAIN_IN_SPATIAL,
                                 YOLO_CONV3_CHAIN_SRC_GROUP, 1u) &&
           yolo_dma_ddr_to_wgt(C3_WGT_DDR, WGT_BASE, YOLO_CONV3_CHAIN_WGT_WORDS) &&
           yolo_run_conv2d_qparams(ACT_BASE, WGT_BASE, OUT_BASE,
                                   YOLO_CONV3_CHAIN_IN_W, YOLO_CONV3_CHAIN_IN_H,
                                   YOLO_CONV3_CHAIN_IC, YOLO_CONV3_CHAIN_OC,
                                   YOLO_CONV3_CHAIN_KH, YOLO_CONV3_CHAIN_KW,
                                   YOLO_CONV3_CHAIN_STRIDE, YOLO_CONV3_CHAIN_PAD,
                                   yolo_conv3_chain_bias_q,
                                   yolo_conv3_chain_scale_mul,
                                   yolo_conv3_chain_scale_shift,
                                   NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN) &&
           yolo_dma_out_to_ddr(C3_OUT_DDR, OUT_BASE, YOLO_CONV3_CHAIN_OUT_WORDS, 0u);
}

// Stage s1(glue): conv2 group-1 (cv1) over conv1 output, SiLU, requant-to-glue,
// left resident in Out SRAM at R_SKIP (no drain) as the eltwise skip source.
static int run_stage_s1_glue(void)
{
    uint32_t i;
    for (i = 0u; i < YOLO_C2F_STAGE_WGT_WORDS; i++)
        write_ddr_word128(STAGE_WGT_DDR, i, yolo_c2f_stage_wgt_words[i]);

    yolo_set_silu_requant(YOLO_C2F_ADD_GLUE_REQUANT_MUL,
                          YOLO_C2F_ADD_GLUE_REQUANT_SHIFT,
                          YOLO_C2F_ADD_GLUE_ZP);
    if (!yolo_dma_ddr_to_act(C1_OUT_DDR, ACT_BASE, YOLO_C2F_STAGE_ACT_WORDS) ||
        !yolo_dma_ddr_to_wgt(STAGE_WGT_DDR, WGT_BASE, YOLO_C2F_STAGE_WGT_WORDS) ||
        !yolo_run_pw_conv1x1_qparams(ACT_BASE, WGT_BASE, R_SKIP,
                                     YOLO_C2F_STAGE_IN_W, YOLO_C2F_STAGE_IN_H,
                                     YOLO_C2F_STAGE_IC, YOLO_C2F_STAGE_OC,
                                     yolo_c2f_stage_bias_q,
                                     yolo_c2f_stage_scale_mul,
                                     yolo_c2f_stage_scale_shift,
                                     NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN))
        return 0;
    return 1;
}

// conv4 conv -> SiLU -> requant-to-glue, with HW signed eltwise adding the
// staged s1(glue). Output drained to ADD_OUT_DDR.
static int run_conv4_add(void)
{
    uint32_t i;
    for (i = 0u; i < YOLO_CONV4_CHAIN_WGT_WORDS; i++)
        write_ddr_word128(C4_WGT_DDR, i, yolo_conv4_chain_wgt_words[i]);

    yolo_set_pad_value(YOLO_CONV4_CHAIN_PAD_VALUE);
    yolo_set_silu_requant(YOLO_C2F_ADD_GLUE_REQUANT_MUL,
                          YOLO_C2F_ADD_GLUE_REQUANT_SHIFT,
                          YOLO_C2F_ADD_GLUE_ZP);
    yolo_set_eltwise(YOLO_C2F_ADD_GLUE_ZP, R_SKIP);
    if (!yolo_dma_ddr_to_act(C3_OUT_DDR, ACT_BASE, YOLO_CONV4_CHAIN_ACT_WORDS) ||
        !yolo_dma_ddr_to_wgt(C4_WGT_DDR, WGT_BASE, YOLO_CONV4_CHAIN_WGT_WORDS) ||
        !yolo_run_conv2d_qparams(ACT_BASE, WGT_BASE, OUT_BASE,
                                 YOLO_CONV4_CHAIN_IN_W, YOLO_CONV4_CHAIN_IN_H,
                                 YOLO_CONV4_CHAIN_IC, YOLO_CONV4_CHAIN_OC,
                                 YOLO_CONV4_CHAIN_KH, YOLO_CONV4_CHAIN_KW,
                                 YOLO_CONV4_CHAIN_STRIDE, YOLO_CONV4_CHAIN_PAD,
                                 yolo_conv4_chain_bias_q,
                                 yolo_conv4_chain_scale_mul,
                                 yolo_conv4_chain_scale_shift,
                                 NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN |
                                 NPU_CTRL_ELTWISE_EN | NPU_CTRL_ELT_SIGNED) ||
        !yolo_dma_out_to_ddr(ADD_OUT_DDR, OUT_BASE, YOLO_C2F_ADD_OUT_SPATIAL, 0u))
        return 0;
    return 1;
}

void usercode7(void)
{
    uint32_t i;
    uint32_t pos;
    uint32_t oc;
    uint32_t errors = 0u;

    print_str("YOLO C2F RESIDUAL-ADD (model.2/m.0/Add) CPU SMOKE\n");

    for (i = 0u; i < YOLO_C2F_ADD_OUT_SPATIAL; i++) {
        const uint32_t zero[4] = {0u, 0u, 0u, 0u};
        write_ddr_word128(ADD_OUT_DDR, i, zero);
    }

    if (!run_conv0_to_conv3()) {
        print_str("  prefix conv0->conv3 failed\n");
        errors++;
    }
    // Stage s1(glue) FIRST: it leaves R_SKIP resident in Out SRAM. conv4's
    // eltwise pass then reads R_SKIP while writing Out base 0 (disjoint).
    if (errors == 0u && !run_stage_s1_glue()) {
        print_str("  stage s1(glue) failed\n");
        errors++;
    }
    if (errors == 0u && !run_conv4_add()) {
        print_str("  conv4 eltwise add failed\n");
        errors++;
    }

    for (pos = 0u; pos < YOLO_C2F_ADD_OUT_SPATIAL && errors <= 16u; pos++) {
        for (oc = 0u; oc < YOLO_C2F_ADD_OC; oc++) {
            int32_t got = read_ddr_s8(ADD_OUT_DDR, pos, oc);
            int32_t expect = s8(yolo_c2f_add_expected_rtl[pos][oc]);
            if (abs_diff(got, expect) > YOLO_C2F_ADD_RTL_TOL) {
                errors++;
                print_str("  mismatch pos=");
                print_dec(pos);
                print_str(" oc=");
                print_dec(oc);
                print_str(" got=");
                print_dec(got);
                print_str(" exp=");
                print_dec(expect);
                print_str("\n");
                if (errors > 16u)
                    break;
            }
        }
    }

    if (errors == 0u) {
        print_str("YOLO C2F RESIDUAL-ADD CPU SMOKE PASS\n");
        return;
    }
    print_str("YOLO C2F RESIDUAL-ADD CPU SMOKE FAIL errors=");
    print_dec(errors);
    print_str("\n");
    __asm__ volatile ("ebreak");
}
