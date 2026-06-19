// CPU/NPU smoke closing the first C2f block (model.2):
//   conv0->conv1->conv2->{s0,s1}; s1->conv3->conv4; add=s1+conv4 (M5u);
//   concat(s0, s1, add) -> conv5 (cv2, 1x1, 48->32) on the shared NPU.
//
// The three concat pieces are integer-requantized on the CPU to conv5's
// in_scale/zp (= /model.2/Concat), concatenated tile-major in DDR, then conv5
// runs as a pointwise oc_single pass.

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
#include "yolo_c2f_close_m5v_data.h"
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
#define CONCAT_DDR  0x404C0000u
#define C5_WGT_DDR  0x40540000u
#define C5_OUT_DDR  0x40548000u

#define ACT_BASE 0u
#define WGT_BASE 0u
#define OUT_BASE 0u
#define R_SKIP   4096u

#define SPATIAL YOLO_C2F_CLOSE_SPATIAL

static void write_ddr_word128(uint32_t byte_addr, uint32_t word_idx, const uint32_t lanes[4])
{
    volatile uint32_t *ptr = (volatile uint32_t *)(byte_addr + word_idx * 16u);
    ptr[0] = lanes[0]; ptr[1] = lanes[1]; ptr[2] = lanes[2]; ptr[3] = lanes[3];
}

static void read_ddr_word128(uint32_t byte_addr, uint32_t word_idx, uint32_t lanes[4])
{
    volatile uint32_t *ptr = (volatile uint32_t *)(byte_addr + word_idx * 16u);
    lanes[0] = ptr[0]; lanes[1] = ptr[1]; lanes[2] = ptr[2]; lanes[3] = ptr[3];
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

static int32_t clamp_s8(int32_t v)
{
    if (v > 127) return 127;
    if (v < -128) return -128;
    return v;
}

static int run_conv0_to_conv3(void)
{
    uint32_t i, s;
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
        if (!yolo_run_conv0_strip_from_plan(strip, C0_ACT_DDR, C0_WGT_DDR,
                C0_OUT_DDR + s * YOLO_CONV0_STRIP_OUT_WORDS_PER_STRIP * 16u,
                ACT_BASE, WGT_BASE, OUT_BASE,
                yolo_conv0_strip_bias_q, yolo_conv0_strip_scale_mul,
                yolo_conv0_strip_scale_shift, YOLO_CONV0_STRIP_REQUANT_MUL,
                YOLO_CONV0_STRIP_REQUANT_SHIFT, YOLO_CONV0_STRIP_REQUANT_ZP))
            return 0;
    }

    yolo_set_pad_value(-127);
    yolo_set_silu_requant(YOLO_CONV1_CHAIN_REQUANT_MUL, YOLO_CONV1_CHAIN_REQUANT_SHIFT,
                          YOLO_CONV1_CHAIN_REQUANT_ZP);
    if (!yolo_dma_ddr_to_act(C0_OUT_DDR, ACT_BASE, YOLO_CONV1_CHAIN_ACT_WORDS) ||
        !yolo_dma_ddr_to_wgt(C1_WGT_DDR, WGT_BASE, YOLO_CONV1_CHAIN_WGT_WORDS) ||
        !yolo_run_conv2d_qparams_pads(ACT_BASE, WGT_BASE, OUT_BASE,
                YOLO_CONV1_CHAIN_IN_W, YOLO_CONV1_CHAIN_IN_H, YOLO_CONV1_CHAIN_IC,
                YOLO_CONV1_CHAIN_OC, YOLO_CONV1_CHAIN_KH, YOLO_CONV1_CHAIN_KW,
                YOLO_CONV1_CHAIN_STRIDE, YOLO_CONV1_CHAIN_PAD, YOLO_CONV1_CHAIN_PAD,
                yolo_conv1_chain_bias_q, yolo_conv1_chain_scale_mul,
                yolo_conv1_chain_scale_shift,
                NPU_CTRL_OC_SINGLE | NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN) ||
        !yolo_dma_out_to_ddr(C1_OUT_DDR, OUT_BASE, YOLO_CONV1_CHAIN_OUT_WORDS, 0u))
        return 0;

    yolo_set_pad_value(YOLO_CONV2_CHAIN_PAD_VALUE);
    yolo_set_silu_requant(YOLO_CONV2_CHAIN_REQUANT_MUL, YOLO_CONV2_CHAIN_REQUANT_SHIFT,
                          YOLO_CONV2_CHAIN_REQUANT_ZP);
    if (!yolo_dma_ddr_to_act(C1_OUT_DDR, ACT_BASE, YOLO_CONV2_CHAIN_ACT_WORDS) ||
        !yolo_dma_ddr_to_wgt(C2_WGT_DDR, WGT_BASE, YOLO_CONV2_CHAIN_WGT_WORDS) ||
        !yolo_run_pw_conv1x1_qparams(ACT_BASE, WGT_BASE, OUT_BASE,
                YOLO_CONV2_CHAIN_IN_W, YOLO_CONV2_CHAIN_IN_H, YOLO_CONV2_CHAIN_IC,
                YOLO_CONV2_CHAIN_OC, yolo_conv2_chain_bias_q,
                yolo_conv2_chain_scale_mul, yolo_conv2_chain_scale_shift,
                NPU_CTRL_OC_SINGLE | NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN) ||
        !yolo_dma_out_to_ddr(C2_OUT_DDR, OUT_BASE, YOLO_CONV2_CHAIN_OUT_WORDS, 0u))
        return 0;

    yolo_set_pad_value(YOLO_CONV3_CHAIN_PAD_VALUE);
    yolo_set_silu_requant(YOLO_CONV3_CHAIN_REQUANT_MUL, YOLO_CONV3_CHAIN_REQUANT_SHIFT,
                          YOLO_CONV3_CHAIN_REQUANT_ZP);
    return yolo_slice_ddr_to_act(C2_OUT_DDR, ACT_BASE, YOLO_CONV3_CHAIN_IN_SPATIAL,
                                 YOLO_CONV3_CHAIN_SRC_GROUP, 1u) &&
           yolo_dma_ddr_to_wgt(C3_WGT_DDR, WGT_BASE, YOLO_CONV3_CHAIN_WGT_WORDS) &&
           yolo_run_conv2d_qparams(ACT_BASE, WGT_BASE, OUT_BASE,
                YOLO_CONV3_CHAIN_IN_W, YOLO_CONV3_CHAIN_IN_H, YOLO_CONV3_CHAIN_IC,
                YOLO_CONV3_CHAIN_OC, YOLO_CONV3_CHAIN_KH, YOLO_CONV3_CHAIN_KW,
                YOLO_CONV3_CHAIN_STRIDE, YOLO_CONV3_CHAIN_PAD,
                yolo_conv3_chain_bias_q, yolo_conv3_chain_scale_mul,
                yolo_conv3_chain_scale_shift,
                NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN) &&
           yolo_dma_out_to_ddr(C3_OUT_DDR, OUT_BASE, YOLO_CONV3_CHAIN_OUT_WORDS, 0u);
}

static int run_stage_s1_glue(void)
{
    uint32_t i;
    for (i = 0u; i < YOLO_C2F_STAGE_WGT_WORDS; i++)
        write_ddr_word128(STAGE_WGT_DDR, i, yolo_c2f_stage_wgt_words[i]);
    yolo_set_silu_requant(YOLO_C2F_ADD_GLUE_REQUANT_MUL, YOLO_C2F_ADD_GLUE_REQUANT_SHIFT,
                          YOLO_C2F_ADD_GLUE_ZP);
    return yolo_dma_ddr_to_act(C1_OUT_DDR, ACT_BASE, YOLO_C2F_STAGE_ACT_WORDS) &&
           yolo_dma_ddr_to_wgt(STAGE_WGT_DDR, WGT_BASE, YOLO_C2F_STAGE_WGT_WORDS) &&
           yolo_run_pw_conv1x1_qparams(ACT_BASE, WGT_BASE, R_SKIP,
                YOLO_C2F_STAGE_IN_W, YOLO_C2F_STAGE_IN_H, YOLO_C2F_STAGE_IC,
                YOLO_C2F_STAGE_OC, yolo_c2f_stage_bias_q, yolo_c2f_stage_scale_mul,
                yolo_c2f_stage_scale_shift,
                NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN);
}

static int run_conv4_add(void)
{
    uint32_t i;
    for (i = 0u; i < YOLO_CONV4_CHAIN_WGT_WORDS; i++)
        write_ddr_word128(C4_WGT_DDR, i, yolo_conv4_chain_wgt_words[i]);
    yolo_set_pad_value(YOLO_CONV4_CHAIN_PAD_VALUE);
    yolo_set_silu_requant(YOLO_C2F_ADD_GLUE_REQUANT_MUL, YOLO_C2F_ADD_GLUE_REQUANT_SHIFT,
                          YOLO_C2F_ADD_GLUE_ZP);
    yolo_set_eltwise(YOLO_C2F_ADD_GLUE_ZP, R_SKIP);
    return yolo_dma_ddr_to_act(C3_OUT_DDR, ACT_BASE, YOLO_CONV4_CHAIN_ACT_WORDS) &&
           yolo_dma_ddr_to_wgt(C4_WGT_DDR, WGT_BASE, YOLO_CONV4_CHAIN_WGT_WORDS) &&
           yolo_run_conv2d_qparams(ACT_BASE, WGT_BASE, OUT_BASE,
                YOLO_CONV4_CHAIN_IN_W, YOLO_CONV4_CHAIN_IN_H, YOLO_CONV4_CHAIN_IC,
                YOLO_CONV4_CHAIN_OC, YOLO_CONV4_CHAIN_KH, YOLO_CONV4_CHAIN_KW,
                YOLO_CONV4_CHAIN_STRIDE, YOLO_CONV4_CHAIN_PAD,
                yolo_conv4_chain_bias_q, yolo_conv4_chain_scale_mul,
                yolo_conv4_chain_scale_shift,
                NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN |
                NPU_CTRL_ELTWISE_EN | NPU_CTRL_ELT_SIGNED) &&
           yolo_dma_out_to_ddr(ADD_OUT_DDR, OUT_BASE, YOLO_C2F_ADD_OUT_SPATIAL, 0u);
}

// CPU integer requant of one 128-bit word (16 lanes) to the concat scale.
static void requant_word(uint32_t src_addr, uint32_t src_word,
                         uint32_t mul, int32_t in_zp,
                         uint32_t dst_addr, uint32_t dst_word)
{
    uint32_t in_lanes[4];
    uint32_t out_lanes[4] = {0u, 0u, 0u, 0u};
    uint32_t k;
    read_ddr_word128(src_addr, src_word, in_lanes);
    for (k = 0u; k < 16u; k++) {
        int32_t q = s8(in_lanes[k >> 2] >> ((k & 3u) * 8u));
        int32_t v = (((q - in_zp) * (int32_t)mul) >> YOLO_C2F_CAT_REQ_SHIFT) + YOLO_C2F_CAT_ZP;
        uint32_t b = (uint32_t)(clamp_s8(v) & 0xFF);
        out_lanes[k >> 2] |= b << ((k & 3u) * 8u);
    }
    write_ddr_word128(dst_addr, dst_word, out_lanes);
}

// Build concat[48ch] tile-major in DDR: group0=s0, group1=s1, group2=add.
static void build_concat(void)
{
    uint32_t pos;
    for (pos = 0u; pos < SPATIAL; pos++) {
        // s0 = C2_OUT group0 (word pos); s1 = group1 (word SPATIAL+pos)
        requant_word(C2_OUT_DDR, pos, YOLO_C2F_CAT_MUL_S0S1,
                     YOLO_C2F_CAT_INZP_S0S1, CONCAT_DDR, pos);
        requant_word(C2_OUT_DDR, SPATIAL + pos, YOLO_C2F_CAT_MUL_S0S1,
                     YOLO_C2F_CAT_INZP_S0S1, CONCAT_DDR, SPATIAL + pos);
        requant_word(ADD_OUT_DDR, pos, YOLO_C2F_CAT_MUL_ADD,
                     YOLO_C2F_CAT_INZP_ADD, CONCAT_DDR, 2u * SPATIAL + pos);
    }
}

static int run_conv5(void)
{
    uint32_t i;
    for (i = 0u; i < YOLO_C2F_CONV5_WGT_WORDS; i++)
        write_ddr_word128(C5_WGT_DDR, i, yolo_c2f_conv5_wgt_words[i]);
    yolo_set_silu_requant(YOLO_C2F_CONV5_REQUANT_MUL, YOLO_C2F_CONV5_REQUANT_SHIFT,
                          YOLO_C2F_CONV5_REQUANT_ZP);
    return yolo_dma_ddr_to_act(CONCAT_DDR, ACT_BASE, SPATIAL * 3u) &&
           yolo_dma_ddr_to_wgt(C5_WGT_DDR, WGT_BASE, YOLO_C2F_CONV5_WGT_WORDS) &&
           yolo_run_pw_conv1x1_qparams(ACT_BASE, WGT_BASE, OUT_BASE,
                YOLO_C2F_CLOSE_IN_W, YOLO_C2F_CLOSE_IN_H, YOLO_C2F_CLOSE_IC,
                YOLO_C2F_CLOSE_OC, yolo_c2f_conv5_bias_q, yolo_c2f_conv5_scale_mul,
                yolo_c2f_conv5_scale_shift,
                NPU_CTRL_OC_SINGLE | NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN) &&
           yolo_dma_out_to_ddr(C5_OUT_DDR, OUT_BASE, SPATIAL * 2u, 0u);
}

void usercode7(void)
{
    uint32_t pos, oc, errors = 0u;

    print_str("YOLO C2F CLOSE (model.2 concat->conv5) CPU SMOKE\n");

    if (!run_conv0_to_conv3())            { print_str("  prefix failed\n"); errors++; }
    if (errors == 0u && !run_stage_s1_glue()) { print_str("  stage failed\n"); errors++; }
    if (errors == 0u && !run_conv4_add()) { print_str("  conv4 add failed\n"); errors++; }
    if (errors == 0u) build_concat();
    if (errors == 0u && !run_conv5())     { print_str("  conv5 failed\n"); errors++; }

    for (pos = 0u; pos < SPATIAL && errors <= 16u; pos++) {
        for (oc = 0u; oc < YOLO_C2F_CLOSE_OC; oc++) {
            // conv5 output tile-major: word = (oc/16)*SPATIAL + pos, byte = oc%16
            int32_t got = read_ddr_s8(C5_OUT_DDR, (oc >> 4) * SPATIAL + pos, oc & 15u);
            int32_t expect = s8(yolo_c2f_close_expected_rtl[pos][oc]);
            if (abs_diff(got, expect) > YOLO_C2F_CLOSE_RTL_TOL) {
                errors++;
                print_str("  mismatch pos="); print_dec(pos);
                print_str(" oc="); print_dec(oc);
                print_str(" got="); print_dec(got);
                print_str(" exp="); print_dec(expect); print_str("\n");
                if (errors > 16u) break;
            }
        }
    }

    if (errors == 0u) {
        print_str("YOLO C2F CLOSE CPU SMOKE PASS\n");
        return;
    }
    print_str("YOLO C2F CLOSE CPU SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
