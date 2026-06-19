// CPU/NPU smoke for conv7 (model.4 cv1): 64->64, 1x1 pointwise.
// Input = conv6 (model.3) output strip, baked in as a fixture.
// Exercises IC=64 (4 ic-groups) OC=64 oc_single pointwise on the shared NPU.

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_conv7_from_conv6_m5x_data.h"
#include <stdint.h>

#define C7_ACT_DDR 0x40090000u
#define C7_WGT_DDR 0x40140000u
#define C7_OUT_DDR 0x40200000u

#define ACT_BASE 0u
#define WGT_BASE 0u
#define OUT_BASE 0u

static void write_ddr_word128(uint32_t byte_addr, uint32_t word_idx, const uint32_t lanes[4])
{
    volatile uint32_t *ptr = (volatile uint32_t *)(byte_addr + word_idx * 16u);
    ptr[0] = lanes[0]; ptr[1] = lanes[1]; ptr[2] = lanes[2]; ptr[3] = lanes[3];
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
    uint32_t i, pos, oc, errors = 0u;

    print_str("YOLO CONV7 (model.4 cv1, 64->64 1x1) CPU SMOKE\n");

    for (i = 0u; i < YOLO_CONV7_ACT_WORDS; i++)
        write_ddr_word128(C7_ACT_DDR, i, yolo_conv7_act_words[i]);
    for (i = 0u; i < YOLO_CONV7_WGT_WORDS; i++)
        write_ddr_word128(C7_WGT_DDR, i, yolo_conv7_wgt_words[i]);

    yolo_set_silu_requant(YOLO_CONV7_REQUANT_MUL, YOLO_CONV7_REQUANT_SHIFT,
                          YOLO_CONV7_REQUANT_ZP);
    if (!yolo_dma_ddr_to_act(C7_ACT_DDR, ACT_BASE, YOLO_CONV7_ACT_WORDS) ||
        !yolo_dma_ddr_to_wgt(C7_WGT_DDR, WGT_BASE, YOLO_CONV7_WGT_WORDS) ||
        !yolo_run_pw_conv1x1_qparams(ACT_BASE, WGT_BASE, OUT_BASE,
                                     YOLO_CONV7_IN_W, YOLO_CONV7_IN_H,
                                     YOLO_CONV7_IC, YOLO_CONV7_OC,
                                     yolo_conv7_bias_q, yolo_conv7_scale_mul,
                                     yolo_conv7_scale_shift,
                                     NPU_CTRL_OC_SINGLE | NPU_CTRL_SILU_EN |
                                     NPU_CTRL_SILU_REQUANT_EN) ||
        !yolo_dma_out_to_ddr(C7_OUT_DDR, OUT_BASE, YOLO_CONV7_OUT_WORDS, 0u)) {
        print_str("  conv7 run failed\n");
        errors++;
    }

    for (pos = 0u; pos < YOLO_CONV7_OUT_SPATIAL && errors <= 16u; pos++) {
        for (oc = 0u; oc < YOLO_CONV7_OC; oc++) {
            int32_t got = read_ddr_s8(C7_OUT_DDR, (oc >> 4) * YOLO_CONV7_OUT_SPATIAL + pos, oc & 15u);
            int32_t expect = s8(yolo_conv7_expected_rtl[pos][oc]);
            if (abs_diff(got, expect) > YOLO_CONV7_RTL_TOL) {
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
        print_str("YOLO CONV7 CPU SMOKE PASS\n");
        return;
    }
    print_str("YOLO CONV7 CPU SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
