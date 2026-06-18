// CPU/NPU smoke for a YOLO concat-channel pointwise block.
// Uses conv5: OC32 x IC48 x 1x1, representing a post-concat 3-IC-group input.

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_plan.h"
#include "yolo_concat_channel_real_data.h"
#include <stdint.h>

#define ACT_DDR 0x40082000u
#define WGT_DDR 0x40086000u
#define OUT_DDR 0x4008C000u

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
    uint32_t pos;
    uint32_t oc;
    uint32_t errors = 0u;

    print_str("YOLO CONCAT-CHANNEL PWCONV CREF CPU SMOKE\n");

    for (i = 0u; i < YOLO_CONCAT_CH_ACT_WORDS; i++)
        write_ddr_word128(ACT_DDR, i, yolo_concat_ch_act_words[i]);
    for (i = 0u; i < YOLO_CONCAT_CH_WGT_WORDS; i++)
        write_ddr_word128(WGT_DDR, i, yolo_concat_ch_wgt_words[i]);
    for (i = 0u; i < YOLO_CONCAT_CH_OUT_WORDS; i++) {
        const uint32_t zero[4] = {0u, 0u, 0u, 0u};
        write_ddr_word128(OUT_DDR, i, zero);
    }

    {
        const yolo_conv_desc_t desc = {
            ACT_DDR,
            WGT_DDR,
            OUT_DDR,
            ACT_BASE,
            WGT_BASE,
            OUT_BASE,
            0u,
            YOLO_CONCAT_CH_IN_W,
            YOLO_CONCAT_CH_IN_H,
            YOLO_CONCAT_CH_IC,
            YOLO_CONCAT_CH_OC,
            1u,
            1u,
            1u,
            0u,
            (uint32_t)YOLO_CONCAT_CH_IN_ZP,
            YOLO_CONCAT_CH_ACT_WORDS,
            YOLO_CONCAT_CH_WGT_WORDS,
            YOLO_CONCAT_CH_OUT_WORDS,
            yolo_concat_ch_bias_q,
            yolo_concat_ch_scale_mul,
            yolo_concat_ch_scale_shift,
            YOLO_CONCAT_CH_REQUANT_MUL,
            YOLO_CONCAT_CH_REQUANT_SHIFT,
            YOLO_CONCAT_CH_REQUANT_ZP,
            NPU_CTRL_OC_SINGLE | NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN
        };

        if (!yolo_run_conv_desc(&desc)) {
            print_str("  concat-channel pwconv path failed\n");
            errors++;
        }
    }

    for (pos = 0u; pos < YOLO_CONCAT_CH_OUT_SPATIAL; pos++) {
        for (oc = 0u; oc < YOLO_CONCAT_CH_OC; oc++) {
            uint32_t group = oc >> 4;
            uint32_t lane = oc & 15u;
            uint32_t word_idx = group * YOLO_CONCAT_CH_OUT_SPATIAL + pos;
            int32_t got = read_ddr_s8(OUT_DDR, word_idx, lane);
            int32_t expect = s8(yolo_concat_ch_expected_cref[pos][oc]);
            uint32_t diff = abs_diff(got, expect);
            if (diff > YOLO_CONCAT_CH_CREF_TOL) {
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
        print_str("YOLO CONCAT-CHANNEL PWCONV CREF CPU SMOKE PASS\n");
        return;
    }

    print_str("YOLO CONCAT-CHANNEL PWCONV CREF CPU SMOKE FAIL errors=");
    print_dec(errors);
    print_str("\n");
    __asm__ volatile ("ebreak");
}
