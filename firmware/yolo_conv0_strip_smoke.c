// CPU/NPU smoke for a full-width YOLO conv0 top strip.
// Shape: 640x16x3 -> 320x8x16, stride=2, pad=1.

#include "firmware.h"
#include "yolo_plan.h"
#include "yolo_conv0_strip_real_data.h"
#include <stdint.h>

#define ACT_DDR 0x40090000u
#define WGT_DDR 0x400C0000u
#define OUT_DDR 0x400C4000u

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

    print_str("YOLO CONV0 FULL-WIDTH STRIP RTLINT CPU SMOKE\n");

    for (i = 0u; i < YOLO_CONV0_STRIP_ACT_WORDS; i++)
        write_ddr_word128(ACT_DDR, i, yolo_conv0_strip_act_words[i]);
    for (i = 0u; i < YOLO_CONV0_STRIP_WGT_WORDS; i++)
        write_ddr_word128(WGT_DDR, i, yolo_conv0_strip_wgt_words[i]);
    for (i = 0u; i < YOLO_CONV0_STRIP_OUT_WORDS; i++) {
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
            YOLO_CONV0_STRIP_IN_W,
            YOLO_CONV0_STRIP_IN_H,
            YOLO_CONV0_STRIP_IC,
            YOLO_CONV0_STRIP_OC,
            YOLO_CONV0_STRIP_KH,
            YOLO_CONV0_STRIP_KW,
            YOLO_CONV0_STRIP_STRIDE,
            YOLO_CONV0_STRIP_PAD,
            (uint32_t)YOLO_CONV0_STRIP_IN_ZP,
            YOLO_CONV0_STRIP_ACT_WORDS,
            YOLO_CONV0_STRIP_WGT_WORDS,
            YOLO_CONV0_STRIP_OUT_WORDS,
            yolo_conv0_strip_bias_q,
            yolo_conv0_strip_scale_mul,
            yolo_conv0_strip_scale_shift,
            YOLO_CONV0_STRIP_REQUANT_MUL,
            YOLO_CONV0_STRIP_REQUANT_SHIFT,
            YOLO_CONV0_STRIP_REQUANT_ZP,
            NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN
        };

        if (!yolo_run_conv_desc(&desc)) {
            print_str("  conv0 strip path failed\n");
            errors++;
        }
    }

    for (pos = 0u; pos < YOLO_CONV0_STRIP_OUT_WORDS; pos++) {
        for (oc = 0u; oc < YOLO_CONV0_STRIP_OC; oc++) {
            int32_t got = read_ddr_s8(OUT_DDR, pos, oc);
            int32_t expect = s8(yolo_conv0_strip_expected_rtl[pos][oc]);
            uint32_t diff = abs_diff(got, expect);
            if (diff > YOLO_CONV0_STRIP_RTL_TOL) {
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
        if (errors > 16u)
            break;
    }

    if (errors == 0u) {
        print_str("YOLO CONV0 FULL-WIDTH STRIP RTLINT CPU SMOKE PASS\n");
        return;
    }

    print_str("YOLO CONV0 FULL-WIDTH STRIP RTLINT CPU SMOKE FAIL errors=");
    print_dec(errors);
    print_str("\n");
    __asm__ volatile ("ebreak");
}
