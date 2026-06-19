// CPU/NPU smoke for running conv0 strips from generated strip-plan metadata.

#include "firmware.h"
#include "yolo_block_plan.h"
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
    uint32_t s;
    uint32_t pos;
    uint32_t oc;
    uint32_t errors = 0u;
    const yolo_block_plan_entry_t *conv0 = &yolo_block_plan[0];

    print_str("YOLO CONV0 STRIP PLAN CPU SMOKE\n");

    if (conv0->idx != 0u || conv0->strip_count < YOLO_CONV0_STRIP_TEST_COUNT ||
        conv0->strip_offset + YOLO_CONV0_STRIP_TEST_COUNT > YOLO_STRIP_PLAN_COUNT) {
        print_str("  bad conv0 generic strip lookup\n");
        errors++;
    }

    for (i = 0u; i < YOLO_CONV0_STRIP_ACT_WORDS; i++)
        write_ddr_word128(ACT_DDR, i, yolo_conv0_strip_act_words[i]);
    for (i = 0u; i < YOLO_CONV0_STRIP_WGT_WORDS; i++)
        write_ddr_word128(WGT_DDR, i, yolo_conv0_strip_wgt_words[i]);
    for (i = 0u; i < YOLO_CONV0_STRIP_OUT_WORDS_PER_STRIP * 2u; i++) {
        const uint32_t zero[4] = {0u, 0u, 0u, 0u};
        write_ddr_word128(OUT_DDR, i, zero);
    }

    for (s = 0u; s < YOLO_CONV0_STRIP_TEST_COUNT; s++) {
        const yolo_strip_plan_entry_t *strip = &yolo_strip_plan[conv0->strip_offset + s];
        if (!yolo_run_conv0_strip_from_plan(strip,
                                            ACT_DDR,
                                            WGT_DDR,
                                            OUT_DDR + s * YOLO_CONV0_STRIP_OUT_WORDS_PER_STRIP * 16u,
                                            ACT_BASE,
                                            WGT_BASE,
                                            OUT_BASE,
                                            yolo_conv0_strip_bias_q,
                                            yolo_conv0_strip_scale_mul,
                                            yolo_conv0_strip_scale_shift,
                                            YOLO_CONV0_STRIP_REQUANT_MUL,
                                            YOLO_CONV0_STRIP_REQUANT_SHIFT,
                                            YOLO_CONV0_STRIP_REQUANT_ZP)) {
            print_str("  strip descriptor run failed s=");
            print_dec(s);
            print_str("\n");
            errors++;
            break;
        }
    }

    for (pos = 0u; pos < YOLO_CONV0_STRIP_OUT_WORDS_PER_STRIP * YOLO_CONV0_STRIP_TEST_COUNT; pos++) {
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
        print_str("YOLO CONV0 STRIP PLAN CPU SMOKE PASS\n");
        return;
    }

    print_str("YOLO CONV0 STRIP PLAN CPU SMOKE FAIL errors=");
    print_dec(errors);
    print_str("\n");
    __asm__ volatile ("ebreak");
}
