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

void usercode7(void)
{
    uint32_t i;
    uint32_t errors = 0u;
    const yolo_strip_plan_entry_t *strip0 = &yolo_conv0_strip_plan[0];
    const yolo_strip_plan_entry_t *strip1 = &yolo_conv0_strip_plan[1];

    print_str("YOLO CONV0 STRIP PLAN CPU SMOKE\n");

    for (i = 0u; i < YOLO_CONV0_STRIP_ACT_WORDS; i++)
        write_ddr_word128(ACT_DDR, i, yolo_conv0_strip_act_words[i]);
    for (i = 0u; i < YOLO_CONV0_STRIP_WGT_WORDS; i++)
        write_ddr_word128(WGT_DDR, i, yolo_conv0_strip_wgt_words[i]);

    if (!yolo_run_conv0_strip_from_plan(strip0,
                                        ACT_DDR,
                                        WGT_DDR,
                                        OUT_DDR,
                                        ACT_BASE,
                                        WGT_BASE,
                                        OUT_BASE,
                                        yolo_conv0_strip_bias_q,
                                        yolo_conv0_strip_scale_mul,
                                        yolo_conv0_strip_scale_shift,
                                        YOLO_CONV0_STRIP_REQUANT_MUL,
                                        YOLO_CONV0_STRIP_REQUANT_SHIFT,
                                        YOLO_CONV0_STRIP_REQUANT_ZP)) {
        print_str("  strip0 descriptor run failed\n");
        errors++;
    }

    if (!yolo_run_conv0_strip_from_plan(strip1,
                                        ACT_DDR,
                                        WGT_DDR,
                                        OUT_DDR + YOLO_CONV0_STRIP_OUT_WORDS * 16u,
                                        ACT_BASE,
                                        WGT_BASE,
                                        OUT_BASE,
                                        yolo_conv0_strip_bias_q,
                                        yolo_conv0_strip_scale_mul,
                                        yolo_conv0_strip_scale_shift,
                                        YOLO_CONV0_STRIP_REQUANT_MUL,
                                        YOLO_CONV0_STRIP_REQUANT_SHIFT,
                                        YOLO_CONV0_STRIP_REQUANT_ZP)) {
        print_str("  strip1 descriptor run failed\n");
        errors++;
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
