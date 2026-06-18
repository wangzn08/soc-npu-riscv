// CPU smoke for the generated YOLO block-plan table.

#include "firmware.h"
#include "yolo_block_plan.h"
#include <stdint.h>

static int check_flag(uint32_t flags, uint32_t flag)
{
    return (flags & flag) != 0u;
}

void usercode7(void)
{
    uint32_t i;
    uint32_t errors = 0u;
    const yolo_block_plan_entry_t *conv0 = &yolo_block_plan[0];
    const yolo_block_plan_entry_t *conv5 = &yolo_block_plan[5];

    print_str("YOLO BLOCK PLAN CPU SMOKE\n");

    if (YOLO_BLOCK_PLAN_COUNT != 63u) {
        print_str("  bad plan count\n");
        errors++;
    }
    if (conv0->idx != 0u || conv0->strip_rows != 8u || conv0->strip_count != 40u ||
        !check_flag(conv0->flags, YOLO_PLAN_FLAG_HW_PAD)) {
        print_str("  bad conv0 strip/pad plan\n");
        errors++;
    }
    if (conv5->idx != 5u ||
        conv5->in_w != 160u || conv5->in_h != 160u || conv5->in_c != 48u ||
        conv5->out_w != 160u || conv5->out_h != 160u || conv5->out_c != 32u ||
        conv5->input_words != ((160u * 160u * 48u) / 16u) ||
        conv5->output_words != ((160u * 160u * 32u) / 16u) ||
        conv5->weight_words != ((32u * 48u) / 16u) ||
        !check_flag(conv5->flags, YOLO_PLAN_FLAG_PW_EN) ||
        !check_flag(conv5->flags, YOLO_PLAN_FLAG_OC_SINGLE) ||
        !check_flag(conv5->flags, YOLO_PLAN_FLAG_SILU_REQUANT)) {
        print_str("  bad conv5 concat-channel plan\n");
        errors++;
    }

    for (i = 1u; i < YOLO_BLOCK_PLAN_COUNT; i++) {
        uint32_t prev_end = yolo_block_plan[i - 1u].output_ddr +
                            yolo_block_plan[i - 1u].output_words * 16u;
        if (prev_end > yolo_block_plan[i].output_ddr) {
            print_str("  output DDR overlap at plan index ");
            print_dec(i);
            print_str("\n");
            errors++;
            break;
        }
    }

    if (errors == 0u) {
        print_str("YOLO BLOCK PLAN CPU SMOKE PASS\n");
        return;
    }

    print_str("YOLO BLOCK PLAN CPU SMOKE FAIL errors=");
    print_dec(errors);
    print_str("\n");
    __asm__ volatile ("ebreak");
}
