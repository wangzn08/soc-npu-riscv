// CPU/NPU cooperative smoke test for YOLO channel-group slice at block level.

#include "firmware.h"
#include "yolo_ops.h"
#include <stdint.h>

#define SRC_DDR       0x4002A000u
#define DST_DDR       0x4002C000u
#define ACT_DST_BASE  768u

#define IN_W          4u
#define IN_H          3u
#define SPATIAL       (IN_W * IN_H)
#define SRC_GROUPS    4u
#define SLICE_GROUP   2u
#define SLICE_GROUPS  1u
#define DST_WORDS     (SPATIAL * SLICE_GROUPS)

static uint32_t rep8(uint32_t b)
{
    b &= 0xFFu;
    return b | (b << 8) | (b << 16) | (b << 24);
}

static uint32_t src_pattern(uint32_t group, uint32_t pos)
{
    return rep8(0x30u + group * 0x20u + pos);
}

static void write_ddr_word128(uint32_t byte_addr, uint32_t word_idx, uint32_t value)
{
    volatile uint32_t *ptr = (volatile uint32_t *)(byte_addr + word_idx * 16u);
    ptr[0] = value;
    ptr[1] = value;
    ptr[2] = value;
    ptr[3] = value;
}

static uint32_t read_ddr_word128_lane(uint32_t byte_addr, uint32_t word_idx, uint32_t lane)
{
    volatile uint32_t *ptr = (volatile uint32_t *)(byte_addr + word_idx * 16u);
    return ptr[lane];
}

void usercode7(void)
{
    uint32_t group;
    uint32_t pos;
    uint32_t lane;
    uint32_t errors = 0u;

    print_str("YOLO SLICE CPU SMOKE\n");

    for (group = 0u; group < SRC_GROUPS; group++) {
        for (pos = 0u; pos < SPATIAL; pos++)
            write_ddr_word128(SRC_DDR, group * SPATIAL + pos, src_pattern(group, pos));
    }
    for (pos = 0u; pos < DST_WORDS; pos++)
        write_ddr_word128(DST_DDR, pos, 0u);

    if (!yolo_slice_ddr_to_act(SRC_DDR, ACT_DST_BASE, SPATIAL, SLICE_GROUP, SLICE_GROUPS)) {
        print_str("  slice DMA load timeout\n");
        errors++;
    }

    if (errors == 0u && !yolo_dma_act_to_ddr(DST_DDR, ACT_DST_BASE, DST_WORDS)) {
        print_str("  slice DMA drain timeout\n");
        errors++;
    }

    for (pos = 0u; pos < SPATIAL; pos++) {
        uint32_t expect = src_pattern(SLICE_GROUP, pos);
        for (lane = 0u; lane < 4u; lane++) {
            uint32_t got = read_ddr_word128_lane(DST_DDR, pos, lane);
            if (got != expect) {
                errors++;
                print_str("  mismatch pos=");
                print_dec(pos);
                print_str(" lane=");
                print_dec(lane);
                print_str(" got=0x");
                print_hex(got, 8);
                print_str(" exp=0x");
                print_hex(expect, 8);
                print_str("\n");
            }
        }
    }

    if (errors == 0u) {
        print_str("YOLO SLICE CPU SMOKE PASS\n");
        return;
    }

    print_str("YOLO SLICE CPU SMOKE FAIL errors=");
    print_dec(errors);
    print_str("\n");
    __asm__ volatile ("ebreak");
}
