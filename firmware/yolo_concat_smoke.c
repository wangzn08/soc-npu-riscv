// CPU/NPU cooperative smoke test for YOLO channel concat at block level.
// Two DDR-resident tile-major tensors are DMA-loaded into one Act-SRAM
// destination layout:
//   dst groups = [src0 groups..., src1 groups...]

#include "firmware.h"
#include "yolo_ops.h"
#include <stdint.h>

#define SRC0_DDR      0x40024000u
#define SRC1_DDR      0x40026000u
#define DST_DDR       0x40028000u
#define ACT_DST_BASE  512u

#define IN_W          3u
#define IN_H          2u
#define SPATIAL       (IN_W * IN_H)
#define SRC0_GROUPS   2u
#define SRC1_GROUPS   3u
#define DST_GROUPS    (SRC0_GROUPS + SRC1_GROUPS)
#define DST_WORDS     (DST_GROUPS * SPATIAL)

static uint32_t rep8(uint32_t b)
{
    b &= 0xFFu;
    return b | (b << 8) | (b << 16) | (b << 24);
}

static uint32_t src0_pattern(uint32_t group, uint32_t pos)
{
    return rep8(0x20u + group * 0x10u + pos);
}

static uint32_t src1_pattern(uint32_t group, uint32_t pos)
{
    return rep8(0x80u + group * 0x10u + pos);
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

static uint32_t expected_concat(uint32_t dst_group, uint32_t pos)
{
    if (dst_group < SRC0_GROUPS)
        return src0_pattern(dst_group, pos);
    return src1_pattern(dst_group - SRC0_GROUPS, pos);
}

void usercode7(void)
{
    uint32_t group;
    uint32_t pos;
    uint32_t lane;
    uint32_t errors = 0u;

    print_str("YOLO CONCAT CPU SMOKE\n");

    for (group = 0u; group < SRC0_GROUPS; group++) {
        for (pos = 0u; pos < SPATIAL; pos++)
            write_ddr_word128(SRC0_DDR, group * SPATIAL + pos, src0_pattern(group, pos));
    }
    for (group = 0u; group < SRC1_GROUPS; group++) {
        for (pos = 0u; pos < SPATIAL; pos++)
            write_ddr_word128(SRC1_DDR, group * SPATIAL + pos, src1_pattern(group, pos));
    }
    for (pos = 0u; pos < DST_WORDS; pos++)
        write_ddr_word128(DST_DDR, pos, 0u);

    if (!yolo_concat2_ddr_to_act(SRC0_DDR, SRC1_DDR, ACT_DST_BASE,
                                 SPATIAL, SRC0_GROUPS, SRC1_GROUPS)) {
        print_str("  concat DMA load timeout\n");
        errors++;
    }

    if (errors == 0u && !yolo_dma_act_to_ddr(DST_DDR, ACT_DST_BASE, DST_WORDS)) {
        print_str("  concat DMA drain timeout\n");
        errors++;
    }

    for (group = 0u; group < DST_GROUPS; group++) {
        for (pos = 0u; pos < SPATIAL; pos++) {
            uint32_t dst_idx = group * SPATIAL + pos;
            uint32_t expect = expected_concat(group, pos);
            for (lane = 0u; lane < 4u; lane++) {
                uint32_t got = read_ddr_word128_lane(DST_DDR, dst_idx, lane);
                if (got != expect) {
                    errors++;
                    print_str("  mismatch group=");
                    print_dec(group);
                    print_str(" pos=");
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
    }

    if (errors == 0u) {
        print_str("YOLO CONCAT CPU SMOKE PASS\n");
        return;
    }

    print_str("YOLO CONCAT CPU SMOKE FAIL errors=");
    print_dec(errors);
    print_str("\n");
    __asm__ volatile ("ebreak");
}
