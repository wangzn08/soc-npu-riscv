// CPU/NPU cooperative smoke test for the shared 2x upsample engine.
// The CPU writes source data to DDR, uses the NPU DMA to move it into Act SRAM,
// triggers the on-chip upsample engine, DMA-drains the result back to DDR, and
// compares the output in software.

#include "firmware.h"
#include <stdint.h>

#define SRC_DDR       0x40020000u
#define DST_DDR       0x40022000u
#define ACT_SRC_BASE  4u
#define ACT_DST_BASE  64u
#define IN_W          3u
#define IN_H          2u
#define IC_GROUPS     2u
#define SRC_WORDS     (IN_W * IN_H * IC_GROUPS)
#define DST_WORDS     ((IN_W * 2u) * (IN_H * 2u) * IC_GROUPS)

#define SMOKE_TIMEOUT 200000u

static inline void npu_wr(uint32_t addr, uint32_t data)
{
    *(volatile uint32_t *)addr = data;
}

static inline uint32_t npu_rd(uint32_t addr)
{
    return *(volatile uint32_t *)addr;
}

static uint32_t rep8(uint32_t b)
{
    b &= 0xFFu;
    return b | (b << 8) | (b << 16) | (b << 24);
}

static uint32_t src_pattern(uint32_t group, uint32_t y, uint32_t x)
{
    return rep8(0x10u + group * 0x40u + y * 0x08u + x);
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

static int wait_dma_status(uint32_t mask)
{
    uint32_t timeout = SMOKE_TIMEOUT;
    while (timeout-- != 0u) {
        if ((npu_rd(NPU_DMA_STATUS) & mask) != 0u)
            return 1;
    }
    return 0;
}

static int dma_ddr_to_act(uint32_t ddr_addr, uint32_t act_base, uint32_t words)
{
    npu_wr(NPU_DMA_PING_SEL, 0u);
    npu_wr(NPU_DMA_SRAM_SEL, 0u);
    npu_wr(NPU_DMA_RD_DDR_ADDR, ddr_addr);
    npu_wr(NPU_DMA_RD_LEN, words - 1u);
    npu_wr(NPU_DMA_RD_SRAM_BASE, act_base);
    npu_wr(NPU_DMA_RD_TRIG, 1u);
    return wait_dma_status(NPU_DMA_STATUS_RD_DONE);
}

static int dma_act_to_ddr(uint32_t ddr_addr, uint32_t act_base, uint32_t words)
{
    npu_wr(NPU_DMA_PING_SEL, 0u);
    npu_wr(NPU_DMA_PATH_CTL, 0x2u);  // DMA write source = Act SRAM
    npu_wr(NPU_DMA_WR_DDR_ADDR, ddr_addr);
    npu_wr(NPU_DMA_WR_LEN, words - 1u);
    npu_wr(NPU_DMA_WR_SRAM_BASE, act_base);
    npu_wr(NPU_DMA_WR_TRIG, 1u);
    return wait_dma_status(NPU_DMA_STATUS_WR_DONE);
}

static int run_upsample(void)
{
    npu_wr(NPU_DMA_PING_SEL, 0u);
    npu_wr(NPU_DMA_RD_SRAM_BASE, ACT_SRC_BASE);
    npu_wr(NPU_DMA_WR_SRAM_BASE, ACT_DST_BASE);
    npu_wr(NPU_UPSAMPLE_CFG0, (IN_H << 16) | IN_W);
    npu_wr(NPU_UPSAMPLE_CFG1, IC_GROUPS);
    npu_wr(NPU_DMA_UPSAMPLE_TRIG, 1u);
    return wait_dma_status(NPU_DMA_STATUS_UPSAMPLE_DONE);
}

void usercode7(void)
{
    uint32_t group;
    uint32_t y;
    uint32_t x;
    uint32_t dy;
    uint32_t dx;
    uint32_t lane;
    uint32_t errors = 0u;

    print_str("UPSAMPLE2X CPU SMOKE\n");

    for (uint32_t word = 0u; word < DST_WORDS; word++)
        write_ddr_word128(DST_DDR, word, 0u);

    for (group = 0u; group < IC_GROUPS; group++) {
        for (y = 0u; y < IN_H; y++) {
            for (x = 0u; x < IN_W; x++) {
                uint32_t src_idx = group * (IN_W * IN_H) + y * IN_W + x;
                write_ddr_word128(SRC_DDR, src_idx, src_pattern(group, y, x));
            }
        }
    }

    if (!dma_ddr_to_act(SRC_DDR, ACT_SRC_BASE, SRC_WORDS)) {
        print_str("  DMA DDR->ACT timeout\n");
        errors++;
    }

    if (errors == 0u && !run_upsample()) {
        print_str("  UPSAMPLE timeout\n");
        errors++;
    }

    if (errors == 0u && !dma_act_to_ddr(DST_DDR, ACT_DST_BASE, DST_WORDS)) {
        print_str("  DMA ACT->DDR timeout\n");
        errors++;
    }

    for (group = 0u; group < IC_GROUPS; group++) {
        for (y = 0u; y < IN_H; y++) {
            for (x = 0u; x < IN_W; x++) {
                uint32_t expect = src_pattern(group, y, x);
                for (dy = 0u; dy < 2u; dy++) {
                    for (dx = 0u; dx < 2u; dx++) {
                        uint32_t out_y = y * 2u + dy;
                        uint32_t out_x = x * 2u + dx;
                        uint32_t dst_idx = group * (IN_W * 2u) * (IN_H * 2u) +
                                           out_y * (IN_W * 2u) + out_x;
                        for (lane = 0u; lane < 4u; lane++) {
                            uint32_t got = read_ddr_word128_lane(DST_DDR, dst_idx, lane);
                            if (got != expect) {
                                errors++;
                                print_str("  mismatch idx=");
                                print_dec(dst_idx);
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
            }
        }
    }

    if (errors == 0u) {
        print_str("UPSAMPLE2X CPU SMOKE PASS\n");
        return;
    }

    print_str("UPSAMPLE2X CPU SMOKE FAIL errors=");
    print_dec(errors);
    print_str("\n");
    __asm__ volatile ("ebreak");
}
