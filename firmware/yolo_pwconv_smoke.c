// CPU/NPU cooperative smoke test for a YOLO-style 1x1 pointwise conv block.
// The CPU writes deterministic activations and weights to DDR, uses NPU DMA
// to load Act/Wgt SRAM, starts the shared NPU pointwise path, drains Out SRAM
// to DDR, and checks exact INT8 output bytes.

#include "firmware.h"
#include "yolo_ops.h"
#include <stdint.h>

#define ACT_DDR      0x4002A000u
#define WGT_DDR      0x4002C000u
#define OUT_DDR      0x4002E000u

#define ACT_BASE     0u
#define WGT_BASE     0u
#define OUT_BASE     0u

#define IN_W         3u
#define IN_H         2u
#define SPATIAL      (IN_W * IN_H)
#define IC           16u
#define OC           16u
#define WGT_WORDS    16u

static uint32_t rep8(uint32_t b)
{
    b &= 0xFFu;
    return b | (b << 8) | (b << 16) | (b << 24);
}

static void write_ddr_word128(uint32_t byte_addr, uint32_t word_idx,
                              uint32_t lane0, uint32_t lane1,
                              uint32_t lane2, uint32_t lane3)
{
    volatile uint32_t *ptr = (volatile uint32_t *)(byte_addr + word_idx * 16u);
    ptr[0] = lane0;
    ptr[1] = lane1;
    ptr[2] = lane2;
    ptr[3] = lane3;
}

static uint32_t read_ddr_byte(uint32_t byte_addr, uint32_t word_idx, uint32_t byte_idx)
{
    volatile uint32_t *ptr = (volatile uint32_t *)(byte_addr + word_idx * 16u);
    uint32_t lane = ptr[byte_idx >> 2];
    return (lane >> ((byte_idx & 3u) * 8u)) & 0xFFu;
}

static uint32_t expected_byte(uint32_t pos, uint32_t oc)
{
    return (pos + 1u) * ((oc % 4u) + 1u);
}

void usercode7(void)
{
    int32_t bias[16];
    uint32_t pos;
    uint32_t oc;
    uint32_t ic;
    uint32_t errors = 0u;

    print_str("YOLO PWCONV CPU SMOKE\n");

    for (pos = 0u; pos < SPATIAL; pos++) {
        uint32_t v = rep8(pos + 1u);
        write_ddr_word128(ACT_DDR, pos, v, v, v, v);
        write_ddr_word128(OUT_DDR, pos, 0u, 0u, 0u, 0u);
    }

    for (oc = 0u; oc < OC; oc++) {
        uint32_t lanes[4] = {0u, 0u, 0u, 0u};
        uint32_t active = (oc % 4u) + 1u;
        for (ic = 0u; ic < IC; ic++) {
            if (ic < active)
                lanes[ic >> 2] |= 1u << ((ic & 3u) * 8u);
        }
        write_ddr_word128(WGT_DDR, oc, lanes[0], lanes[1], lanes[2], lanes[3]);
    }

    for (oc = 0u; oc < OC; oc++)
        bias[oc] = 0;

    if (!yolo_dma_ddr_to_act(ACT_DDR, ACT_BASE, SPATIAL)) {
        print_str("  act DMA timeout\n");
        errors++;
    }
    if (errors == 0u && !yolo_dma_ddr_to_wgt(WGT_DDR, WGT_BASE, WGT_WORDS)) {
        print_str("  wgt DMA timeout\n");
        errors++;
    }
    if (errors == 0u &&
        !yolo_run_pw_conv1x1(ACT_BASE, WGT_BASE, OUT_BASE, IN_W, IN_H, IC, OC,
                             bias, 1u, 0u, NPU_CTRL_RELU_EN)) {
        print_str("  pwconv run timeout/error\n");
        errors++;
    }
    if (errors == 0u && !yolo_dma_out_to_ddr(OUT_DDR, OUT_BASE, SPATIAL, 0u)) {
        print_str("  out DMA timeout\n");
        errors++;
    }

    for (pos = 0u; pos < SPATIAL; pos++) {
        for (oc = 0u; oc < OC; oc++) {
            uint32_t got = read_ddr_byte(OUT_DDR, pos, oc);
            uint32_t expect = expected_byte(pos, oc);
            if (got != expect) {
                errors++;
                print_str("  mismatch pos=");
                print_dec(pos);
                print_str(" oc=");
                print_dec(oc);
                print_str(" got=");
                print_dec(got);
                print_str(" exp=");
                print_dec(expect);
                print_str("\n");
            }
        }
    }

    if (errors == 0u) {
        print_str("YOLO PWCONV CPU SMOKE PASS\n");
        return;
    }

    print_str("YOLO PWCONV CPU SMOKE FAIL errors=");
    print_dec(errors);
    print_str("\n");
    __asm__ volatile ("ebreak");
}
