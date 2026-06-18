// CPU/NPU cooperative smoke for a tiny YOLO-style subgraph:
//   pointwise conv -> upsample2x -> concat skip -> pointwise conv.
//
// The tensors are intentionally tiny, but every step goes through the shared
// CPU, MMIO, DMA, Act/Wgt/Out SRAM, and NPU datapath used by larger YOLO blocks.

#include "firmware.h"
#include "yolo_ops.h"
#include <stdint.h>

#define IN_DDR         0x40036000u
#define WGT0_DDR       0x40038000u
#define CONV0_DDR      0x4003A000u
#define UP_DDR         0x4003C000u
#define SKIP_DDR       0x4003E000u
#define WGT1_DDR       0x40040000u
#define OUT_DDR        0x40042000u

#define ACT_IN_BASE    0u
#define ACT_UP_SRC     128u
#define ACT_UP_DST     256u
#define ACT_CONCAT     512u
#define WGT_BASE       0u
#define OUT_BASE       0u

#define IN_W           2u
#define IN_H           2u
#define UP_W           4u
#define UP_H           4u
#define IC             16u
#define OC             16u
#define GROUPS_16      1u
#define CONCAT_IC      32u
#define SP0            (IN_W * IN_H)
#define SP1            (UP_W * UP_H)

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

static uint32_t src_value(uint32_t pos)
{
    return pos + 1u;
}

static uint32_t skip_value(uint32_t pos)
{
    return 1u + (pos & 3u);
}

static uint32_t upsampled_value(uint32_t out_pos)
{
    uint32_t y = out_pos / UP_W;
    uint32_t x = out_pos % UP_W;
    return src_value((y >> 1) * IN_W + (x >> 1));
}

static uint32_t expected_final(uint32_t pos)
{
    return upsampled_value(pos) + 2u * skip_value(pos);
}

static void write_conv0_weights(void)
{
    uint32_t oc;

    for (oc = 0u; oc < OC; oc++) {
        uint32_t lanes[4] = {0u, 0u, 0u, 0u};
        lanes[0] = 0x00000001u;  // IC0 weight = 1, all other IC lanes = 0
        write_ddr_word128(WGT0_DDR, oc, lanes[0], lanes[1], lanes[2], lanes[3]);
    }
}

static void write_conv1_weights(void)
{
    uint32_t oc;

    for (oc = 0u; oc < OC; oc++) {
        uint32_t g0[4] = {0u, 0u, 0u, 0u};
        uint32_t g1[4] = {0u, 0u, 0u, 0u};
        g0[0] = 0x00000001u;  // group0 lane0: upsample branch
        g1[0] = 0x00000002u;  // group1 lane0: skip branch
        write_ddr_word128(WGT1_DDR, oc * 2u, g0[0], g0[1], g0[2], g0[3]);
        write_ddr_word128(WGT1_DDR, oc * 2u + 1u, g1[0], g1[1], g1[2], g1[3]);
    }
}

void usercode7(void)
{
    int32_t bias[16];
    uint32_t pos;
    uint32_t oc;
    uint32_t errors = 0u;

    print_str("YOLO SUBGRAPH CPU SMOKE\n");

    for (oc = 0u; oc < OC; oc++)
        bias[oc] = 0;

    for (pos = 0u; pos < SP0; pos++)
        write_ddr_word128(IN_DDR, pos, src_value(pos), 0u, 0u, 0u);
    for (pos = 0u; pos < SP1; pos++) {
        uint32_t v = rep8(skip_value(pos));
        write_ddr_word128(SKIP_DDR, pos, v, v, v, v);
        write_ddr_word128(OUT_DDR, pos, 0u, 0u, 0u, 0u);
    }

    write_conv0_weights();
    write_conv1_weights();

    if (!yolo_dma_ddr_to_act(IN_DDR, ACT_IN_BASE, SP0) ||
        !yolo_dma_ddr_to_wgt(WGT0_DDR, WGT_BASE, OC) ||
        !yolo_run_pw_conv1x1(ACT_IN_BASE, WGT_BASE, OUT_BASE,
                             IN_W, IN_H, IC, OC, bias, 1u, 0u, NPU_CTRL_RELU_EN) ||
        !yolo_dma_out_to_ddr(CONV0_DDR, OUT_BASE, SP0, 0u)) {
        print_str("  conv0 path failed\n");
        errors++;
    }

    if (errors == 0u &&
        (!yolo_dma_ddr_to_act(CONV0_DDR, ACT_UP_SRC, SP0) ||
         !yolo_run_upsample2x(ACT_UP_SRC, ACT_UP_DST, IN_W, IN_H, GROUPS_16) ||
         !yolo_dma_act_to_ddr(UP_DDR, ACT_UP_DST, SP1))) {
        print_str("  upsample path failed\n");
        errors++;
    }

    if (errors == 0u &&
        (!yolo_concat2_ddr_to_act(UP_DDR, SKIP_DDR, ACT_CONCAT, SP1, 1u, 1u) ||
         !yolo_dma_ddr_to_wgt(WGT1_DDR, WGT_BASE, OC * 2u) ||
         !yolo_run_pw_conv1x1(ACT_CONCAT, WGT_BASE, OUT_BASE,
                              UP_W, UP_H, CONCAT_IC, OC, bias, 1u, 0u, NPU_CTRL_RELU_EN) ||
         !yolo_dma_out_to_ddr(OUT_DDR, OUT_BASE, SP1, 0u))) {
        print_str("  concat/final conv path failed\n");
        errors++;
    }

    for (pos = 0u; pos < SP1; pos++) {
        uint32_t expect = expected_final(pos);
        for (oc = 0u; oc < OC; oc++) {
            uint32_t got = read_ddr_byte(OUT_DDR, pos, oc);
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
        print_str("YOLO SUBGRAPH CPU SMOKE PASS\n");
        return;
    }

    print_str("YOLO SUBGRAPH CPU SMOKE FAIL errors=");
    print_dec(errors);
    print_str("\n");
    __asm__ volatile ("ebreak");
}
