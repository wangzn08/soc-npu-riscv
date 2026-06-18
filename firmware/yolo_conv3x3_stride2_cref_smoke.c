// CPU/NPU smoke for one real YOLO stride-2 3x3 conv tile against C-reference.
// Uses conv1 OC0..15: OC16 x IC16 x 3x3, stride=2, pad=1.

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_conv3x3_stride2_real_data.h"
#include <stdint.h>

#define ACT_DDR 0x40068000u
#define WGT_DDR 0x4006A000u
#define OUT_DDR 0x4006E000u

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

    print_str("YOLO REAL CONV3X3 STRIDE2 CREF CPU SMOKE\n");

    for (i = 0u; i < YOLO_CONV3X3_S2_ACT_WORDS; i++)
        write_ddr_word128(ACT_DDR, i, yolo_conv3x3_s2_act_words[i]);
    for (i = 0u; i < YOLO_CONV3X3_S2_WGT_WORDS; i++)
        write_ddr_word128(WGT_DDR, i, yolo_conv3x3_s2_wgt_words[i]);
    for (i = 0u; i < YOLO_CONV3X3_S2_OUT_WORDS; i++) {
        const uint32_t zero[4] = {0u, 0u, 0u, 0u};
        write_ddr_word128(OUT_DDR, i, zero);
    }

    yolo_set_silu_requant(YOLO_CONV3X3_S2_REQUANT_MUL,
                          YOLO_CONV3X3_S2_REQUANT_SHIFT,
                          YOLO_CONV3X3_S2_REQUANT_ZP);
    yolo_set_pad_value(YOLO_CONV3X3_S2_IN_ZP);

    if (!yolo_dma_ddr_to_act(ACT_DDR, ACT_BASE, YOLO_CONV3X3_S2_ACT_WORDS) ||
        !yolo_dma_ddr_to_wgt(WGT_DDR, WGT_BASE, YOLO_CONV3X3_S2_WGT_WORDS) ||
        !yolo_run_conv2d_qparams(ACT_BASE, WGT_BASE, OUT_BASE,
                                 YOLO_CONV3X3_S2_IN_W, YOLO_CONV3X3_S2_IN_H,
                                 YOLO_CONV3X3_S2_IC, YOLO_CONV3X3_S2_OC,
                                 YOLO_CONV3X3_S2_KH, YOLO_CONV3X3_S2_KW,
                                 YOLO_CONV3X3_S2_STRIDE, YOLO_CONV3X3_S2_PAD,
                                 yolo_conv3x3_s2_bias_q,
                                 yolo_conv3x3_s2_scale_mul,
                                 yolo_conv3x3_s2_scale_shift,
                                 NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN) ||
        !yolo_dma_out_to_ddr(OUT_DDR, OUT_BASE, YOLO_CONV3X3_S2_OUT_WORDS, 0u)) {
        print_str("  real conv3x3 stride2 path failed\n");
        errors++;
    }

    for (pos = 0u; pos < YOLO_CONV3X3_S2_OUT_WORDS; pos++) {
        for (oc = 0u; oc < YOLO_CONV3X3_S2_OC; oc++) {
            int32_t got = read_ddr_s8(OUT_DDR, pos, oc);
            int32_t expect = s8(yolo_conv3x3_s2_expected_cref[pos][oc]);
            uint32_t diff = abs_diff(got, expect);
            if (diff > YOLO_CONV3X3_S2_CREF_TOL) {
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
        print_str("YOLO REAL CONV3X3 STRIDE2 CREF CPU SMOKE PASS\n");
        return;
    }

    print_str("YOLO REAL CONV3X3 STRIDE2 CREF CPU SMOKE FAIL errors=");
    print_dec(errors);
    print_str("\n");
    __asm__ volatile ("ebreak");
}
