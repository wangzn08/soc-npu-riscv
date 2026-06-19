// CPU/NPU smoke for conv6 (model.3): 32->64, 3x3, stride 2, pad 1.
// Input = the closed C2f (model.2) conv5 output strip, baked in as a fixture.
// Exercises OC=64 oc_single 3x3 stride-2 on the shared NPU.

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_conv6_from_c2f_m5w_data.h"
#include <stdint.h>

#define C6_ACT_DDR 0x40090000u
#define C6_WGT_DDR 0x40140000u
#define C6_OUT_DDR 0x40200000u

#define ACT_BASE 0u
#define WGT_BASE 0u
#define OUT_BASE 0u

static void write_ddr_word128(uint32_t byte_addr, uint32_t word_idx, const uint32_t lanes[4])
{
    volatile uint32_t *ptr = (volatile uint32_t *)(byte_addr + word_idx * 16u);
    ptr[0] = lanes[0]; ptr[1] = lanes[1]; ptr[2] = lanes[2]; ptr[3] = lanes[3];
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
    uint32_t i, pos, oc, errors = 0u;

    print_str("YOLO CONV6 (model.3, 32->64 s2) CPU SMOKE\n");

    for (i = 0u; i < YOLO_CONV6_ACT_WORDS; i++)
        write_ddr_word128(C6_ACT_DDR, i, yolo_conv6_act_words[i]);
    for (i = 0u; i < YOLO_CONV6_WGT_WORDS; i++)
        write_ddr_word128(C6_WGT_DDR, i, yolo_conv6_wgt_words[i]);

    yolo_set_pad_value(YOLO_CONV6_PAD_VALUE);
    yolo_set_silu_requant(YOLO_CONV6_REQUANT_MUL, YOLO_CONV6_REQUANT_SHIFT,
                          YOLO_CONV6_REQUANT_ZP);
    if (!yolo_dma_ddr_to_act(C6_ACT_DDR, ACT_BASE, YOLO_CONV6_ACT_WORDS) ||
        !yolo_dma_ddr_to_wgt(C6_WGT_DDR, WGT_BASE, YOLO_CONV6_WGT_WORDS) ||
        !yolo_run_conv2d_qparams(ACT_BASE, WGT_BASE, OUT_BASE,
                                 YOLO_CONV6_IN_W, YOLO_CONV6_IN_H,
                                 YOLO_CONV6_IC, YOLO_CONV6_OC,
                                 YOLO_CONV6_KH, YOLO_CONV6_KW,
                                 YOLO_CONV6_STRIDE, YOLO_CONV6_PAD,
                                 yolo_conv6_bias_q, yolo_conv6_scale_mul,
                                 yolo_conv6_scale_shift,
                                 NPU_CTRL_OC_SINGLE | NPU_CTRL_SILU_EN |
                                 NPU_CTRL_SILU_REQUANT_EN) ||
        !yolo_dma_out_to_ddr(C6_OUT_DDR, OUT_BASE, YOLO_CONV6_OUT_WORDS, 0u)) {
        print_str("  conv6 run failed\n");
        errors++;
    }

    // DIAGNOSTIC: global error histogram across all (pos,oc).
    {
        uint32_t n0 = 0u, n5 = 0u, n10 = 0u, n30 = 0u, n60 = 0u, ntot = 0u;
        for (oc = 0u; oc < YOLO_CONV6_OC; oc++) {
            for (pos = 0u; pos < YOLO_CONV6_OUT_SPATIAL; pos++) {
                int32_t got = read_ddr_s8(C6_OUT_DDR, (oc >> 4) * YOLO_CONV6_OUT_SPATIAL + pos, oc & 15u);
                int32_t expect = s8(yolo_conv6_expected_rtl[pos][oc]);
                uint32_t d = abs_diff(got, expect);
                ntot++;
                if (d == 0u) n0++;
                else if (d <= 5u) n5++;
                else if (d <= 10u) n10++;
                else if (d <= 30u) n30++;
                else if (d <= 60u) n60++;
                else errors++;  // >60
            }
        }
        print_str("HIST total="); print_dec(ntot);
        print_str(" eq0="); print_dec(n0);
        print_str(" le5="); print_dec(n5);
        print_str(" le10="); print_dec(n10);
        print_str(" le30="); print_dec(n30);
        print_str(" le60="); print_dec(n60);
        print_str(" gt60="); print_dec(errors); print_str("\n");
    }

    // DIAGNOSTIC: exact-rate per output row (probe vertical stride bug).
    {
        uint32_t oy;
        for (oy = 0u; oy < YOLO_CONV6_OUT_H; oy++) {
            uint32_t eq = 0u, tot = 0u;
            for (oc = 0u; oc < YOLO_CONV6_OC; oc++) {
                uint32_t ox;
                for (ox = 0u; ox < YOLO_CONV6_OUT_W; ox++) {
                    uint32_t p = oy * YOLO_CONV6_OUT_W + ox;
                    int32_t got = read_ddr_s8(C6_OUT_DDR, (oc >> 4) * YOLO_CONV6_OUT_SPATIAL + p, oc & 15u);
                    int32_t e = s8(yolo_conv6_expected_rtl[p][oc]);
                    tot++;
                    if (abs_diff(got, e) <= 2u) eq++;
                }
            }
            print_str("row "); print_dec(oy);
            print_str(" exact<=2: "); print_dec(eq);
            print_str("/"); print_dec(tot); print_str("\n");
        }
    }

    // DIAGNOSTIC: read back stored params for oc 37/38/39 vs golden.
    {
        uint32_t c;
        for (c = 37u; c <= 39u; c++) {
            uint32_t hb = *(volatile uint32_t *)NPU_BIAS(c);
            uint32_t hs = *(volatile uint32_t *)NPU_SCALE(c);
            uint32_t hh = *(volatile uint32_t *)NPU_SHIFT(c);
            print_str("oc="); print_dec(c);
            print_str(" biasHW="); print_dec(hb);
            print_str(" biasGD="); print_dec((uint32_t)yolo_conv6_bias_q[c]);
            print_str(" sclHW="); print_dec(hs);
            print_str(" sclGD="); print_dec(yolo_conv6_scale_mul[c]);
            print_str(" shHW="); print_dec(hh);
            print_str(" shGD="); print_dec(yolo_conv6_scale_shift[c]);
            print_str("\n");
        }
    }

    // DIAGNOSTIC: raw oc=38 got/exp for positions 0..31.
    print_str("oc38 raw (pos got exp):\n");
    for (pos = 0u; pos < 32u; pos++) {
        int32_t got = read_ddr_s8(C6_OUT_DDR, 2u * YOLO_CONV6_OUT_SPATIAL + pos, 6u);
        int32_t expect = s8(yolo_conv6_expected_rtl[pos][38]);
        print_str("  "); print_dec(pos);
        print_str(" "); print_dec((uint32_t)got);
        print_str(" "); print_dec((uint32_t)expect); print_str("\n");
    }

    if (errors == 0u) {
        print_str("YOLO CONV6 CPU SMOKE PASS\n");
        return;
    }
    print_str("YOLO CONV6 CPU SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
