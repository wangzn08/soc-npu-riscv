// conv20 (model.7): 128->256 3x3 stride2 pad1, input = c2f_6 output (baked).
// OC=256 -> 4-chunk OC>64 conv.

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_conv20_data.h"
#include <stdint.h>

#define ACT_DDR 0x40090000u
#define WGT_DDR 0x40160000u
#define OUT_DDR 0x40240000u
#define ACT_BASE 0u
#define WGT_BASE 0u

static void wrw(uint32_t a, uint32_t w, const uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); p[0]=l[0];p[1]=l[1];p[2]=l[2];p[3]=l[3]; }
static int32_t rs8(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); uint32_t v=p[b>>2]>>((b&3u)*8u)&0xFFu; return (v&0x80u)?((int32_t)v-256):(int32_t)v; }
static int32_t s8(uint32_t b){ b&=0xFFu; return (b&0x80u)?((int32_t)b-256):(int32_t)b; }
static uint32_t ad(int32_t a,int32_t b){ int32_t d=a-b; return (uint32_t)(d<0?-d:d); }

void usercode7(void)
{
    uint32_t i, pos, oc, errors = 0u;

    print_str("YOLO CONV20 (model.7, 128->256 s2, OC=256) CPU SMOKE\n");
    for (i = 0u; i < YOLO_CONV20_ACT_WORDS; i++) wrw(ACT_DDR, i, yolo_conv20_act_words[i]);
    for (i = 0u; i < YOLO_CONV20_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_conv20_wgt_words[i]);

    yolo_set_pad_value(YOLO_CONV20_PAD_VALUE);
    yolo_set_silu_requant(YOLO_CONV20_REQUANT_MUL, YOLO_CONV20_REQUANT_SHIFT, YOLO_CONV20_REQUANT_ZP);
    if (!yolo_dma_ddr_to_act(ACT_DDR, ACT_BASE, YOLO_CONV20_ACT_WORDS) ||
        !yolo_run_conv2d_oc_chunks(ACT_BASE, WGT_DDR, WGT_BASE, OUT_DDR,
                                   YOLO_CONV20_IN_W, YOLO_CONV20_IN_H,
                                   YOLO_CONV20_IC, YOLO_CONV20_OC,
                                   YOLO_CONV20_KH, YOLO_CONV20_KW,
                                   YOLO_CONV20_STRIDE, YOLO_CONV20_PAD,
                                   yolo_conv20_bias_q, yolo_conv20_scale_mul,
                                   yolo_conv20_scale_shift,
                                   NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN,
                                   YOLO_CONV20_WGT_PER_OC, YOLO_CONV20_OUT_SPATIAL)) {
        print_str("  conv20 run failed\n"); errors++;
    }

    for (pos = 0u; pos < YOLO_CONV20_OUT_SPATIAL && errors <= 16u; pos++)
        for (oc = 0u; oc < YOLO_CONV20_OC; oc++) {
            int32_t got = rs8(OUT_DDR, (oc>>4)*YOLO_CONV20_OUT_SPATIAL + pos, oc&15u);
            int32_t exp = s8(yolo_conv20_expected_rtl[pos][oc]);
            if (ad(got, exp) > YOLO_CONV20_RTL_TOL) {
                errors++;
                print_str("  mismatch pos="); print_dec(pos);
                print_str(" oc="); print_dec(oc);
                print_str(" got="); print_dec(got);
                print_str(" exp="); print_dec(exp); print_str("\n");
                if (errors > 16u) break;
            }
        }

    if (errors == 0u) { print_str("YOLO CONV20 CPU SMOKE PASS\n"); return; }
    print_str("YOLO CONV20 CPU SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
