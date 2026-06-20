// conv13 (model.5): 64->128 3x3 stride2 pad1, run via the TILED strip primitive
// with FORCED strip_out_rows=2 (out_h=4 -> 2 strips). Exercises cross-strip halo
// + materialized vertical boundary padding + OC>64 chunking. Output must match
// the same golden as the non-tiled conv13 smoke (RTL_TOL).

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_conv13_m6b_data.h"
#include <stdint.h>

#define ACT_DDR     0x40090000u
#define WGT_DDR     0x40140000u
#define OUT_DDR     0x40200000u
#define PAD_ROW_DDR 0x40300000u
#define WGT_BASE    0u

#define STRIP_OUT_ROWS 2u

static void wrw(uint32_t a, uint32_t w, const uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); p[0]=l[0];p[1]=l[1];p[2]=l[2];p[3]=l[3]; }
static int32_t rs8(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); uint32_t v=p[b>>2]>>((b&3u)*8u)&0xFFu; return (v&0x80u)?((int32_t)v-256):(int32_t)v; }
static int32_t s8(uint32_t b){ b&=0xFFu; return (b&0x80u)?((int32_t)b-256):(int32_t)b; }
static uint32_t ad(int32_t a,int32_t b){ int32_t d=a-b; return (uint32_t)(d<0?-d:d); }

void usercode7(void)
{
    uint32_t i, pos, oc, errors = 0u;

    print_str("YOLO CONV13 TILED (model.5, 64->128 s2, strip=2) CPU SMOKE\n");
    for (i = 0u; i < YOLO_CONV13_ACT_WORDS; i++) wrw(ACT_DDR, i, yolo_conv13_act_words[i]);
    for (i = 0u; i < YOLO_CONV13_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_conv13_wgt_words[i]);

    yolo_set_pad_value(YOLO_CONV13_PAD_VALUE);
    yolo_set_silu_requant(YOLO_CONV13_REQUANT_MUL, YOLO_CONV13_REQUANT_SHIFT, YOLO_CONV13_REQUANT_ZP);

    if (!yolo_run_conv2d_tiled(ACT_DDR, WGT_DDR, WGT_BASE, OUT_DDR, PAD_ROW_DDR,
                               YOLO_CONV13_IN_W, YOLO_CONV13_IN_H,
                               YOLO_CONV13_IC, YOLO_CONV13_OC,
                               YOLO_CONV13_KH, YOLO_CONV13_KW,
                               YOLO_CONV13_STRIDE, YOLO_CONV13_PAD,
                               yolo_conv13_bias_q, yolo_conv13_scale_mul,
                               yolo_conv13_scale_shift,
                               NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN,
                               YOLO_CONV13_WGT_PER_OC, STRIP_OUT_ROWS,
                               YOLO_CONV13_PAD_VALUE)) {
        print_str("  conv13 tiled run failed\n"); errors++;
    }

    for (pos = 0u; pos < YOLO_CONV13_OUT_SPATIAL && errors <= 16u; pos++)
        for (oc = 0u; oc < YOLO_CONV13_OC; oc++) {
            int32_t got = rs8(OUT_DDR, (oc>>4)*YOLO_CONV13_OUT_SPATIAL + pos, oc&15u);
            int32_t exp = s8(yolo_conv13_expected_rtl[pos][oc]);
            if (ad(got, exp) > YOLO_CONV13_RTL_TOL) {
                errors++;
                print_str("  mismatch pos="); print_dec(pos);
                print_str(" oc="); print_dec(oc);
                print_str(" got="); print_dec(got);
                print_str(" exp="); print_dec(exp); print_str("\n");
                if (errors > 16u) break;
            }
        }

    if (errors == 0u) { print_str("YOLO CONV13 TILED CPU SMOKE PASS\n"); return; }
    print_str("YOLO CONV13 TILED CPU SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
