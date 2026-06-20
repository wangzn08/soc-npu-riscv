// Generic @320 single trunk-conv tiled measurement (data from gen_yolo_layer320.py).
// Measures real per-layer NPU cost (perf counters) + validates vs RTL-model golden.

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_layer320_data.h"
#include <stdint.h>

#define ACT_DDR     0x40090000u
#define WGT_DDR     0x40140000u
#define OUT_DDR     0x40200000u
#define PAD_ROW_DDR 0x40300000u
#define WGT_BASE    0u
#define STRIP_OUT_ROWS 16u

static void wrw(uint32_t a, uint32_t w, const uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); p[0]=l[0];p[1]=l[1];p[2]=l[2];p[3]=l[3]; }
static int32_t rs8(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); uint32_t v=p[b>>2]>>((b&3u)*8u)&0xFFu; return (v&0x80u)?((int32_t)v-256):(int32_t)v; }
static int32_t s8(uint32_t b){ b&=0xFFu; return (b&0x80u)?((int32_t)b-256):(int32_t)b; }
static uint32_t ad(int32_t a,int32_t b){ int32_t d=a-b; return (uint32_t)(d<0?-d:d); }
static uint32_t rdp(uint32_t a){ return *(volatile uint32_t*)a; }

void usercode7(void)
{
    uint32_t i, pos, oc, errors = 0u;

    print_str("YOLO LAYER320 conv"); print_dec(LYR_CI); print_str(" TILED SMOKE\n");
    for (i = 0u; i < LYR_ACT_WORDS; i++) wrw(ACT_DDR, i, yolo_layer320_act_words[i]);
    for (i = 0u; i < LYR_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_layer320_wgt_words[i]);

    yolo_set_pad_value(LYR_PAD_VALUE);
    yolo_set_silu_requant(LYR_REQUANT_MUL, LYR_REQUANT_SHIFT, LYR_REQUANT_ZP);

    *(volatile uint32_t *)NPU_PERF_CLR = 1u;

    if (!yolo_run_conv2d_tiled(ACT_DDR, WGT_DDR, WGT_BASE, OUT_DDR, PAD_ROW_DDR,
                               LYR_IN_W, LYR_IN_H, LYR_IC, LYR_OC,
                               LYR_KH, LYR_KW, LYR_STRIDE, LYR_PAD,
                               yolo_layer320_bias_q, yolo_layer320_scale_mul,
                               yolo_layer320_scale_shift,
                               NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN,
                               LYR_WGT_PER_OC, STRIP_OUT_ROWS, LYR_PAD_VALUE)) {
        print_str("  tiled run failed\n"); errors++;
    }

    print_str("PERF cyc_total="); print_dec(rdp(NPU_PERF_CYC_TOTAL));
    print_str(" npu_busy=");      print_dec(rdp(NPU_PERF_CYC_BUSY));
    print_str(" arr=");           print_dec(rdp(NPU_PERF_CYC_ARR));
    print_str(" rd_beats=");      print_dec(rdp(NPU_PERF_RD_BEATS));
    print_str(" wr_beats=");      print_dec(rdp(NPU_PERF_WR_BEATS));
    print_str("\n");

    for (pos = 0u; pos < LYR_OUT_SPATIAL && errors <= 16u; pos++)
        for (oc = 0u; oc < LYR_OC; oc++) {
            int32_t got = rs8(OUT_DDR, (oc>>4)*LYR_OUT_SPATIAL + pos, oc&15u);
            int32_t exp = s8(yolo_layer320_golden[pos][oc]);
            if (ad(got, exp) > LYR_RTL_TOL) {
                errors++;
                print_str("  mismatch pos="); print_dec(pos);
                print_str(" oc="); print_dec(oc);
                print_str(" got="); print_dec(got);
                print_str(" exp="); print_dec(exp); print_str("\n");
                if (errors > 16u) break;
            }
        }

    if (errors == 0u) { print_str("YOLO LAYER320 SMOKE PASS\n"); return; }
    print_str("YOLO LAYER320 SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
