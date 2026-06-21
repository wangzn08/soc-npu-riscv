// conv1 (model.1, 16->32 3x3 s2 pad1) at FULL 320 resolution with the EXACT
// per-layer SiLU LUT (CTRL[22]). Proves the exact-SiLU fix on a standalone tiled
// conv: conv1 is the worst legacy-LUT saturator (~20%), and its exact golden
// aligns to the C float dump within +-1 LSB. Same tiled datapath as the legacy
// conv1 smoke, but loads the per-layer LUT and runs NPU_CTRL_SILU_EXACT_EN.

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_conv1_320_exact_data.h"
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

    print_str("YOLO CONV1 @320 EXACT-SiLU SMOKE (160x160x16 -> 80x80x32)\n");
    for (i = 0u; i < C1E_ACT_WORDS; i++) wrw(ACT_DDR, i, yolo_conv1_320e_act_words[i]);
    for (i = 0u; i < C1E_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_conv1_320e_wgt_words[i]);

    yolo_set_pad_value(C1E_PAD_VALUE);
    yolo_load_silu_lut(yolo_conv1_320e_silu_lut);
    yolo_set_silu_requant(0u, 0u, C1E_OUT_ZP);   /* exact: only the out zp matters */

    *(volatile uint32_t *)NPU_PERF_CLR = 1u;

    if (!yolo_run_conv2d_tiled(ACT_DDR, WGT_DDR, WGT_BASE, OUT_DDR, PAD_ROW_DDR,
                               C1E_IN_W, C1E_IN_H, C1E_IC, C1E_OC,
                               3u, 3u, 2u, 1u,
                               yolo_conv1_320e_bias_q, yolo_conv1_320e_scale_mul,
                               yolo_conv1_320e_scale_shift,
                               NPU_CTRL_SILU_EXACT_EN,
                               C1E_WGT_PER_OC, STRIP_OUT_ROWS, C1E_PAD_VALUE)) {
        print_str("  tiled run failed\n"); errors++;
    }

    print_str("PERF cyc_total="); print_dec(rdp(NPU_PERF_CYC_TOTAL));
    print_str(" npu_busy=");      print_dec(rdp(NPU_PERF_CYC_BUSY));
    print_str("\n");

    for (pos = 0u; pos < C1E_OUT_SPATIAL && errors <= 16u; pos++)
        for (oc = 0u; oc < C1E_OC; oc++) {
            int32_t got = rs8(OUT_DDR, (oc>>4)*C1E_OUT_SPATIAL + pos, oc&15u);
            int32_t exp = s8(yolo_conv1_320e_golden[pos][oc]);
            if (ad(got, exp) > C1E_RTL_TOL) {
                errors++;
                print_str("  mismatch pos="); print_dec(pos);
                print_str(" oc="); print_dec(oc);
                print_str(" got="); print_dec(got);
                print_str(" exp="); print_dec(exp); print_str("\n");
                if (errors > 16u) break;
            }
        }

    if (errors == 0u) { print_str("YOLO CONV1 @320 EXACT-SiLU SMOKE PASS\n"); return; }
    print_str("YOLO CONV1 @320 EXACT-SiLU SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
