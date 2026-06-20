// Detect-head scale-0 bbox branch: pan_p3 -> conv36(3x3 SiLU) -> conv38(3x3 SiLU)
// -> conv41(1x1 LINEAR). conv41 uses the linear-requant path (SILU_REQUANT_EN
// without SILU_EN): out = clamp_s8(s2 + out_zp). Input = c2f_15 output (baked).

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_head_bbox0_data.h"
#include <stdint.h>

#define IN_DDR  0x40090000u
#define C36_DDR 0x40120000u
#define C38_DDR 0x40160000u
#define OUT_DDR 0x401A0000u
#define WGT_DDR 0x401E0000u
#define ACT_BASE 0u
#define WGT_BASE 0u
#define W YOLO_HB0_IN_W
#define H YOLO_HB0_IN_H
#define SP YOLO_HB0_SPATIAL

static void wrw(uint32_t a, uint32_t w, const uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); p[0]=l[0];p[1]=l[1];p[2]=l[2];p[3]=l[3]; }
static int32_t rs8(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); uint32_t v=p[b>>2]>>((b&3u)*8u)&0xFFu; return (v&0x80u)?((int32_t)v-256):(int32_t)v; }
static int32_t s8(uint32_t b){ b&=0xFFu; return (b&0x80u)?((int32_t)b-256):(int32_t)b; }
static uint32_t ad(int32_t a,int32_t b){ int32_t d=a-b; return (uint32_t)(d<0?-d:d); }

void usercode7(void)
{
    uint32_t i, pos, oc, errors = 0u;

    print_str("YOLO HEAD bbox0 (conv36->38->41 LINEAR) CPU SMOKE\n");
    for (i = 0u; i < YOLO_HB0_BLKIN_WORDS; i++) wrw(IN_DDR, i, yolo_hb0_blkin_words[i]);

    // conv36 (3x3 SiLU)
    for (i = 0u; i < YOLO_HB0_C36_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_hb0_c36_wgt[i]);
    yolo_set_pad_value(YOLO_HB0_C36_PAD);
    yolo_set_silu_requant(YOLO_HB0_C36_RQ_MUL, 12u, YOLO_HB0_C36_RQ_ZP);
    if (!yolo_dma_ddr_to_act(IN_DDR, ACT_BASE, SP*4u) ||
        !yolo_dma_ddr_to_wgt(WGT_DDR, WGT_BASE, YOLO_HB0_C36_WGT_WORDS) ||
        !yolo_run_conv2d_qparams(ACT_BASE, WGT_BASE, 0u, W, H, 64u, 64u, 3u,3u,1u,1u,
                                 yolo_hb0_c36_bias, yolo_hb0_c36_mul, yolo_hb0_c36_shift,
                                 NPU_CTRL_OC_SINGLE|NPU_CTRL_SILU_EN|NPU_CTRL_SILU_REQUANT_EN) ||
        !yolo_dma_out_to_ddr(C36_DDR, 0u, SP*4u, 0u)) { print_str("  c36 fail\n"); errors++; }

    // conv38 (3x3 SiLU)
    if (errors==0u) {
        for (i = 0u; i < YOLO_HB0_C38_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_hb0_c38_wgt[i]);
        yolo_set_pad_value(YOLO_HB0_C38_PAD);
        yolo_set_silu_requant(YOLO_HB0_C38_RQ_MUL, 12u, YOLO_HB0_C38_RQ_ZP);
        if (!yolo_dma_ddr_to_act(C36_DDR, ACT_BASE, SP*4u) ||
            !yolo_dma_ddr_to_wgt(WGT_DDR, WGT_BASE, YOLO_HB0_C38_WGT_WORDS) ||
            !yolo_run_conv2d_qparams(ACT_BASE, WGT_BASE, 0u, W, H, 64u, 64u, 3u,3u,1u,1u,
                                     yolo_hb0_c38_bias, yolo_hb0_c38_mul, yolo_hb0_c38_shift,
                                     NPU_CTRL_OC_SINGLE|NPU_CTRL_SILU_EN|NPU_CTRL_SILU_REQUANT_EN) ||
            !yolo_dma_out_to_ddr(C38_DDR, 0u, SP*4u, 0u)) { print_str("  c38 fail\n"); errors++; }
    }

    // conv41 (1x1 LINEAR): SILU_REQUANT_EN without SILU_EN; zp = out_zp.
    if (errors==0u) {
        for (i = 0u; i < YOLO_HB0_C41_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_hb0_c41_wgt[i]);
        yolo_set_silu_requant(0u, 0u, YOLO_HB0_C41_OUT_ZP);
        if (!yolo_dma_ddr_to_act(C38_DDR, ACT_BASE, SP*4u) ||
            !yolo_dma_ddr_to_wgt(WGT_DDR, WGT_BASE, YOLO_HB0_C41_WGT_WORDS) ||
            !yolo_run_pw_conv1x1_qparams(ACT_BASE, WGT_BASE, 0u, W, H, 64u, 64u,
                                         yolo_hb0_c41_bias, yolo_hb0_c41_mul, yolo_hb0_c41_shift,
                                         NPU_CTRL_OC_SINGLE|NPU_CTRL_SILU_REQUANT_EN) ||
            !yolo_dma_out_to_ddr(OUT_DDR, 0u, SP*4u, 0u)) { print_str("  c41 fail\n"); errors++; }
    }

    for (pos = 0u; pos < SP && errors <= 16u; pos++)
        for (oc = 0u; oc < 64u; oc++) {
            int32_t got = rs8(OUT_DDR, (oc>>4)*SP + pos, oc&15u);
            int32_t exp = s8(yolo_hb0_expected_rtl[pos][oc]);
            if (ad(got, exp) > YOLO_HB0_RTL_TOL) {
                errors++;
                print_str("  mismatch pos="); print_dec(pos);
                print_str(" oc="); print_dec(oc);
                print_str(" got="); print_dec(got);
                print_str(" exp="); print_dec(exp); print_str("\n");
                if (errors > 16u) break;
            }
        }

    if (errors == 0u) { print_str("YOLO HEAD bbox0 CPU SMOKE PASS\n"); return; }
    print_str("YOLO HEAD bbox0 CPU SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
