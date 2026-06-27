// 1x1 pointwise conv (real YOLOv8n PW weights, IC=32 OC=16) via the descriptor
// runtime. Validates the 1x1 PW path (kh=kw=1, pad=0, no halo, no HW pad) and
// the legacy SiLU (CTRL[18]) activation mode through OP_ACTIVATION_CFG.
// Golden = the NPU SILU_EN output (yolo_silu_real_expected).
//
// Build: touch .yolo_ddr; bash run_all.sh sim yolo_pwconv_desc_smoke.c yolo_ops.c yolo_desc.c

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_desc.h"
#include "yolo_pwconv_silu_real_data.h"
#include <stdint.h>

#define ACT_DDR     0x40090000u
#define WGT_DDR     0x40094000u
#define OUT_DDR     0x40098000u
#define PAD_ROW_DDR 0x4009C000u
#define WGT_BASE    0u

static void wrw(uint32_t a, uint32_t w, const uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); p[0]=l[0];p[1]=l[1];p[2]=l[2];p[3]=l[3]; }
static uint32_t rb(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); return (p[b>>2]>>((b&3u)*8u))&0xFFu; }

void usercode7(void)
{
    uint32_t i, pos, oc, errors = 0u;

    print_str("YOLO PWCONV (1x1) DESCRIPTOR SMOKE (2x2x32 -> 2x2x16, SILU_EN)\n");
    for (i = 0u; i < YOLO_SILU_REAL_ACT_WORDS; i++) wrw(ACT_DDR, i, yolo_silu_real_act_words[i]);
    for (i = 0u; i < YOLO_SILU_REAL_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_silu_real_wgt_words[i]);

    *(volatile uint32_t *)NPU_PERF_CLR = 1u;

    // 1x1 PW, IC/OC from data, stride 1, pad 0; wgt_per_oc = icg*1*1 = 2.
    if (!yolo_run_conv2d_tiled_desc(ACT_DDR, WGT_DDR, WGT_BASE, OUT_DDR, PAD_ROW_DDR,
                                    YOLO_SILU_REAL_IN_W, YOLO_SILU_REAL_IN_H,
                                    YOLO_SILU_REAL_IC, YOLO_SILU_REAL_OC,
                                    1u, 1u, 1u, 0u,
                                    yolo_silu_real_bias, yolo_silu_real_scale_mul,
                                    yolo_silu_real_scale_shift,
                                    NPU_CTRL_SILU_EN,
                                    YOLO_SILU_REAL_IC / 16u, 16u, 0,
                                    0u, 0u, 0)) {
        print_str("  desc pw run failed\n"); errors++;
    }

    for (pos = 0u; pos < YOLO_SILU_REAL_OUT_WORDS; pos++)
        for (oc = 0u; oc < YOLO_SILU_REAL_OC; oc++) {
            uint32_t got = rb(OUT_DDR, pos, oc);
            uint32_t exp = yolo_silu_real_expected[pos][oc];
            if (got != exp) {
                errors++;
                print_str("  mismatch pos="); print_dec(pos);
                print_str(" oc="); print_dec(oc);
                print_str(" got=0x"); print_hex(got, 2);
                print_str(" exp=0x"); print_hex(exp, 2); print_str("\n");
            }
        }

    if (errors == 0u) { print_str("YOLO PWCONV DESC SMOKE PASS\n"); return; }
    print_str("YOLO PWCONV DESC SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
