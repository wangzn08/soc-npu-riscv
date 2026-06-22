// conv6 (model.3 downsample, 32->64 3x3 s2) standalone @320 with exact SiLU.
// Input = dump conv5 (c2f_2 output, baked); WEIGHTS come from the DDR blob
// (WGT_OF(6), no bake). Validates the general per-conv exact generator + DDR
// weight blob for a new backbone layer via the position-weighted checksum.
//
// Build: touch .yolo_ddr; bash run_all.sh sim yolo_conv6_exact_smoke.c yolo_ops.c

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_conv6_exact_data.h"
#include "yolo_weight_map.h"
#include <stdint.h>

#define ACT_DDR     0x40090000u
#define OUT_DDR     0x40200000u
#define PAD_ROW_DDR 0x40300000u
#define WGT_BASE    0u
#define WGT_OF(ci) (YOLO_WGT_DDR_BASE + yolo_wgt_map[ci].off * 16u)

static void wrw(uint32_t a, uint32_t w, const uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); p[0]=l[0];p[1]=l[1];p[2]=l[2];p[3]=l[3]; }
static int32_t rs8(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); uint32_t v=p[b>>2]>>((b&3u)*8u)&0xFFu; return (v&0x80u)?((int32_t)v-256):(int32_t)v; }
static uint32_t rdp(uint32_t a){ return *(volatile uint32_t*)a; }

void usercode7(void)
{
    uint32_t i, pos, oc, errors = 0u;

    print_str("YOLO CONV6 EXACT SMOKE (80x80x32 -> 40x40x64, weights from DDR blob)\n");
    for (i = 0u; i < C6E_ACT_WORDS; i++) wrw(ACT_DDR, i, yolo_conv6e_act_words[i]);

    yolo_set_pad_value(C6E_PAD_VALUE);
    yolo_load_silu_lut(yolo_conv6e_silu_lut);
    yolo_set_silu_requant(0u, 0u, 0);

    if (!yolo_run_conv2d_tiled(ACT_DDR, WGT_OF(6), WGT_BASE, OUT_DDR, PAD_ROW_DDR,
                               C6E_IN_W, C6E_IN_H, C6E_IC, C6E_OC,
                               3u, 3u, C6E_STRIDE, 1u,
                               yolo_conv6e_bias_q, yolo_conv6e_scale_mul,
                               yolo_conv6e_scale_shift,
                               NPU_CTRL_SILU_EXACT_EN, C6E_WGT_PER_OC, 16u, C6E_PAD_VALUE)) {
        print_str("  tiled run failed\n"); errors++;
    }

    {
        uint32_t chk = 0u, idx = 0u;
        for (pos = 0u; pos < C6E_OUT_SPATIAL; pos++)
            for (oc = 0u; oc < C6E_OC; oc++) {
                int32_t got = rs8(OUT_DDR, (oc>>4)*C6E_OUT_SPATIAL + pos, oc&15u);
                chk += ((uint32_t)(got + 128) * (idx + 1u));
                idx++;
            }
        if (chk != C6E_GOLDEN_CHK) {
            errors++;
            print_str("  checksum mismatch got=0x"); print_hex(chk, 8);
            print_str(" exp=0x"); print_hex(C6E_GOLDEN_CHK, 8); print_str("\n");
        }
    }
    (void)rdp;

    if (errors == 0u) { print_str("YOLO CONV6 EXACT SMOKE PASS\n"); return; }
    print_str("YOLO CONV6 EXACT SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
