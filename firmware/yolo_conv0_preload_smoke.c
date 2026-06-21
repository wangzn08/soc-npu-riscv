// Verifies the YOLO DDR image preload (+define+YOLO_DDR). The 320x320x3 image is
// $readmemh-loaded into DDR at 0x4040_0000 by the shared-memory model, so this
// firmware does NOT bake or write the 1.6MB image -- it only loads the small conv0
// weights and runs conv0 directly on the preloaded DDR image, checking the same
// position-weighted checksum as yolo_conv0_320_exact_smoke.c.
//
// Build: YOLO_DDR=1 bash run_all.sh sim yolo_conv0_preload_smoke.c yolo_ops.c

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_conv0_320_noact_data.h"
#include <stdint.h>

#define ACT_DDR     0x40400000u   // preloaded image (DDR word base 0x40000)
#define WGT_DDR     0x405C0000u
#define OUT_DDR     0x40600000u
#define PAD_ROW_DDR 0x40780000u
#define WGT_BASE    0u
#define STRIP_OUT_ROWS 16u

static void wrw(uint32_t a, uint32_t w, const uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); p[0]=l[0];p[1]=l[1];p[2]=l[2];p[3]=l[3]; }
static int32_t rs8(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); uint32_t v=p[b>>2]>>((b&3u)*8u)&0xFFu; return (v&0x80u)?((int32_t)v-256):(int32_t)v; }
static uint32_t rdp(uint32_t a){ return *(volatile uint32_t*)a; }

void usercode7(void)
{
    uint32_t i, pos, oc, errors = 0u;

    print_str("YOLO CONV0 DDR-PRELOAD SMOKE (image $readmemh'd into DDR)\n");
    // Sanity: first preloaded image word should equal the baked act_words[0].
    if (rdp(ACT_DDR) != C0E_ACT_WORD0_0) {
        print_str("  preload word0 mismatch got=0x"); print_hex(rdp(ACT_DDR), 8); print_str("\n");
        errors++;
    }
    for (i = 0u; i < C0E_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_conv0_320e_wgt_words[i]);

    yolo_set_pad_value(C0E_PAD_VALUE);
    yolo_load_silu_lut(yolo_conv0_320e_silu_lut);
    yolo_set_silu_requant(0u, 0u, C0E_OUT_ZP);

    if (!yolo_run_conv2d_tiled(ACT_DDR, WGT_DDR, WGT_BASE, OUT_DDR, PAD_ROW_DDR,
                               C0E_IN_W, C0E_IN_H, C0E_IC, C0E_OC,
                               3u, 3u, 2u, 1u,
                               yolo_conv0_320e_bias_q, yolo_conv0_320e_scale_mul,
                               yolo_conv0_320e_scale_shift,
                               NPU_CTRL_SILU_EXACT_EN,
                               C0E_WGT_PER_OC, STRIP_OUT_ROWS, C0E_PAD_VALUE)) {
        print_str("  tiled run failed\n"); errors++;
    }

    {
        uint32_t chk = 0u, idx = 0u;
        for (pos = 0u; pos < C0E_OUT_SPATIAL; pos++)
            for (oc = 0u; oc < C0E_OC; oc++) {
                int32_t got = rs8(OUT_DDR, (oc>>4)*C0E_OUT_SPATIAL + pos, oc&15u);
                chk += ((uint32_t)(got + 128) * (idx + 1u));
                idx++;
            }
        if (chk != C0E_GOLDEN_CHK) {
            errors++;
            print_str("  checksum mismatch got=0x"); print_hex(chk, 8);
            print_str(" exp=0x"); print_hex(C0E_GOLDEN_CHK, 8); print_str("\n");
        }
    }

    if (errors == 0u) { print_str("YOLO CONV0 DDR-PRELOAD SMOKE PASS\n"); return; }
    print_str("YOLO CONV0 DDR-PRELOAD SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
