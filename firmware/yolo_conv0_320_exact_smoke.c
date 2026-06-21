// conv0 (model.0, 3->16 3x3 s2 pad1) at FULL 320 resolution with the EXACT
// per-layer SiLU LUT (CTRL[22]). Input = the real 320x320x3 image quantized like
// the C engine (q=pixel-128), 3 channels packed into 16 lanes (lanes 3..15 zero-
// weighted). conv0 is stride-2 (serial tiled path, row_par auto-off) and the
// second-worst LUT saturator; its exact golden aligns to dump320/conv0.bin within
// +-2. Proves the front-of-net (image quant + stem conv0) on-SoC.

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_conv0_320_exact_data.h"
#include <stdint.h>

#define ACT_DDR     0x40400000u
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

    print_str("YOLO CONV0 @320 EXACT-SiLU SMOKE (320x320x3 -> 160x160x16)\n");
    for (i = 0u; i < C0E_ACT_WORDS; i++) wrw(ACT_DDR, i, yolo_conv0_320e_act_words[i]);
    for (i = 0u; i < C0E_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_conv0_320e_wgt_words[i]);

    yolo_set_pad_value(C0E_PAD_VALUE);
    yolo_load_silu_lut(yolo_conv0_320e_silu_lut);
    yolo_set_silu_requant(0u, 0u, C0E_OUT_ZP);

    *(volatile uint32_t *)NPU_PERF_CLR = 1u;

    if (!yolo_run_conv2d_tiled(ACT_DDR, WGT_DDR, WGT_BASE, OUT_DDR, PAD_ROW_DDR,
                               C0E_IN_W, C0E_IN_H, C0E_IC, C0E_OC,
                               3u, 3u, 2u, 1u,
                               yolo_conv0_320e_bias_q, yolo_conv0_320e_scale_mul,
                               yolo_conv0_320e_scale_shift,
                               NPU_CTRL_SILU_EXACT_EN,
                               C0E_WGT_PER_OC, STRIP_OUT_ROWS, C0E_PAD_VALUE)) {
        print_str("  tiled run failed\n"); errors++;
    }

    print_str("PERF cyc_total="); print_dec(rdp(NPU_PERF_CYC_TOTAL));
    print_str(" npu_busy=");      print_dec(rdp(NPU_PERF_CYC_BUSY));
    print_str("\n");

#ifdef C0E_HAVE_GOLDEN
    {
        uint32_t dbg_err = 0u;
        for (pos = 0u; pos < C0E_OUT_SPATIAL; pos++)
            for (oc = 0u; oc < C0E_OC; oc++) {
                int32_t got = rs8(OUT_DDR, pos, oc);
                int32_t exp = yolo_conv0_320e_golden[pos][oc]; exp = (exp&0x80)?exp-256:exp;
                int32_t d = got-exp; if (d<0) d=-d;
                if ((uint32_t)d > C0E_RTL_TOL) {
                    errors++;
                    if (dbg_err++ < 12u) {
                        print_str("  pos="); print_dec(pos);
                        print_str(" oc="); print_dec(oc);
                        print_str(" got="); print_dec((uint32_t)got);
                        print_str(" exp="); print_dec((uint32_t)exp); print_str("\n");
                    }
                }
            }
        if (errors == 0u) { print_str("YOLO CONV0 @320 EXACT-SiLU SMOKE PASS\n"); return; }
        print_str("YOLO CONV0 @320 EXACT-SiLU SMOKE FAIL errors="); print_dec(errors); print_str("\n");
        __asm__ volatile ("ebreak");
    }
#endif

    /* Position-weighted checksum over the whole output (the 1.6MB image leaves no
     * room to bake the golden tensor; per-element exactness is proven on conv1). */
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

    if (errors == 0u) { print_str("YOLO CONV0 @320 EXACT-SiLU SMOKE PASS\n"); return; }
    print_str("YOLO CONV0 @320 EXACT-SiLU SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
