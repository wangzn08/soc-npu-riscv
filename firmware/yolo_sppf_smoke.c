// SPPF (model.9): conv25(256->128) -> 3x MaxPool5x5(s1,p2) -> concat(512) ->
// conv26(512->256). conv25/26 on NPU (OC-chunked); maxpool + concat on CPU.
// Input = c2f_8 output (baked).

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_sppf_data.h"
#include <stdint.h>

#define IN_DDR   0x40090000u
#define CV1_DDR  0x400C0000u
#define M0_DDR   0x400E0000u
#define M1_DDR   0x40100000u
#define M2_DDR   0x40120000u
#define CAT_DDR  0x40140000u
#define OUT_DDR  0x40180000u
#define WGT_DDR  0x401C0000u
#define ACT_BASE 0u
#define WGT_BASE 0u

#define H YOLO_SPPF_IN_H
#define W YOLO_SPPF_IN_W
#define SP YOLO_SPPF_SPATIAL
#define C25_GROUPS (YOLO_SPPF_C25_OC / 16u)  /* 8 */

static void wrw(uint32_t a, uint32_t w, const uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); p[0]=l[0];p[1]=l[1];p[2]=l[2];p[3]=l[3]; }
static void rdw(uint32_t a, uint32_t w, uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); l[0]=p[0];l[1]=p[1];l[2]=p[2];l[3]=p[3]; }
static int32_t rs8(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); uint32_t v=p[b>>2]>>((b&3u)*8u)&0xFFu; return (v&0x80u)?((int32_t)v-256):(int32_t)v; }
static int32_t s8(uint32_t b){ b&=0xFFu; return (b&0x80u)?((int32_t)b-256):(int32_t)b; }
static uint32_t ad(int32_t a,int32_t b){ int32_t d=a-b; return (uint32_t)(d<0?-d:d); }

// 5x5 stride-1 pad-2 int8 max-pool on tile-major DDR (groups of 16 channels).
static void maxpool5(uint32_t src, uint32_t dst, uint32_t groups)
{
    uint32_t g, oh, ow, kh, kw, k;
    for (g = 0u; g < groups; g++)
        for (oh = 0u; oh < H; oh++)
            for (ow = 0u; ow < W; ow++) {
                int32_t mx[16];
                uint32_t o[4] = {0u,0u,0u,0u};
                for (k = 0u; k < 16u; k++) mx[k] = -128;
                for (kh = 0u; kh < 5u; kh++)
                    for (kw = 0u; kw < 5u; kw++) {
                        int32_t ih = (int32_t)oh - 2 + (int32_t)kh;
                        int32_t iw = (int32_t)ow - 2 + (int32_t)kw;
                        if (ih >= 0 && ih < (int32_t)H && iw >= 0 && iw < (int32_t)W) {
                            uint32_t in[4];
                            rdw(src, g * SP + (uint32_t)ih * W + (uint32_t)iw, in);
                            for (k = 0u; k < 16u; k++) {
                                int32_t v = s8(in[k>>2] >> ((k&3u)*8u));
                                if (v > mx[k]) mx[k] = v;
                            }
                        }
                    }
                for (k = 0u; k < 16u; k++)
                    o[k>>2] |= ((uint32_t)(mx[k] & 0xFF)) << ((k&3u)*8u);
                wrw(dst, g * SP + oh * W + ow, o);
            }
}

// concat 4 same-scale parts (tile-major) into one [4*groups] tensor.
static void concat4(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t dst, uint32_t groups)
{
    uint32_t srcs[4]; uint32_t i, n;
    srcs[0]=a; srcs[1]=b; srcs[2]=c; srcs[3]=d;
    for (i = 0u; i < 4u; i++)
        for (n = 0u; n < groups * SP; n++) {
            uint32_t w[4]; rdw(srcs[i], n, w);
            wrw(dst, i * groups * SP + n, w);
        }
}

void usercode7(void)
{
    uint32_t i, pos, oc, errors = 0u;

    print_str("YOLO SPPF (model.9) CPU SMOKE\n");
    for (i = 0u; i < YOLO_SPPF_BLKIN_WORDS; i++) wrw(IN_DDR, i, yolo_sppf_blkin_words[i]);
    for (i = 0u; i < YOLO_SPPF_C25_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_sppf_c25_wgt[i]);

    // cv1 = conv25 (1x1 256->128, OC-chunked)
    yolo_set_silu_requant(YOLO_SPPF_C25_RQ_MUL, YOLO_SPPF_C25_RQ_SHIFT, YOLO_SPPF_C25_RQ_ZP);
    if (!yolo_dma_ddr_to_act(IN_DDR, ACT_BASE, SP * (YOLO_SPPF_C25_IC/16u)) ||
        !yolo_run_pw_conv1x1_oc_chunks(ACT_BASE, WGT_DDR, WGT_BASE, CV1_DDR,
                                       W, H, YOLO_SPPF_C25_IC, YOLO_SPPF_C25_OC,
                                       yolo_sppf_c25_bias, yolo_sppf_c25_mul, yolo_sppf_c25_shift,
                                       NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN, SP)) {
        print_str("  conv25 failed\n"); errors++;
    }

    if (errors == 0u) {
        maxpool5(CV1_DDR, M0_DDR, C25_GROUPS);
        maxpool5(M0_DDR, M1_DDR, C25_GROUPS);
        maxpool5(M1_DDR, M2_DDR, C25_GROUPS);
        concat4(CV1_DDR, M0_DDR, M1_DDR, M2_DDR, CAT_DDR, C25_GROUPS);
    }

    // cv2 = conv26 (1x1 512->256, OC-chunked)
    if (errors == 0u) {
        for (i = 0u; i < YOLO_SPPF_C26_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_sppf_c26_wgt[i]);
        yolo_set_silu_requant(YOLO_SPPF_C26_RQ_MUL, YOLO_SPPF_C26_RQ_SHIFT, YOLO_SPPF_C26_RQ_ZP);
        if (!yolo_dma_ddr_to_act(CAT_DDR, ACT_BASE, SP * (YOLO_SPPF_C26_IC/16u)) ||
            !yolo_run_pw_conv1x1_oc_chunks(ACT_BASE, WGT_DDR, WGT_BASE, OUT_DDR,
                                           W, H, YOLO_SPPF_C26_IC, YOLO_SPPF_C26_OC,
                                           yolo_sppf_c26_bias, yolo_sppf_c26_mul, yolo_sppf_c26_shift,
                                           NPU_CTRL_SILU_EN | NPU_CTRL_SILU_REQUANT_EN, SP)) {
            print_str("  conv26 failed\n"); errors++;
        }
    }

    for (pos = 0u; pos < SP && errors <= 16u; pos++)
        for (oc = 0u; oc < YOLO_SPPF_C26_OC; oc++) {
            int32_t got = rs8(OUT_DDR, (oc>>4)*SP + pos, oc&15u);
            int32_t exp = s8(yolo_sppf_expected_rtl[pos][oc]);
            if (ad(got, exp) > YOLO_SPPF_RTL_TOL) {
                errors++;
                print_str("  mismatch pos="); print_dec(pos);
                print_str(" oc="); print_dec(oc);
                print_str(" got="); print_dec(got);
                print_str(" exp="); print_dec(exp); print_str("\n");
                if (errors > 16u) break;
            }
        }

    if (errors == 0u) { print_str("YOLO SPPF CPU SMOKE PASS\n"); return; }
    print_str("YOLO SPPF CPU SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
