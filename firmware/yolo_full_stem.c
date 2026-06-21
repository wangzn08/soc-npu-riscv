// yolo_full.c stem: FULL on-SoC stem chain conv0 -> conv1 -> c2f_2, all through
// DDR with exact-SiLU. conv0 reads the DDR-PRELOADED 320x320x3 image (no bake;
// requires `touch .yolo_ddr` so the shared-mem model $readmemh's the image into
// 0x4040_0000). Each layer's RTL output feeds the next directly (scale/zp/tile-
// major layout match by construction). Final compared to the c2f_2 golden with a
// propagation-aware tolerance (conv0/conv1 quantization +-1..2 propagates).
//
// Build: touch .yolo_ddr; bash run_all.sh sim yolo_full_stem.c yolo_c2f.c yolo_ops.c

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_c2f.h"
#include "yolo_conv0_320_noact_data.h"
#include "yolo_conv1_320_exact_data.h"
#include "yolo_c2f2_320_data.h"
#include <stdint.h>

#define IMG       0x40400000u   // preloaded image (conv0 input)
#define C0_WGT    0x405C0000u
#define C0_OUT    0x40600000u   // = conv1 input
#define C1_WGT    0x40680000u
#define C1_OUT    0x406C0000u   // = c2f_2 input
#define CV1_OUT   0x40700000u
#define BN_OUT    0x40740000u
#define MCV2_DDR  0x40760000u
#define ADD0_DDR  0x40780000u
#define CONCAT    0x407A0000u
#define C2F_OUT   0x40820000u
#define C2F_WGT   0x40880000u
#define PAD_ROW   0x408C0000u
#define WGT_BASE  0u

static void wrw(uint32_t a, uint32_t w, const uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); p[0]=l[0];p[1]=l[1];p[2]=l[2];p[3]=l[3]; }
static int32_t rs8(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); uint32_t v=p[b>>2]>>((b&3u)*8u)&0xFFu; return (v&0x80u)?((int32_t)v-256):(int32_t)v; }
static int32_t s8(uint32_t b){ b&=0xFFu; return (b&0x80u)?((int32_t)b-256):(int32_t)b; }
static uint32_t ad(int32_t a,int32_t b){ int32_t d=a-b; return (uint32_t)(d<0?-d:d); }

void usercode7(void)
{
    uint32_t i, pos, oc, errors = 0u, maxd = 0u;
    yolo_c2f_cfg_t cfg;

    print_str("YOLO FULL STEM: conv0 -> conv1 -> c2f_2 (on-SoC, DDR, preloaded img)\n");

    // ---------- Stage 0: conv0 (preloaded image) -> C0_OUT ----------
    for (i = 0u; i < C0E_WGT_WORDS; i++) wrw(C0_WGT, i, yolo_conv0_320e_wgt_words[i]);
    yolo_set_pad_value(C0E_PAD_VALUE);
    yolo_load_silu_lut(yolo_conv0_320e_silu_lut);
    yolo_set_silu_requant(0u, 0u, C0E_OUT_ZP);
    if (!yolo_run_conv2d_tiled(IMG, C0_WGT, WGT_BASE, C0_OUT, PAD_ROW,
                               C0E_IN_W, C0E_IN_H, C0E_IC, C0E_OC,
                               3u, 3u, 2u, 1u,
                               yolo_conv0_320e_bias_q, yolo_conv0_320e_scale_mul,
                               yolo_conv0_320e_scale_shift,
                               NPU_CTRL_SILU_EXACT_EN, C0E_WGT_PER_OC, 16u, C0E_PAD_VALUE)) {
        print_str("  conv0 fail\n"); errors++;
    }

    // ---------- Stage 1: conv1 reads C0_OUT -> C1_OUT ----------
    for (i = 0u; i < C1E_WGT_WORDS; i++) wrw(C1_WGT, i, yolo_conv1_320e_wgt_words[i]);
    yolo_set_pad_value(C1E_PAD_VALUE);
    yolo_load_silu_lut(yolo_conv1_320e_silu_lut);
    yolo_set_silu_requant(0u, 0u, C1E_OUT_ZP);
    if (errors == 0u && !yolo_run_conv2d_tiled(C0_OUT, C1_WGT, WGT_BASE, C1_OUT, PAD_ROW,
                               C1E_IN_W, C1E_IN_H, C1E_IC, C1E_OC,
                               3u, 3u, 2u, 1u,
                               yolo_conv1_320e_bias_q, yolo_conv1_320e_scale_mul,
                               yolo_conv1_320e_scale_shift,
                               NPU_CTRL_SILU_EXACT_EN, C1E_WGT_PER_OC, 16u, C1E_PAD_VALUE)) {
        print_str("  conv1 fail\n"); errors++;
    }

    // ---------- Stage 2: c2f_2 reads C1_OUT ----------
    cfg.in_w=YOLO_C2F2_IN_W; cfg.in_h=YOLO_C2F2_IN_H; cfg.spatial=YOLO_C2F2_SPATIAL;
    cfg.full_c=YOLO_C2F2_FULL_C; cfg.n_bottleneck=1u; cfg.shortcut=1u;
    cfg.in_ddr=C1_OUT; cfg.cv1_ic=YOLO_C2F2_CV1_IC; cfg.cv1_out_ddr=CV1_OUT;
    cfg.bn_out_ddr=BN_OUT; cfg.mcv2_ddr=MCV2_DDR; cfg.add_ddr[0]=ADD0_DDR;
    cfg.concat_ddr=CONCAT; cfg.out_ddr=C2F_OUT; cfg.wgt_ddr=C2F_WGT;
    cfg.pad_row_ddr=PAD_ROW; cfg.strip=16u; cfg.silu_exact=1u;
    cfg.cv1_wgt=yolo_c2f2_cv1_wgt; cfg.cv1_wgt_words=YOLO_C2F2_CV1_WGT_WORDS;
    cfg.cv1_bias=yolo_c2f2_cv1_bias; cfg.cv1_mul=yolo_c2f2_cv1_mul; cfg.cv1_shift=yolo_c2f2_cv1_shift;
    cfg.cv1_rq_mul=YOLO_C2F2_CV1_RQ_MUL; cfg.cv1_rq_shift=YOLO_C2F2_CV1_RQ_SHIFT; cfg.cv1_rq_zp=YOLO_C2F2_CV1_RQ_ZP;
    cfg.cv1_silu_lut=yolo_c2f2_cv1_silu_lut;
    cfg.mcv1_wgt_words=YOLO_C2F2_MCV1_WGT_WORDS;
    cfg.mcv1_wgt[0]=yolo_c2f2_mcv1_0_wgt; cfg.mcv1_bias[0]=yolo_c2f2_mcv1_0_bias;
    cfg.mcv1_mul[0]=yolo_c2f2_mcv1_0_mul; cfg.mcv1_shift[0]=yolo_c2f2_mcv1_0_shift;
    cfg.mcv1_rq_mul[0]=YOLO_C2F2_MCV1_0_RQ_MUL; cfg.mcv1_rq_shift[0]=YOLO_C2F2_GLUE_RQ_SHIFT;
    cfg.mcv1_rq_zp[0]=YOLO_C2F2_MCV1_0_RQ_ZP; cfg.mcv1_pad_value[0]=YOLO_C2F2_MCV1_0_PAD;
    cfg.mcv1_silu_lut[0]=yolo_c2f2_mcv1_0_silu_lut;
    cfg.mcv2_wgt_words=YOLO_C2F2_MCV2_WGT_WORDS;
    cfg.mcv2_wgt[0]=yolo_c2f2_mcv2_0_wgt; cfg.mcv2_bias[0]=yolo_c2f2_mcv2_0_bias;
    cfg.mcv2_mul[0]=yolo_c2f2_mcv2_0_mul; cfg.mcv2_shift[0]=yolo_c2f2_mcv2_0_shift;
    cfg.mcv2_pad_value[0]=YOLO_C2F2_MCV2_0_PAD;
    cfg.mcv2_silu_lut[0]=yolo_c2f2_mcv2_0_silu_lut;
    cfg.glue_rq_mul[0]=YOLO_C2F2_GLUE0_RQ_MUL; cfg.glue_rq_shift[0]=YOLO_C2F2_GLUE_RQ_SHIFT; cfg.glue_zp[0]=YOLO_C2F2_GLUE0_ZP;
    cfg.add_ratio_shift=YOLO_C2F2_ADD_RATIO_SHIFT;
    cfg.add_ratio_mul[0]=YOLO_C2F2_ADD0_RATIO_MUL; cfg.add_prev_zp[0]=YOLO_C2F2_ADD0_PREV_ZP;
    cfg.cat_req_shift=YOLO_C2F2_CAT_SHIFT; cfg.cat_zp=YOLO_C2F2_CAT_ZP;
    cfg.cat_mul_s0s1=YOLO_C2F2_CAT_MUL_S0S1; cfg.cat_inzp_s0s1=YOLO_C2F2_CAT_INZP_S0S1;
    cfg.cat_mul_add[0]=YOLO_C2F2_CAT_MUL_ADD0; cfg.cat_inzp_add[0]=YOLO_C2F2_CAT_INZP_ADD0;
    cfg.cv2_ic=YOLO_C2F2_CV2_IC; cfg.cv2_oc=YOLO_C2F2_CV2_OC;
    cfg.cv2_wgt=yolo_c2f2_cv2_wgt; cfg.cv2_wgt_words=YOLO_C2F2_CV2_WGT_WORDS;
    cfg.cv2_bias=yolo_c2f2_cv2_bias; cfg.cv2_mul=yolo_c2f2_cv2_mul; cfg.cv2_shift=yolo_c2f2_cv2_shift;
    cfg.cv2_rq_mul=YOLO_C2F2_CV2_RQ_MUL; cfg.cv2_rq_shift=YOLO_C2F2_CV2_RQ_SHIFT; cfg.cv2_rq_zp=YOLO_C2F2_CV2_RQ_ZP;
    cfg.cv2_silu_lut=yolo_c2f2_cv2_silu_lut;
    if (errors == 0u && !yolo_run_c2f_block(&cfg)) { print_str("  c2f2 fail\n"); errors++; }

    // ---------- validate full-stem c2f_2 output vs golden ----------
    for (pos = 0u; pos < YOLO_C2F2_SPATIAL && errors <= 16u; pos++)
        for (oc = 0u; oc < YOLO_C2F2_CV2_OC; oc++) {
            int32_t got = rs8(C2F_OUT, (oc>>4)*YOLO_C2F2_SPATIAL + pos, oc&15u);
            int32_t exp = s8(yolo_c2f2_expected_rtl[pos][oc]);
            uint32_t d = ad(got, exp);
            if (d > maxd) maxd = d;
            if (d > 32u) {   /* conv0+conv1 +-1..2 propagate through c2f_2 */
                errors++;
                if (errors <= 8u) {
                    print_str("  pos="); print_dec(pos); print_str(" oc="); print_dec(oc);
                    print_str(" got="); print_dec((uint32_t)got); print_str(" exp="); print_dec((uint32_t)exp); print_str("\n");
                }
            }
        }

    print_str("full-stem c2f_2 vs golden maxdiff="); print_dec(maxd); print_str("\n");
    if (errors == 0u) { print_str("YOLO FULL STEM PASS (conv0->conv1->c2f_2 chained)\n"); return; }
    print_str("YOLO FULL STEM FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
