// c2f_15 (model.15, PAN p3, n=1, shortcut=0) via the generic C2f runner.
// Input = cat2 (upsample(c2f_12 out)+concat(p4), baked).

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_c2f.h"
#include "yolo_c2f15_data.h"
#include <stdint.h>

#define PAD_ROW    0x40080000u
#define IN_DDR     0x40090000u
#define CV1_OUT    0x40120000u
#define BN_OUT     0x40160000u
#define MCV2_DDR   0x40180000u
#define ADD0_DDR   0x401A0000u
#define CONCAT_DDR 0x401C0000u
#define OUT_DDR    0x40220000u
#define WGT_DDR    0x40260000u

static void wrw(uint32_t a, uint32_t w, const uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); p[0]=l[0];p[1]=l[1];p[2]=l[2];p[3]=l[3]; }
static int32_t rs8(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); uint32_t v=p[b>>2]>>((b&3u)*8u)&0xFFu; return (v&0x80u)?((int32_t)v-256):(int32_t)v; }
static int32_t s8(uint32_t b){ b&=0xFFu; return (b&0x80u)?((int32_t)b-256):(int32_t)b; }
static uint32_t ad(int32_t a,int32_t b){ int32_t d=a-b; return (uint32_t)(d<0?-d:d); }

void usercode7(void)
{
    uint32_t i, pos, oc, errors = 0u;
    yolo_c2f_cfg_t cfg;

    print_str("YOLO C2F15 (model.15, PAN p3, shortcut=0) CPU SMOKE\n");
    for (i = 0u; i < YOLO_C2F15_BLKIN_WORDS; i++) wrw(IN_DDR, i, yolo_c2f15_blkin_words[i]);

    cfg.in_w=YOLO_C2F15_IN_W; cfg.in_h=YOLO_C2F15_IN_H; cfg.spatial=YOLO_C2F15_SPATIAL;
    cfg.full_c=YOLO_C2F15_FULL_C; cfg.n_bottleneck=1u; cfg.shortcut=0u;
    cfg.in_ddr=IN_DDR; cfg.cv1_ic=YOLO_C2F15_CV1_IC; cfg.cv1_out_ddr=CV1_OUT;
    cfg.bn_out_ddr=BN_OUT; cfg.mcv2_ddr=MCV2_DDR; cfg.add_ddr[0]=ADD0_DDR;
    cfg.concat_ddr=CONCAT_DDR; cfg.out_ddr=OUT_DDR; cfg.wgt_ddr=WGT_DDR;
    cfg.pad_row_ddr=PAD_ROW; cfg.strip=16u; cfg.silu_exact=0u; cfg.wgt_in_blob=0u;

    cfg.cv1_wgt=yolo_c2f15_cv1_wgt; cfg.cv1_wgt_words=YOLO_C2F15_CV1_WGT_WORDS;
    cfg.cv1_bias=yolo_c2f15_cv1_bias; cfg.cv1_mul=yolo_c2f15_cv1_mul; cfg.cv1_shift=yolo_c2f15_cv1_shift;
    cfg.cv1_rq_mul=YOLO_C2F15_CV1_RQ_MUL; cfg.cv1_rq_shift=YOLO_C2F15_CV1_RQ_SHIFT; cfg.cv1_rq_zp=YOLO_C2F15_CV1_RQ_ZP;

    cfg.mcv1_wgt_words=YOLO_C2F15_MCV1_WGT_WORDS;
    cfg.mcv1_wgt[0]=yolo_c2f15_mcv1_0_wgt; cfg.mcv1_bias[0]=yolo_c2f15_mcv1_0_bias;
    cfg.mcv1_mul[0]=yolo_c2f15_mcv1_0_mul; cfg.mcv1_shift[0]=yolo_c2f15_mcv1_0_shift;
    cfg.mcv1_rq_mul[0]=YOLO_C2F15_MCV1_0_RQ_MUL; cfg.mcv1_rq_shift[0]=YOLO_C2F15_GLUE_RQ_SHIFT;
    cfg.mcv1_rq_zp[0]=YOLO_C2F15_MCV1_0_RQ_ZP; cfg.mcv1_pad_value[0]=YOLO_C2F15_MCV1_0_PAD;

    cfg.mcv2_wgt_words=YOLO_C2F15_MCV2_WGT_WORDS;
    cfg.mcv2_wgt[0]=yolo_c2f15_mcv2_0_wgt; cfg.mcv2_bias[0]=yolo_c2f15_mcv2_0_bias;
    cfg.mcv2_mul[0]=yolo_c2f15_mcv2_0_mul; cfg.mcv2_shift[0]=yolo_c2f15_mcv2_0_shift;
    cfg.mcv2_pad_value[0]=YOLO_C2F15_MCV2_0_PAD;

    cfg.glue_rq_mul[0]=YOLO_C2F15_GLUE0_RQ_MUL; cfg.glue_rq_shift[0]=YOLO_C2F15_GLUE_RQ_SHIFT; cfg.glue_zp[0]=YOLO_C2F15_GLUE0_ZP;
    cfg.add_ratio_shift=0u; cfg.add_ratio_mul[0]=0u; cfg.add_prev_zp[0]=0;

    cfg.cat_req_shift=YOLO_C2F15_CAT_SHIFT; cfg.cat_zp=YOLO_C2F15_CAT_ZP;
    cfg.cat_mul_s0s1=YOLO_C2F15_CAT_MUL_S0S1; cfg.cat_inzp_s0s1=YOLO_C2F15_CAT_INZP_S0S1;
    cfg.cat_mul_add[0]=YOLO_C2F15_CAT_MUL_ADD0; cfg.cat_inzp_add[0]=YOLO_C2F15_CAT_INZP_ADD0;

    cfg.cv2_ic=YOLO_C2F15_CV2_IC; cfg.cv2_oc=YOLO_C2F15_CV2_OC;
    cfg.cv2_wgt=yolo_c2f15_cv2_wgt; cfg.cv2_wgt_words=YOLO_C2F15_CV2_WGT_WORDS;
    cfg.cv2_bias=yolo_c2f15_cv2_bias; cfg.cv2_mul=yolo_c2f15_cv2_mul; cfg.cv2_shift=yolo_c2f15_cv2_shift;
    cfg.cv2_rq_mul=YOLO_C2F15_CV2_RQ_MUL; cfg.cv2_rq_shift=YOLO_C2F15_CV2_RQ_SHIFT; cfg.cv2_rq_zp=YOLO_C2F15_CV2_RQ_ZP;

    if (!yolo_run_c2f_block(&cfg)) { print_str("  c2f15 runner failed\n"); errors++; }

    for (pos = 0u; pos < YOLO_C2F15_SPATIAL && errors <= 16u; pos++)
        for (oc = 0u; oc < YOLO_C2F15_CV2_OC; oc++) {
            int32_t got = rs8(OUT_DDR, (oc>>4)*YOLO_C2F15_SPATIAL + pos, oc&15u);
            int32_t exp = s8(yolo_c2f15_expected_rtl[pos][oc]);
            if (ad(got, exp) > YOLO_C2F15_RTL_TOL) {
                errors++;
                print_str("  mismatch pos="); print_dec(pos);
                print_str(" oc="); print_dec(oc);
                print_str(" got="); print_dec(got);
                print_str(" exp="); print_dec(exp); print_str("\n");
                if (errors > 16u) break;
            }
        }

    if (errors == 0u) { print_str("YOLO C2F15 CPU SMOKE PASS\n"); return; }
    print_str("YOLO C2F15 CPU SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
