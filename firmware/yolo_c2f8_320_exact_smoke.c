// c2f_8 (model.8, n=1, 256ch) @320 standalone, exact-SiLU + baked weights.
// cv1 IC=256 (icg=16), cv2 IC=384 (icg=24): exercises >16 IC-group PW streaming.
// Per-stage checksums (CV1/ADD0/CONCAT/OUT) localize the first diverging stage.

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_c2f.h"
#include "yolo_c2f8_exact_data.h"
#include <stdint.h>

#define PAD_ROW    0x40080000u
#define IN_DDR     0x40090000u
#define CV1_OUT    0x40140000u
#define BN_OUT     0x40180000u
#define MCV2_DDR   0x401A0000u
#define ADD0_DDR   0x401C0000u
#define ADD1_DDR   0x401E0000u
#define CONCAT_DDR 0x40200000u
#define OUT_DDR    0x40280000u
#define WGT_DDR    0x402C0000u

static void wrw(uint32_t a, uint32_t w, const uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); p[0]=l[0];p[1]=l[1];p[2]=l[2];p[3]=l[3]; }
static int32_t rs8(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); uint32_t v=p[b>>2]>>((b&3u)*8u)&0xFFu; return (v&0x80u)?((int32_t)v-256):(int32_t)v; }
static uint32_t rub(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); return (p[b>>2]>>((b&3u)*8u))&0xFFu; }
static int32_t s8(uint32_t b){ b&=0xFFu; return (b&0x80u)?((int32_t)b-256):(int32_t)b; }
static uint32_t ad(int32_t a,int32_t b){ int32_t d=a-b; return (uint32_t)(d<0?-d:d); }

static void stage_ck(const char *nm, uint32_t base, uint32_t C, uint32_t sp,
                     uint32_t gsum, uint32_t ghash)
{
    uint32_t ch, pos, sum=0u, hash=0u;
    for (ch=0u; ch<C; ch++)
        for (pos=0u; pos<sp; pos++) {
            uint32_t ub = rub(base, (ch>>4)*sp+pos, ch&15u);
            sum += ub; hash += ub*(ch*131u + pos*7u + 1u);
        }
    print_str(nm); print_str(" sum="); print_dec(sum); print_str(" hash="); print_dec(hash);
    if (sum==gsum && hash==ghash) print_str(" OK\n");
    else { print_str(" MISMATCH gsum="); print_dec(gsum); print_str(" ghash="); print_dec(ghash); print_str("\n"); }
}

void usercode7(void)
{
    uint32_t i, pos, oc, errors = 0u, maxd = 0u;
    yolo_c2f_cfg_t cfg;

    print_str("YOLO C2F8 @320 EXACT standalone (n=1, baked weights, icg16/24)\n");
    for (i = 0u; i < YOLO_C2F8_BLKIN_WORDS; i++) wrw(IN_DDR, i, yolo_c2f8_blkin[i]);

    cfg.in_w=YOLO_C2F8_IN_W; cfg.in_h=YOLO_C2F8_IN_H; cfg.spatial=YOLO_C2F8_SPATIAL;
    cfg.full_c=YOLO_C2F8_FULL_C; cfg.n_bottleneck=YOLO_C2F8_N; cfg.shortcut=1u;
    cfg.in_ddr=IN_DDR; cfg.cv1_ic=YOLO_C2F8_CV1_IC; cfg.cv1_out_ddr=CV1_OUT;
    cfg.bn_out_ddr=BN_OUT; cfg.mcv2_ddr=MCV2_DDR; cfg.add_ddr[0]=ADD0_DDR; cfg.add_ddr[1]=ADD1_DDR;
    cfg.concat_ddr=CONCAT_DDR; cfg.out_ddr=OUT_DDR; cfg.wgt_ddr=WGT_DDR;
    cfg.pad_row_ddr=PAD_ROW; cfg.strip=16u; cfg.silu_exact=1u; cfg.wgt_in_blob=0u;
    cfg.cv1_wgt=yolo_c2f8_cv1_wgt; cfg.cv1_wgt_words=YOLO_C2F8_CV1_WGT_WORDS;
    cfg.cv1_bias=yolo_c2f8_cv1_bias; cfg.cv1_mul=yolo_c2f8_cv1_mul; cfg.cv1_shift=yolo_c2f8_cv1_shift;
    cfg.cv1_silu_lut=yolo_c2f8_cv1_lut; cfg.cv1_rq_zp=YOLO_C2F8_CV1_RQ_ZP;
    cfg.mcv1_wgt_words=YOLO_C2F8_MCV_WGT_WORDS; cfg.mcv2_wgt_words=YOLO_C2F8_MCV_WGT_WORDS;
    cfg.mcv1_wgt[0]=yolo_c2f8_mcv1_0_wgt; cfg.mcv1_bias[0]=yolo_c2f8_mcv1_0_bias;
    cfg.mcv1_mul[0]=yolo_c2f8_mcv1_0_mul; cfg.mcv1_shift[0]=yolo_c2f8_mcv1_0_shift;
    cfg.mcv1_silu_lut[0]=yolo_c2f8_mcv1_0_lut; cfg.mcv1_rq_zp[0]=YOLO_C2F8_MCV1_0_RQ_ZP; cfg.mcv1_pad_value[0]=YOLO_C2F8_MCV1_0_PAD;
    cfg.mcv1_rq_mul[0]=0u; cfg.mcv1_rq_shift[0]=YOLO_C2F8_GLUE_RQ_SHIFT;
    cfg.mcv2_wgt[0]=yolo_c2f8_mcv2_0_wgt; cfg.mcv2_bias[0]=yolo_c2f8_mcv2_0_bias;
    cfg.mcv2_mul[0]=yolo_c2f8_mcv2_0_mul; cfg.mcv2_shift[0]=yolo_c2f8_mcv2_0_shift;
    cfg.mcv2_silu_lut[0]=yolo_c2f8_mcv2_0_lut; cfg.mcv2_pad_value[0]=YOLO_C2F8_MCV2_0_PAD;
    cfg.glue_rq_mul[0]=0u; cfg.glue_rq_shift[0]=YOLO_C2F8_GLUE_RQ_SHIFT;
    cfg.glue_zp[0]=YOLO_C2F8_GLUE0_ZP;
    cfg.add_ratio_shift=YOLO_C2F8_ADD_RATIO_SHIFT;
    cfg.add_ratio_mul[0]=YOLO_C2F8_ADD0_RATIO_MUL; cfg.add_prev_zp[0]=YOLO_C2F8_ADD0_PREV_ZP;
    cfg.cat_req_shift=YOLO_C2F8_CAT_SHIFT; cfg.cat_zp=YOLO_C2F8_CAT_ZP;
    cfg.cat_mul_s0s1=YOLO_C2F8_CAT_MUL_S0S1; cfg.cat_inzp_s0s1=YOLO_C2F8_CAT_INZP_S0S1;
    cfg.cat_mul_add[0]=YOLO_C2F8_CAT_MUL_ADD0; cfg.cat_inzp_add[0]=YOLO_C2F8_CAT_INZP_ADD0;
    cfg.cv2_ic=YOLO_C2F8_CV2_IC; cfg.cv2_oc=YOLO_C2F8_CV2_OC; cfg.cv2_wgt_words=YOLO_C2F8_CV2_WGT_WORDS;
    cfg.cv2_wgt=yolo_c2f8_cv2_wgt; cfg.cv2_bias=yolo_c2f8_cv2_bias; cfg.cv2_mul=yolo_c2f8_cv2_mul; cfg.cv2_shift=yolo_c2f8_cv2_shift;
    cfg.cv2_silu_lut=yolo_c2f8_cv2_lut; cfg.cv2_rq_zp=YOLO_C2F8_CV2_RQ_ZP;

    if (!yolo_run_c2f_block(&cfg)) { print_str("  c2f8 runner failed\n"); errors++; }

    stage_ck("[ck CV1]   ", CV1_OUT,    YOLO_C2F8_FULL_C,    YOLO_C2F8_SPATIAL, YOLO_C2F8_CK_CV1_SUM,    YOLO_C2F8_CK_CV1_HASH);
    stage_ck("[ck ADD0]  ", ADD0_DDR,   YOLO_C2F8_FULL_C/2u, YOLO_C2F8_SPATIAL, YOLO_C2F8_CK_ADD0_SUM,   YOLO_C2F8_CK_ADD0_HASH);
    stage_ck("[ck CONCAT]", CONCAT_DDR, YOLO_C2F8_CV2_IC,    YOLO_C2F8_SPATIAL, YOLO_C2F8_CK_CONCAT_SUM, YOLO_C2F8_CK_CONCAT_HASH);
    stage_ck("[ck OUT]   ", OUT_DDR,    YOLO_C2F8_CV2_OC,    YOLO_C2F8_SPATIAL, YOLO_C2F8_CK_OUT_SUM,    YOLO_C2F8_CK_OUT_HASH);

    for (pos = 0u; pos < YOLO_C2F8_SPATIAL && errors <= 16u; pos++)
        for (oc = 0u; oc < YOLO_C2F8_CV2_OC; oc++) {
            int32_t got = rs8(OUT_DDR, (oc>>4)*YOLO_C2F8_SPATIAL + pos, oc&15u);
            int32_t exp = s8(yolo_c2f8_golden[pos][oc]);
            uint32_t d = ad(got, exp); if (d > maxd) maxd = d;
            if (d > YOLO_C2F8_RTL_TOL) { errors++; if (errors<=4u){ print_str("  pos="); print_dec(pos); print_str(" oc="); print_dec(oc); print_str(" got="); print_dec((uint32_t)got); print_str(" exp="); print_dec((uint32_t)exp); print_str("\n"); } }
        }
    print_str("c2f8 standalone maxdiff="); print_dec(maxd); print_str("\n");
    if (errors == 0u) { print_str("YOLO C2F8 @320 STANDALONE PASS\n"); return; }
    print_str("YOLO C2F8 @320 STANDALONE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
