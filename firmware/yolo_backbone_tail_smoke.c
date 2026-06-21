// Integrated multi-block RTL run (M7 step): conv20 -> c2f_8 -> SPPF chained in
// ONE firmware, intermediates flowing through DDR (NOT re-baked between blocks).
// Input = c2f_6 output (baked, conv20 input). Final compared to the SPPF golden.
// Proves the three blocks (~10 convs + maxpools + concat) chain bit-exactly in RTL.

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_c2f.h"
#include "yolo_conv20_data.h"
#include "yolo_c2f8_data.h"
#include "yolo_sppf_data.h"
#include <stdint.h>

#define PAD_ROW    0x40080000u   // tiled vertical-pad scratch row
#define C20_IN     0x40090000u   // conv20 input (c2f_6 out, baked)
#define C20_OUT    0x40120000u   // conv20 out = c2f_8 in
#define CV1_OUT    0x40140000u
#define BN_OUT     0x40160000u
#define MCV2_DDR   0x40180000u
#define ADD0_DDR   0x401A0000u
#define CONCAT8    0x401C0000u
#define C2F8_OUT   0x40220000u   // c2f_8 out = SPPF in
#define SP_CV1     0x40240000u
#define SP_M0      0x40260000u
#define SP_M1      0x40280000u
#define SP_M2      0x402A0000u
#define SP_CAT     0x402C0000u
#define SPPF_OUT   0x40320000u
#define WGT_DDR    0x40360000u
#define ACT_BASE 0u
#define WGT_BASE 0u

static void wrw(uint32_t a, uint32_t w, const uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); p[0]=l[0];p[1]=l[1];p[2]=l[2];p[3]=l[3]; }
static void rdw(uint32_t a, uint32_t w, uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); l[0]=p[0];l[1]=p[1];l[2]=p[2];l[3]=p[3]; }
static int32_t rs8(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); uint32_t v=p[b>>2]>>((b&3u)*8u)&0xFFu; return (v&0x80u)?((int32_t)v-256):(int32_t)v; }
static int32_t s8(uint32_t b){ b&=0xFFu; return (b&0x80u)?((int32_t)b-256):(int32_t)b; }
static uint32_t ad(int32_t a,int32_t b){ int32_t d=a-b; return (uint32_t)(d<0?-d:d); }

#define SPP_W YOLO_SPPF_IN_W
#define SPP_H YOLO_SPPF_IN_H
#define SPP_SP YOLO_SPPF_SPATIAL

static void maxpool5(uint32_t src, uint32_t dst, uint32_t groups)
{
    uint32_t g, oh, ow, kh, kw, k;
    for (g = 0u; g < groups; g++)
        for (oh = 0u; oh < SPP_H; oh++)
            for (ow = 0u; ow < SPP_W; ow++) {
                int32_t mx[16]; uint32_t o[4]={0u,0u,0u,0u};
                for (k=0u;k<16u;k++) mx[k]=-128;
                for (kh=0u;kh<5u;kh++) for (kw=0u;kw<5u;kw++){
                    int32_t ih=(int32_t)oh-2+(int32_t)kh, iw=(int32_t)ow-2+(int32_t)kw;
                    if (ih>=0&&ih<(int32_t)SPP_H&&iw>=0&&iw<(int32_t)SPP_W){
                        uint32_t in[4]; rdw(src, g*SPP_SP+(uint32_t)ih*SPP_W+(uint32_t)iw, in);
                        for (k=0u;k<16u;k++){ int32_t v=s8(in[k>>2]>>((k&3u)*8u)); if(v>mx[k])mx[k]=v; }
                    }
                }
                for (k=0u;k<16u;k++) o[k>>2]|=((uint32_t)(mx[k]&0xFF))<<((k&3u)*8u);
                wrw(dst, g*SPP_SP+oh*SPP_W+ow, o);
            }
}
static void concat4(uint32_t a,uint32_t b,uint32_t c,uint32_t d,uint32_t dst,uint32_t groups)
{
    uint32_t srcs[4]={a,b,c,d}, i, n;
    for(i=0u;i<4u;i++) for(n=0u;n<groups*SPP_SP;n++){ uint32_t w[4]; rdw(srcs[i],n,w); wrw(dst,i*groups*SPP_SP+n,w); }
}

void usercode7(void)
{
    uint32_t i, pos, oc, errors = 0u;
    yolo_c2f_cfg_t cfg;

    print_str("YOLO BACKBONE TAIL (conv20->c2f_8->SPPF integrated) CPU SMOKE\n");

    // ---- conv20 (input baked = c2f_6 out) -> C20_OUT ----
    for (i = 0u; i < YOLO_CONV20_ACT_WORDS; i++) wrw(C20_IN, i, yolo_conv20_act_words[i]);
    for (i = 0u; i < YOLO_CONV20_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_conv20_wgt_words[i]);
    yolo_set_pad_value(YOLO_CONV20_PAD_VALUE);
    yolo_set_silu_requant(YOLO_CONV20_REQUANT_MUL, YOLO_CONV20_REQUANT_SHIFT, YOLO_CONV20_REQUANT_ZP);
    if (!yolo_dma_ddr_to_act(C20_IN, ACT_BASE, YOLO_CONV20_ACT_WORDS) ||
        !yolo_run_conv2d_oc_chunks(ACT_BASE, WGT_DDR, WGT_BASE, C20_OUT,
                                   YOLO_CONV20_IN_W, YOLO_CONV20_IN_H, YOLO_CONV20_IC, YOLO_CONV20_OC,
                                   YOLO_CONV20_KH, YOLO_CONV20_KW, YOLO_CONV20_STRIDE, YOLO_CONV20_PAD,
                                   yolo_conv20_bias_q, yolo_conv20_scale_mul, yolo_conv20_scale_shift,
                                   NPU_CTRL_SILU_EN|NPU_CTRL_SILU_REQUANT_EN,
                                   YOLO_CONV20_WGT_PER_OC, YOLO_CONV20_OUT_SPATIAL)) { print_str(" conv20 fail\n"); errors++; }

    // ---- c2f_8 (input = conv20 RTL output C20_OUT) -> C2F8_OUT ----
    if (errors==0u) {
        cfg.in_w=YOLO_C2F8_IN_W; cfg.in_h=YOLO_C2F8_IN_H; cfg.spatial=YOLO_C2F8_SPATIAL;
        cfg.full_c=YOLO_C2F8_FULL_C; cfg.n_bottleneck=1u; cfg.shortcut=1u;
        cfg.in_ddr=C20_OUT; cfg.cv1_ic=YOLO_C2F8_CV1_IC; cfg.cv1_out_ddr=CV1_OUT;
        cfg.bn_out_ddr=BN_OUT; cfg.mcv2_ddr=MCV2_DDR; cfg.add_ddr[0]=ADD0_DDR;
        cfg.concat_ddr=CONCAT8; cfg.out_ddr=C2F8_OUT; cfg.wgt_ddr=WGT_DDR;
        cfg.pad_row_ddr=PAD_ROW; cfg.strip=16u; cfg.silu_exact=0u;
        cfg.cv1_wgt=yolo_c2f8_cv1_wgt; cfg.cv1_wgt_words=YOLO_C2F8_CV1_WGT_WORDS;
        cfg.cv1_bias=yolo_c2f8_cv1_bias; cfg.cv1_mul=yolo_c2f8_cv1_mul; cfg.cv1_shift=yolo_c2f8_cv1_shift;
        cfg.cv1_rq_mul=YOLO_C2F8_CV1_RQ_MUL; cfg.cv1_rq_shift=YOLO_C2F8_CV1_RQ_SHIFT; cfg.cv1_rq_zp=YOLO_C2F8_CV1_RQ_ZP;
        cfg.mcv1_wgt_words=YOLO_C2F8_MCV1_WGT_WORDS;
        cfg.mcv1_wgt[0]=yolo_c2f8_mcv1_0_wgt; cfg.mcv1_bias[0]=yolo_c2f8_mcv1_0_bias;
        cfg.mcv1_mul[0]=yolo_c2f8_mcv1_0_mul; cfg.mcv1_shift[0]=yolo_c2f8_mcv1_0_shift;
        cfg.mcv1_rq_mul[0]=YOLO_C2F8_MCV1_0_RQ_MUL; cfg.mcv1_rq_shift[0]=YOLO_C2F8_GLUE_RQ_SHIFT;
        cfg.mcv1_rq_zp[0]=YOLO_C2F8_MCV1_0_RQ_ZP; cfg.mcv1_pad_value[0]=YOLO_C2F8_MCV1_0_PAD;
        cfg.mcv2_wgt_words=YOLO_C2F8_MCV2_WGT_WORDS;
        cfg.mcv2_wgt[0]=yolo_c2f8_mcv2_0_wgt; cfg.mcv2_bias[0]=yolo_c2f8_mcv2_0_bias;
        cfg.mcv2_mul[0]=yolo_c2f8_mcv2_0_mul; cfg.mcv2_shift[0]=yolo_c2f8_mcv2_0_shift;
        cfg.mcv2_pad_value[0]=YOLO_C2F8_MCV2_0_PAD;
        cfg.glue_rq_mul[0]=YOLO_C2F8_GLUE0_RQ_MUL; cfg.glue_rq_shift[0]=YOLO_C2F8_GLUE_RQ_SHIFT; cfg.glue_zp[0]=YOLO_C2F8_GLUE0_ZP;
        cfg.add_ratio_shift=YOLO_C2F8_ADD_RATIO_SHIFT; cfg.add_ratio_mul[0]=YOLO_C2F8_ADD0_RATIO_MUL; cfg.add_prev_zp[0]=YOLO_C2F8_ADD0_PREV_ZP;
        cfg.cat_req_shift=YOLO_C2F8_CAT_SHIFT; cfg.cat_zp=YOLO_C2F8_CAT_ZP;
        cfg.cat_mul_s0s1=YOLO_C2F8_CAT_MUL_S0S1; cfg.cat_inzp_s0s1=YOLO_C2F8_CAT_INZP_S0S1;
        cfg.cat_mul_add[0]=YOLO_C2F8_CAT_MUL_ADD0; cfg.cat_inzp_add[0]=YOLO_C2F8_CAT_INZP_ADD0;
        cfg.cv2_ic=YOLO_C2F8_CV2_IC; cfg.cv2_oc=YOLO_C2F8_CV2_OC;
        cfg.cv2_wgt=yolo_c2f8_cv2_wgt; cfg.cv2_wgt_words=YOLO_C2F8_CV2_WGT_WORDS;
        cfg.cv2_bias=yolo_c2f8_cv2_bias; cfg.cv2_mul=yolo_c2f8_cv2_mul; cfg.cv2_shift=yolo_c2f8_cv2_shift;
        cfg.cv2_rq_mul=YOLO_C2F8_CV2_RQ_MUL; cfg.cv2_rq_shift=YOLO_C2F8_CV2_RQ_SHIFT; cfg.cv2_rq_zp=YOLO_C2F8_CV2_RQ_ZP;
        if (!yolo_run_c2f_block(&cfg)) { print_str(" c2f8 fail\n"); errors++; }
    }

    // ---- SPPF (input = c2f_8 RTL output C2F8_OUT) -> SPPF_OUT ----
    if (errors==0u) {
        for (i = 0u; i < YOLO_SPPF_C25_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_sppf_c25_wgt[i]);
        yolo_set_silu_requant(YOLO_SPPF_C25_RQ_MUL, YOLO_SPPF_C25_RQ_SHIFT, YOLO_SPPF_C25_RQ_ZP);
        if (!yolo_dma_ddr_to_act(C2F8_OUT, ACT_BASE, SPP_SP*(YOLO_SPPF_C25_IC/16u)) ||
            !yolo_run_pw_conv1x1_oc_chunks(ACT_BASE, WGT_DDR, WGT_BASE, SP_CV1,
                                           SPP_W, SPP_H, YOLO_SPPF_C25_IC, YOLO_SPPF_C25_OC,
                                           yolo_sppf_c25_bias, yolo_sppf_c25_mul, yolo_sppf_c25_shift,
                                           NPU_CTRL_SILU_EN|NPU_CTRL_SILU_REQUANT_EN, SPP_SP)) { print_str(" c25 fail\n"); errors++; }
    }
    if (errors==0u) {
        maxpool5(SP_CV1, SP_M0, YOLO_SPPF_C25_OC/16u);
        maxpool5(SP_M0, SP_M1, YOLO_SPPF_C25_OC/16u);
        maxpool5(SP_M1, SP_M2, YOLO_SPPF_C25_OC/16u);
        concat4(SP_CV1, SP_M0, SP_M1, SP_M2, SP_CAT, YOLO_SPPF_C25_OC/16u);
        for (i = 0u; i < YOLO_SPPF_C26_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_sppf_c26_wgt[i]);
        yolo_set_silu_requant(YOLO_SPPF_C26_RQ_MUL, YOLO_SPPF_C26_RQ_SHIFT, YOLO_SPPF_C26_RQ_ZP);
        if (!yolo_dma_ddr_to_act(SP_CAT, ACT_BASE, SPP_SP*(YOLO_SPPF_C26_IC/16u)) ||
            !yolo_run_pw_conv1x1_oc_chunks(ACT_BASE, WGT_DDR, WGT_BASE, SPPF_OUT,
                                           SPP_W, SPP_H, YOLO_SPPF_C26_IC, YOLO_SPPF_C26_OC,
                                           yolo_sppf_c26_bias, yolo_sppf_c26_mul, yolo_sppf_c26_shift,
                                           NPU_CTRL_SILU_EN|NPU_CTRL_SILU_REQUANT_EN, SPP_SP)) { print_str(" c26 fail\n"); errors++; }
    }

    for (pos = 0u; pos < SPP_SP && errors <= 16u; pos++)
        for (oc = 0u; oc < YOLO_SPPF_C26_OC; oc++) {
            int32_t got = rs8(SPPF_OUT, (oc>>4)*SPP_SP + pos, oc&15u);
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

    if (errors == 0u) { print_str("YOLO BACKBONE TAIL INTEGRATED CPU SMOKE PASS\n"); return; }
    print_str("YOLO BACKBONE TAIL FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
