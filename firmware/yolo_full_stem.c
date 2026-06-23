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
#include "yolo_conv6_exact_data.h"
#include "yolo_c2f4_exact_data.h"
#include "yolo_conv13_exact_data.h"
#include "yolo_c2f6_exact_data.h"
#include "yolo_conv20_exact_data.h"
#include "yolo_c2f8_exact_data.h"
#include "yolo_sppf_exact_data.h"
#include "yolo_neck_data.h"
#include "yolo_weight_map.h"
#include <stdint.h>

// conv0/conv1 read weights DIRECTLY from the DDR weight blob (no bake); the tiled
// conv takes a DDR weight base. c2f_2 weights stay baked (C2f runner takes arrays).
#define WGT_OF(ci) (YOLO_WGT_DDR_BASE + yolo_wgt_map[ci].off * 16u)

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
// NOTE: image preload = 0x4040_0000..0x405A_0000, weight blob = 0x4080_0000..
// 0x40B0_1000. ALL written buffers MUST avoid both. Free: 0x405A_0000..0x4080_0000
// (between) and 0x40C0_0000..0x4100_0000 (above blob).
#define PAD_ROW   0x40C00000u   // moved ABOVE the blob (was in-blob -> corrupted weights)
#define C2F_OUT   0x40C40000u   // c2f_2 out (80x80x32) = conv6 input
#define C6_OUT    0x40D00000u   // conv6 out 40x40x64 = c2f_4 input
#define B_CV1     0x40D40000u
#define B_BN      0x40D80000u
#define B_MCV2    0x40DA0000u
#define B_ADD0    0x40DC0000u
#define B_ADD1    0x40DE0000u
#define B_CONCAT  0x40E00000u
#define C2F4_OUT  0x40E80000u   // c2f_4 out 40x40x64 = conv13 input
#define C2F_WGT   0x40EC0000u   // unused in blob mode (wgt_in_blob=1), just a valid scratch
// ---- backbone extension: conv13 -> c2f_6 (all in 0x40F0_0000..0x4100_0000, 1MB) ----
#define C13_OUT   0x40F00000u   // conv13 out 20x20x128 = c2f_6 input
#define D_CV1     0x40F10000u   // c2f_6 cv1 (256ch @ 20x20)
#define D_BN      0x40F30000u
#define D_MCV2    0x40F38000u
#define D_ADD0    0x40F40000u
#define D_ADD1    0x40F48000u
#define D_CONCAT  0x40F50000u   // 512ch @ 20x20
#define C2F6_OUT  0x40F90000u   // c2f_6 out 20x20x128
#define D_PSUM    0x40FA0000u   // ic_stream INT32 psum scratch (c2f_6 bottleneck icg4 -> unused)
// ---- backbone extension: conv20 -> c2f_8 (in the free 0x405A_0000..0x4080_0000, 2.4MB) ----
#define C20_OUT   0x405C0000u   // conv20 out 10x10x256 = c2f_8 input
#define C20_PSUM  0x40600000u   // conv20 (large-IC stride2) ic_stream INT32 psum
#define E_CV1     0x40640000u   // c2f_8 cv1 (256ch @ 10x10)
#define E_BN      0x40660000u
#define E_MCV2    0x40670000u
#define E_ADD0    0x40680000u
#define E_CONCAT  0x40690000u   // 384ch @ 10x10
#define C2F8_OUT  0x406C0000u   // c2f_8 out 10x10x256 = SPPF input
#define E_PSUM    0x40700000u   // c2f_8 bottleneck (icg8) ic_stream INT32 psum
// ---- SPPF (model.9): conv25 -> 3x maxpool5 -> concat -> conv26 (10x10) ----
#define S_CV1     0x40720000u   // conv25 out 128ch
#define S_M0      0x40730000u
#define S_M1      0x40740000u
#define S_M2      0x40750000u
#define S_CAT     0x40760000u   // concat 512ch
#define SPPF_OUT  0x40780000u   // conv26 out 256ch (P5)
// ---- NECK (FPN/PAN) buffers. backbone taps stay alive: P5tap=C2F8_OUT,
//      cat1-tap=C2F6_OUT (c2f6 out), cat2-tap=C2F4_OUT (c2f4 out). ----
#define NK_UP1    0x40790000u   // upsample(SPPF) 20x20x256
#define NK_CAT1   0x407B0000u   // 20x20x384
#define NK_FMID   0x407E0000u   // fpn_mid 20x20x128
#define NK_UP2    0x40B01000u   // upsample(fpn_mid) 40x40x128
#define NK_CAT2   0x40B40000u   // 40x40x192
#define NK_PANP3  0x40B90000u   // pan_p3 40x40x64 (HEAD P3)
#define NK_C35    0x40FA0000u   // conv35 out 20x20x64
#define NK_CAT3   0x40FA8000u   // 20x20x192
#define NK_PANP4  0x40FC0000u   // pan_p4 20x20x128 (HEAD P4)
#define NK_C46    0x40FD0000u   // conv46 out 10x10x128
#define NK_CAT4   0x40FD8000u   // 10x10x384
#define NK_PANP5  0x40FF0000u   // pan_p5 10x10x256 (HEAD P5)
#define NK_PSUM   0x40000000u   // ic_stream psum for conv46 (icg8 s2) + c2f_21 bottleneck
// neck C2f internal scratch (reused per stage; in dead backbone-buffer space < C2F4_OUT)
#define NK_SCR_CV1  0x40C40000u
#define NK_SCR_BN   0x40C80000u
#define NK_SCR_MCV2 0x40CA0000u
#define NK_SCR_ADD  0x40CC0000u
#define NK_SCR_CAT  0x40CE0000u
#define WGT_BASE  0u

static int32_t rs8(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); uint32_t v=p[b>>2]>>((b&3u)*8u)&0xFFu; return (v&0x80u)?((int32_t)v-256):(int32_t)v; }
static int32_t s8(uint32_t b){ b&=0xFFu; return (b&0x80u)?((int32_t)b-256):(int32_t)b; }
static uint32_t ad(int32_t a,int32_t b){ int32_t d=a-b; return (uint32_t)(d<0?-d:d); }

// SPPF 5x5 stride-1 pad-2 int8 max-pool, tile-major DDR (groups of 16 channels).
static void maxpool5(uint32_t src, uint32_t dst, uint32_t groups, uint32_t H, uint32_t W)
{
    uint32_t SP = H*W, g, oh, ow, kh, kw, k;
    for (g = 0u; g < groups; g++)
        for (oh = 0u; oh < H; oh++)
            for (ow = 0u; ow < W; ow++) {
                int32_t mx[16]; uint32_t o[4] = {0u,0u,0u,0u};
                for (k = 0u; k < 16u; k++) mx[k] = -128;
                for (kh = 0u; kh < 5u; kh++)
                    for (kw = 0u; kw < 5u; kw++) {
                        int32_t ih = (int32_t)oh-2+(int32_t)kh, iw = (int32_t)ow-2+(int32_t)kw;
                        if (ih>=0 && ih<(int32_t)H && iw>=0 && iw<(int32_t)W) {
                            volatile uint32_t *p=(volatile uint32_t*)(src+(g*SP+(uint32_t)ih*W+(uint32_t)iw)*16u);
                            for (k = 0u; k < 16u; k++) { int32_t v=s8(p[k>>2]>>((k&3u)*8u)); if (v>mx[k]) mx[k]=v; }
                        }
                    }
                for (k = 0u; k < 16u; k++) o[k>>2] |= ((uint32_t)(mx[k]&0xFF))<<((k&3u)*8u);
                { volatile uint32_t *q=(volatile uint32_t*)(dst+(g*SP+oh*W+ow)*16u); q[0]=o[0];q[1]=o[1];q[2]=o[2];q[3]=o[3]; }
            }
}
// concat 4 same-scale tile-major parts -> [4*groups] tensor.
static void concat4(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t dst,
                    uint32_t groups, uint32_t SP)
{
    uint32_t srcs[4]; uint32_t i, n; srcs[0]=a;srcs[1]=b;srcs[2]=c;srcs[3]=d;
    for (i = 0u; i < 4u; i++)
        for (n = 0u; n < groups*SP; n++) {
            volatile uint32_t *s=(volatile uint32_t*)(srcs[i]+n*16u);
            volatile uint32_t *t=(volatile uint32_t*)(dst+(i*groups*SP+n)*16u);
            t[0]=s[0];t[1]=s[1];t[2]=s[2];t[3]=s[3];
        }
}
// nearest-neighbor 2x upsample (tile-major DDR, scale/zp unchanged): HxW -> 2Hx2W.
static void upsample2(uint32_t src, uint32_t dst, uint32_t groups, uint32_t H, uint32_t W)
{
    uint32_t g, y, x; uint32_t OW = W*2u;
    for (g = 0u; g < groups; g++)
        for (y = 0u; y < H; y++)
            for (x = 0u; x < W; x++) {
                volatile uint32_t *s=(volatile uint32_t*)(src+(g*H*W + y*W + x)*16u);
                uint32_t v0=s[0],v1=s[1],v2=s[2],v3=s[3];
                uint32_t base = g*(H*2u)*OW;
                uint32_t d00=base+(y*2u)*OW+x*2u, d01=d00+1u, d10=d00+OW, d11=d10+1u;
                uint32_t dd[4]; uint32_t k; dd[0]=d00;dd[1]=d01;dd[2]=d10;dd[3]=d11;
                for (k=0u;k<4u;k++){ volatile uint32_t *t=(volatile uint32_t*)(dst+dd[k]*16u); t[0]=v0;t[1]=v1;t[2]=v2;t[3]=v3; }
            }
}
// signed-int8 requant one lane: clamp((q-inzp)*mul>>sh + catzp)
static int32_t rq_lane(int32_t q, int32_t inzp, uint32_t mul, uint32_t sh, int32_t catzp)
{ int32_t v=(((q-inzp)*(int32_t)mul)>>sh)+catzp; return v>127?127:(v<-128?-128:v); }
// concat2 with per-part requant to a common (cat) scale -> [up_g+tap_g] tile-major.
static void concat2_rq(uint32_t up, uint32_t up_g, uint32_t up_mul, int32_t up_iz,
                       uint32_t tap, uint32_t tap_g, uint32_t tap_mul, int32_t tap_iz,
                       int32_t catzp, uint32_t sh, uint32_t dst, uint32_t SP)
{
    uint32_t n,k; uint32_t parts[2]={up,tap}, pg[2]={up_g,tap_g}, pm[2]={up_mul,tap_mul};
    int32_t pz[2]={up_iz,tap_iz}; uint32_t base=0u,p;
    for (p=0u;p<2u;p++){
        for (n=0u;n<pg[p]*SP;n++){
            volatile uint32_t *s=(volatile uint32_t*)(parts[p]+n*16u);
            uint32_t o[4]={0u,0u,0u,0u};
            for (k=0u;k<16u;k++){ int32_t q=s8(s[k>>2]>>((k&3u)*8u));
                o[k>>2]|=((uint32_t)(rq_lane(q,pz[p],pm[p],sh,catzp)&0xFF))<<((k&3u)*8u); }
            volatile uint32_t *t=(volatile uint32_t*)(dst+(base+n)*16u);
            t[0]=o[0];t[1]=o[1];t[2]=o[2];t[3]=o[3];
        }
        base += pg[p]*SP;
    }
}
// stage checksum vs golden (sum + position/channel hash), prints OK/MISMATCH.
static void ck_stage(const char *nm, uint32_t base, uint32_t SP, uint32_t OC,
                     const uint8_t *golden, uint32_t gOC, uint32_t tol)
{
    uint32_t pos, oc, errs=0u, maxd=0u;
    for (pos=0u; pos<SP; pos++)
        for (oc=0u; oc<OC; oc++) {
            int32_t got=rs8(base,(oc>>4)*SP+pos,oc&15u), exp=s8(golden[pos*gOC+oc]);
            uint32_t d=ad(got,exp); if(d>maxd)maxd=d; if(d>tol)errs++;
        }
    print_str(nm); print_str(" maxd="); print_dec(maxd);
    print_str(errs?" MISMATCH errs=":" OK errs="); print_dec(errs); print_str("\n");
}

/* Set up + run one neck C2f (shortcut=0, n=1, exact, blob weights). P/p = upper/
 * lower data prefix; weights from the blob at the 4 conv offsets. */
#define NECK_C2F(P, p, INDDR, OUTDDR, Wc1, Wm1, Wm2, Wc2) do { \
    cfg.in_w=P##_IN_W; cfg.in_h=P##_IN_H; cfg.spatial=P##_SPATIAL; \
    cfg.full_c=P##_FULL_C; cfg.n_bottleneck=P##_N; cfg.shortcut=0u; \
    cfg.in_ddr=(INDDR); cfg.cv1_ic=P##_CV1_IC; cfg.cv1_out_ddr=NK_SCR_CV1; \
    cfg.bn_out_ddr=NK_SCR_BN; cfg.mcv2_ddr=NK_SCR_MCV2; cfg.add_ddr[0]=NK_SCR_ADD; \
    cfg.concat_ddr=NK_SCR_CAT; cfg.out_ddr=(OUTDDR); cfg.wgt_ddr=C2F_WGT; \
    cfg.pad_row_ddr=PAD_ROW; cfg.strip=16u; cfg.silu_exact=1u; cfg.wgt_in_blob=1u; cfg.psum_ddr=NK_PSUM; \
    cfg.cv1_wgt_ddr=(Wc1); cfg.mcv1_wgt_ddr[0]=(Wm1); cfg.mcv2_wgt_ddr[0]=(Wm2); cfg.cv2_wgt_ddr=(Wc2); \
    cfg.cv1_wgt_words=P##_FULL_C*(P##_CV1_IC/16u); \
    cfg.cv1_bias=p##_cv1_bias; cfg.cv1_mul=p##_cv1_mul; cfg.cv1_shift=p##_cv1_shift; \
    cfg.cv1_silu_lut=p##_cv1_lut; cfg.cv1_rq_zp=P##_CV1_ZP; \
    cfg.mcv1_wgt_words=(P##_FULL_C/2u)*((P##_FULL_C/2u)/16u)*9u; cfg.mcv2_wgt_words=cfg.mcv1_wgt_words; \
    cfg.mcv1_bias[0]=p##_mcv1_0_bias; cfg.mcv1_mul[0]=p##_mcv1_0_mul; cfg.mcv1_shift[0]=p##_mcv1_0_shift; \
    cfg.mcv1_silu_lut[0]=p##_mcv1_0_lut; cfg.mcv1_rq_zp[0]=P##_MCV1_0_ZP; cfg.mcv1_pad_value[0]=P##_MCV1_0_PAD; \
    cfg.mcv2_bias[0]=p##_mcv2_0_bias; cfg.mcv2_mul[0]=p##_mcv2_0_mul; cfg.mcv2_shift[0]=p##_mcv2_0_shift; \
    cfg.mcv2_silu_lut[0]=p##_mcv2_0_lut; cfg.mcv2_pad_value[0]=P##_MCV2_0_PAD; \
    cfg.mcv1_rq_mul[0]=0u; cfg.mcv1_rq_shift[0]=12u; cfg.glue_rq_mul[0]=0u; cfg.glue_rq_shift[0]=12u; cfg.glue_zp[0]=P##_MCV2_0_ZP; \
    cfg.add_ratio_shift=16u; cfg.add_ratio_mul[0]=0u; cfg.add_prev_zp[0]=0; \
    cfg.cat_req_shift=P##_CAT_SHIFT; cfg.cat_zp=P##_CAT_ZP; \
    cfg.cat_mul_s0s1=P##_CAT_MUL_S0S1; cfg.cat_inzp_s0s1=P##_CAT_INZP_S0S1; \
    cfg.cat_mul_add[0]=P##_CAT_MUL_ADD0; cfg.cat_inzp_add[0]=P##_CAT_INZP_ADD0; \
    cfg.cv2_ic=P##_CV2_IC; cfg.cv2_oc=P##_CV2_OC; cfg.cv2_wgt_words=P##_CV2_OC*(P##_CV2_IC/16u); \
    cfg.cv2_bias=p##_cv2_bias; cfg.cv2_mul=p##_cv2_mul; cfg.cv2_shift=p##_cv2_shift; \
    cfg.cv2_silu_lut=p##_cv2_lut; cfg.cv2_rq_zp=P##_CV2_ZP; \
    if (errors==0u && !yolo_run_c2f_block(&cfg)) { print_str("  neck c2f fail\n"); errors++; } \
} while(0)

void usercode7(void)
{
    uint32_t errors = 0u;
    yolo_c2f_cfg_t cfg;

    print_str("YOLO FULL STEM: conv0 -> ... -> SPPF (on-SoC, DDR, preloaded img)\n");
    /* intermediate goldens are checkpoints; only the final (SPPF) one is validated */
    (void)yolo_c2f4_golden; (void)yolo_c2f6_golden; (void)yolo_c2f8_golden;

    // ---------- Stage 0: conv0 (preloaded image, weights from DDR blob) -> C0_OUT ----------
    yolo_set_pad_value(C0E_PAD_VALUE);
    yolo_load_silu_lut(yolo_conv0_320e_silu_lut);
    yolo_set_silu_requant(0u, 0u, 0);
    if (!yolo_run_conv2d_tiled(IMG, WGT_OF(0), WGT_BASE, C0_OUT, PAD_ROW,
                               C0E_IN_W, C0E_IN_H, C0E_IC, C0E_OC,
                               3u, 3u, 2u, 1u,
                               yolo_conv0_320e_bias_q, yolo_conv0_320e_scale_mul,
                               yolo_conv0_320e_scale_shift,
                               NPU_CTRL_SILU_EXACT_EN, C0E_WGT_PER_OC, 16u, C0E_PAD_VALUE)) {
        print_str("  conv0 fail\n"); errors++;
    }

    // ---------- Stage 1: conv1 reads C0_OUT, weights from DDR blob -> C1_OUT ----------
    yolo_set_pad_value(C1E_PAD_VALUE);
    yolo_load_silu_lut(yolo_conv1_320e_silu_lut);
    yolo_set_silu_requant(0u, 0u, 0);
    if (errors == 0u && !yolo_run_conv2d_tiled(C0_OUT, WGT_OF(1), WGT_BASE, C1_OUT, PAD_ROW,
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
    cfg.wgt_in_blob=1u;   // c2f_2 weights (conv2/3/4/5) from the DDR blob too
    cfg.cv1_wgt_ddr=WGT_OF(2); cfg.mcv1_wgt_ddr[0]=WGT_OF(3);
    cfg.mcv2_wgt_ddr[0]=WGT_OF(4); cfg.cv2_wgt_ddr=WGT_OF(5);
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
    print_str("  [stage2 c2f_2 done]\n");

    // ---------- Stage 3: conv6 (downsample 80x80x32 -> 40x40x64) reads C2F_OUT ----------
    yolo_set_pad_value(C6E_PAD_VALUE);
    yolo_load_silu_lut(yolo_conv6e_silu_lut);
    yolo_set_silu_requant(0u, 0u, 0);
    if (errors == 0u && !yolo_run_conv2d_tiled(C2F_OUT, WGT_OF(6), WGT_BASE, C6_OUT, PAD_ROW,
                               C6E_IN_W, C6E_IN_H, C6E_IC, C6E_OC, 3u, 3u, C6E_STRIDE, 1u,
                               yolo_conv6e_bias_q, yolo_conv6e_scale_mul, yolo_conv6e_scale_shift,
                               NPU_CTRL_SILU_EXACT_EN, C6E_WGT_PER_OC, 16u, C6E_PAD_VALUE)) {
        print_str("  conv6 fail\n"); errors++;
    }
    print_str("  [stage3 conv6 done]\n");

    // ---------- Stage 4: c2f_4 (model.4, n=2) reads C6_OUT, blob weights ----------
    cfg.in_w=YOLO_C2F4_IN_W; cfg.in_h=YOLO_C2F4_IN_H; cfg.spatial=YOLO_C2F4_SPATIAL;
    cfg.full_c=YOLO_C2F4_FULL_C; cfg.n_bottleneck=YOLO_C2F4_N; cfg.shortcut=1u;
    cfg.in_ddr=C6_OUT; cfg.cv1_ic=YOLO_C2F4_CV1_IC; cfg.cv1_out_ddr=B_CV1;
    cfg.bn_out_ddr=B_BN; cfg.mcv2_ddr=B_MCV2; cfg.add_ddr[0]=B_ADD0; cfg.add_ddr[1]=B_ADD1;
    cfg.concat_ddr=B_CONCAT; cfg.out_ddr=C2F4_OUT; cfg.wgt_ddr=C2F_WGT;
    cfg.pad_row_ddr=PAD_ROW; cfg.strip=16u; cfg.silu_exact=1u; cfg.wgt_in_blob=1u;
    cfg.cv1_wgt_ddr=WGT_OF(7); cfg.cv2_wgt_ddr=WGT_OF(12);
    cfg.mcv1_wgt_ddr[0]=WGT_OF(8);  cfg.mcv2_wgt_ddr[0]=WGT_OF(9);
    cfg.mcv1_wgt_ddr[1]=WGT_OF(10); cfg.mcv2_wgt_ddr[1]=WGT_OF(11);
    cfg.cv1_wgt_words=YOLO_C2F4_CV1_WGT_WORDS;
    cfg.cv1_bias=yolo_c2f4_cv1_bias; cfg.cv1_mul=yolo_c2f4_cv1_mul; cfg.cv1_shift=yolo_c2f4_cv1_shift;
    cfg.cv1_silu_lut=yolo_c2f4_cv1_lut; cfg.cv1_rq_zp=YOLO_C2F4_CV1_RQ_ZP;
    cfg.mcv1_wgt_words=YOLO_C2F4_MCV_WGT_WORDS; cfg.mcv2_wgt_words=YOLO_C2F4_MCV_WGT_WORDS;
    cfg.mcv1_bias[0]=yolo_c2f4_mcv1_0_bias; cfg.mcv1_mul[0]=yolo_c2f4_mcv1_0_mul; cfg.mcv1_shift[0]=yolo_c2f4_mcv1_0_shift;
    cfg.mcv1_silu_lut[0]=yolo_c2f4_mcv1_0_lut; cfg.mcv1_rq_zp[0]=YOLO_C2F4_MCV1_0_RQ_ZP; cfg.mcv1_pad_value[0]=YOLO_C2F4_MCV1_0_PAD;
    cfg.mcv1_bias[1]=yolo_c2f4_mcv1_1_bias; cfg.mcv1_mul[1]=yolo_c2f4_mcv1_1_mul; cfg.mcv1_shift[1]=yolo_c2f4_mcv1_1_shift;
    cfg.mcv1_silu_lut[1]=yolo_c2f4_mcv1_1_lut; cfg.mcv1_rq_zp[1]=YOLO_C2F4_MCV1_1_RQ_ZP; cfg.mcv1_pad_value[1]=YOLO_C2F4_MCV1_1_PAD;
    cfg.mcv1_rq_mul[0]=0u; cfg.mcv1_rq_mul[1]=0u; cfg.mcv1_rq_shift[0]=YOLO_C2F4_GLUE_RQ_SHIFT; cfg.mcv1_rq_shift[1]=YOLO_C2F4_GLUE_RQ_SHIFT;
    cfg.mcv2_bias[0]=yolo_c2f4_mcv2_0_bias; cfg.mcv2_mul[0]=yolo_c2f4_mcv2_0_mul; cfg.mcv2_shift[0]=yolo_c2f4_mcv2_0_shift;
    cfg.mcv2_silu_lut[0]=yolo_c2f4_mcv2_0_lut; cfg.mcv2_pad_value[0]=YOLO_C2F4_MCV2_0_PAD;
    cfg.mcv2_bias[1]=yolo_c2f4_mcv2_1_bias; cfg.mcv2_mul[1]=yolo_c2f4_mcv2_1_mul; cfg.mcv2_shift[1]=yolo_c2f4_mcv2_1_shift;
    cfg.mcv2_silu_lut[1]=yolo_c2f4_mcv2_1_lut; cfg.mcv2_pad_value[1]=YOLO_C2F4_MCV2_1_PAD;
    cfg.glue_rq_mul[0]=0u; cfg.glue_rq_mul[1]=0u; cfg.glue_rq_shift[0]=YOLO_C2F4_GLUE_RQ_SHIFT; cfg.glue_rq_shift[1]=YOLO_C2F4_GLUE_RQ_SHIFT;
    cfg.glue_zp[0]=YOLO_C2F4_GLUE0_ZP; cfg.glue_zp[1]=YOLO_C2F4_GLUE1_ZP;
    cfg.add_ratio_shift=YOLO_C2F4_ADD_RATIO_SHIFT;
    cfg.add_ratio_mul[0]=YOLO_C2F4_ADD0_RATIO_MUL; cfg.add_prev_zp[0]=YOLO_C2F4_ADD0_PREV_ZP;
    cfg.add_ratio_mul[1]=YOLO_C2F4_ADD1_RATIO_MUL; cfg.add_prev_zp[1]=YOLO_C2F4_ADD1_PREV_ZP;
    cfg.cat_req_shift=YOLO_C2F4_CAT_SHIFT; cfg.cat_zp=YOLO_C2F4_CAT_ZP;
    cfg.cat_mul_s0s1=YOLO_C2F4_CAT_MUL_S0S1; cfg.cat_inzp_s0s1=YOLO_C2F4_CAT_INZP_S0S1;
    cfg.cat_mul_add[0]=YOLO_C2F4_CAT_MUL_ADD0; cfg.cat_inzp_add[0]=YOLO_C2F4_CAT_INZP_ADD0;
    cfg.cat_mul_add[1]=YOLO_C2F4_CAT_MUL_ADD1; cfg.cat_inzp_add[1]=YOLO_C2F4_CAT_INZP_ADD1;
    cfg.cv2_ic=YOLO_C2F4_CV2_IC; cfg.cv2_oc=YOLO_C2F4_CV2_OC;
    cfg.cv2_wgt_words=YOLO_C2F4_CV2_WGT_WORDS;
    cfg.cv2_bias=yolo_c2f4_cv2_bias; cfg.cv2_mul=yolo_c2f4_cv2_mul; cfg.cv2_shift=yolo_c2f4_cv2_shift;
    cfg.cv2_silu_lut=yolo_c2f4_cv2_lut; cfg.cv2_rq_zp=YOLO_C2F4_CV2_RQ_ZP;
    if (errors == 0u && !yolo_run_c2f_block(&cfg)) { print_str("  c2f4 fail\n"); errors++; }
    print_str("  [stage4 c2f_4 done]\n");

    // ---------- Stage 5: conv13 (downsample 40x40x64 -> 20x20x128) reads C2F4_OUT ----------
    yolo_set_pad_value(C13E_PAD_VALUE);
    yolo_load_silu_lut(yolo_conv13e_silu_lut);
    yolo_set_silu_requant(0u, 0u, 0);
    if (errors == 0u && !yolo_run_conv2d_tiled(C2F4_OUT, WGT_OF(13), WGT_BASE, C13_OUT, PAD_ROW,
                               C13E_IN_W, C13E_IN_H, C13E_IC, C13E_OC, 3u, 3u, C13E_STRIDE, 1u,
                               yolo_conv13e_bias_q, yolo_conv13e_scale_mul, yolo_conv13e_scale_shift,
                               NPU_CTRL_SILU_EXACT_EN, C13E_WGT_PER_OC, 16u, C13E_PAD_VALUE)) {
        print_str("  conv13 fail\n"); errors++;
    }
    print_str("  [stage5 conv13 done]\n");

    // ---------- Stage 6: c2f_6 (model.6, n=2, 128ch) reads C13_OUT, blob weights ----------
    cfg.in_w=YOLO_C2F6_IN_W; cfg.in_h=YOLO_C2F6_IN_H; cfg.spatial=YOLO_C2F6_SPATIAL;
    cfg.full_c=YOLO_C2F6_FULL_C; cfg.n_bottleneck=YOLO_C2F6_N; cfg.shortcut=1u;
    cfg.in_ddr=C13_OUT; cfg.cv1_ic=YOLO_C2F6_CV1_IC; cfg.cv1_out_ddr=D_CV1;
    cfg.bn_out_ddr=D_BN; cfg.mcv2_ddr=D_MCV2; cfg.add_ddr[0]=D_ADD0; cfg.add_ddr[1]=D_ADD1;
    cfg.concat_ddr=D_CONCAT; cfg.out_ddr=C2F6_OUT; cfg.wgt_ddr=C2F_WGT;
    cfg.pad_row_ddr=PAD_ROW; cfg.strip=16u; cfg.silu_exact=1u; cfg.wgt_in_blob=1u;
    cfg.psum_ddr=D_PSUM;
    cfg.cv1_wgt_ddr=WGT_OF(14); cfg.cv2_wgt_ddr=WGT_OF(19);
    cfg.mcv1_wgt_ddr[0]=WGT_OF(15); cfg.mcv2_wgt_ddr[0]=WGT_OF(16);
    cfg.mcv1_wgt_ddr[1]=WGT_OF(17); cfg.mcv2_wgt_ddr[1]=WGT_OF(18);
    cfg.cv1_wgt_words=YOLO_C2F6_CV1_WGT_WORDS;
    cfg.cv1_bias=yolo_c2f6_cv1_bias; cfg.cv1_mul=yolo_c2f6_cv1_mul; cfg.cv1_shift=yolo_c2f6_cv1_shift;
    cfg.cv1_silu_lut=yolo_c2f6_cv1_lut; cfg.cv1_rq_zp=YOLO_C2F6_CV1_RQ_ZP;
    cfg.mcv1_wgt_words=YOLO_C2F6_MCV_WGT_WORDS; cfg.mcv2_wgt_words=YOLO_C2F6_MCV_WGT_WORDS;
    cfg.mcv1_bias[0]=yolo_c2f6_mcv1_0_bias; cfg.mcv1_mul[0]=yolo_c2f6_mcv1_0_mul; cfg.mcv1_shift[0]=yolo_c2f6_mcv1_0_shift;
    cfg.mcv1_silu_lut[0]=yolo_c2f6_mcv1_0_lut; cfg.mcv1_rq_zp[0]=YOLO_C2F6_MCV1_0_RQ_ZP; cfg.mcv1_pad_value[0]=YOLO_C2F6_MCV1_0_PAD;
    cfg.mcv1_bias[1]=yolo_c2f6_mcv1_1_bias; cfg.mcv1_mul[1]=yolo_c2f6_mcv1_1_mul; cfg.mcv1_shift[1]=yolo_c2f6_mcv1_1_shift;
    cfg.mcv1_silu_lut[1]=yolo_c2f6_mcv1_1_lut; cfg.mcv1_rq_zp[1]=YOLO_C2F6_MCV1_1_RQ_ZP; cfg.mcv1_pad_value[1]=YOLO_C2F6_MCV1_1_PAD;
    cfg.mcv1_rq_mul[0]=0u; cfg.mcv1_rq_mul[1]=0u; cfg.mcv1_rq_shift[0]=YOLO_C2F6_GLUE_RQ_SHIFT; cfg.mcv1_rq_shift[1]=YOLO_C2F6_GLUE_RQ_SHIFT;
    cfg.mcv2_bias[0]=yolo_c2f6_mcv2_0_bias; cfg.mcv2_mul[0]=yolo_c2f6_mcv2_0_mul; cfg.mcv2_shift[0]=yolo_c2f6_mcv2_0_shift;
    cfg.mcv2_silu_lut[0]=yolo_c2f6_mcv2_0_lut; cfg.mcv2_pad_value[0]=YOLO_C2F6_MCV2_0_PAD;
    cfg.mcv2_bias[1]=yolo_c2f6_mcv2_1_bias; cfg.mcv2_mul[1]=yolo_c2f6_mcv2_1_mul; cfg.mcv2_shift[1]=yolo_c2f6_mcv2_1_shift;
    cfg.mcv2_silu_lut[1]=yolo_c2f6_mcv2_1_lut; cfg.mcv2_pad_value[1]=YOLO_C2F6_MCV2_1_PAD;
    cfg.glue_rq_mul[0]=0u; cfg.glue_rq_mul[1]=0u; cfg.glue_rq_shift[0]=YOLO_C2F6_GLUE_RQ_SHIFT; cfg.glue_rq_shift[1]=YOLO_C2F6_GLUE_RQ_SHIFT;
    cfg.glue_zp[0]=YOLO_C2F6_GLUE0_ZP; cfg.glue_zp[1]=YOLO_C2F6_GLUE1_ZP;
    cfg.add_ratio_shift=YOLO_C2F6_ADD_RATIO_SHIFT;
    cfg.add_ratio_mul[0]=YOLO_C2F6_ADD0_RATIO_MUL; cfg.add_prev_zp[0]=YOLO_C2F6_ADD0_PREV_ZP;
    cfg.add_ratio_mul[1]=YOLO_C2F6_ADD1_RATIO_MUL; cfg.add_prev_zp[1]=YOLO_C2F6_ADD1_PREV_ZP;
    cfg.cat_req_shift=YOLO_C2F6_CAT_SHIFT; cfg.cat_zp=YOLO_C2F6_CAT_ZP;
    cfg.cat_mul_s0s1=YOLO_C2F6_CAT_MUL_S0S1; cfg.cat_inzp_s0s1=YOLO_C2F6_CAT_INZP_S0S1;
    cfg.cat_mul_add[0]=YOLO_C2F6_CAT_MUL_ADD0; cfg.cat_inzp_add[0]=YOLO_C2F6_CAT_INZP_ADD0;
    cfg.cat_mul_add[1]=YOLO_C2F6_CAT_MUL_ADD1; cfg.cat_inzp_add[1]=YOLO_C2F6_CAT_INZP_ADD1;
    cfg.cv2_ic=YOLO_C2F6_CV2_IC; cfg.cv2_oc=YOLO_C2F6_CV2_OC;
    cfg.cv2_wgt_words=YOLO_C2F6_CV2_WGT_WORDS;
    cfg.cv2_bias=yolo_c2f6_cv2_bias; cfg.cv2_mul=yolo_c2f6_cv2_mul; cfg.cv2_shift=yolo_c2f6_cv2_shift;
    cfg.cv2_silu_lut=yolo_c2f6_cv2_lut; cfg.cv2_rq_zp=YOLO_C2F6_CV2_RQ_ZP;
    if (errors == 0u && !yolo_run_c2f_block(&cfg)) { print_str("  c2f6 fail\n"); errors++; }
    print_str("  [stage6 c2f_6 done]\n");

    // ---------- Stage 7: conv20 (downsample 20x20x128 -> 10x10x256, large-IC s2) ----------
    // icg=8 (>ICG_BUF) + stride2 3x3 -> IC-chunk streaming + CPU INT32 psum accumulate.
    if (errors == 0u && !yolo_run_conv2d_ic_stream(C2F6_OUT, WGT_OF(20), WGT_BASE, C20_OUT,
                               C20_PSUM, PAD_ROW, C20E_IN_W, C20E_IN_H, C20E_IC, C20E_OC,
                               3u, 3u, C20E_STRIDE, 1u,
                               yolo_conv20e_bias_q, yolo_conv20e_scale_mul, yolo_conv20e_scale_shift,
                               yolo_conv20e_silu_lut, C20E_PAD_VALUE)) {
        print_str("  conv20 fail\n"); errors++;
    }
    print_str("  [stage7 conv20 done]\n");

    // ---------- Stage 8: c2f_8 (model.8, n=1, 256ch) reads C20_OUT, blob weights ----------
    cfg.in_w=YOLO_C2F8_IN_W; cfg.in_h=YOLO_C2F8_IN_H; cfg.spatial=YOLO_C2F8_SPATIAL;
    cfg.full_c=YOLO_C2F8_FULL_C; cfg.n_bottleneck=YOLO_C2F8_N; cfg.shortcut=1u;
    cfg.in_ddr=C20_OUT; cfg.cv1_ic=YOLO_C2F8_CV1_IC; cfg.cv1_out_ddr=E_CV1;
    cfg.bn_out_ddr=E_BN; cfg.mcv2_ddr=E_MCV2; cfg.add_ddr[0]=E_ADD0;
    cfg.concat_ddr=E_CONCAT; cfg.out_ddr=C2F8_OUT; cfg.wgt_ddr=C2F_WGT;
    cfg.pad_row_ddr=PAD_ROW; cfg.strip=16u; cfg.silu_exact=1u; cfg.wgt_in_blob=1u;
    cfg.psum_ddr=E_PSUM;
    cfg.cv1_wgt_ddr=WGT_OF(21); cfg.cv2_wgt_ddr=WGT_OF(24);
    cfg.mcv1_wgt_ddr[0]=WGT_OF(22); cfg.mcv2_wgt_ddr[0]=WGT_OF(23);
    cfg.cv1_wgt_words=YOLO_C2F8_CV1_WGT_WORDS;
    cfg.cv1_bias=yolo_c2f8_cv1_bias; cfg.cv1_mul=yolo_c2f8_cv1_mul; cfg.cv1_shift=yolo_c2f8_cv1_shift;
    cfg.cv1_silu_lut=yolo_c2f8_cv1_lut; cfg.cv1_rq_zp=YOLO_C2F8_CV1_RQ_ZP;
    cfg.mcv1_wgt_words=YOLO_C2F8_MCV_WGT_WORDS; cfg.mcv2_wgt_words=YOLO_C2F8_MCV_WGT_WORDS;
    cfg.mcv1_bias[0]=yolo_c2f8_mcv1_0_bias; cfg.mcv1_mul[0]=yolo_c2f8_mcv1_0_mul; cfg.mcv1_shift[0]=yolo_c2f8_mcv1_0_shift;
    cfg.mcv1_silu_lut[0]=yolo_c2f8_mcv1_0_lut; cfg.mcv1_rq_zp[0]=YOLO_C2F8_MCV1_0_RQ_ZP; cfg.mcv1_pad_value[0]=YOLO_C2F8_MCV1_0_PAD;
    cfg.mcv1_rq_mul[0]=0u; cfg.mcv1_rq_shift[0]=YOLO_C2F8_GLUE_RQ_SHIFT;
    cfg.mcv2_bias[0]=yolo_c2f8_mcv2_0_bias; cfg.mcv2_mul[0]=yolo_c2f8_mcv2_0_mul; cfg.mcv2_shift[0]=yolo_c2f8_mcv2_0_shift;
    cfg.mcv2_silu_lut[0]=yolo_c2f8_mcv2_0_lut; cfg.mcv2_pad_value[0]=YOLO_C2F8_MCV2_0_PAD;
    cfg.glue_rq_mul[0]=0u; cfg.glue_rq_shift[0]=YOLO_C2F8_GLUE_RQ_SHIFT; cfg.glue_zp[0]=YOLO_C2F8_GLUE0_ZP;
    cfg.add_ratio_shift=YOLO_C2F8_ADD_RATIO_SHIFT;
    cfg.add_ratio_mul[0]=YOLO_C2F8_ADD0_RATIO_MUL; cfg.add_prev_zp[0]=YOLO_C2F8_ADD0_PREV_ZP;
    cfg.cat_req_shift=YOLO_C2F8_CAT_SHIFT; cfg.cat_zp=YOLO_C2F8_CAT_ZP;
    cfg.cat_mul_s0s1=YOLO_C2F8_CAT_MUL_S0S1; cfg.cat_inzp_s0s1=YOLO_C2F8_CAT_INZP_S0S1;
    cfg.cat_mul_add[0]=YOLO_C2F8_CAT_MUL_ADD0; cfg.cat_inzp_add[0]=YOLO_C2F8_CAT_INZP_ADD0;
    cfg.cv2_ic=YOLO_C2F8_CV2_IC; cfg.cv2_oc=YOLO_C2F8_CV2_OC;
    cfg.cv2_wgt_words=YOLO_C2F8_CV2_WGT_WORDS;
    cfg.cv2_bias=yolo_c2f8_cv2_bias; cfg.cv2_mul=yolo_c2f8_cv2_mul; cfg.cv2_shift=yolo_c2f8_cv2_shift;
    cfg.cv2_silu_lut=yolo_c2f8_cv2_lut; cfg.cv2_rq_zp=YOLO_C2F8_CV2_RQ_ZP;
    if (errors == 0u && !yolo_run_c2f_block(&cfg)) { print_str("  c2f8 fail\n"); errors++; }
    print_str("  [stage8 c2f_8 done]\n");

    // ---------- Stage 9: SPPF (model.9) conv25 -> 3x maxpool5 -> concat -> conv26 ----------
    // conv25 (1x1 256->128, icg16 PW stream, exact); maxpool/concat on CPU (scale-
    // preserving); conv26 (1x1 512->256, icg32 PW stream, exact). Reads C2F8_OUT.
    yolo_load_silu_lut(yolo_sppf_e_c25_lut);
    yolo_set_silu_requant(0u, 0u, 0);
    if (errors == 0u && !yolo_run_conv2d_tiled(C2F8_OUT, WGT_OF(25), WGT_BASE, S_CV1, PAD_ROW,
                               SPPFE_IN_W, SPPFE_IN_H, SPPFE_C25_IC, SPPFE_C25_OC, 1u, 1u, 1u, 0u,
                               yolo_sppf_e_c25_bias, yolo_sppf_e_c25_mul, yolo_sppf_e_c25_shift,
                               NPU_CTRL_SILU_EXACT_EN, SPPFE_C25_IC/16u, 16u, 0)) {
        print_str("  conv25 fail\n"); errors++;
    }
    if (errors == 0u) {
        uint32_t g25 = SPPFE_C25_OC / 16u;
        maxpool5(S_CV1, S_M0, g25, SPPFE_IN_H, SPPFE_IN_W);
        maxpool5(S_M0,  S_M1, g25, SPPFE_IN_H, SPPFE_IN_W);
        maxpool5(S_M1,  S_M2, g25, SPPFE_IN_H, SPPFE_IN_W);
        concat4(S_CV1, S_M0, S_M1, S_M2, S_CAT, g25, SPPFE_SPATIAL);
    }
    yolo_load_silu_lut(yolo_sppf_e_c26_lut);
    yolo_set_silu_requant(0u, 0u, 0);
    if (errors == 0u && !yolo_run_conv2d_tiled(S_CAT, WGT_OF(26), WGT_BASE, SPPF_OUT, PAD_ROW,
                               SPPFE_IN_W, SPPFE_IN_H, SPPFE_C26_IC, SPPFE_C26_OC, 1u, 1u, 1u, 0u,
                               yolo_sppf_e_c26_bias, yolo_sppf_e_c26_mul, yolo_sppf_e_c26_shift,
                               NPU_CTRL_SILU_EXACT_EN, SPPFE_C26_IC/16u, 16u, 0)) {
        print_str("  conv26 fail\n"); errors++;
    }
    print_str("  [stage9 SPPF done]\n");
    ck_stage("[ck SPPF]", SPPF_OUT, SPPFE_SPATIAL, SPPFE_C26_OC, &yolo_sppf_e_golden[0][0], SPPFE_C26_OC, 120u);

    // ================= NECK (FPN/PAN, model.10-21) =================
    // ---- FPN: P5 up + concat(P4=c2f6) -> c2f_12 -> fpn_mid (20x20x128) ----
    upsample2(SPPF_OUT, NK_UP1, 256u/16u, 10u, 10u);
    concat2_rq(NK_UP1, 256u/16u, YOLO_NK_CAT1_MUL_UP, YOLO_NK_CAT1_INZP_UP,
               C2F6_OUT, 128u/16u, YOLO_NK_CAT1_MUL_TAP, YOLO_NK_CAT1_INZP_TAP,
               YOLO_NK_CAT1_CAT_ZP, YOLO_NK_CAT1_CAT_SHIFT, NK_CAT1, 20u*20u);
    NECK_C2F(YOLO_NK_C12, yolo_nk_c12, NK_CAT1, NK_FMID, WGT_OF(27),WGT_OF(28),WGT_OF(29),WGT_OF(30));
    print_str("  [neck c2f_12 done]\n");

    // ---- FPN: fpn_mid up + concat(P3=c2f4) -> c2f_15 -> pan_p3 (40x40x64) ----
    upsample2(NK_FMID, NK_UP2, 128u/16u, 20u, 20u);
    concat2_rq(NK_UP2, 128u/16u, YOLO_NK_CAT2_MUL_UP, YOLO_NK_CAT2_INZP_UP,
               C2F4_OUT, 64u/16u, YOLO_NK_CAT2_MUL_TAP, YOLO_NK_CAT2_INZP_TAP,
               YOLO_NK_CAT2_CAT_ZP, YOLO_NK_CAT2_CAT_SHIFT, NK_CAT2, 40u*40u);
    NECK_C2F(YOLO_NK_C15, yolo_nk_c15, NK_CAT2, NK_PANP3, WGT_OF(31),WGT_OF(32),WGT_OF(33),WGT_OF(34));
    ck_stage("[ck pan_p3]", NK_PANP3, YOLO_NK_P3_SP, YOLO_NK_P3_OC, &yolo_nk_pan_p3_golden[0][0], YOLO_NK_P3_OC, 120u);

    // ---- PAN: conv35(pan_p3 3x3 s2) + concat(fpn_mid) -> c2f_18 -> pan_p4 (20x20x128) ----
    yolo_set_pad_value(YOLO_NK_C35_PAD);
    yolo_load_silu_lut(yolo_nk_c35_lut); yolo_set_silu_requant(0u,0u,0);
    if (errors==0u && !yolo_run_conv2d_tiled(NK_PANP3, WGT_OF(35), WGT_BASE, NK_C35, PAD_ROW,
                               40u,40u, 64u,64u, 3u,3u, 2u,1u,
                               yolo_nk_c35_bias, yolo_nk_c35_mul, yolo_nk_c35_shift,
                               NPU_CTRL_SILU_EXACT_EN, 4u*9u, 16u, YOLO_NK_C35_PAD)) { print_str("  conv35 fail\n"); errors++; }
    concat2_rq(NK_C35, 64u/16u, YOLO_NK_CAT3_MUL_UP, YOLO_NK_CAT3_INZP_UP,
               NK_FMID, 128u/16u, YOLO_NK_CAT3_MUL_TAP, YOLO_NK_CAT3_INZP_TAP,
               YOLO_NK_CAT3_CAT_ZP, YOLO_NK_CAT3_CAT_SHIFT, NK_CAT3, 20u*20u);
    NECK_C2F(YOLO_NK_C18, yolo_nk_c18, NK_CAT3, NK_PANP4, WGT_OF(40),WGT_OF(43),WGT_OF(44),WGT_OF(45));
    ck_stage("[ck pan_p4]", NK_PANP4, YOLO_NK_P4_SP, YOLO_NK_P4_OC, &yolo_nk_pan_p4_golden[0][0], YOLO_NK_P4_OC, 120u);

    // ---- PAN: conv46(pan_p4 3x3 s2, large-IC) + concat(c2f8) -> c2f_21 -> pan_p5 (10x10x256) ----
    if (errors==0u && !yolo_run_conv2d_ic_stream(NK_PANP4, WGT_OF(46), WGT_BASE, NK_C46, NK_PSUM, PAD_ROW,
                               20u,20u, 128u,128u, 3u,3u, 2u,1u,
                               yolo_nk_c46_bias, yolo_nk_c46_mul, yolo_nk_c46_shift,
                               yolo_nk_c46_lut, YOLO_NK_C46_PAD)) { print_str("  conv46 fail\n"); errors++; }
    concat2_rq(NK_C46, 128u/16u, YOLO_NK_CAT4_MUL_UP, YOLO_NK_CAT4_INZP_UP,
               C2F8_OUT, 256u/16u, YOLO_NK_CAT4_MUL_TAP, YOLO_NK_CAT4_INZP_TAP,
               YOLO_NK_CAT4_CAT_ZP, YOLO_NK_CAT4_CAT_SHIFT, NK_CAT4, 10u*10u);
    NECK_C2F(YOLO_NK_C21, yolo_nk_c21, NK_CAT4, NK_PANP5, WGT_OF(51),WGT_OF(54),WGT_OF(55),WGT_OF(56));
    ck_stage("[ck pan_p5]", NK_PANP5, YOLO_NK_P5_SP, YOLO_NK_P5_OC, &yolo_nk_pan_p5_golden[0][0], YOLO_NK_P5_OC, 120u);

    print_str("  [neck done]\n");
    if (errors == 0u) { print_str("YOLO BACKBONE+SPPF+NECK PASS (conv0->...->pan_p5)\n"); return; }
    print_str("YOLO NECK FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
