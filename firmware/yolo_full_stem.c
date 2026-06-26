// yolo_full.c stem: FULL on-SoC stem chain conv0 -> conv1 -> c2f_2, all through
// DDR with exact-SiLU. conv0 reads the DDR-PRELOADED 320x320x3 image (no bake;
// requires `touch .yolo_ddr` so the shared-mem model $readmemh's the image into
// 0x4040_0000). Each layer's RTL output feeds the next directly (scale/zp/tile-
// major layout match by construction). Final compared to the c2f_2 golden with a
// propagation-aware tolerance (conv0/conv1 quantization +-1..2 propagates).
//
// Build: touch .yolo_ddr; bash run_all.sh sim yolo_full_stem.c yolo_c2f.c yolo_ops.c

#include "firmware.h"
#include "npu_desc.h"
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
#include "yolo_head_data.h"
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
#define S_POOL_ACT 2048u        // Act-SRAM scratch for generic 5x5 maxpool engine
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
// ---- HEAD (model.22) buffers (dead backbone region < C2F4_OUT) ----
#define HD_SCR1   0x40D10000u   // stem out (reused per branch)
#define HD_SCR2   0x40D40000u   // mid out (reused per branch)
#define HD_BB_P3  0x40D70000u   // bbox int8 per scale
#define HD_CL_P3  0x40DA0000u
#define HD_BB_P4  0x40DD0000u
#define HD_CL_P4  0x40DE0000u
#define HD_BB_P5  0x40DF0000u
#define HD_CL_P5  0x40E00000u
#define HD_PSUM   0x40E20000u   // ic_stream psum for head large-IC convs
#define HD_DFL_IN  0x40E60000u  // reordered [anchor][coord] DFL input scratch
#define HD_DFL_OUT 0x40E64000u  // packed Q8.8 DFL output scratch
#define HD_DFL_ACT 4096u        // Act-SRAM scratch for generic DFL engine
#define WGT_BASE  0u

static int32_t rs8(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); uint32_t v=p[b>>2]>>((b&3u)*8u)&0xFFu; return (v&0x80u)?((int32_t)v-256):(int32_t)v; }
static int32_t s8(uint32_t b){ b&=0xFFu; return (b&0x80u)?((int32_t)b-256):(int32_t)b; }
static uint32_t ad(int32_t a,int32_t b) __attribute__((unused));
static uint32_t ad(int32_t a,int32_t b){ int32_t d=a-b; return (uint32_t)(d<0?-d:d); }

// ---- Phase-0 per-stage profiler (reads free-running RTL perf counters) ----
// dcyc = wall cycles for the stage; dnpu = NPU-FSM-busy cycles in the stage;
// (dcyc - dnpu) ~= CPU/DMA software-glue cost. drd/dwr = AXI DDR beats.
// YOLO-only file (not compiled for MNIST), so this never perturbs the baseline.
static uint32_t pf_tot, pf_busy, pf_rd, pf_wr;
static void prof_reset(void)
{
    pf_tot  = *(volatile uint32_t*)NPU_PERF_CYC_TOTAL;
    pf_busy = *(volatile uint32_t*)NPU_PERF_CYC_BUSY;
    pf_rd   = *(volatile uint32_t*)NPU_PERF_RD_BEATS;
    pf_wr   = *(volatile uint32_t*)NPU_PERF_WR_BEATS;
}
static void prof_mark(const char *tag)
{
    uint32_t t = *(volatile uint32_t*)NPU_PERF_CYC_TOTAL;
    uint32_t b = *(volatile uint32_t*)NPU_PERF_CYC_BUSY;
    uint32_t r = *(volatile uint32_t*)NPU_PERF_RD_BEATS;
    uint32_t w = *(volatile uint32_t*)NPU_PERF_WR_BEATS;
    print_str("[PROF] "); print_str(tag);
    print_str(" dcyc="); print_dec(t - pf_tot);
    print_str(" dnpu="); print_dec(b - pf_busy);
    print_str(" drd=");  print_dec(r - pf_rd);
    print_str(" dwr=");  print_dec(w - pf_wr);
    print_str("\n");
    pf_tot = t; pf_busy = b; pf_rd = r; pf_wr = w;
}

// SPPF 5x5 stride-1 pad-2 int8 max-pool, tile-major DDR (groups of 16 channels).
// 5x5 stride-1 pad-2 int8 maxpool, SEPARABLE: a 1x5 horizontal max into a local
// buffer, then a 5x1 vertical max -> 10 taps/pixel instead of 25 (~2.5x fewer
// scalar max ops; the stage is CPU-arithmetic-bound, not memory-bound). The src
// plane is staged into fast local RAM once per group. Output bit-identical.
static void maxpool5(uint32_t src, uint32_t dst, uint32_t groups, uint32_t H, uint32_t W) __attribute__((unused));
static void maxpool5(uint32_t src, uint32_t dst, uint32_t groups, uint32_t H, uint32_t W)
{
    static int8_t plane[256*16];   // src group plane (SP<=256, 16 lanes), signed
    static int8_t hmax[256*16];    // horizontal 1x5 max
    uint32_t SP = H*W, g, oh, ow, kk, k, p;
    for (g = 0u; g < groups; g++) {
        volatile uint32_t *sp = (volatile uint32_t*)(src + g*SP*16u);
        for (p = 0u; p < SP; p++) {                       // DDR -> local int8 (once)
            uint32_t w0=sp[p*4u],w1=sp[p*4u+1u],w2=sp[p*4u+2u],w3=sp[p*4u+3u];
            int8_t *pl=&plane[p*16u];
            for (k=0u;k<4u;k++) pl[k]   =(int8_t)(w0>>(k*8u));
            for (k=0u;k<4u;k++) pl[4u+k]=(int8_t)(w1>>(k*8u));
            for (k=0u;k<4u;k++) pl[8u+k]=(int8_t)(w2>>(k*8u));
            for (k=0u;k<4u;k++) pl[12u+k]=(int8_t)(w3>>(k*8u));
        }
        // horizontal 1x5 max -> hmax
        for (oh = 0u; oh < H; oh++)
            for (ow = 0u; ow < W; ow++) {
                int8_t *o = &hmax[(oh*W+ow)*16u];
                for (k=0u;k<16u;k++) o[k] = -128;
                for (kk = 0u; kk < 5u; kk++) {
                    int32_t iw = (int32_t)ow-2+(int32_t)kk;
                    if (iw>=0 && iw<(int32_t)W) {
                        int8_t *q=&plane[(oh*W+(uint32_t)iw)*16u];
                        for (k=0u;k<16u;k++) if (q[k]>o[k]) o[k]=q[k];
                    }
                }
            }
        // vertical 5x1 max -> dst
        for (oh = 0u; oh < H; oh++)
            for (ow = 0u; ow < W; ow++) {
                int32_t mx[16]; uint32_t out[4]={0u,0u,0u,0u};
                for (k=0u;k<16u;k++) mx[k]=-128;
                for (kk = 0u; kk < 5u; kk++) {
                    int32_t ih = (int32_t)oh-2+(int32_t)kk;
                    if (ih>=0 && ih<(int32_t)H) {
                        int8_t *q=&hmax[((uint32_t)ih*W+ow)*16u];
                        for (k=0u;k<16u;k++) if (q[k]>mx[k]) mx[k]=q[k];
                    }
                }
                for (k=0u;k<16u;k++) out[k>>2] |= ((uint32_t)((uint8_t)mx[k]))<<((k&3u)*8u);
                { volatile uint32_t *t=(volatile uint32_t*)(dst+(g*SP+oh*W+ow)*16u); t[0]=out[0];t[1]=out[1];t[2]=out[2];t[3]=out[3]; }
            }
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
static void upsample2(uint32_t src, uint32_t dst, uint32_t groups, uint32_t H, uint32_t W) __attribute__((unused));
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
// plain tile-major tensor copy (groups*SP 128-bit words), no requant. Used to lay
// the backbone tap next to the upsampled feature in the FPN/PAN cat buffer; the cat
// requant is folded into the neck c2f cv1 weights, so no per-lane math here.
static void copy_groups(uint32_t src, uint32_t dst, uint32_t groups, uint32_t SP) __attribute__((unused));
static void copy_groups(uint32_t src, uint32_t dst, uint32_t groups, uint32_t SP)
{
    uint32_t n, words = groups*SP;
    for (n = 0u; n < words; n++) {
        volatile uint32_t *s=(volatile uint32_t*)(src+n*16u);
        volatile uint32_t *t=(volatile uint32_t*)(dst+n*16u);
        t[0]=s[0];t[1]=s[1];t[2]=s[2];t[3]=s[3];
    }
}
// signed-int8 requant one lane: clamp((q-inzp)*mul>>sh + catzp)
static int32_t rq_lane(int32_t q, int32_t inzp, uint32_t mul, uint32_t sh, int32_t catzp) __attribute__((unused));
static int32_t rq_lane(int32_t q, int32_t inzp, uint32_t mul, uint32_t sh, int32_t catzp)
{ int32_t v=(((q-inzp)*(int32_t)mul + (1<<(sh-1u)))>>sh)+catzp; return v>127?127:(v<-128?-128:v); }
// concat2 with per-part requant to a common (cat) scale -> [up_g+tap_g] tile-major.
// Superseded by the cv1 cat-fold (copy_groups + folded weights); kept for reference.
static void concat2_rq(uint32_t up, uint32_t up_g, uint32_t up_mul, int32_t up_iz,
                       uint32_t tap, uint32_t tap_g, uint32_t tap_mul, int32_t tap_iz,
                       int32_t catzp, uint32_t sh, uint32_t dst, uint32_t SP) __attribute__((unused));
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
// Validation-only (gated by YOLO_DEBUG_CK at the call sites); may be unused.
static void ck_stage(const char *nm, uint32_t base, uint32_t SP, uint32_t OC,
                     const uint8_t *golden, uint32_t gOC, uint32_t tol) __attribute__((unused));
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
    cfg.cv2_silu_lut=p##_cv2_lut; cfg.cv2_rq_zp=P##_CV2_ZP; cfg.cv2_folded=P##_CV2_FOLDED; \
    if (errors==0u && !yolo_run_c2f_block(&cfg)) { print_str("  neck c2f fail\n"); errors++; } \
} while(0)

/* One head conv (exact): 3x3 large-IC -> ic_stream; else conv2d_tiled resident /
 * 1x1 PW (handles pw-stream internally). lut = exact-SiLU or linear LUT. */
static int run_head_conv(uint32_t in, uint32_t wgt, uint32_t out, uint32_t inw, uint32_t inh,
                         uint32_t ic, uint32_t oc, uint32_t kh, uint32_t stride, uint32_t pad,
                         const int32_t *bias, const uint32_t *mul, const uint32_t *shift,
                         const uint8_t *lut, int32_t inzp)
{
    if (kh == 3u && (ic/16u) > YOLO_ICG_BUF)
        return yolo_run_conv2d_ic_stream(in, wgt, WGT_BASE, out, HD_PSUM, PAD_ROW,
                   inw, inh, ic, oc, 3u, 3u, stride, pad, bias, mul, shift, lut, inzp);
    yolo_set_pad_value(inzp); yolo_load_silu_lut(lut); yolo_set_silu_requant(0u,0u,0);
    return yolo_run_conv2d_tiled(in, wgt, WGT_BASE, out, PAD_ROW, inw, inh, ic, oc,
               kh, kh, stride, pad, bias, mul, shift, NPU_CTRL_SILU_EXACT_EN,
               (ic/16u)*kh*kh, 16u, inzp);
}
/* expf approx (bare-metal, no libm): exp(x)=2^(x*log2e), poly for frac + float exp. */
static float my_exp(float x)
{
    if (x < -30.0f) return 0.0f;
    if (x > 30.0f) x = 30.0f;
    float t = x * 1.44269504f; int i = (int)(t < 0.0f ? t - 1.0f : t); float f = t - (float)i;
    float p = 1.0f + f*(0.6931472f + f*(0.2402265f + f*(0.0555041f + f*0.0096181f)));
    union { float fl; uint32_t u; } v; int e = 127 + i;
    if (e < 0) e = 0;
    if (e > 254) e = 254;
    v.u = ((uint32_t)e) << 23; return p * v.fl;
}
static void dfl_prepare_tables(float scale, int16_t wk[16], uint16_t elut[256])
{
    uint32_t i;
    float e = 1.0f;
    float step = my_exp(-scale);
    for (i = 0u; i < 16u; i++)
        wk[i] = (int16_t)(yolo_hd_dfl_w[i] * 256.0f + 0.5f);
    for (i = 0u; i < 256u; i++) {
        uint32_t q = (uint32_t)(e * 32768.0f + 0.5f);
        if (q > 65535u) q = 65535u;
        elut[i] = (uint16_t)q;
        e *= step;
    }
}
static float q8_8_to_float(uint16_t q)
{
    return ((float)((int16_t)q)) * (1.0f / 256.0f);
}
static float dq(uint32_t base, uint32_t sp, uint32_t ch, uint32_t pos, int32_t zp, float sc)
{ return (float)(rs8(base, (ch>>4)*sp + pos, ch&15u) - zp) * sc; }

void usercode7(void)
{
    uint32_t errors = 0u;
    yolo_c2f_cfg_t cfg;

    print_str("YOLO FULL STEM: conv0 -> ... -> SPPF (on-SoC, DDR, preloaded img)\n");
    /* intermediate goldens are checkpoints; only the final (SPPF) one is validated */
    (void)yolo_c2f4_golden; (void)yolo_c2f6_golden; (void)yolo_c2f8_golden;
    /* Stage checksums (ck_stage) are validation-only; gate them out of the deploy
     * path (define YOLO_DEBUG_CK to re-enable). The final 4-box check is the real
     * functional gate. These (void) casts keep -Werror happy when ck is gated. */
    (void)yolo_sppf_e_golden; (void)yolo_nk_pan_p3_golden;
    (void)yolo_nk_pan_p4_golden; (void)yolo_nk_pan_p5_golden;
    prof_reset();

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
    prof_mark("s0_conv0");

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
    prof_mark("s1_conv1");

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
    cfg.cv2_folded=YOLO_C2F2_CV2_FOLDED;   // concat folded into cv2 weights (blob has folded conv5)
    cfg.cv2_silu_lut=yolo_c2f2_cv2_silu_lut;
    if (errors == 0u && !yolo_run_c2f_block(&cfg)) { print_str("  c2f2 fail\n"); errors++; }
    print_str("  [stage2 c2f_2 done]\n");
    prof_mark("s2_c2f2");

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
    prof_mark("s3_conv6");

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
    cfg.cv2_folded=YOLO_C2F4_CV2_FOLDED;   // concat requant folded into cv2 weights (blob has w2f)
    if (errors == 0u && !yolo_run_c2f_block(&cfg)) { print_str("  c2f4 fail\n"); errors++; }
    print_str("  [stage4 c2f_4 done]\n");
    prof_mark("s4_c2f4");

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
    prof_mark("s5_conv13");

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
    cfg.cv2_folded=YOLO_C2F6_CV2_FOLDED;
    if (errors == 0u && !yolo_run_c2f_block(&cfg)) { print_str("  c2f6 fail\n"); errors++; }
    print_str("  [stage6 c2f_6 done]\n");
    prof_mark("s6_c2f6");

    // ---------- Stage 7: conv20 (downsample 20x20x128 -> 10x10x256, large-IC s2) ----------
    // icg=8 now fits the resident path (ICG_MAX/ICG_BUF=8): tiled conv (stride2 serial),
    // no CPU INT32 psum. (void)C20_PSUM.
    (void)C20_PSUM;
    yolo_set_pad_value(C20E_PAD_VALUE);
    yolo_load_silu_lut(yolo_conv20e_silu_lut); yolo_set_silu_requant(0u, 0u, 0);
    if (errors == 0u && !yolo_run_conv2d_tiled(C2F6_OUT, WGT_OF(20), WGT_BASE, C20_OUT, PAD_ROW,
                               C20E_IN_W, C20E_IN_H, C20E_IC, C20E_OC, 3u, 3u, C20E_STRIDE, 1u,
                               yolo_conv20e_bias_q, yolo_conv20e_scale_mul, yolo_conv20e_scale_shift,
                               NPU_CTRL_SILU_EXACT_EN, (C20E_IC/16u)*9u, 16u, C20E_PAD_VALUE)) {
        print_str("  conv20 fail\n"); errors++;
    }
    print_str("  [stage7 conv20 done]\n");
    prof_mark("s7_conv20_icstream");

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
    cfg.cv2_folded=YOLO_C2F8_CV2_FOLDED;
    if (errors == 0u && !yolo_run_c2f_block(&cfg)) { print_str("  c2f8 fail\n"); errors++; }
    print_str("  [stage8 c2f_8 done]\n");
    prof_mark("s8_c2f8");

    // ---------- Stage 9: SPPF (model.9) conv25 -> 3x maxpool5 -> concat -> conv26 ----------
    // conv25 (1x1 256->128, icg16 PW stream, exact); maxpool/concat on CPU (scale-
    // preserving); conv26 (1x1 512->256, icg32 PW stream, exact). Reads C2F8_OUT.
    yolo_load_silu_lut(yolo_sppf_e_c25_lut);
    yolo_set_silu_requant(0u, 0u, 0);
    {
        const npu_desc_t d = {
            .op = NPU_DESC_OP_CONV2D_TILED,
            .src0 = C2F8_OUT,
            .wgt = WGT_OF(25),
            .scratch0 = WGT_BASE,
            .dst = S_CV1,
            .scratch1 = PAD_ROW,
            .in_w = SPPFE_IN_W,
            .in_h = SPPFE_IN_H,
            .in_c = SPPFE_C25_IC,
            .out_c = SPPFE_C25_OC,
            .kh = 1u,
            .kw = 1u,
            .stride = 1u,
            .pad = 0u,
            .bias = yolo_sppf_e_c25_bias,
            .scale_mul = yolo_sppf_e_c25_mul,
            .scale_shift = yolo_sppf_e_c25_shift,
            .flags = NPU_CTRL_SILU_EXACT_EN,
            .wgt_words_per_oc = SPPFE_C25_IC/16u,
            .strip_out_rows = 16u,
            .pad_value = 0
        };
        if (errors == 0u && !npu_desc_run(&d)) {
            print_str("  conv25 desc fail\n"); errors++;
        }
    }
    prof_mark("s9_sppf_conv25");
    if (errors == 0u) {
        uint32_t g25 = SPPFE_C25_OC / 16u;
        if (!yolo_run_maxpool5x5(S_CV1, S_M0, S_POOL_ACT, SPPFE_IN_W, SPPFE_IN_H, g25)) { print_str("  sppf pool0 fail\n"); errors++; }
        if (errors == 0u && !yolo_run_maxpool5x5(S_M0, S_M1, S_POOL_ACT, SPPFE_IN_W, SPPFE_IN_H, g25)) { print_str("  sppf pool1 fail\n"); errors++; }
        if (errors == 0u && !yolo_run_maxpool5x5(S_M1, S_M2, S_POOL_ACT, SPPFE_IN_W, SPPFE_IN_H, g25)) { print_str("  sppf pool2 fail\n"); errors++; }
        concat4(S_CV1, S_M0, S_M1, S_M2, S_CAT, g25, SPPFE_SPATIAL);
    }
    prof_mark("s9_sppf_maxpool_concat_CPU");
    yolo_load_silu_lut(yolo_sppf_e_c26_lut);
    yolo_set_silu_requant(0u, 0u, 0);
    {
        const npu_desc_t d = {
            .op = NPU_DESC_OP_CONV2D_TILED,
            .src0 = S_CAT,
            .wgt = WGT_OF(26),
            .scratch0 = WGT_BASE,
            .dst = SPPF_OUT,
            .scratch1 = PAD_ROW,
            .in_w = SPPFE_IN_W,
            .in_h = SPPFE_IN_H,
            .in_c = SPPFE_C26_IC,
            .out_c = SPPFE_C26_OC,
            .kh = 1u,
            .kw = 1u,
            .stride = 1u,
            .pad = 0u,
            .bias = yolo_sppf_e_c26_bias,
            .scale_mul = yolo_sppf_e_c26_mul,
            .scale_shift = yolo_sppf_e_c26_shift,
            .flags = NPU_CTRL_SILU_EXACT_EN,
            .wgt_words_per_oc = SPPFE_C26_IC/16u,
            .strip_out_rows = 16u,
            .pad_value = 0
        };
        if (errors == 0u && !npu_desc_run(&d)) {
            print_str("  conv26 desc fail\n"); errors++;
        }
    }
    prof_mark("s9_sppf_conv26");
    print_str("  [stage9 SPPF done]\n");
#ifdef YOLO_DEBUG_CK
    ck_stage("[ck SPPF]", SPPF_OUT, SPPFE_SPATIAL, SPPFE_C26_OC, &yolo_sppf_e_golden[0][0], SPPFE_C26_OC, 120u);
    prof_mark("ck_sppf_DEBUG");
#endif

    // ================= NECK (FPN/PAN, model.10-21) =================
    // ---- FPN: P5 up + concat(P4=c2f6) -> c2f_12 -> fpn_mid (20x20x128) ----
    // cat folded into c2f_12 cv1 weights: build raw [up|tap] native (no requant).
    {
        const npu_desc_t descs[] = {
            {
                .op = NPU_DESC_OP_UPSAMPLE2X_DDR,
                .src0 = SPPF_OUT,
                .dst = NK_CAT1,
                .scratch0 = 0u,
                .in_w = 10u,
                .in_h = 10u,
                .in_c = 256u
            },
            {
                .op = NPU_DESC_OP_COPY_DDR_TO_DDR,
                .src0 = C2F6_OUT,
                .dst = NK_CAT1 + (256u/16u)*(20u*20u)*16u,
                .scratch0 = 0u,
                .words = (128u/16u)*(20u*20u)
            }
        };
        if (errors == 0u && !npu_desc_run_many(descs, 2u)) {
            print_str("  neck cat1 desc fail\n"); errors++;
        }
    }
    prof_mark("neck_up1_cat1_DMA");
    NECK_C2F(YOLO_NK_C12, yolo_nk_c12, NK_CAT1, NK_FMID, WGT_OF(27),WGT_OF(28),WGT_OF(29),WGT_OF(30));
    print_str("  [neck c2f_12 done]\n");
    prof_mark("neck_c2f12");

    // ---- FPN: fpn_mid up + concat(P3=c2f4) -> c2f_15 -> pan_p3 (40x40x64) ----
    {
        const npu_desc_t descs[] = {
            {
                .op = NPU_DESC_OP_UPSAMPLE2X_DDR,
                .src0 = NK_FMID,
                .dst = NK_CAT2,
                .scratch0 = 0u,
                .in_w = 20u,
                .in_h = 20u,
                .in_c = 128u
            },
            {
                .op = NPU_DESC_OP_COPY_DDR_TO_DDR,
                .src0 = C2F4_OUT,
                .dst = NK_CAT2 + (128u/16u)*(40u*40u)*16u,
                .scratch0 = 0u,
                .words = (64u/16u)*(40u*40u)
            }
        };
        if (errors == 0u && !npu_desc_run_many(descs, 2u)) {
            print_str("  neck cat2 desc fail\n"); errors++;
        }
    }
    prof_mark("neck_up2_cat2_DMA");
    NECK_C2F(YOLO_NK_C15, yolo_nk_c15, NK_CAT2, NK_PANP3, WGT_OF(31),WGT_OF(32),WGT_OF(33),WGT_OF(34));
    prof_mark("neck_c2f15");
#ifdef YOLO_DEBUG_CK
    ck_stage("[ck pan_p3]", NK_PANP3, YOLO_NK_P3_SP, YOLO_NK_P3_OC, &yolo_nk_pan_p3_golden[0][0], YOLO_NK_P3_OC, 120u);
    prof_mark("ck_panp3_DEBUG");
#endif

    // ---- PAN: conv35(pan_p3 3x3 s2) + concat(fpn_mid) -> c2f_18 -> pan_p4 (20x20x128) ----
    yolo_set_pad_value(YOLO_NK_C35_PAD);
    yolo_load_silu_lut(yolo_nk_c35_lut); yolo_set_silu_requant(0u,0u,0);
    if (errors==0u && !yolo_run_conv2d_tiled(NK_PANP3, WGT_OF(35), WGT_BASE, NK_C35, PAD_ROW,
                               40u,40u, 64u,64u, 3u,3u, 2u,1u,
                               yolo_nk_c35_bias, yolo_nk_c35_mul, yolo_nk_c35_shift,
                               NPU_CTRL_SILU_EXACT_EN, 4u*9u, 16u, YOLO_NK_C35_PAD)) { print_str("  conv35 fail\n"); errors++; }
    prof_mark("neck_conv35_s2");
    {
        const npu_desc_t descs[] = {
            {
                .op = NPU_DESC_OP_COPY_DDR_TO_DDR,
                .src0 = NK_C35,
                .dst = NK_CAT3,
                .scratch0 = 0u,
                .words = (64u/16u)*(20u*20u)
            },
            {
                .op = NPU_DESC_OP_COPY_DDR_TO_DDR,
                .src0 = NK_FMID,
                .dst = NK_CAT3 + (64u/16u)*(20u*20u)*16u,
                .scratch0 = 0u,
                .words = (128u/16u)*(20u*20u)
            }
        };
        if (errors == 0u && !npu_desc_run_many(descs, 2u)) {
            print_str("  neck cat3 desc fail\n"); errors++;
        }
    }
    prof_mark("neck_cat3_DMA");
    NECK_C2F(YOLO_NK_C18, yolo_nk_c18, NK_CAT3, NK_PANP4, WGT_OF(40),WGT_OF(43),WGT_OF(44),WGT_OF(45));
    prof_mark("neck_c2f18");
#ifdef YOLO_DEBUG_CK
    ck_stage("[ck pan_p4]", NK_PANP4, YOLO_NK_P4_SP, YOLO_NK_P4_OC, &yolo_nk_pan_p4_golden[0][0], YOLO_NK_P4_OC, 120u);
    prof_mark("ck_panp4_DEBUG");
#endif

    // ---- PAN: conv46(pan_p4 3x3 s2, icg8) + concat(c2f8) -> c2f_21 -> pan_p5 (10x10x256) ----
    // icg8 now resident (ICG_MAX/ICG_BUF=8): tiled conv instead of CPU ic_stream. (void)NK_PSUM.
    (void)NK_PSUM;
    yolo_set_pad_value(YOLO_NK_C46_PAD);
    yolo_load_silu_lut(yolo_nk_c46_lut); yolo_set_silu_requant(0u, 0u, 0);
    if (errors==0u && !yolo_run_conv2d_tiled(NK_PANP4, WGT_OF(46), WGT_BASE, NK_C46, PAD_ROW,
                               20u,20u, 128u,128u, 3u,3u, 2u,1u,
                               yolo_nk_c46_bias, yolo_nk_c46_mul, yolo_nk_c46_shift,
                               NPU_CTRL_SILU_EXACT_EN, (128u/16u)*9u, 16u, YOLO_NK_C46_PAD)) { print_str("  conv46 fail\n"); errors++; }
    prof_mark("neck_conv46_s2_icstream");
    {
        const npu_desc_t descs[] = {
            {
                .op = NPU_DESC_OP_COPY_DDR_TO_DDR,
                .src0 = NK_C46,
                .dst = NK_CAT4,
                .scratch0 = 0u,
                .words = (128u/16u)*(10u*10u)
            },
            {
                .op = NPU_DESC_OP_COPY_DDR_TO_DDR,
                .src0 = C2F8_OUT,
                .dst = NK_CAT4 + (128u/16u)*(10u*10u)*16u,
                .scratch0 = 0u,
                .words = (256u/16u)*(10u*10u)
            }
        };
        if (errors == 0u && !npu_desc_run_many(descs, 2u)) {
            print_str("  neck cat4 desc fail\n"); errors++;
        }
    }
    prof_mark("neck_cat4_DMA");
    NECK_C2F(YOLO_NK_C21, yolo_nk_c21, NK_CAT4, NK_PANP5, WGT_OF(51),WGT_OF(54),WGT_OF(55),WGT_OF(56));
    prof_mark("neck_c2f21");
#ifdef YOLO_DEBUG_CK
    ck_stage("[ck pan_p5]", NK_PANP5, YOLO_NK_P5_SP, YOLO_NK_P5_OC, &yolo_nk_pan_p5_golden[0][0], YOLO_NK_P5_OC, 120u);
    prof_mark("ck_panp5_DEBUG");
#endif

    print_str("  [neck done]\n");

    // ================= HEAD (model.22) =================
    // 6 branches: bbox(64)/cls(80) per scale = stem(3x3)->mid(3x3)->out(1x1 linear).
    #define HBR(IN,INW,INH, P, p, BB_CI,BBM_CI,BBO_CI, CL_CI,CLM_CI,CLO_CI, BBDST,CLDST) do { \
      if(errors==0u && !run_head_conv(IN,WGT_OF(BB_CI),HD_SCR1,INW,INH,YOLO_HD_##P##_STEM_IC,YOLO_HD_##P##_BB_STEM_OC,3u,1u,1u, \
            p##_bb_stem_bias,p##_bb_stem_mul,p##_bb_stem_shift,p##_bb_stem_lut,YOLO_HD_##P##_BB_STEM_PAD)) {print_str(" hbbstem fail\n");errors++;} \
      prof_mark("h_bbstem3x3"); \
      if(errors==0u && !run_head_conv(HD_SCR1,WGT_OF(BBM_CI),HD_SCR2,INW,INH,YOLO_HD_##P##_BB_STEM_OC,YOLO_HD_##P##_BB_STEM_OC,3u,1u,1u, \
            p##_bb_mid_bias,p##_bb_mid_mul,p##_bb_mid_shift,p##_bb_mid_lut,YOLO_HD_##P##_BB_MID_PAD)) {print_str(" hbbmid fail\n");errors++;} \
      prof_mark("h_bbmid3x3"); \
      if(errors==0u && !run_head_conv(HD_SCR2,WGT_OF(BBO_CI),BBDST,INW,INH,YOLO_HD_##P##_BB_STEM_OC,64u,1u,1u,0u, \
            p##_bb_out_bias,p##_bb_out_mul,p##_bb_out_shift,p##_bb_out_lut,YOLO_HD_##P##_BB_OUT_PAD)) {print_str(" hbbout fail\n");errors++;} \
      prof_mark("h_bbout1x1"); \
      if(errors==0u && !run_head_conv(IN,WGT_OF(CL_CI),HD_SCR1,INW,INH,YOLO_HD_##P##_STEM_IC,YOLO_HD_##P##_CL_STEM_OC,3u,1u,1u, \
            p##_cl_stem_bias,p##_cl_stem_mul,p##_cl_stem_shift,p##_cl_stem_lut,YOLO_HD_##P##_CL_STEM_PAD)) {print_str(" hclstem fail\n");errors++;} \
      prof_mark("h_clstem3x3"); \
      if(errors==0u && !run_head_conv(HD_SCR1,WGT_OF(CLM_CI),HD_SCR2,INW,INH,YOLO_HD_##P##_CL_STEM_OC,YOLO_HD_##P##_CL_STEM_OC,3u,1u,1u, \
            p##_cl_mid_bias,p##_cl_mid_mul,p##_cl_mid_shift,p##_cl_mid_lut,YOLO_HD_##P##_CL_MID_PAD)) {print_str(" hclmid fail\n");errors++;} \
      prof_mark("h_clmid3x3"); \
      if(errors==0u && !run_head_conv(HD_SCR2,WGT_OF(CLO_CI),CLDST,INW,INH,YOLO_HD_##P##_CL_STEM_OC,80u,1u,1u,0u, \
            p##_cl_out_bias,p##_cl_out_mul,p##_cl_out_shift,p##_cl_out_lut,YOLO_HD_##P##_CL_OUT_PAD)) {print_str(" hclout fail\n");errors++;} \
      prof_mark("h_clout1x1"); \
    } while(0)
    HBR(NK_PANP3,40u,40u, P3, yolo_hd_p3, 36,38,41, 37,39,42, HD_BB_P3,HD_CL_P3);
    prof_mark("head_P3_6conv");
    HBR(NK_PANP4,20u,20u, P4, yolo_hd_p4, 47,49,52, 48,50,53, HD_BB_P4,HD_CL_P4);
    prof_mark("head_P4_6conv");
    HBR(NK_PANP5,10u,10u, P5, yolo_hd_p5, 57,59,61, 58,60,62, HD_BB_P5,HD_CL_P5);
    prof_mark("head_P5_6conv");
    print_str("  [head convs done]\n");

    // ---- decode: class0 gate on CPU, DFL expectation on the generic NPU DFL engine ----
    {
        static float dx1[128],dy1[128],dx2[128],dy2[128]; uint32_t nd=0u;
        static uint16_t dfl_elut[256];
        static int16_t dfl_wk[16];
        static uint16_t cand_ax[128], cand_ay[128];
        uint32_t bbB[3]={HD_BB_P3,HD_BB_P4,HD_BB_P5}, clB[3]={HD_CL_P3,HD_CL_P4,HD_CL_P5};
        uint32_t hw[3]={40u,20u,10u}, st[3]={8u,16u,32u};
        float bbs[3]={YOLO_HD_P3_BB_OUTS,YOLO_HD_P4_BB_OUTS,YOLO_HD_P5_BB_OUTS};
        float cls[3]={YOLO_HD_P3_CL_OUTS,YOLO_HD_P4_CL_OUTS,YOLO_HD_P5_CL_OUTS};
        int32_t clz[3]={YOLO_HD_P3_CL_OUTZP,YOLO_HD_P4_CL_OUTZP,YOLO_HD_P5_CL_OUTZP};
        uint32_t s2;
        for (s2=0u; s2<3u; s2++) {
            uint32_t H=hw[s2], Wd=H, SP=H*Wd, pos, local_n=0u;
            dfl_prepare_tables(bbs[s2], dfl_wk, dfl_elut);
            yolo_dfl_load_weights(dfl_wk);
            yolo_dfl_load_exp_lut(dfl_elut);
            for (pos=0u; pos<SP && nd + local_n < 128u; pos++) {
                uint32_t ay=pos/Wd, ax=pos%Wd, c;
                /* cheapest first: class-0 (person) gate. sigmoid is monotonic, so
                 * sigmoid(lg0)<0.25 <=> lg0 < ln(1/3); compare the raw logit and skip
                 * non-candidate anchors entirely. */
                float lg0=dq(clB[s2],SP,0u,pos,clz[s2],cls[s2]);
                if(lg0 < -1.09861229f) continue;   /* sigmoid(lg0) < 0.25 */
                cand_ax[local_n] = (uint16_t)ax;
                cand_ay[local_n] = (uint16_t)ay;
                for (c=0u; c<4u; c++) {
                    volatile uint32_t *src = (volatile uint32_t *)(bbB[s2] + (c*SP + pos)*16u);
                    volatile uint32_t *dst = (volatile uint32_t *)(HD_DFL_IN + (local_n*4u + c)*16u);
                    dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3];
                }
                local_n++;
            }
            if (local_n != 0u) {
                uint32_t q;
                if (!yolo_run_dfl_ddr(HD_DFL_IN, HD_DFL_OUT, HD_DFL_ACT, local_n*4u)) {
                    print_str("  dfl fail\n"); errors++;
                    break;
                }
                for (q=0u; q<local_n; q++) {
                    volatile uint32_t *op = (volatile uint32_t *)(HD_DFL_OUT + q*16u);
                    uint32_t lo=op[0], hi=op[1];
                    float coord0=q8_8_to_float((uint16_t)(lo & 0xFFFFu));
                    float coord1=q8_8_to_float((uint16_t)(lo >> 16));
                    float coord2=q8_8_to_float((uint16_t)(hi & 0xFFFFu));
                    float coord3=q8_8_to_float((uint16_t)(hi >> 16));
                    float axf=(float)cand_ax[q], ayf=(float)cand_ay[q];
                    dx1[nd]=(axf+0.5f-coord0)*(float)st[s2];
                    dy1[nd]=(ayf+0.5f-coord1)*(float)st[s2];
                    dx2[nd]=(axf+0.5f+coord2)*(float)st[s2];
                    dy2[nd]=(ayf+0.5f+coord3)*(float)st[s2];
                    nd++;
                }
            }
        }
        // greedy NMS (class-agnostic; golden are all one class)
        uint32_t i,j; static uint8_t sup[128];
        for(i=0u;i<nd;i++) sup[i]=0u;
        // print NMS-deduped det centers (works for ANY image; 14px greedy dedup)
        print_str("  boxes(cx,cy):");
        for(i=0u;i<nd;i++){
            if(sup[i])continue;
            float fcx=(dx1[i]+dx2[i])*0.5f, fcy=(dy1[i]+dy2[i])*0.5f;
            int32_t cx=(int32_t)fcx, cy=(int32_t)fcy; if(cx<0)cx=0; if(cy<0)cy=0;
            print_str(" ("); print_dec((uint32_t)cx); print_str(","); print_dec((uint32_t)cy); print_str(")");
            for(j=i+1u;j<nd;j++){ float ex=(dx1[j]+dx2[j])*0.5f-fcx, ey=(dy1[j]+dy2[j])*0.5f-fcy;
                if(ex<0)ex=-ex;
                if(ey<0)ey=-ey;
                if(ex<14.0f&&ey<14.0f) sup[j]=1u; }
        }
        print_str("\n");
        for(i=0u;i<nd;i++) sup[i]=0u;
        uint32_t matched=0u;
        // for each golden box, see if a surviving det matches center within 14px
        for(j=0u;j<4u;j++){
            float gcx=yolo_hd_golden[j][0], gcy=yolo_hd_golden[j][1]; uint32_t hit=0u;
            for(i=0u;i<nd;i++){ if(sup[i])continue; float cx=(dx1[i]+dx2[i])*0.5f, cy=(dy1[i]+dy2[i])*0.5f;
                float ex=cx-gcx, ey=cy-gcy; if(ex<0)ex=-ex; if(ey<0)ey=-ey; if(ex<14.0f&&ey<14.0f){hit=1u;break;} }
            if(hit) matched++;
        }
        print_str("  head dets="); print_dec(nd); print_str(" golden-matched="); print_dec(matched); print_str("/4\n");
        prof_mark("decode_dfl_sigmoid_nms_CPU");
        if (errors==0u && matched>=4u) { print_str("YOLO FULL NET PASS (4 boxes match C oracle)\n"); return; }
        print_str("YOLO FULL NET: only "); print_dec(matched); print_str("/4 boxes matched\n");
    }
    if (errors) { print_str("HEAD conv errors="); print_dec(errors); print_str("\n"); }
    __asm__ volatile ("ebreak");
}
