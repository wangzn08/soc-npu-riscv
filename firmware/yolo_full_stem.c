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
#define C2F8_OUT  0x406C0000u   // c2f_8 out 10x10x256
#define E_PSUM    0x40700000u   // c2f_8 bottleneck (icg8) ic_stream INT32 psum
#define WGT_BASE  0u

static int32_t rs8(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); uint32_t v=p[b>>2]>>((b&3u)*8u)&0xFFu; return (v&0x80u)?((int32_t)v-256):(int32_t)v; }
static int32_t s8(uint32_t b){ b&=0xFFu; return (b&0x80u)?((int32_t)b-256):(int32_t)b; }
static uint32_t ad(int32_t a,int32_t b){ int32_t d=a-b; return (uint32_t)(d<0?-d:d); }

void usercode7(void)
{
    uint32_t pos, oc, errors = 0u, maxd = 0u;
    yolo_c2f_cfg_t cfg;

    print_str("YOLO FULL STEM: conv0 -> conv1 -> c2f_2 (on-SoC, DDR, preloaded img)\n");

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

    // ---------- validate c2f_8 output vs golden (integration; 8-block drift propagates) ----------
    for (pos = 0u; pos < YOLO_C2F8_SPATIAL && errors <= 16u; pos++)
        for (oc = 0u; oc < YOLO_C2F8_CV2_OC; oc++) {
            int32_t got = rs8(C2F8_OUT, (oc>>4)*YOLO_C2F8_SPATIAL + pos, oc&15u);
            int32_t exp = s8(yolo_c2f8_golden[pos][oc]);
            uint32_t d = ad(got, exp);
            if (d > maxd) maxd = d;
            if (d > 120u) {   /* integration: 8-block preact-scale drift accumulates */
                errors++;
                if (errors <= 8u) {
                    print_str("  pos="); print_dec(pos); print_str(" oc="); print_dec(oc);
                    print_str(" got="); print_dec((uint32_t)got); print_str(" exp="); print_dec((uint32_t)exp); print_str("\n");
                }
            }
        }

    print_str("backbone c2f_8 vs golden maxdiff="); print_dec(maxd); print_str("\n");
    if (errors == 0u) { print_str("YOLO BACKBONE PASS (conv0->...->c2f_8 chained)\n"); return; }
    print_str("YOLO BACKBONE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
