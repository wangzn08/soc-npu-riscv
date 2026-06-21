#ifndef YOLO_C2F_H
#define YOLO_C2F_H

#include <stdint.h>

// Generic C2f block runner. Convs (cv1, bottleneck m_cv1/m_cv2, cv2) run on the
// shared NPU; the residual add and the concat requant run on the CPU (uniform,
// faithful to the C reference for arbitrary per-bottleneck glue scales).
//
// Block: cv1(1x1) -> split s0,s1(half) -> for i<n: m_cv1_i(3x3)->m_cv2_i(3x3,
//        requant to glue[i]); if shortcut: add_i = clamp_s8(round((prev_q -
//        prev_zp)*ratio_i) + mcv2_q), prev=s1(i=0)/add_{i-1}(i>0); else add_i =
//        mcv2 out. Then concat(s0, s1, add_0..add_{n-1}) requant to cv2 in-scale
//        -> cv2(1x1). All convs use SiLU + SiLU-requant.
//
// Weight words tile-major: 1x1 => oc*(ic/16); 3x3 => oc*(ic/16)*9.
#define YOLO_C2F_MAX_BN 4

typedef struct {
    // ----- geometry -----
    uint32_t in_w, in_h, spatial;   // cv1 input spatial (= block input)
    uint32_t full_c;                // cv1 output channels (split into 2 halves)
    uint32_t n_bottleneck;
    uint32_t shortcut;

    // ----- DDR scratch (caller-owned, disjoint) -----
    uint32_t in_ddr, cv1_ic, cv1_out_ddr;
    uint32_t bn_out_ddr;            // m_cv1 output scratch (half_c)
    uint32_t mcv2_ddr;              // m_cv2 output scratch (half_c, glue scale)
    uint32_t add_ddr[YOLO_C2F_MAX_BN];
    uint32_t concat_ddr, out_ddr, wgt_ddr;
    uint32_t pad_row_ddr;           // one-row scratch for tiled vertical padding
    uint32_t strip;                 // tiled output-row strip height (0 => default 16)

    // ----- exact per-layer SiLU LUT (0 = legacy Q4.4 SiLU + requant) -----
    // When silu_exact != 0, each conv loads its 256-entry out-grid SiLU LUT,
    // runs with NPU_CTRL_SILU_EXACT_EN, and the *_mul/*_shift/*_bias arrays are
    // the LINEAR out-grid qparams (s2 == round(preact/out_scale)); the *_rq_zp /
    // glue_zp fields supply the output zero-point. Legacy *_rq_mul are unused.
    uint32_t silu_exact;
    const uint8_t *cv1_silu_lut;
    const uint8_t *mcv1_silu_lut[YOLO_C2F_MAX_BN];
    const uint8_t *mcv2_silu_lut[YOLO_C2F_MAX_BN];
    const uint8_t *cv2_silu_lut;

    // ----- weights-from-DDR-blob (0 = use the cv*_wgt C arrays via push_wgt) -----
    // When wgt_in_blob != 0, each conv reads weights DIRECTLY from its DDR blob
    // address (no push_wgt copy); the cv*_wgt C-array pointers are then unused.
    uint32_t wgt_in_blob;
    uint32_t cv1_wgt_ddr;
    uint32_t mcv1_wgt_ddr[YOLO_C2F_MAX_BN];
    uint32_t mcv2_wgt_ddr[YOLO_C2F_MAX_BN];
    uint32_t cv2_wgt_ddr;

    // ----- cv1 (1x1) -----
    const uint32_t (*cv1_wgt)[4]; uint32_t cv1_wgt_words;
    const int32_t *cv1_bias; const uint32_t *cv1_mul; const uint32_t *cv1_shift;
    uint32_t cv1_rq_mul, cv1_rq_shift; int32_t cv1_rq_zp;

    // ----- per-bottleneck m_cv1 / m_cv2 (3x3, pad 1, half_c->half_c) -----
    const uint32_t (*mcv1_wgt[YOLO_C2F_MAX_BN])[4]; uint32_t mcv1_wgt_words;
    const int32_t *mcv1_bias[YOLO_C2F_MAX_BN];
    const uint32_t *mcv1_mul[YOLO_C2F_MAX_BN]; const uint32_t *mcv1_shift[YOLO_C2F_MAX_BN];
    uint32_t mcv1_rq_mul[YOLO_C2F_MAX_BN], mcv1_rq_shift[YOLO_C2F_MAX_BN];
    int32_t mcv1_rq_zp[YOLO_C2F_MAX_BN]; int32_t mcv1_pad_value[YOLO_C2F_MAX_BN];

    const uint32_t (*mcv2_wgt[YOLO_C2F_MAX_BN])[4]; uint32_t mcv2_wgt_words;
    const int32_t *mcv2_bias[YOLO_C2F_MAX_BN];
    const uint32_t *mcv2_mul[YOLO_C2F_MAX_BN]; const uint32_t *mcv2_shift[YOLO_C2F_MAX_BN];
    int32_t mcv2_pad_value[YOLO_C2F_MAX_BN];
    // m_cv2 requant target = glue[i] scale.
    uint32_t glue_rq_mul[YOLO_C2F_MAX_BN], glue_rq_shift[YOLO_C2F_MAX_BN];
    int32_t glue_zp[YOLO_C2F_MAX_BN];

    // ----- CPU residual add: add_i = clamp(round((prev_q-prev_zp)*ratio)+mcv2_q) -----
    uint32_t add_ratio_mul[YOLO_C2F_MAX_BN], add_ratio_shift;
    int32_t  add_prev_zp[YOLO_C2F_MAX_BN];   // prev's own zero-point

    // ----- concat requant to cv2 in-scale (per piece) -----
    uint32_t cat_req_shift; int32_t cat_zp;
    uint32_t cat_mul_s0s1;  int32_t cat_inzp_s0s1;             // s0,s1 at cv1 out scale
    uint32_t cat_mul_add[YOLO_C2F_MAX_BN]; int32_t cat_inzp_add[YOLO_C2F_MAX_BN]; // add_i at glue[i]

    // ----- cv2 (1x1) -----
    uint32_t cv2_ic, cv2_oc;
    const uint32_t (*cv2_wgt)[4]; uint32_t cv2_wgt_words;
    const int32_t *cv2_bias; const uint32_t *cv2_mul; const uint32_t *cv2_shift;
    uint32_t cv2_rq_mul, cv2_rq_shift; int32_t cv2_rq_zp;
} yolo_c2f_cfg_t;

int yolo_run_c2f_block(const yolo_c2f_cfg_t *cfg);

#endif
