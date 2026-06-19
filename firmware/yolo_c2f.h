#ifndef YOLO_C2F_H
#define YOLO_C2F_H

#include <stdint.h>

// Generic C2f block runner config. Reuses the shared NPU conv/eltwise path.
//
// Block: cv1(1x1) -> split s0,s1(half) -> for i<n: m_cv1_i(3x3)->m_cv2_i(3x3)
//        [+ residual add of the bottleneck input via HW signed eltwise if
//        shortcut] -> concat(s0, s1, add_0..add_{n-1}) requant to cv2 in-scale
//        -> cv2(1x1). All convs use SiLU + SiLU-requant.
//
// Weight words are tile-major: 1x1 => oc*(ic/16) words; 3x3 => oc*(ic/16)*9.
#define YOLO_C2F_MAX_BN 4

typedef struct {
    // ----- geometry -----
    uint32_t in_w, in_h, spatial;   // cv1 input spatial (= block input)
    uint32_t full_c;                // cv1 output channels (split into 2 halves)
    uint32_t n_bottleneck;
    uint32_t shortcut;              // 1 => residual add each bottleneck

    // ----- DDR scratch bases (caller-owned, must be disjoint) -----
    uint32_t in_ddr;                // cv1 input (block input), full_c... actually cv1_ic
    uint32_t cv1_ic;                // cv1 input channels
    uint32_t cv1_out_ddr;           // cv1 output (full_c, tile-major groups)
    uint32_t bn_in_ddr;             // bottleneck input scratch (half_c)
    uint32_t bn_out_ddr;            // bottleneck output scratch (half_c)
    uint32_t add_ddr[YOLO_C2F_MAX_BN]; // residual-add outputs (half_c each, glue scale)
    uint32_t concat_ddr;            // concat buffer (full_c + n*half_c groups)
    uint32_t out_ddr;               // cv2 output (block output)
    uint32_t wgt_ddr;               // weight staging DDR (reused per conv)
    uint32_t skip_out_base;         // Out-SRAM word base for residual staging

    // ----- cv1 (1x1) -----
    const uint32_t (*cv1_wgt)[4]; uint32_t cv1_wgt_words;
    const int32_t *cv1_bias; const uint32_t *cv1_mul; const uint32_t *cv1_shift;
    uint32_t cv1_rq_mul, cv1_rq_shift; int32_t cv1_rq_zp;

    // ----- per-bottleneck m_cv1 / m_cv2 (3x3, pad 1, half_c->half_c) -----
    const uint32_t (*mcv1_wgt[YOLO_C2F_MAX_BN])[4]; uint32_t mcv1_wgt_words;
    const int32_t *mcv1_bias[YOLO_C2F_MAX_BN];
    const uint32_t *mcv1_mul[YOLO_C2F_MAX_BN]; const uint32_t *mcv1_shift[YOLO_C2F_MAX_BN];
    uint32_t mcv1_rq_mul[YOLO_C2F_MAX_BN], mcv1_rq_shift[YOLO_C2F_MAX_BN];
    int32_t mcv1_rq_zp[YOLO_C2F_MAX_BN];
    int32_t mcv1_pad_value[YOLO_C2F_MAX_BN];

    const uint32_t (*mcv2_wgt[YOLO_C2F_MAX_BN])[4]; uint32_t mcv2_wgt_words;
    const int32_t *mcv2_bias[YOLO_C2F_MAX_BN];
    const uint32_t *mcv2_mul[YOLO_C2F_MAX_BN]; const uint32_t *mcv2_shift[YOLO_C2F_MAX_BN];
    int32_t mcv2_pad_value[YOLO_C2F_MAX_BN];
    // m_cv2 requant target = the glue (add) scale so the eltwise add operands align.
    uint32_t glue_rq_mul[YOLO_C2F_MAX_BN], glue_rq_shift[YOLO_C2F_MAX_BN];
    int32_t glue_zp[YOLO_C2F_MAX_BN];

    // ----- residual-add staging: re-run cv1 group-1 to glue scale -----
    // (weights = cv1 OC half_c..full_c-1; reuses cv1 input). One per bottleneck.
    const uint32_t (*stage_wgt[YOLO_C2F_MAX_BN])[4]; uint32_t stage_wgt_words;
    const int32_t *stage_bias[YOLO_C2F_MAX_BN];
    const uint32_t *stage_mul[YOLO_C2F_MAX_BN]; const uint32_t *stage_shift[YOLO_C2F_MAX_BN];
    int32_t stage_pad_value;

    // ----- concat requant to cv2 in-scale (per piece: s0, s1, add_i) -----
    uint32_t cat_req_shift; int32_t cat_zp;
    uint32_t cat_mul_s0s1;  int32_t cat_inzp_s0s1;   // s0,s1 at cv1 out scale
    uint32_t cat_mul_add;   int32_t cat_inzp_add;    // add_i at glue scale

    // ----- cv2 (1x1, (2+n)*half_c -> out_c) -----
    uint32_t cv2_ic, cv2_oc;
    const uint32_t (*cv2_wgt)[4]; uint32_t cv2_wgt_words;
    const int32_t *cv2_bias; const uint32_t *cv2_mul; const uint32_t *cv2_shift;
    uint32_t cv2_rq_mul, cv2_rq_shift; int32_t cv2_rq_zp;
} yolo_c2f_cfg_t;

// Runs the whole block. Returns 1 on success. Block output lands at cfg->out_ddr
// (cv2_oc channels, tile-major). Requires the block input already at cfg->in_ddr.
int yolo_run_c2f_block(const yolo_c2f_cfg_t *cfg);

#endif
