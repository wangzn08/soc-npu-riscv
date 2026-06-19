// Generic C2f runner proof: run c2f_2 (model.2) via yolo_run_c2f_block and
// compare to the hand-orchestrated M5v golden. Block input (conv1 output) is
// baked in; cfg is wired from the existing c2f_2 fixtures.

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_c2f.h"
#include "yolo_c2f2_blockin_data.h"
#include "yolo_conv2_from_conv1_chain_data.h"
#include "yolo_conv3_from_conv2_chain_data.h"
#include "yolo_conv4_from_conv3_chain_data.h"
#include "yolo_c2f_add_m5u_data.h"
#include "yolo_c2f_close_m5v_data.h"
#include <stdint.h>

#define IN_DDR     0x40090000u
#define CV1_OUT    0x40200000u
#define BN_OUT     0x40280000u
#define ADD0_DDR   0x402C0000u
#define CONCAT_DDR 0x40300000u
#define OUT_DDR    0x403C0000u
#define WGT_DDR    0x40440000u
#define SKIP_BASE  4096u

static void write_ddr_word128(uint32_t a, uint32_t w, const uint32_t l[4])
{
    volatile uint32_t *p = (volatile uint32_t *)(a + w * 16u);
    p[0]=l[0]; p[1]=l[1]; p[2]=l[2]; p[3]=l[3];
}
static int32_t read_ddr_s8(uint32_t a, uint32_t w, uint32_t b)
{
    volatile uint32_t *p = (volatile uint32_t *)(a + w * 16u);
    uint32_t lane = p[b >> 2];
    uint32_t byte = (lane >> ((b & 3u) * 8u)) & 0xFFu;
    return (byte & 0x80u) ? ((int32_t)byte - 256) : (int32_t)byte;
}
static int32_t s8(uint32_t b) { b &= 0xFFu; return (b & 0x80u) ? ((int32_t)b - 256) : (int32_t)b; }
static uint32_t absd(int32_t a, int32_t b) { int32_t d = a - b; return (uint32_t)(d < 0 ? -d : d); }

void usercode7(void)
{
    uint32_t i, pos, oc, errors = 0u;
    yolo_c2f_cfg_t cfg;

    print_str("YOLO C2F GENERIC RUNNER (c2f_2) CPU SMOKE\n");

    for (i = 0u; i < YOLO_C2F2_BLOCKIN_WORDS; i++)
        write_ddr_word128(IN_DDR, i, yolo_c2f2_blockin_words[i]);

    // ---- geometry ----
    cfg.in_w = YOLO_CONV2_CHAIN_IN_W; cfg.in_h = YOLO_CONV2_CHAIN_IN_H;
    cfg.spatial = YOLO_CONV2_CHAIN_IN_W * YOLO_CONV2_CHAIN_IN_H;
    cfg.full_c = YOLO_CONV2_CHAIN_OC;   // 32
    cfg.n_bottleneck = 1u; cfg.shortcut = 1u;

    // ---- DDR scratch ----
    cfg.in_ddr = IN_DDR; cfg.cv1_ic = YOLO_CONV2_CHAIN_IC; cfg.cv1_out_ddr = CV1_OUT;
    cfg.bn_in_ddr = 0u; cfg.bn_out_ddr = BN_OUT; cfg.add_ddr[0] = ADD0_DDR;
    cfg.concat_ddr = CONCAT_DDR; cfg.out_ddr = OUT_DDR; cfg.wgt_ddr = WGT_DDR;
    cfg.skip_out_base = SKIP_BASE;

    // ---- cv1 = conv2 (1x1, 32->32), requant to conv2 out scale ----
    cfg.cv1_wgt = yolo_conv2_chain_wgt_words; cfg.cv1_wgt_words = YOLO_CONV2_CHAIN_WGT_WORDS;
    cfg.cv1_bias = yolo_conv2_chain_bias_q; cfg.cv1_mul = yolo_conv2_chain_scale_mul;
    cfg.cv1_shift = yolo_conv2_chain_scale_shift;
    cfg.cv1_rq_mul = YOLO_CONV2_CHAIN_REQUANT_MUL; cfg.cv1_rq_shift = YOLO_CONV2_CHAIN_REQUANT_SHIFT;
    cfg.cv1_rq_zp = YOLO_CONV2_CHAIN_REQUANT_ZP;

    // ---- bottleneck 0: m_cv1 = conv3, m_cv2 = conv4 (both 3x3) ----
    cfg.mcv1_wgt_words = YOLO_CONV3_CHAIN_WGT_WORDS;
    cfg.mcv1_wgt[0] = yolo_conv3_chain_wgt_words; cfg.mcv1_bias[0] = yolo_conv3_chain_bias_q;
    cfg.mcv1_mul[0] = yolo_conv3_chain_scale_mul; cfg.mcv1_shift[0] = yolo_conv3_chain_scale_shift;
    cfg.mcv1_rq_mul[0] = YOLO_CONV3_CHAIN_REQUANT_MUL; cfg.mcv1_rq_shift[0] = YOLO_CONV3_CHAIN_REQUANT_SHIFT;
    cfg.mcv1_rq_zp[0] = YOLO_CONV3_CHAIN_REQUANT_ZP; cfg.mcv1_pad_value[0] = YOLO_CONV3_CHAIN_PAD_VALUE;

    cfg.mcv2_wgt_words = YOLO_CONV4_CHAIN_WGT_WORDS;
    cfg.mcv2_wgt[0] = yolo_conv4_chain_wgt_words; cfg.mcv2_bias[0] = yolo_conv4_chain_bias_q;
    cfg.mcv2_mul[0] = yolo_conv4_chain_scale_mul; cfg.mcv2_shift[0] = yolo_conv4_chain_scale_shift;
    cfg.mcv2_pad_value[0] = YOLO_CONV4_CHAIN_PAD_VALUE;

    // glue (residual add) scale = /model.2/m.0/Add
    cfg.glue_rq_mul[0] = YOLO_C2F_ADD_GLUE_REQUANT_MUL; cfg.glue_rq_shift[0] = YOLO_C2F_ADD_GLUE_REQUANT_SHIFT;
    cfg.glue_zp[0] = YOLO_C2F_ADD_GLUE_ZP;

    // staging = re-run cv1 group-1 (conv2 OC 16..31) to glue scale
    cfg.stage_wgt_words = YOLO_C2F_STAGE_WGT_WORDS; cfg.stage_wgt[0] = yolo_c2f_stage_wgt_words;
    cfg.stage_bias[0] = yolo_c2f_stage_bias_q; cfg.stage_mul[0] = yolo_c2f_stage_scale_mul;
    cfg.stage_shift[0] = yolo_c2f_stage_scale_shift; cfg.stage_pad_value = 0;

    // ---- concat requant to cv2 in-scale ----
    cfg.cat_req_shift = YOLO_C2F_CAT_REQ_SHIFT; cfg.cat_zp = YOLO_C2F_CAT_ZP;
    cfg.cat_mul_s0s1 = YOLO_C2F_CAT_MUL_S0S1; cfg.cat_inzp_s0s1 = YOLO_C2F_CAT_INZP_S0S1;
    cfg.cat_mul_add = YOLO_C2F_CAT_MUL_ADD; cfg.cat_inzp_add = YOLO_C2F_CAT_INZP_ADD;

    // ---- cv2 = conv5 (1x1, 48->32) ----
    cfg.cv2_ic = YOLO_C2F_CLOSE_IC; cfg.cv2_oc = YOLO_C2F_CLOSE_OC;
    cfg.cv2_wgt = yolo_c2f_conv5_wgt_words; cfg.cv2_wgt_words = YOLO_C2F_CONV5_WGT_WORDS;
    cfg.cv2_bias = yolo_c2f_conv5_bias_q; cfg.cv2_mul = yolo_c2f_conv5_scale_mul;
    cfg.cv2_shift = yolo_c2f_conv5_scale_shift;
    cfg.cv2_rq_mul = YOLO_C2F_CONV5_REQUANT_MUL; cfg.cv2_rq_shift = YOLO_C2F_CONV5_REQUANT_SHIFT;
    cfg.cv2_rq_zp = YOLO_C2F_CONV5_REQUANT_ZP;

    if (!yolo_run_c2f_block(&cfg)) {
        print_str("  c2f runner failed\n");
        errors++;
    }

    for (pos = 0u; pos < YOLO_C2F_CLOSE_SPATIAL && errors <= 16u; pos++) {
        for (oc = 0u; oc < YOLO_C2F_CLOSE_OC; oc++) {
            int32_t got = read_ddr_s8(OUT_DDR, (oc >> 4) * YOLO_C2F_CLOSE_SPATIAL + pos, oc & 15u);
            int32_t expect = s8(yolo_c2f_close_expected_rtl[pos][oc]);
            if (absd(got, expect) > YOLO_C2F_CLOSE_RTL_TOL) {
                errors++;
                print_str("  mismatch pos="); print_dec(pos);
                print_str(" oc="); print_dec(oc);
                print_str(" got="); print_dec(got);
                print_str(" exp="); print_dec(expect); print_str("\n");
                if (errors > 16u) break;
            }
        }
    }

    if (errors == 0u) { print_str("YOLO C2F GENERIC RUNNER CPU SMOKE PASS\n"); return; }
    print_str("YOLO C2F GENERIC RUNNER CPU SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
