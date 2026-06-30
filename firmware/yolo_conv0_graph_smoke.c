// conv0 @320 as a one-submit graph descriptor smoke.

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_desc.h"
#include "yolo_conv0_320_exact_data.h"
#include <stdint.h>

#define ACT_DDR     0x40400000u
#define WGT_DDR     0x405C0000u
#define OUT_DDR     0x40600000u
#define PAD_ROW_DDR 0x40780000u
#define WGT_BASE    0u
#define STRIP_OUT_ROWS 16u

static void wrw(uint32_t a, uint32_t w, const uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); p[0]=l[0];p[1]=l[1];p[2]=l[2];p[3]=l[3]; }
static int32_t rs8(uint32_t a, uint32_t w, uint32_t b)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); uint32_t v=p[b>>2]>>((b&3u)*8u)&0xFFu; return (v&0x80u)?((int32_t)v-256):(int32_t)v; }

void usercode7(void)
{
    uint32_t i, pos, oc, errors = 0u;

    print_str("YOLO CONV0 @320 GRAPH-DESC SMOKE\n");

    for (i = 0u; i < C0E_ACT_WORDS; i++) wrw(ACT_DDR, i, yolo_conv0_320e_act_words[i]);
    for (i = 0u; i < C0E_WGT_WORDS; i++) wrw(WGT_DDR, i, yolo_conv0_320e_wgt_words[i]);

    yolo_desc_reset();
    yolo_desc_graph_begin();
    yolo_desc_load_silu_lut(yolo_conv0_320e_silu_lut);

    *(volatile uint32_t *)NPU_PERF_CLR = 1u;

    if (!yolo_run_conv2d_tiled_desc(ACT_DDR, WGT_DDR, WGT_BASE, OUT_DDR, PAD_ROW_DDR,
                                    C0E_IN_W, C0E_IN_H, C0E_IC, C0E_OC,
                                    3u, 3u, 2u, 1u,
                                    yolo_conv0_320e_bias_q, yolo_conv0_320e_scale_mul,
                                    yolo_conv0_320e_scale_shift,
                                    NPU_CTRL_SILU_EXACT_EN,
                                    C0E_WGT_PER_OC, STRIP_OUT_ROWS, C0E_PAD_VALUE,
                                    0u, 0u, 0)) {
        print_str("  graph build failed\n"); errors++;
    }
    if (errors == 0u && !yolo_desc_graph_end_and_submit()) {
        print_str("  graph submit failed\n"); errors++;
    }
    print_str("submit_count="); print_dec(yolo_desc_submit_count()); print_str("\n");
    if (errors == 0u && yolo_desc_submit_count() != 1u)
        errors++;

    print_str("PERF cyc_total="); print_dec(*(volatile uint32_t *)NPU_PERF_CYC_TOTAL);
    print_str(" npu_busy=");      print_dec(*(volatile uint32_t *)NPU_PERF_CYC_BUSY); print_str("\n");

    {
        uint32_t chk = 0u, idx = 0u;
        for (pos = 0u; pos < C0E_OUT_SPATIAL; pos++)
            for (oc = 0u; oc < C0E_OC; oc++) {
                int32_t got = rs8(OUT_DDR, (oc>>4)*C0E_OUT_SPATIAL + pos, oc&15u);
                chk += ((uint32_t)(got + 128) * (idx + 1u));
                idx++;
            }
        if (chk != C0E_GOLDEN_CHK) {
            errors++;
            print_str("  checksum mismatch got=0x"); print_hex(chk, 8);
            print_str(" exp=0x"); print_hex(C0E_GOLDEN_CHK, 8); print_str("\n");
        }
    }

    if (errors == 0u) { print_str("YOLO CONV0 GRAPH-DESC SMOKE PASS\n"); return; }
    print_str("YOLO CONV0 GRAPH-DESC SMOKE FAIL errors="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
