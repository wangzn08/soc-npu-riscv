// Root-cause probe for the conv0 <16-real-IC issue. Runs a tiny 4x4x16 conv
// (only input lanes 0..2 non-zero) with bias=0, scale_mul=1, scale_shift=0 and
// INT32_OUT, so the NPU writes the RAW int32 accumulator per OC. Compares to the
// pure 3-channel MAC golden. If acc differs, the NPU corrupts zero input lanes.

#include "firmware.h"
#include "yolo_ops.h"
#include "conv0_accprobe_data.h"
#include <stdint.h>

#define ACT_DDR  0x40090000u
#define WGT_DDR  0x400A0000u
#define OUT_DDR  0x400B0000u
#define ACT_BASE 0u
#define WGT_BASE 0u

static void wrw(uint32_t a, uint32_t w, const uint32_t l[4])
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); p[0]=l[0];p[1]=l[1];p[2]=l[2];p[3]=l[3]; }
static int32_t rint32(uint32_t a, uint32_t w, uint32_t lane)
{ volatile uint32_t *p=(volatile uint32_t*)(a+w*16u); return (int32_t)p[lane&3u]; }

static int32_t bias0[16];
static uint32_t mul1[16];
static uint32_t shift0[16];

void usercode7(void)
{
    uint32_t i, pos, oc, errors = 0u;

    print_str("CONV0 ACC PROBE (4x4x16, lanes 0..2, INT32 acc)\n");
    for (i = 0u; i < AP_ACT_WORDS; i++) wrw(ACT_DDR, i, ap_act_words[i]);
    for (i = 0u; i < AP_WGT_WORDS; i++) wrw(WGT_DDR, i, ap_wgt_words[i]);
    for (i = 0u; i < 16u; i++) { bias0[i]=0; mul1[i]=1u; shift0[i]=0u; }

    if (!yolo_dma_ddr_to_act(ACT_DDR, ACT_BASE, AP_ACT_WORDS) ||
        !yolo_dma_ddr_to_wgt(WGT_DDR, WGT_BASE, AP_WGT_WORDS)) {
        print_str("  dma fail\n"); errors++;
    }
    if (!yolo_run_conv2d_qparams_pads(ACT_BASE, WGT_BASE, 0u,
                                      AP_IN_W, AP_IN_H, 16u, AP_OC,
                                      3u, 3u, 1u, 0u, 0u,
                                      bias0, mul1, shift0,
                                      NPU_CTRL_INT32_OUT | NPU_CTRL_OC_SINGLE)) {
        print_str("  conv fail\n"); errors++;
    }
    // INT32_OUT: 4 Out words per position (16 int32). Drain all positions.
    if (!yolo_dma_out_to_ddr(OUT_DDR, 0u, AP_OUT_SPATIAL * 4u, 0u)) {
        print_str("  drain fail\n"); errors++;
    }

    for (pos = 0u; pos < AP_OUT_SPATIAL; pos++)
        for (oc = 0u; oc < AP_OC; oc++) {
            int32_t got = rint32(OUT_DDR, (oc>>2)*AP_OUT_SPATIAL + pos, oc&3u);
            int32_t exp = ap_acc_golden[pos][oc];
            if (got != exp) {
                errors++;
                print_str("  pos="); print_dec(pos);
                print_str(" oc="); print_dec(oc);
                print_str(" got="); print_dec((uint32_t)got);
                print_str(" exp="); print_dec((uint32_t)exp); print_str("\n");
                if (errors > 24u) { print_str("...\n"); goto done; }
            }
        }
done:
    if (errors == 0u) { print_str("CONV0 ACC PROBE PASS (acc correct)\n"); return; }
    print_str("CONV0 ACC PROBE: acc mismatches="); print_dec(errors); print_str("\n");
    __asm__ volatile ("ebreak");
}
