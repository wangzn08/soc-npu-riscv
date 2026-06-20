// On-chip DFL end-to-end smoke through the real MMIO path:
// CPU stages 16xINT8 logit words into Act SRAM, loads EXP_LUT + W_k, triggers
// dfl_unit, drains the packed Q8.8 distances back to DDR, and compares to the
// integer golden from tools/gen_dfl_vectors.py. Proves the full register path.

#include "firmware.h"
#include "yolo_ops.h"
#include "yolo_dfl_smoke_data.h"
#include <stdint.h>

#define ACT_DDR  0x40090000u
#define OUT_DDR  0x40200000u
#define SRC_BASE 0u
#define DST_BASE 512u

static int32_t rd_s16(uint32_t ddr_word_byte, uint32_t elem)
{
    /* elem 0..3 = 4 packed 16-bit halfwords in one 128-bit word */
    volatile uint32_t *p = (volatile uint32_t *)ddr_word_byte;
    uint32_t w = p[elem >> 1];
    uint32_t h = (elem & 1u) ? (w >> 16) : (w & 0xFFFFu);
    return (h & 0x8000u) ? ((int32_t)h - 65536) : (int32_t)h;
}

static int32_t g_s16(uint32_t v)
{
    v &= 0xFFFFu;
    return (v & 0x8000u) ? ((int32_t)v - 65536) : (int32_t)v;
}

void usercode7(void)
{
    uint32_t i, errors = 0u;

    print_str("YOLO DFL HW SMOKE\n");

    /* stage input words into DDR, then DMA into Act SRAM SRC_BASE */
    for (i = 0u; i < DFL_NWORDS; i++) {
        volatile uint32_t *p = (volatile uint32_t *)(ACT_DDR + i * 16u);
        p[0] = DFL_ACT[i][0]; p[1] = DFL_ACT[i][1];
        p[2] = DFL_ACT[i][2]; p[3] = DFL_ACT[i][3];
    }
    if (!yolo_dma_ddr_to_act(ACT_DDR, SRC_BASE, DFL_NWORDS)) {
        print_str("  dma_to_act fail\n"); errors++;
    }

    yolo_dfl_load_exp_lut(DFL_EXP_LUT);
    yolo_dfl_load_weights(DFL_WK);

    if (!yolo_run_dfl(SRC_BASE, DST_BASE, DFL_NWORDS)) {
        print_str("  dfl run fail\n"); errors++;
    }

    /* drain packed distances (DFL_NWORDS/4 output words) to DDR */
    if (!yolo_dma_act_to_ddr(OUT_DDR, DST_BASE, DFL_NWORDS / 4u)) {
        print_str("  drain fail\n"); errors++;
    }

    for (i = 0u; i < DFL_NWORDS && errors <= 16u; i++) {
        uint32_t j = i >> 2;          /* output word */
        uint32_t k = i & 3u;          /* coord within anchor */
        int32_t got = rd_s16(OUT_DDR + j * 16u, k);
        int32_t exp = g_s16(DFL_GOLD[i]);
        if (got != exp) {
            errors++;
            print_str("  mismatch i="); print_dec(i);
            print_str(" got="); print_dec(got);
            print_str(" exp="); print_dec(exp); print_str("\n");
        }
    }

    if (errors == 0u) { print_str("YOLO DFL HW SMOKE PASS\n"); return; }
    print_str("YOLO DFL HW SMOKE FAIL\n");
    __asm__ volatile ("ebreak");
}
