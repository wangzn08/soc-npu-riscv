// Descriptor-driven YOLO runtime: build per-layer hardware-descriptor programs
// and hand them to the on-chip descriptor engine, instead of per-op CPU MMIO.
// Reuses the descriptor ops added to descriptor_engine.v (DMA/CONV2D/ACT_CFG/
// movement). Mirrors the CPU helpers in yolo_ops.c so a layer can switch paths.
#ifndef YOLO_DESC_H
#define YOLO_DESC_H

#include <stdint.h>

// Scratch DDR for the descriptor program + resident qparam table. Must stay
// clear of the layer's image/weights/output buffers (free: 0x4000_0000+).
#define YOLO_DESC_DDR    0x40000000u
#define YOLO_QPARAM_DDR  0x40080000u

// Strip-tiled conv driven entirely by the descriptor queue. Same arguments as
// yolo_run_conv2d_tiled (yolo_ops.c) plus the SiLU requant config. The SiLU
// mode is taken from ctrl_flags (SILU_EXACT_EN, or SILU_EN + SILU_REQUANT_EN);
// for exact-SiLU the LUT must be CPU-preloaded (yolo_load_silu_lut) and the
// requant args are 0. Handles 3x3 (HW pad + DMA'd vertical halo) and 1x1 PW
// (kh=kw=1, pad=0), OC chunking, and per-chunk weight reload when the layer's
// weights exceed Wgt SRAM. Returns 1 on success, 0 on descriptor error/timeout.
int yolo_run_conv2d_tiled_desc(uint32_t in_ddr, uint32_t wgt_all_ddr,
                               uint32_t wgt_base, uint32_t out_ddr,
                               uint32_t pad_row_ddr,
                               uint32_t in_w, uint32_t in_h,
                               uint32_t in_c, uint32_t out_c,
                               uint32_t kernel_h, uint32_t kernel_w,
                               uint32_t stride, uint32_t pad,
                               const int32_t *bias, const uint32_t *scale_mul,
                               const uint32_t *scale_shift, uint32_t ctrl_flags,
                               uint32_t wgt_words_per_oc, uint32_t strip_out_rows,
                               int32_t pad_value,
                               uint32_t silu_requant_mul,
                               uint32_t silu_requant_shift,
                               int32_t silu_requant_zp);

// 5x5 stride-1 signed maxpool (DDR->DDR) via one descriptor program (DMA-in +
// HW maxpool + DMA-out). Same args/result as yolo_run_maxpool5x5 (yolo_ops.c).
int yolo_run_maxpool5x5_desc(uint32_t src_ddr, uint32_t dst_ddr,
                             uint32_t scratch_act_base, uint32_t in_w,
                             uint32_t in_h, uint32_t ic_groups);

// Signed eltwise residual add (DDR->DDR) via one descriptor program. Same args
// and result as yolo_run_eltwise_add_ddr (yolo_ops.c).
int yolo_run_eltwise_add_desc(uint32_t src0_ddr, uint32_t src1_ddr,
                              uint32_t dst_ddr, uint32_t scratch_act_base,
                              uint32_t words, int32_t zp, uint32_t ratio_en,
                              uint32_t ratio_mul, uint32_t ratio_shift);

// Phase-0 probe: print accumulated conv-desc CPU-build vs engine-run cycles.
void yolo_desc_prof_print(void);

// ---- Pre-compiled descriptor image (A2 record/replay). Word idx =
// (cpu_addr - 0x40000000) / 16. Regions audited collision-free (see plan). ----
#define DESC_IMAGE_BASE    0x40000000u   // descriptor records blob (cap 1MB)
#define DESC_QPARAM_BASE   0x40200000u   // per-layer qparam blob   (cap 256KB)
#define DESC_CATALOG_BASE  0x40280000u   // catalog: N x {off,count}
#define DESC_CATALOG_MAX   256u

// Reset the record/replay program cursor + bump allocators at net start.
void yolo_desc_reset(void);

#endif
