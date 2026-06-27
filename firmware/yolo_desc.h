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

// Strip-tiled exact-SiLU conv driven entirely by the descriptor queue. Same
// arguments as yolo_run_conv2d_tiled (yolo_ops.c). The exact-SiLU LUT must be
// CPU-preloaded (yolo_load_silu_lut) before the call; it persists in the NPU.
// out_c must be <= 64 (single OC chunk); weights are DMA'd to Wgt SRAM once.
// Returns 1 on success, 0 on a descriptor error/timeout.
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
                               int32_t pad_value);

#endif
