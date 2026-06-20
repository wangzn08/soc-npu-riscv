#ifndef YOLO_OPS_H
#define YOLO_OPS_H

#include <stdint.h>

int yolo_dma_ddr_to_act(uint32_t ddr_addr, uint32_t act_base, uint32_t words);
int yolo_dma_ddr_to_wgt(uint32_t ddr_addr, uint32_t wgt_base, uint32_t words);
int yolo_dma_act_to_ddr(uint32_t ddr_addr, uint32_t act_base, uint32_t words);
int yolo_dma_out_to_ddr(uint32_t ddr_addr,
                        uint32_t out_base,
                        uint32_t words,
                        uint32_t out_pong);
int yolo_concat2_ddr_to_act(uint32_t src0_ddr,
                            uint32_t src1_ddr,
                            uint32_t dst_act_base,
                            uint32_t spatial_words,
                            uint32_t src0_groups,
                            uint32_t src1_groups);
int yolo_slice_ddr_to_act(uint32_t src_ddr,
                          uint32_t dst_act_base,
                          uint32_t spatial_words,
                          uint32_t first_group,
                          uint32_t group_count);
int yolo_run_upsample2x(uint32_t src_act_base,
                        uint32_t dst_act_base,
                        uint32_t in_w,
                        uint32_t in_h,
                        uint32_t ic_groups);
void yolo_set_silu_requant(uint32_t mul, uint32_t shift, int32_t zp);
void yolo_set_pad_value(int32_t pad_value);
void yolo_set_eltwise(int32_t zp, uint32_t skip_base);

// 3x3/1x1 conv with out_c > 64: loops 64-OC chunks (oc_single per chunk, params
// reloaded), since the resident param regfile holds only 64 OCs. Act must already
// be in Act SRAM (caller DMAs it + sets pad_value/silu_requant). Weights for all
// out_c OCs live in DDR at wgt_all_ddr (tile-major, wgt_words_per_oc each); output
// drained tile-major to out_ddr. out_c need not be a multiple of 64.
int yolo_run_pw_conv1x1_oc_chunks(uint32_t act_base, uint32_t wgt_all_ddr, uint32_t wgt_base,
                                  uint32_t out_ddr, uint32_t in_w, uint32_t in_h,
                                  uint32_t in_c, uint32_t out_c,
                                  const int32_t *bias, const uint32_t *scale_mul,
                                  const uint32_t *scale_shift, uint32_t ctrl_flags,
                                  uint32_t out_spatial);

int yolo_run_conv2d_oc_chunks(uint32_t act_base, uint32_t wgt_all_ddr, uint32_t wgt_base,
                              uint32_t out_ddr, uint32_t in_w, uint32_t in_h,
                              uint32_t in_c, uint32_t out_c,
                              uint32_t kernel_h, uint32_t kernel_w,
                              uint32_t stride, uint32_t pad,
                              const int32_t *bias, const uint32_t *scale_mul,
                              const uint32_t *scale_shift, uint32_t ctrl_flags,
                              uint32_t wgt_words_per_oc, uint32_t out_spatial);

// Tiled (output-row strip) 3x3/1x1 conv for feature maps too big for Act SRAM.
// Activation lives in DDR (tile-major [ic_group][row][col], 16ch/word). Each
// output-row strip stages (strip_in_h = (so-1)*stride+kh) input rows into Act
// SRAM, materializing vertical pad rows from pad_row_ddr (filled here with
// pad_value) so the conv runs with pad_h=0 (horizontal pad stays HW). OC>64 is
// chunked. Output drained tile-major to out_ddr. Caller sets silu_requant.
// NOTE: routes through the im2col conv datapath (yolo_run_conv2d_qparams_pads),
// so it targets 3x3/strided convs. 1x1 pointwise tiling (via the PW engine) is a
// future extension; pointwise layers in the net run at low resolution and
// currently fit Act SRAM without strips.
int yolo_run_conv2d_tiled(uint32_t in_ddr, uint32_t wgt_all_ddr, uint32_t wgt_base,
                          uint32_t out_ddr, uint32_t pad_row_ddr,
                          uint32_t in_w, uint32_t in_h, uint32_t in_c, uint32_t out_c,
                          uint32_t kernel_h, uint32_t kernel_w,
                          uint32_t stride, uint32_t pad,
                          const int32_t *bias, const uint32_t *scale_mul,
                          const uint32_t *scale_shift, uint32_t ctrl_flags,
                          uint32_t wgt_words_per_oc, uint32_t strip_out_rows,
                          int32_t pad_value);

int yolo_run_conv2d_qparams(uint32_t act_base,
                            uint32_t wgt_base,
                            uint32_t out_base,
                            uint32_t in_w,
                            uint32_t in_h,
                            uint32_t in_c,
                            uint32_t out_c,
                            uint32_t kernel_h,
                            uint32_t kernel_w,
                            uint32_t stride,
                            uint32_t pad,
                            const int32_t *bias,
                            const uint32_t *scale_mul,
                            const uint32_t *scale_shift,
                            uint32_t ctrl_flags);
int yolo_run_conv2d_qparams_pads(uint32_t act_base,
                                 uint32_t wgt_base,
                                 uint32_t out_base,
                                 uint32_t in_w,
                                 uint32_t in_h,
                                 uint32_t in_c,
                                 uint32_t out_c,
                                 uint32_t kernel_h,
                                 uint32_t kernel_w,
                                 uint32_t stride,
                                 uint32_t pad_h,
                                 uint32_t pad_w,
                                 const int32_t *bias,
                                 const uint32_t *scale_mul,
                                 const uint32_t *scale_shift,
                                 uint32_t ctrl_flags);
int yolo_run_pw_conv1x1(uint32_t act_base,
                        uint32_t wgt_base,
                        uint32_t out_base,
                        uint32_t in_w,
                        uint32_t in_h,
                        uint32_t in_c,
                        uint32_t out_c,
                        const int32_t *bias,
                        uint32_t scale_mul,
                        uint32_t scale_shift,
                        uint32_t ctrl_flags);
int yolo_run_pw_conv1x1_qparams(uint32_t act_base,
                                uint32_t wgt_base,
                                uint32_t out_base,
                                uint32_t in_w,
                                uint32_t in_h,
                                uint32_t in_c,
                                uint32_t out_c,
                                const int32_t *bias,
                                const uint32_t *scale_mul,
                                const uint32_t *scale_shift,
                                uint32_t ctrl_flags);

#endif
