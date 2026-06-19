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
