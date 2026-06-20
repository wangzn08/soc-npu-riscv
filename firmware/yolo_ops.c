// Reusable CPU-side YOLO block scheduling helpers.
// These helpers use the existing shared SoC/NPU DMA and Act SRAM layout; they
// do not introduce a model-specific hardware path.

#include "firmware.h"
#include "yolo_ops.h"

#define YOLO_DMA_TIMEOUT 500000u
#define YOLO_NPU_TIMEOUT 500000u
#define YOLO_DMA_MAX_BEATS 256u

static inline void npu_wr(uint32_t addr, uint32_t data)
{
    *(volatile uint32_t *)addr = data;
}

static inline uint32_t npu_rd(uint32_t addr)
{
    return *(volatile uint32_t *)addr;
}

static int wait_dma_status(uint32_t mask)
{
    uint32_t timeout = YOLO_DMA_TIMEOUT;
    while (timeout-- != 0u) {
        if ((npu_rd(NPU_DMA_STATUS) & mask) != 0u)
            return 1;
    }
    return 0;
}

int yolo_dma_ddr_to_act(uint32_t ddr_addr, uint32_t act_base, uint32_t words)
{
    uint32_t sent = 0u;

    npu_wr(NPU_DMA_PING_SEL, 0u);
    npu_wr(NPU_DMA_SRAM_SEL, 0u);

    while (sent < words) {
        uint32_t chunk = words - sent;
        if (chunk > YOLO_DMA_MAX_BEATS)
            chunk = YOLO_DMA_MAX_BEATS;

        npu_wr(NPU_DMA_RD_DDR_ADDR, ddr_addr + sent * 16u);
        npu_wr(NPU_DMA_RD_LEN, chunk - 1u);
        npu_wr(NPU_DMA_RD_SRAM_BASE, act_base + sent);
        npu_wr(NPU_DMA_RD_TRIG, 1u);

        if (!wait_dma_status(NPU_DMA_STATUS_RD_DONE))
            return 0;

        sent += chunk;
    }

    return 1;
}

int yolo_dma_ddr_to_wgt(uint32_t ddr_addr, uint32_t wgt_base, uint32_t words)
{
    uint32_t sent = 0u;

    npu_wr(NPU_DMA_PING_SEL, 0u);
    npu_wr(NPU_DMA_SRAM_SEL, 1u);
    npu_wr(NPU_DMA_PATH_CTL, 0x4u);

    while (sent < words) {
        uint32_t chunk = words - sent;
        if (chunk > YOLO_DMA_MAX_BEATS)
            chunk = YOLO_DMA_MAX_BEATS;

        npu_wr(NPU_DMA_RD_DDR_ADDR, ddr_addr + sent * 16u);
        npu_wr(NPU_DMA_RD_LEN, chunk - 1u);
        npu_wr(NPU_DMA_RD_SRAM_BASE, wgt_base + sent);
        npu_wr(NPU_DMA_RD_TRIG, 1u);

        if (!wait_dma_status(NPU_DMA_STATUS_RD_DONE))
            return 0;

        sent += chunk;
    }

    return 1;
}

int yolo_dma_act_to_ddr(uint32_t ddr_addr, uint32_t act_base, uint32_t words)
{
    uint32_t sent = 0u;

    npu_wr(NPU_DMA_PING_SEL, 0u);
    npu_wr(NPU_DMA_PATH_CTL, 0x2u);

    while (sent < words) {
        uint32_t chunk = words - sent;
        if (chunk > YOLO_DMA_MAX_BEATS)
            chunk = YOLO_DMA_MAX_BEATS;

        npu_wr(NPU_DMA_WR_DDR_ADDR, ddr_addr + sent * 16u);
        npu_wr(NPU_DMA_WR_LEN, chunk - 1u);
        npu_wr(NPU_DMA_WR_SRAM_BASE, act_base + sent);
        npu_wr(NPU_DMA_WR_TRIG, 1u);

        if (!wait_dma_status(NPU_DMA_STATUS_WR_DONE))
            return 0;

        sent += chunk;
    }

    return 1;
}

int yolo_dma_out_to_ddr(uint32_t ddr_addr,
                        uint32_t out_base,
                        uint32_t words,
                        uint32_t out_pong)
{
    uint32_t sent = 0u;

    npu_wr(NPU_DMA_SRAM_SEL, 0u);
    npu_wr(NPU_DMA_PATH_CTL, 0x1u);
    npu_wr(NPU_DMA_PING_SEL, out_pong != 0u ? 0x4u : 0x0u);

    while (sent < words) {
        uint32_t chunk = words - sent;
        if (chunk > YOLO_DMA_MAX_BEATS)
            chunk = YOLO_DMA_MAX_BEATS;

        npu_wr(NPU_DMA_WR_DDR_ADDR, ddr_addr + sent * 16u);
        npu_wr(NPU_DMA_WR_LEN, chunk - 1u);
        npu_wr(NPU_DMA_WR_SRAM_BASE, out_base + sent);
        npu_wr(NPU_DMA_WR_TRIG, 1u);

        if (!wait_dma_status(NPU_DMA_STATUS_WR_DONE))
            return 0;

        sent += chunk;
    }

    return 1;
}

int yolo_concat2_ddr_to_act(uint32_t src0_ddr,
                            uint32_t src1_ddr,
                            uint32_t dst_act_base,
                            uint32_t spatial_words,
                            uint32_t src0_groups,
                            uint32_t src1_groups)
{
    uint32_t src0_words = spatial_words * src0_groups;
    uint32_t src1_words = spatial_words * src1_groups;

    if (src0_words != 0u) {
        if (!yolo_dma_ddr_to_act(src0_ddr, dst_act_base, src0_words))
            return 0;
    }

    if (src1_words != 0u) {
        uint32_t dst1_base = dst_act_base + src0_words;
        if (!yolo_dma_ddr_to_act(src1_ddr, dst1_base, src1_words))
            return 0;
    }

    return 1;
}

int yolo_slice_ddr_to_act(uint32_t src_ddr,
                          uint32_t dst_act_base,
                          uint32_t spatial_words,
                          uint32_t first_group,
                          uint32_t group_count)
{
    uint32_t src_offset_words;

    if (spatial_words == 0u || group_count == 0u)
        return 0;

    src_offset_words = first_group * spatial_words;
    return yolo_dma_ddr_to_act(src_ddr + src_offset_words * 16u,
                               dst_act_base,
                               spatial_words * group_count);
}

int yolo_run_upsample2x(uint32_t src_act_base,
                        uint32_t dst_act_base,
                        uint32_t in_w,
                        uint32_t in_h,
                        uint32_t ic_groups)
{
    if (in_w == 0u || in_h == 0u || ic_groups == 0u)
        return 0;

    npu_wr(NPU_DMA_PING_SEL, 0u);
    npu_wr(NPU_DMA_RD_SRAM_BASE, src_act_base);
    npu_wr(NPU_DMA_WR_SRAM_BASE, dst_act_base);
    npu_wr(NPU_UPSAMPLE_CFG0, (in_h << 16) | in_w);
    npu_wr(NPU_UPSAMPLE_CFG1, ic_groups);
    npu_wr(NPU_DMA_UPSAMPLE_TRIG, 1u);

    return wait_dma_status(NPU_DMA_STATUS_UPSAMPLE_DONE);
}

void yolo_set_silu_requant(uint32_t mul, uint32_t shift, int32_t zp)
{
    npu_wr(NPU_SILU_REQUANT_CFG,
           ((uint32_t)zp & 0xFFu) << 24 |
           (shift & 0x3Fu) << 16 |
           (mul & 0xFFFFu));
}

void yolo_set_pad_value(int32_t pad_value)
{
    npu_wr(NPU_PAD_VALUE, (uint32_t)pad_value & 0xFFu);
}

void yolo_set_eltwise(int32_t zp, uint32_t skip_base)
{
    npu_wr(NPU_ELTWISE_ZP, (uint32_t)zp & 0xFFu);
    npu_wr(NPU_SKIP_BASE, skip_base);
}

static int wait_npu_done(void)
{
    uint32_t timeout = YOLO_NPU_TIMEOUT;
    while (timeout-- != 0u) {
        if (npu_irq_flag != 0u) {
            uint32_t status = npu_rd(NPU_STATUS);
            if ((status & (NPU_STATUS_DMA_RD_ERR | NPU_STATUS_DMA_WR_ERR)) != 0u)
                return 0;
            return 1;
        }
    }
    return 0;
}

static int run_pw_conv1x1_common(uint32_t act_base,
                                 uint32_t wgt_base,
                                 uint32_t out_base,
                                 uint32_t in_w,
                                 uint32_t in_h,
                                 uint32_t in_c,
                                 uint32_t out_c,
                                 const int32_t *bias,
                                 const uint32_t *scale_mul,
                                 const uint32_t *scale_shift,
                                 uint32_t uniform_scale_mul,
                                 uint32_t uniform_scale_shift,
                                 uint32_t use_per_channel,
                                 uint32_t ctrl_flags)
{
    uint32_t ch;

    if (in_w == 0u || in_h == 0u || in_c == 0u || out_c == 0u || out_c > 64u)
        return 0;
    if (out_c > 16u && (ctrl_flags & NPU_CTRL_OC_SINGLE) == 0u)
        return 0;

    npu_wr(NPU_IN_W, in_w);
    npu_wr(NPU_IN_H, in_h);
    npu_wr(NPU_IC, in_c);
    npu_wr(NPU_OC, out_c);
    npu_wr(NPU_KERNEL, (1u << 8) | 1u);
    npu_wr(NPU_STRIDE, (1u << 8) | 1u);
    npu_wr(NPU_PAD, 0u);
    npu_wr(NPU_ACT_ADDR_A, act_base);
    npu_wr(NPU_WGT_ADDR_A, wgt_base);
    npu_wr(NPU_OUT_ADDR_A, out_base);

    for (ch = 0u; ch < out_c; ch++) {
        int32_t b = bias != (const int32_t *)0 ? bias[ch] : 0;
        uint32_t mul = use_per_channel != 0u ? scale_mul[ch] : uniform_scale_mul;
        uint32_t sh = use_per_channel != 0u ? scale_shift[ch] : uniform_scale_shift;
        npu_wr(NPU_BIAS(ch), (uint32_t)b);
        npu_wr(NPU_SCALE(ch), mul);
        npu_wr(NPU_SHIFT(ch), sh);
    }

    npu_irq_flag = 0u;
    npu_wr(NPU_CTRL, NPU_CTRL_START | NPU_CTRL_PW_EN | ctrl_flags);
    return wait_npu_done();
}

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
                        uint32_t ctrl_flags)
{
    return run_pw_conv1x1_common(act_base, wgt_base, out_base,
                                 in_w, in_h, in_c, out_c,
                                 bias, (const uint32_t *)0, (const uint32_t *)0,
                                 scale_mul, scale_shift, 0u, ctrl_flags);
}

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
                                uint32_t ctrl_flags)
{
    if (scale_mul == (const uint32_t *)0 || scale_shift == (const uint32_t *)0)
        return 0;

    return run_pw_conv1x1_common(act_base, wgt_base, out_base,
                                 in_w, in_h, in_c, out_c,
                                 bias, scale_mul, scale_shift,
                                 0u, 0u, 1u, ctrl_flags);
}

int yolo_run_pw_conv1x1_oc_chunks(uint32_t act_base, uint32_t wgt_all_ddr, uint32_t wgt_base,
                                  uint32_t out_ddr, uint32_t in_w, uint32_t in_h,
                                  uint32_t in_c, uint32_t out_c,
                                  const int32_t *bias, const uint32_t *scale_mul,
                                  const uint32_t *scale_shift, uint32_t ctrl_flags,
                                  uint32_t out_spatial)
{
    uint32_t wpo = in_c / 16u;   // 1x1 weight words per OC
    uint32_t done = 0u;
    while (done < out_c) {
        uint32_t chunk = out_c - done;
        if (chunk > 64u) chunk = 64u;
        if (!yolo_dma_ddr_to_wgt(wgt_all_ddr + done * wpo * 16u, wgt_base, chunk * wpo))
            return 0;
        if (!yolo_run_pw_conv1x1_qparams(act_base, wgt_base, 0u,
                                         in_w, in_h, in_c, chunk,
                                         bias + done, scale_mul + done, scale_shift + done,
                                         ctrl_flags | NPU_CTRL_OC_SINGLE))
            return 0;
        if (!yolo_dma_out_to_ddr(out_ddr + (done / 16u) * out_spatial * 16u,
                                 0u, (chunk / 16u) * out_spatial, 0u))
            return 0;
        done += chunk;
    }
    return 1;
}

int yolo_run_conv2d_oc_chunks(uint32_t act_base, uint32_t wgt_all_ddr, uint32_t wgt_base,
                              uint32_t out_ddr, uint32_t in_w, uint32_t in_h,
                              uint32_t in_c, uint32_t out_c,
                              uint32_t kernel_h, uint32_t kernel_w,
                              uint32_t stride, uint32_t pad,
                              const int32_t *bias, const uint32_t *scale_mul,
                              const uint32_t *scale_shift, uint32_t ctrl_flags,
                              uint32_t wgt_words_per_oc, uint32_t out_spatial)
{
    uint32_t done = 0u;
    while (done < out_c) {
        uint32_t chunk = out_c - done;
        if (chunk > 64u) chunk = 64u;
        if (!yolo_dma_ddr_to_wgt(wgt_all_ddr + done * wgt_words_per_oc * 16u,
                                 wgt_base, chunk * wgt_words_per_oc))
            return 0;
        if (!yolo_run_conv2d_qparams(act_base, wgt_base, 0u,
                                     in_w, in_h, in_c, chunk,
                                     kernel_h, kernel_w, stride, pad,
                                     bias + done, scale_mul + done, scale_shift + done,
                                     ctrl_flags | NPU_CTRL_OC_SINGLE))
            return 0;
        if (!yolo_dma_out_to_ddr(out_ddr + (done / 16u) * out_spatial * 16u,
                                 0u, (chunk / 16u) * out_spatial, 0u))
            return 0;
        done += chunk;
    }
    return 1;
}

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
                            uint32_t ctrl_flags)
{
    return yolo_run_conv2d_qparams_pads(act_base, wgt_base, out_base,
                                        in_w, in_h, in_c, out_c,
                                        kernel_h, kernel_w, stride,
                                        pad, pad,
                                        bias, scale_mul, scale_shift,
                                        ctrl_flags);
}

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
                                 uint32_t ctrl_flags)
{
    uint32_t ch;
    uint32_t cfg_w = in_w + pad_w * 2u;
    uint32_t cfg_h = in_h + pad_h * 2u;

    if (in_w == 0u || in_h == 0u || in_c == 0u || out_c == 0u ||
        out_c > 64u || kernel_h == 0u || kernel_w == 0u ||
        kernel_h > 3u || kernel_w > 3u || stride == 0u ||
        scale_mul == (const uint32_t *)0 || scale_shift == (const uint32_t *)0)
        return 0;

    if (out_c > 16u && (ctrl_flags & NPU_CTRL_OC_SINGLE) == 0u)
        return 0;

    npu_wr(NPU_IN_W, cfg_w);
    npu_wr(NPU_IN_H, cfg_h);
    npu_wr(NPU_IC, in_c);
    npu_wr(NPU_OC, out_c);
    npu_wr(NPU_KERNEL, (kernel_h << 8) | kernel_w);
    npu_wr(NPU_STRIDE, (stride << 8) | stride);
    npu_wr(NPU_PAD, (pad_h << 8) | pad_w);
    npu_wr(NPU_ACT_ADDR_A, act_base);
    npu_wr(NPU_WGT_ADDR_A, wgt_base);
    npu_wr(NPU_OUT_ADDR_A, out_base);

    for (ch = 0u; ch < out_c; ch++) {
        int32_t b = (bias != (const int32_t *)0) ? bias[ch] : 0;
        npu_wr(NPU_BIAS(ch), (uint32_t)b);
        npu_wr(NPU_SCALE(ch), scale_mul[ch]);
        npu_wr(NPU_SHIFT(ch), scale_shift[ch]);
    }

    npu_irq_flag = 0u;
    npu_wr(NPU_CTRL, NPU_CTRL_START | NPU_CTRL_HW_PAD | ctrl_flags);
    return wait_npu_done();
}
