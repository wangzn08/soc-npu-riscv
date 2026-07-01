// Reusable CPU-side YOLO block scheduling helpers.
// These helpers use the existing shared SoC/NPU DMA and Act SRAM layout; they
// do not introduce a model-specific hardware path.

#include "firmware.h"
#include "yolo_ops.h"

/* YOLO_ICG_BUF is defined in yolo_ops.h (mirrors RTL wgt_reader ICG_BUF /
 * im2col ICG_MAX). 1x1 PW with ic_groups>ICG_BUF streams weights per IC group
 * (non-oc_single OC-16 tiling); 3x3 streams via yolo_run_conv2d_ic_stream. */

#define YOLO_DMA_TIMEOUT 4000000u
#define YOLO_NPU_TIMEOUT 60000000u   /* big 320 layers (ic128/oc256, no row_par) run many M cycles */
#define YOLO_DMA_MAX_BEATS 256u
#define YOLO_ACT_SRAM_WORDS 65536u

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

int yolo_run_upsample2x_ddr(uint32_t src_ddr,
                            uint32_t dst_ddr,
                            uint32_t scratch_act_base,
                            uint32_t in_w,
                            uint32_t in_h,
                            uint32_t ic_groups)
{
    uint32_t in_words = in_w * in_h * ic_groups;
    uint32_t out_words = in_words * 4u;
    uint32_t src_act_base = scratch_act_base + out_words;

    if (in_w == 0u || in_h == 0u || ic_groups == 0u)
        return 0;
    if (src_act_base + in_words > YOLO_ACT_SRAM_WORDS)
        return 0;

    if (!yolo_dma_ddr_to_act(src_ddr, src_act_base, in_words))
        return 0;
    if (!yolo_run_upsample2x(src_act_base, scratch_act_base, in_w, in_h, ic_groups))
        return 0;

    return yolo_dma_act_to_ddr(dst_ddr, scratch_act_base, out_words);
}

int yolo_copy_ddr_to_ddr_via_act(uint32_t src_ddr,
                                 uint32_t dst_ddr,
                                 uint32_t scratch_act_base,
                                 uint32_t words)
{
    uint32_t done = 0u;

    if (words == 0u)
        return 1;
    if (scratch_act_base >= YOLO_ACT_SRAM_WORDS)
        return 0;

    while (done < words) {
        uint32_t cap = YOLO_ACT_SRAM_WORDS - scratch_act_base;
        uint32_t chunk = words - done;
        if (chunk > YOLO_DMA_MAX_BEATS)
            chunk = YOLO_DMA_MAX_BEATS;
        if (chunk > cap)
            chunk = cap;
        if (chunk == 0u)
            return 0;

        if (!yolo_dma_ddr_to_act(src_ddr + done * 16u, scratch_act_base, chunk))
            return 0;
        if (!yolo_dma_act_to_ddr(dst_ddr + done * 16u, scratch_act_base, chunk))
            return 0;

        done += chunk;
    }

    return 1;
}

int yolo_run_maxpool5x5(uint32_t src_ddr,
                        uint32_t dst_ddr,
                        uint32_t scratch_act_base,
                        uint32_t in_w,
                        uint32_t in_h,
                        uint32_t ic_groups)
{
    uint32_t words = in_w * in_h * ic_groups;
    uint32_t dst_act_base = scratch_act_base + words;

    if (in_w == 0u || in_h == 0u || ic_groups == 0u)
        return 1;

    if (!yolo_dma_ddr_to_act(src_ddr, scratch_act_base, words))
        return 0;

    npu_wr(NPU_DMA_RD_SRAM_BASE, scratch_act_base);
    npu_wr(NPU_DMA_WR_SRAM_BASE, dst_act_base);
    npu_wr(NPU_UPSAMPLE_CFG0, (in_h << 16) | in_w);
    npu_wr(NPU_UPSAMPLE_CFG1, ic_groups);
    npu_wr(NPU_DMA_UPSAMPLE_TRIG, 2u);

    if (!wait_dma_status(NPU_DMA_STATUS_MAXPOOL5_DONE))
        return 0;

    return yolo_dma_act_to_ddr(dst_ddr, dst_act_base, words);
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

void yolo_dfl_load_weights(const int16_t wk[16])
{
    uint32_t i;
    for (i = 0u; i < 16u; i++)
        npu_wr(NPU_DFL_WLOAD, (i << 16) | ((uint32_t)(uint16_t)wk[i]));
}

void yolo_dfl_load_exp_lut(const uint16_t e[256])
{
    uint32_t i;
    for (i = 0u; i < 256u; i++)
        npu_wr(NPU_DFL_ELOAD, (i << 16) | (uint32_t)e[i]);
}

int yolo_run_dfl(uint32_t src, uint32_t dst, uint32_t n)
{
    npu_wr(NPU_DFL_SRC, src);
    npu_wr(NPU_DFL_DST, dst);
    npu_wr(NPU_DFL_CNT, n);
    npu_wr(NPU_DFL_TRIG, 1u);
    return wait_dma_status(NPU_DMA_STATUS_DFL_DONE);
}

int yolo_run_dfl_ddr(uint32_t src_ddr,
                     uint32_t dst_ddr,
                     uint32_t scratch_act_base,
                     uint32_t in_words)
{
    uint32_t out_words = in_words >> 2;
    uint32_t dst_act_base = scratch_act_base + in_words;

    if (in_words == 0u)
        return 1;
    if ((in_words & 3u) != 0u)
        return 0;

    if (!yolo_dma_ddr_to_act(src_ddr, scratch_act_base, in_words))
        return 0;
    if (!yolo_run_dfl(scratch_act_base, dst_act_base, in_words))
        return 0;
    return yolo_dma_act_to_ddr(dst_ddr, dst_act_base, out_words);
}

int yolo_run_eltwise_add_ddr(uint32_t src0_ddr,
                             uint32_t src1_ddr,
                             uint32_t dst_ddr,
                             uint32_t scratch_act_base,
                             uint32_t words,
                             int32_t zp,
                             uint32_t ratio_en,
                             uint32_t ratio_mul,
                             uint32_t ratio_shift)
{
    uint32_t done = 0u;
    const uint32_t max_chunk = 8192u;

    if (words == 0u)
        return 1;

    while (done < words) {
        uint32_t chunk = words - done;
        uint32_t src1_act;
        uint32_t dst_act;

        if (chunk > max_chunk)
            chunk = max_chunk;
        src1_act = scratch_act_base + chunk;
        dst_act = src1_act + chunk;

        if (!yolo_dma_ddr_to_act(src0_ddr + done * 16u, scratch_act_base, chunk))
            return 0;
        if (!yolo_dma_ddr_to_act(src1_ddr + done * 16u, src1_act, chunk))
            return 0;

        npu_wr(NPU_DMA_RD_SRAM_BASE, scratch_act_base);
        npu_wr(NPU_SKIP_BASE, src1_act);
        npu_wr(NPU_DMA_WR_SRAM_BASE, dst_act);
        npu_wr(NPU_DMA_RD_LEN, chunk);
        npu_wr(NPU_ELTWISE_ZP, NPU_ELT_PACK((uint32_t)zp, ratio_en, ratio_shift, ratio_mul));
        npu_wr(NPU_DMA_UPSAMPLE_TRIG, 4u);

        if (!wait_dma_status(NPU_DMA_STATUS_ELTWISE_DONE))
            return 0;
        if (!yolo_dma_act_to_ddr(dst_ddr + done * 16u, dst_act, chunk))
            return 0;

        done += chunk;
    }

    return 1;
}

void yolo_load_sigmoid_lut(const uint8_t p[256])
{
    uint32_t i;
    for (i = 0u; i < 256u; i++)
        npu_wr(NPU_SIGM_LOAD, (i << 8) | (uint32_t)p[i]);
}

void yolo_load_silu_lut(const uint8_t t[256])
{
    uint32_t i;
    for (i = 0u; i < 256u; i++)
        npu_wr(NPU_SILU_LOAD, (i << 8) | (uint32_t)t[i]);
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

int yolo_run_conv2d_tiled(uint32_t in_ddr, uint32_t wgt_all_ddr, uint32_t wgt_base,
                          uint32_t out_ddr, uint32_t pad_row_ddr,
                          uint32_t in_w, uint32_t in_h, uint32_t in_c, uint32_t out_c,
                          uint32_t kernel_h, uint32_t kernel_w,
                          uint32_t stride, uint32_t pad,
                          const int32_t *bias, const uint32_t *scale_mul,
                          const uint32_t *scale_shift, uint32_t ctrl_flags,
                          uint32_t wgt_words_per_oc, uint32_t strip_out_rows,
                          int32_t pad_value)
{
    uint32_t icg = in_c / 16u;
    uint32_t out_w, out_h, oy0;

    if (icg == 0u || in_w == 0u || in_h == 0u || out_c == 0u ||
        kernel_h == 0u || kernel_w == 0u || stride == 0u || strip_out_rows == 0u)
        return 0;

    out_w = (in_w + 2u * pad - kernel_w) / stride + 1u;
    out_h = (in_h + 2u * pad - kernel_h) / stride + 1u;

    /* row_par (16-row spatial parallelism) is correct here and ~9x faster on deep
     * layers, BUT only for stride-1, normal-size strips:
     *   - it misorders strips smaller than 16 (e.g. the conv13 strip=2 halo test);
     *   - it misorders the drain for stride>1 convs (verified on conv1 @320: the
     *     16-row reorder assumes stride-1 row spacing). Exposed once the exact SiLU
     *     LUT stopped masking edge errors; the serial path is bit-exact for stride2.
     * Auto-enable only when safe; otherwise fall back to the serial path. */
    if (strip_out_rows >= 16u && stride == 1u)
        ctrl_flags |= NPU_CTRL_ROW_PAR;

    /* Materialize one pad row in DDR: in_w words, each lane = pad_value. */
    {
        uint32_t lane = (uint32_t)pad_value & 0xFFu;
        uint32_t word = lane | (lane << 8) | (lane << 16) | (lane << 24);
        volatile uint32_t *p = (volatile uint32_t *)pad_row_ddr;
        uint32_t i;
        for (i = 0u; i < in_w * 4u; i++)
            p[i] = word;
    }

    for (oy0 = 0u; oy0 < out_h; oy0 += strip_out_rows) {
        uint32_t so = out_h - oy0;
        uint32_t strip_in_h;
        int32_t ir0;
        uint32_t g, ri, done;

        if (so > strip_out_rows)
            so = strip_out_rows;
        strip_in_h = (so - 1u) * stride + kernel_h;  /* rows needed for 'so' out rows, pad_h=0 */
        ir0 = (int32_t)(oy0 * stride) - (int32_t)pad; /* first input row (may be < 0 => pad) */

        /* Stage strip input rows (with vertical pad rows) into Act SRAM, tile-major. */
        for (g = 0u; g < icg; g++) {
            for (ri = 0u; ri < strip_in_h; ri++) {
                int32_t r = ir0 + (int32_t)ri;
                uint32_t dst = (g * strip_in_h + ri) * in_w;
                uint32_t src = (r >= 0 && r < (int32_t)in_h)
                    ? in_ddr + (g * in_h + (uint32_t)r) * in_w * 16u
                    : pad_row_ddr;
                if (!yolo_dma_ddr_to_act(src, dst, in_w))
                    return 0;
            }
        }

        /* OC>64 chunks: weights reloaded per chunk; conv runs pad_h=0, pad_w=pad (HW).
         * Large-IC 1x1 (ic_groups>ICG_BUF) can't hold all IC groups resident, so it
         * uses per-IC-group weight streaming (reuse_mode=0 in the FSM). That path is
         * incompatible with oc_single (oc_single forces full IC residency via pf_all
         * in wgt_reader), so streamed PW tiles OC into <=16 (one OC tile per start,
         * no oc_single). IC<=ICG_BUF keeps the fast resident oc_single path.
         * NOTE: 3x3 convs with ic_groups>ICG_BUF are NOT yet supported -- the im2col
         * line buffer (ICG_MAX=4) holds only 4 IC tiles, so the activation window,
         * not the weight buffer, is the limit there (see RESUME doc). Only 1x1 PW
         * (no im2col) streams large IC today. */
        done = 0u;
        while (done < out_c) {
            uint32_t chunk = out_c - done;
            uint32_t sg;
            uint32_t is_pw      = (kernel_h == 1u && kernel_w == 1u);
            uint32_t pw_stream  = is_pw && (icg > YOLO_ICG_BUF);
            uint32_t chunk_cap  = pw_stream ? 16u : 64u;
            if (chunk > chunk_cap)
                chunk = chunk_cap;
            if (!yolo_dma_ddr_to_wgt(wgt_all_ddr + done * wgt_words_per_oc * 16u,
                                     wgt_base, chunk * wgt_words_per_oc))
                return 0;
            if (is_pw) {
                /* 1x1 pointwise: no halo (strip_in_h==so), no pad; PW engine.
                 * PW path is not row_par-aware, so mask it off. oc_single only for
                 * the resident (small-IC) path; streamed PW runs one 16-OC tile. */
                uint32_t pwf = ctrl_flags & ~NPU_CTRL_ROW_PAR;
                if (!pw_stream) pwf |= NPU_CTRL_OC_SINGLE;
                if (!yolo_run_pw_conv1x1_qparams(0u, wgt_base, 0u,
                                                 in_w, strip_in_h, in_c, chunk,
                                                 bias + done, scale_mul + done,
                                                 scale_shift + done, pwf))
                    return 0;
            } else if (!yolo_run_conv2d_qparams_pads(0u, wgt_base, 0u,
                                              in_w, strip_in_h, in_c, chunk,
                                              kernel_h, kernel_w, stride,
                                              0u, pad,
                                              bias + done, scale_mul + done,
                                              scale_shift + done,
                                              ctrl_flags | NPU_CTRL_OC_SINGLE)) {
                return 0;
            }
            /* Drain 'so' rows per OC-group into the full-height out tensor at row oy0. */
            for (sg = 0u; sg < chunk / 16u; sg++) {
                uint32_t oc_grp = done / 16u + sg;
                if (!yolo_dma_out_to_ddr(out_ddr + (oc_grp * out_h + oy0) * out_w * 16u,
                                         sg * so * out_w, so * out_w, 0u))
                    return 0;
            }
            done += chunk;
        }
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * Large-IC im2col conv via IC-chunk streaming + INT32 psum accumulate.
 * The im2col line buffer holds only ICG_MAX IC tiles, so a 3x3 conv with
 * ic_groups>ICG_MAX must process IC in chunks of <=ICG_MAX tiles. Each chunk is
 * a normal serial conv emitting RAW INT32 psum (bias=0, mul=1, shift=0, with
 * int32_out + ic_stream so each output position owns 4 x4-spaced Out-SRAM words);
 * the chunk is drained to DDR. The CPU sums the chunks' INT32 partials, then
 * applies (acc+bias)*mul>>shift + the exact-SiLU LUT -> INT8 (bit-identical to the
 * HW exact post-process). Whole output computed in one pass (no row strip), which
 * is valid for the small-spatial deep layers that need large-IC 3x3 (c2f8/conv20/
 * SPPF-area). stride==1 only. psum_ddr scratch must hold
 * (out_c/16)*out_spatial*4 *2 words (accumulator + per-chunk temp). */
static const int32_t  YOLO_ZERO_BIAS[16]  = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
static const uint32_t YOLO_UNIT_MUL[16]   = {1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u};
static const uint32_t YOLO_ZERO_SHIFT[16] = {0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u};

int yolo_run_conv2d_ic_stream(uint32_t in_ddr, uint32_t wgt_all_ddr, uint32_t wgt_base,
                              uint32_t out_ddr, uint32_t psum_ddr, uint32_t pad_row_ddr,
                              uint32_t in_w, uint32_t in_h, uint32_t in_c, uint32_t out_c,
                              uint32_t kernel_h, uint32_t kernel_w, uint32_t stride, uint32_t pad,
                              const int32_t *bias, const uint32_t *scale_mul,
                              const uint32_t *scale_shift, const uint8_t *silu_lut,
                              int32_t pad_value)
{
    uint32_t icg = in_c / 16u;
    uint32_t ko  = kernel_h * kernel_w;
    uint32_t wpoc_full = icg * ko;          /* weight words per OC, full IC */
    uint32_t out_w, out_h, out_sp, ocg_total;
    uint32_t nchunks, c, oc_done, ocg, p, l, g, ri;
    uint32_t acc_words, acc_i32, temp_ddr;

    if (icg == 0u || in_w == 0u || in_h == 0u || out_c == 0u || (out_c & 15u) != 0u ||
        kernel_h == 0u || kernel_w == 0u || stride == 0u)
        return 0;

    out_w = (in_w + 2u * pad - kernel_w) / stride + 1u;
    out_h = (in_h + 2u * pad - kernel_h) / stride + 1u;
    out_sp = out_w * out_h;
    ocg_total = out_c / 16u;
    nchunks = (icg + YOLO_ICG_BUF - 1u) / YOLO_ICG_BUF;
    acc_words = ocg_total * out_sp * 4u;     /* 128-bit words for the INT32 accumulator */
    acc_i32 = acc_words * 4u;                 /* int32 count */
    temp_ddr = psum_ddr + acc_words * 16u;    /* per-chunk temp region */

    /* Materialize one pad row in DDR (HW handles horizontal pad; vertical via rows). */
    {
        uint32_t lane = (uint32_t)pad_value & 0xFFu;
        uint32_t word = lane | (lane << 8) | (lane << 16) | (lane << 24);
        volatile uint32_t *pr = (volatile uint32_t *)pad_row_ddr;
        uint32_t i;
        for (i = 0u; i < in_w * 4u; i++) pr[i] = word;
    }

    for (c = 0u; c < nchunks; c++) {
        uint32_t g0   = c * YOLO_ICG_BUF;
        uint32_t cicg = (icg - g0 < YOLO_ICG_BUF) ? (icg - g0) : YOLO_ICG_BUF;
        uint32_t cic  = cicg * 16u;
        uint32_t wpoc_chunk = cicg * ko;
        uint32_t dst_psum = (c == 0u) ? psum_ddr : temp_ddr;
        uint32_t strip_in_h = (out_h - 1u) * stride + kernel_h;
        int32_t  ir0 = -(int32_t)pad;

        /* Stage this chunk's IC groups (tile-major) into Act SRAM with vertical pad. */
        for (g = 0u; g < cicg; g++) {
            for (ri = 0u; ri < strip_in_h; ri++) {
                int32_t r = ir0 + (int32_t)ri;
                uint32_t dst = (g * strip_in_h + ri) * in_w;
                uint32_t src = (r >= 0 && r < (int32_t)in_h)
                    ? in_ddr + ((g0 + g) * in_h + (uint32_t)r) * in_w * 16u
                    : pad_row_ddr;
                if (!yolo_dma_ddr_to_act(src, dst, in_w))
                    return 0;
            }
        }

        /* Per 16-OC tile: gather this chunk's weights, run raw-INT32 conv, drain. */
        for (oc_done = 0u; oc_done < out_c; oc_done += 16u) {
            uint32_t o;
            for (o = 0u; o < 16u; o++) {
                if (!yolo_dma_ddr_to_wgt(wgt_all_ddr + ((oc_done + o) * wpoc_full + g0 * ko) * 16u,
                                         wgt_base + o * wpoc_chunk, wpoc_chunk))
                    return 0;
            }
            if (!yolo_run_conv2d_qparams_pads(0u, wgt_base, 0u, in_w, strip_in_h, cic, 16u,
                                              kernel_h, kernel_w, stride, 0u, pad,
                                              YOLO_ZERO_BIAS, YOLO_UNIT_MUL, YOLO_ZERO_SHIFT,
                                              NPU_CTRL_INT32_OUT | NPU_CTRL_IC_STREAM))
                return 0;
            /* drain 4 INT32 words/position for this OC group */
            if (!yolo_dma_out_to_ddr(dst_psum + (oc_done / 16u) * out_sp * 4u * 16u,
                                     0u, out_sp * 4u, 0u))
                return 0;
        }

        /* CPU: accumulate this chunk's partial into the running INT32 sum. */
        if (c != 0u) {
            volatile int32_t *pa = (volatile int32_t *)psum_ddr;
            volatile int32_t *pt = (volatile int32_t *)temp_ddr;
            uint32_t w;
            for (w = 0u; w < acc_i32; w++) pa[w] += pt[w];
        }
    }

    /* CPU final: (acc+bias)*mul>>shift, clamp INT8, exact-SiLU LUT -> tile-major INT8. */
    for (ocg = 0u; ocg < ocg_total; ocg++) {
        for (p = 0u; p < out_sp; p++) {
            uint32_t outw[4] = {0u, 0u, 0u, 0u};
            for (l = 0u; l < 16u; l++) {
                uint32_t oc = ocg * 16u + l;
                volatile int32_t *pp =
                    (volatile int32_t *)(psum_ddr + ((ocg * out_sp + p) * 4u + (l >> 2)) * 16u);
                int32_t acc = pp[l & 3u];
                int64_t v = ((int64_t)(acc + bias[oc]) * (int64_t)(int32_t)scale_mul[oc])
                            >> scale_shift[oc];
                int32_t s2 = (int32_t)v;
                if (s2 > 127) s2 = 127; else if (s2 < -128) s2 = -128;
                outw[l >> 2] |= ((uint32_t)silu_lut[s2 & 0xFF]) << ((l & 3u) * 8u);
            }
            {
                volatile uint32_t *od =
                    (volatile uint32_t *)(out_ddr + (ocg * out_sp + p) * 16u);
                od[0] = outw[0]; od[1] = outw[1]; od[2] = outw[2]; od[3] = outw[3];
            }
        }
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
