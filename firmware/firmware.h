#ifndef FIRMWARE_H
#define FIRMWARE_H

#include <stdint.h>
#include <stdbool.h>

// irq.c
uint32_t *irq(uint32_t *regs, uint32_t irqs);
extern volatile uint32_t npu_irq_flag;
extern volatile uint32_t dma_rd_irq_flag;   // set by IRQ handler when DMA read completes
extern volatile uint32_t dma_wr_irq_flag;   // set by IRQ handler when DMA write completes

// print.c
void print_chr(char ch);
void print_str(const char *p);
void print_dec(unsigned int val);
void print_hex(unsigned int val, int digits);

// usercode.c
void usercode(void);

// usercode2.c
void usercode2(void);

// usercode3.c
void usercode3(void);

// usercode4.c
void usercode4(void);

// usercode5.c
void usercode5(void);

// usercode6.c
void usercode6(void);

// usercode7.c (deepnet deployment)
void usercode7(void);

// ---- NPU 寄存器定义 (0x3000_0000 ~ 0x3000_0FFF) ----
// 16×16 systolic array NPU register map (matches param_regfile.v)
#define NPU_BASE       0x30000000
#define NPU_CTRL       (NPU_BASE + 0x000)  // [0]start [1]ping_pong [2]pool_en [3]eltwise_en [4]clear_done [5]relu_en [6]out_ping [7]gemm_en
#define NPU_STATUS     (NPU_BASE + 0x004)  // [0]done_irq [1]busy [2]dma_rd_err [3]dma_wr_err (read-only)
#define NPU_ACT_ADDR_A (NPU_BASE + 0x008)
#define NPU_ACT_ADDR_B (NPU_BASE + 0x00C)
#define NPU_WGT_ADDR_A (NPU_BASE + 0x010)
#define NPU_WGT_ADDR_B (NPU_BASE + 0x014)
#define NPU_OUT_ADDR_A (NPU_BASE + 0x018)
#define NPU_OUT_ADDR_B (NPU_BASE + 0x01C)
#define NPU_IN_W       (NPU_BASE + 0x020)
#define NPU_IN_H       (NPU_BASE + 0x024)
#define NPU_IC         (NPU_BASE + 0x028)
#define NPU_OC         (NPU_BASE + 0x02C)
#define NPU_KERNEL     (NPU_BASE + 0x030)  // [15:8]KH [7:0]KW
#define NPU_STRIDE     (NPU_BASE + 0x034)  // [15:8]SX [7:0]SY
#define NPU_PAD        (NPU_BASE + 0x150)  // [15:8]pad_h [7:0]pad_w (hardware padding, CTRL[8])
#define NPU_BIAS_ADDR  (NPU_BASE + 0x038)
#define NPU_SCALE_ADDR (NPU_BASE + 0x03C)
// Per-OC bias/scale/shift. ch 0..15 use the legacy 0x040/0x080/0x0C0 blocks;
// ch 16..63 use the decision-O extended region (0x160 / 0x220 / 0x2E0).
#define NPU_BIAS(ch)   (NPU_BASE + ((ch) < 16 ? 0x040 + ((ch) * 4) : 0x160 + (((ch) - 16) * 4)))
#define NPU_SCALE(ch)  (NPU_BASE + ((ch) < 16 ? 0x080 + ((ch) * 4) : 0x220 + (((ch) - 16) * 4)))
#define NPU_SHIFT(ch)  (NPU_BASE + ((ch) < 16 ? 0x0C0 + ((ch) * 4) : 0x2E0 + (((ch) - 16) * 4)))
// DMA control registers (byte offsets matching param_regfile.v case values)
#define NPU_DMA_RD_TRIG      (NPU_BASE + 0x120)  // write any value to trigger DMA read
#define NPU_DMA_RD_DDR_ADDR  (NPU_BASE + 0x124)
#define NPU_DMA_RD_LEN       (NPU_BASE + 0x128)
#define NPU_DMA_RD_SRAM_BASE (NPU_BASE + 0x12C)
#define NPU_DMA_WR_TRIG      (NPU_BASE + 0x130)  // write any value to trigger DMA write
#define NPU_DMA_WR_DDR_ADDR  (NPU_BASE + 0x134)
#define NPU_DMA_WR_LEN       (NPU_BASE + 0x138)
#define NPU_DMA_WR_SRAM_BASE (NPU_BASE + 0x13C)
#define NPU_DMA_STATUS       (NPU_BASE + 0x140)  // [0]rd_done [1]wr_done (read-only)
#define NPU_DMA_SRAM_SEL     (NPU_BASE + 0x144)  // 0=Act, 1=Wgt
#define NPU_DMA_PATH_CTL     (NPU_BASE + 0x148)
#define NPU_DMA_PING_SEL     (NPU_BASE + 0x14C)  // [0]=Act ping, [1]=Wgt ping, [2]=Out ping
#define NPU_DMA_COPY_TRIG    (NPU_BASE + 0x154)  // write any value: trigger on-chip Out->Act copy
#define NPU_DMA_STATUS_COPY_DONE (1 << 2)        // NPU_DMA_STATUS bit2: on-chip copy complete
#define NPU_DMA_EXPAND_TRIG  (NPU_BASE + 0x158)  // write any value: trigger img_expand (byte->word)
#define NPU_DMA_STATUS_EXPAND_DONE (1 << 3)      // NPU_DMA_STATUS bit3: img_expand complete
#define NPU_DMA_TRANSPOSE_TRIG (NPU_BASE + 0x15C) // write any value: trigger Conv6->FC transpose
#define NPU_DMA_STATUS_TRANSPOSE_DONE (1 << 4)   // NPU_DMA_STATUS bit4: transpose complete

// CTRL bits
#define NPU_CTRL_START      (1 << 0)
#define NPU_CTRL_PING_PONG  (1 << 1)
#define NPU_CTRL_POOL_EN    (1 << 2)
#define NPU_CTRL_ELTWISE_EN (1 << 3)
#define NPU_CTRL_CLEAR_DONE (1 << 4)
#define NPU_CTRL_RELU_EN    (1 << 5)
#define NPU_CTRL_OUT_PING   (1 << 6)
#define NPU_CTRL_GEMM_EN    (1 << 7)   // GEMM/FC mode: bypass im2col, vector x matrix
#define NPU_CTRL_HW_PAD     (1 << 8)   // hardware padding: FSM injects border zeros, reads tile-major
#define NPU_CTRL_ROW_PAR    (1 << 9)   // 16-row spatial parallelism (task E): 16 output pixels/group
#define NPU_CTRL_GEMM_REDUCE (1 << 10) // GEMM 16-row IC-reduction (decision M): util 6.25%->~100%
#define NPU_CTRL_ROW_BLOCK   (1 << 11) // row-block packing (#4): pack R output rows, narrow layers 50%->100%
#define NPU_CTRL_OC_SINGLE   (1 << 12) // all-OC-tiles in one start (decision O): im2col+weights reused across OC tiles
#define NPU_CTRL_INT32_OUT   (1 << 13) // raw INT32 output (decision Q): emit scaled un-clamped INT32 (final-FC logits)

// STATUS bits
#define NPU_STATUS_DONE_IRQ   (1 << 0)
#define NPU_STATUS_BUSY       (1 << 1)
#define NPU_STATUS_DMA_RD_ERR (1 << 2)
#define NPU_STATUS_DMA_WR_ERR (1 << 3)

#endif
