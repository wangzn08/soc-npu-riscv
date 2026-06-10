#ifndef FIRMWARE_H
#define FIRMWARE_H

#include <stdint.h>
#include <stdbool.h>

// irq.c
uint32_t *irq(uint32_t *regs, uint32_t irqs);
extern volatile uint32_t npu_irq_flag;

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
#define NPU_CTRL       (NPU_BASE + 0x000)  // [0]start [1]ping_pong [2]pool_en [3]eltwise_en [4]clear_done [5]relu_en
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
#define NPU_BIAS_ADDR  (NPU_BASE + 0x038)
#define NPU_SCALE_ADDR (NPU_BASE + 0x03C)
#define NPU_BIAS(ch)   (NPU_BASE + 0x040 + ((ch) * 4))
#define NPU_SCALE(ch)  (NPU_BASE + 0x080 + ((ch) * 4))
#define NPU_SHIFT(ch)  (NPU_BASE + 0x0C0 + ((ch) * 4))
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
#define NPU_DMA_PATH_CTL    (NPU_BASE + 0x148)

// CTRL bits
#define NPU_CTRL_START      (1 << 0)
#define NPU_CTRL_PING_PONG  (1 << 1)
#define NPU_CTRL_POOL_EN    (1 << 2)
#define NPU_CTRL_ELTWISE_EN (1 << 3)
#define NPU_CTRL_CLEAR_DONE (1 << 4)
#define NPU_CTRL_RELU_EN    (1 << 5)

// STATUS bits
#define NPU_STATUS_DONE_IRQ   (1 << 0)
#define NPU_STATUS_BUSY       (1 << 1)
#define NPU_STATUS_DMA_RD_ERR (1 << 2)
#define NPU_STATUS_DMA_WR_ERR (1 << 3)

#endif
