// deepnet_run.c — SoC path test
// CPU-only 测试 + DMA 往返测试

#include "firmware.h"
#include <stdint.h>

static inline void npu_wr(uint32_t addr, uint32_t data) {
    *(volatile uint32_t *)addr = data;
}
static inline uint32_t npu_rd(uint32_t addr) {
    return *(volatile uint32_t *)addr;
}

void usercode7(void) {
    print_str("=== SoC Basic Path Test (CPU-only) ===\n");

    // ---- Test 1: DDR 写入/读回 ----
    print_str("[Test1] DDR write/readback\n");
    volatile int32_t *ddr = (volatile int32_t *)0x40001000;
    for (int i = 0; i < 16; i++)
        ddr[i] = i * 100;
    int pass1 = 1;
    for (int i = 0; i < 16; i++) {
        if (ddr[i] != i * 100) {
            print_str("  FAIL at "); print_dec(i);
            print_str(": got "); print_dec(ddr[i]);
            print_str(" expected "); print_dec(i * 100);
            print_chr('\n');
            pass1 = 0;
        }
    }
    print_str(pass1 ? "  PASS\n" : "  FAIL\n");

    // ---- Test 2: 1x10 向量点积 (CPU 计算) ----
    print_str("[Test2] 1x10 dot product (CPU)\n");
    volatile int8_t  *A = (volatile int8_t  *)0x40002000;
    volatile int8_t  *B = (volatile int8_t  *)0x40002100;
    volatile int32_t *C = (volatile int32_t *)0x40002200;

    for (int i = 0; i < 10; i++) {
        A[i] = i + 1;
        B[i] = 1;
    }
    int32_t sum = 0;
    for (int i = 0; i < 10; i++)
        sum += (int32_t)A[i] * (int32_t)B[i];
    C[0] = sum;

    print_str("  A = {1..10}, B = {1..1}\n");
    print_str("  dot = "); print_dec((uint32_t)sum);
    print_str(" (expected 55)\n");
    int pass2 = (sum == 55);
    print_str(pass2 ? "  PASS\n" : "  FAIL\n");

    // ---- Test 3: DDR 大块写入/校验 (4KB) ----
    print_str("[Test3] DDR bulk write/verify (4KB)\n");
    volatile int8_t *bulk = (volatile int8_t *)0x40003000;
    int bulk_size = 4096;
    for (int i = 0; i < bulk_size; i++)
        bulk[i] = (int8_t)(i & 0xFF);
    int pass3 = 1;
    for (int i = 0; i < bulk_size; i++) {
        if (bulk[i] != (int8_t)(i & 0xFF)) {
            print_str("  FAIL at offset "); print_dec(i);
            print_str(": got "); print_dec((uint32_t)(uint8_t)bulk[i]);
            print_str(" expected "); print_dec((uint32_t)(uint8_t)(i & 0xFF));
            print_chr('\n');
            pass3 = 0;
            break;
        }
    }
    print_str(pass3 ? "  PASS\n" : "  FAIL\n");

    // ---- Test 4: 顺序写入1-100，倒序读回 ----
    print_str("[Test4] seq write 1-100, reverse read\n");
    volatile int32_t *seq = (volatile int32_t *)0x40004000;
    for (int i = 0; i < 100; i++)
        seq[i] = i + 1;
    int pass4 = 1;
    for (int i = 99; i >= 0; i--) {
        if (seq[i] != i + 1) {
            print_str("  FAIL at "); print_dec(i);
            print_str(": got "); print_dec(seq[i]);
            print_str(" expected "); print_dec(i + 1);
            print_chr('\n');
            pass4 = 0;
            break;
        }
    }
    print_str(pass4 ? "  PASS\n" : "  FAIL\n");

    // ================================================================
    // Helper: DMA read DDR→SRAM, wait for completion
    // ================================================================
    #define DMA_RD_TIMEOUT 50000
    #define DMA_WR_TIMEOUT 50000

    // ---- Test 5: DMA 基本往返 (DDR → Act SRAM → DDR), 4 beats ----
    print_str("[Test5] DMA basic round-trip Act SRAM (4 beats)\n");
    volatile int32_t *dma_src5 = (volatile int32_t *)0x40005000;
    volatile int32_t *dma_dst5 = (volatile int32_t *)0x40006000;
    for (int i = 0; i < 16; i++) {
        dma_src5[i] = 0xAA00 + i;
        dma_dst5[i] = 0xDEAD;
    }

    npu_wr(NPU_DMA_SRAM_SEL, 0);         // DMA 写目标 = Act SRAM
    npu_wr(NPU_DMA_PATH_CTL, 0x2);       // DMA 读源 = Act SRAM
    npu_wr(NPU_DMA_RD_DDR_ADDR, 0x40005000);
    npu_wr(NPU_DMA_RD_LEN, 3);           // arlen=3 → 4 beats
    npu_wr(NPU_DMA_RD_SRAM_BASE, 0);
    npu_wr(NPU_DMA_RD_TRIG, 1);

    int timeout = DMA_RD_TIMEOUT;
    while (timeout-- > 0) {
        if (npu_rd(NPU_DMA_STATUS) & 0x1) break;
    }
    if (timeout <= 0) print_str("  DMA read timeout!\n");

    npu_wr(NPU_DMA_WR_DDR_ADDR, 0x40006000);
    npu_wr(NPU_DMA_WR_LEN, 3);
    npu_wr(NPU_DMA_WR_SRAM_BASE, 0);
    npu_wr(NPU_DMA_WR_TRIG, 1);

    timeout = DMA_WR_TIMEOUT;
    while (timeout-- > 0) {
        if (npu_rd(NPU_DMA_STATUS) & 0x2) break;
    }
    if (timeout <= 0) print_str("  DMA write timeout!\n");

    int pass5 = 1;
    for (int i = 0; i < 16; i++) {
        if (dma_dst5[i] != 0xAA00 + i) {
            print_str("  FAIL at "); print_dec(i);
            print_str(": got "); print_hex(dma_dst5[i], 8);
            print_str(" expected "); print_hex(0xAA00 + i, 8);
            print_chr('\n');
            pass5 = 0;
            break;
        }
    }
    print_str(pass5 ? "  PASS\n" : "  FAIL\n");

    // ---- Test 6: DMA 大 burst (16 beats, 256B, Act SRAM) ----
    print_str("[Test6] DMA large burst Act SRAM (16 beats, 256B)\n");
    // DDR src at 0x40007000, dst at 0x40008000
    volatile int32_t *dma_src6 = (volatile int32_t *)0x40007000;
    volatile int32_t *dma_dst6 = (volatile int32_t *)0x40008000;
    // 16 beats × 128-bit = 16 beats × 4 × 32-bit = 64 个 int32
    for (int i = 0; i < 64; i++) {
        dma_src6[i] = 0x6000 + i;
        dma_dst6[i] = 0xDEAD;
    }

    npu_wr(NPU_DMA_SRAM_SEL, 0);         // Act SRAM
    npu_wr(NPU_DMA_PATH_CTL, 0x2);       // 读源 = Act SRAM
    npu_wr(NPU_DMA_RD_DDR_ADDR, 0x40007000);
    npu_wr(NPU_DMA_RD_LEN, 15);          // arlen=15 → 16 beats
    npu_wr(NPU_DMA_RD_SRAM_BASE, 0);
    npu_wr(NPU_DMA_RD_TRIG, 1);

    timeout = DMA_RD_TIMEOUT;
    while (timeout-- > 0) {
        if (npu_rd(NPU_DMA_STATUS) & 0x1) break;
    }
    if (timeout <= 0) print_str("  DMA read timeout!\n");

    npu_wr(NPU_DMA_WR_DDR_ADDR, 0x40008000);
    npu_wr(NPU_DMA_WR_LEN, 15);
    npu_wr(NPU_DMA_WR_SRAM_BASE, 0);
    npu_wr(NPU_DMA_WR_TRIG, 1);

    timeout = DMA_WR_TIMEOUT;
    while (timeout-- > 0) {
        if (npu_rd(NPU_DMA_STATUS) & 0x2) break;
    }
    if (timeout <= 0) print_str("  DMA write timeout!\n");

    int pass6 = 1;
    for (int i = 0; i < 64; i++) {
        if (dma_dst6[i] != 0x6000 + i) {
            print_str("  FAIL at "); print_dec(i);
            print_str(": got "); print_hex(dma_dst6[i], 8);
            print_str(" expected "); print_hex(0x6000 + i, 8);
            print_chr('\n');
            pass6 = 0;
            break;
        }
    }
    print_str(pass6 ? "  PASS\n" : "  FAIL\n");

    // ---- Test 7: DMA Wgt SRAM 往返 (DDR → Wgt SRAM → DDR) ----
    print_str("[Test7] DMA Wgt SRAM round-trip (8 beats)\n");
    volatile int32_t *dma_src7 = (volatile int32_t *)0x40009000;
    volatile int32_t *dma_dst7 = (volatile int32_t *)0x4000A000;
    // 8 beats × 4 int32 = 32 个 int32
    for (int i = 0; i < 32; i++) {
        dma_src7[i] = 0x7000 + i;
        dma_dst7[i] = 0xDEAD;
    }

    npu_wr(NPU_DMA_SRAM_SEL, 1);         // DMA 写目标 = Wgt SRAM
    npu_wr(NPU_DMA_PATH_CTL, 0x4);       // DMA 读源 = Wgt SRAM (bit[2:1]=10)
    npu_wr(NPU_DMA_RD_DDR_ADDR, 0x40009000);
    npu_wr(NPU_DMA_RD_LEN, 7);           // arlen=7 → 8 beats
    npu_wr(NPU_DMA_RD_SRAM_BASE, 0);
    npu_wr(NPU_DMA_RD_TRIG, 1);

    timeout = DMA_RD_TIMEOUT;
    while (timeout-- > 0) {
        if (npu_rd(NPU_DMA_STATUS) & 0x1) break;
    }
    if (timeout <= 0) print_str("  DMA read timeout!\n");

    npu_wr(NPU_DMA_WR_DDR_ADDR, 0x4000A000);
    npu_wr(NPU_DMA_WR_LEN, 7);
    npu_wr(NPU_DMA_WR_SRAM_BASE, 0);
    npu_wr(NPU_DMA_WR_TRIG, 1);

    timeout = DMA_WR_TIMEOUT;
    while (timeout-- > 0) {
        if (npu_rd(NPU_DMA_STATUS) & 0x2) break;
    }
    if (timeout <= 0) print_str("  DMA write timeout!\n");

    int pass7 = 1;
    for (int i = 0; i < 32; i++) {
        if (dma_dst7[i] != 0x7000 + i) {
            print_str("  FAIL at "); print_dec(i);
            print_str(": got "); print_hex(dma_dst7[i], 8);
            print_str(" expected "); print_hex(0x7000 + i, 8);
            print_chr('\n');
            pass7 = 0;
            break;
        }
    }
    print_str(pass7 ? "  PASS\n" : "  FAIL\n");

    // ---- Test 8: DMA back-to-back 两次读 (Act SRAM) ----
    print_str("[Test8] DMA back-to-back reads (Act SRAM)\n");
    volatile int32_t *dma_src8a = (volatile int32_t *)0x4000B000;
    volatile int32_t *dma_src8b = (volatile int32_t *)0x4000B100;
    volatile int32_t *dma_dst8  = (volatile int32_t *)0x4000C000;
    for (int i = 0; i < 16; i++) {
        dma_src8a[i] = 0x8A00 + i;
        dma_src8b[i] = 0x8B00 + i;
        dma_dst8[i]  = 0xDEAD;
        dma_dst8[i + 16] = 0xDEAD;
    }

    // 第一次读: DDR(0x4000B000) → Act SRAM(addr 0), 4 beats
    npu_wr(NPU_DMA_SRAM_SEL, 0);
    npu_wr(NPU_DMA_PATH_CTL, 0x2);
    npu_wr(NPU_DMA_RD_DDR_ADDR, 0x4000B000);
    npu_wr(NPU_DMA_RD_LEN, 3);
    npu_wr(NPU_DMA_RD_SRAM_BASE, 0);
    npu_wr(NPU_DMA_RD_TRIG, 1);

    timeout = DMA_RD_TIMEOUT;
    while (timeout-- > 0) {
        if (npu_rd(NPU_DMA_STATUS) & 0x1) break;
    }
    if (timeout <= 0) print_str("  DMA read1 timeout!\n");

    // 第二次读: DDR(0x4000B100) → Act SRAM(addr 4), 4 beats, 紧接着
    npu_wr(NPU_DMA_RD_DDR_ADDR, 0x4000B100);
    npu_wr(NPU_DMA_RD_LEN, 3);
    npu_wr(NPU_DMA_RD_SRAM_BASE, 4);
    npu_wr(NPU_DMA_RD_TRIG, 1);

    timeout = DMA_RD_TIMEOUT;
    while (timeout-- > 0) {
        if (npu_rd(NPU_DMA_STATUS) & 0x1) break;
    }
    if (timeout <= 0) print_str("  DMA read2 timeout!\n");

    // 写回: Act SRAM(addr 0) → DDR(0x4000C000), 8 beats
    npu_wr(NPU_DMA_WR_DDR_ADDR, 0x4000C000);
    npu_wr(NPU_DMA_WR_LEN, 7);           // 8 beats (两次读共 8 个 SRAM 条目)
    npu_wr(NPU_DMA_WR_SRAM_BASE, 0);
    npu_wr(NPU_DMA_WR_TRIG, 1);

    timeout = DMA_WR_TIMEOUT;
    while (timeout-- > 0) {
        if (npu_rd(NPU_DMA_STATUS) & 0x2) break;
    }
    if (timeout <= 0) print_str("  DMA write timeout!\n");

    int pass8 = 1;
    // 前 4 个来自 src8a
    for (int i = 0; i < 16; i++) {
        if (dma_dst8[i] != 0x8A00 + i) {
            print_str("  FAIL src8a at "); print_dec(i);
            print_str(": got "); print_hex(dma_dst8[i], 8);
            print_str(" expected "); print_hex(0x8A00 + i, 8);
            print_chr('\n');
            pass8 = 0;
            break;
        }
    }
    // 后 4 个来自 src8b
    if (pass8) {
        for (int i = 0; i < 16; i++) {
            if (dma_dst8[i + 16] != 0x8B00 + i) {
                print_str("  FAIL src8b at "); print_dec(i);
                print_str(": got "); print_hex(dma_dst8[i + 16], 8);
                print_str(" expected "); print_hex(0x8B00 + i, 8);
                print_chr('\n');
                pass8 = 0;
                break;
            }
        }
    }
    print_str(pass8 ? "  PASS\n" : "  FAIL\n");

    // ---- Test 9: DMA 非零 SRAM 基地址 (base=100) ----
    print_str("[Test9] DMA non-zero SRAM base (base=100)\n");
    volatile int32_t *dma_src9 = (volatile int32_t *)0x4000D000;
    volatile int32_t *dma_dst9 = (volatile int32_t *)0x4000E000;
    for (int i = 0; i < 16; i++) {
        dma_src9[i] = 0x9000 + i;
        dma_dst9[i] = 0xDEAD;
    }

    npu_wr(NPU_DMA_SRAM_SEL, 0);
    npu_wr(NPU_DMA_PATH_CTL, 0x2);
    npu_wr(NPU_DMA_RD_DDR_ADDR, 0x4000D000);
    npu_wr(NPU_DMA_RD_LEN, 3);
    npu_wr(NPU_DMA_RD_SRAM_BASE, 100);   // SRAM base = 100
    npu_wr(NPU_DMA_RD_TRIG, 1);

    timeout = DMA_RD_TIMEOUT;
    while (timeout-- > 0) {
        if (npu_rd(NPU_DMA_STATUS) & 0x1) break;
    }
    if (timeout <= 0) print_str("  DMA read timeout!\n");

    npu_wr(NPU_DMA_WR_DDR_ADDR, 0x4000E000);
    npu_wr(NPU_DMA_WR_LEN, 3);
    npu_wr(NPU_DMA_WR_SRAM_BASE, 100);   // 从同一位置读回
    npu_wr(NPU_DMA_WR_TRIG, 1);

    timeout = DMA_WR_TIMEOUT;
    while (timeout-- > 0) {
        if (npu_rd(NPU_DMA_STATUS) & 0x2) break;
    }
    if (timeout <= 0) print_str("  DMA write timeout!\n");

    int pass9 = 1;
    for (int i = 0; i < 16; i++) {
        if (dma_dst9[i] != 0x9000 + i) {
            print_str("  FAIL at "); print_dec(i);
            print_str(": got "); print_hex(dma_dst9[i], 8);
            print_str(" expected "); print_hex(0x9000 + i, 8);
            print_chr('\n');
            pass9 = 0;
            break;
        }
    }
    print_str(pass9 ? "  PASS\n" : "  FAIL\n");

    // ---- Test 10: DMA 特殊数据模式 (全0, 全F, 交替) ----
    print_str("[Test10] DMA special patterns (0x00, 0xFF, alternating)\n");
    volatile int32_t *dma_src10 = (volatile int32_t *)0x4000F000;
    volatile int32_t *dma_dst10 = (volatile int32_t *)0x40010000;
    // 4 beats × 4 int32 = 16 values
    // Pattern: 0x00000000, 0xFFFFFFFF, 0x55555555, 0xAAAAAAAA, repeat
    for (int i = 0; i < 16; i++) {
        switch (i & 3) {
            case 0: dma_src10[i] = 0x00000000; break;
            case 1: dma_src10[i] = 0xFFFFFFFF; break;
            case 2: dma_src10[i] = 0x55555555; break;
            case 3: dma_src10[i] = 0xAAAAAAAA; break;
        }
        dma_dst10[i] = 0xDEAD;
    }

    npu_wr(NPU_DMA_SRAM_SEL, 0);
    npu_wr(NPU_DMA_PATH_CTL, 0x2);
    npu_wr(NPU_DMA_RD_DDR_ADDR, 0x4000F000);
    npu_wr(NPU_DMA_RD_LEN, 3);
    npu_wr(NPU_DMA_RD_SRAM_BASE, 0);
    npu_wr(NPU_DMA_RD_TRIG, 1);

    timeout = DMA_RD_TIMEOUT;
    while (timeout-- > 0) {
        if (npu_rd(NPU_DMA_STATUS) & 0x1) break;
    }
    if (timeout <= 0) print_str("  DMA read timeout!\n");

    npu_wr(NPU_DMA_WR_DDR_ADDR, 0x40010000);
    npu_wr(NPU_DMA_WR_LEN, 3);
    npu_wr(NPU_DMA_WR_SRAM_BASE, 0);
    npu_wr(NPU_DMA_WR_TRIG, 1);

    timeout = DMA_WR_TIMEOUT;
    while (timeout-- > 0) {
        if (npu_rd(NPU_DMA_STATUS) & 0x2) break;
    }
    if (timeout <= 0) print_str("  DMA write timeout!\n");

    int pass10 = 1;
    for (int i = 0; i < 16; i++) {
        int32_t expected;
        switch (i & 3) {
            case 0: expected = 0x00000000; break;
            case 1: expected = (int32_t)0xFFFFFFFF; break;
            case 2: expected = (int32_t)0x55555555; break;
            default: expected = (int32_t)0xAAAAAAAA; break;
        }
        if (dma_dst10[i] != expected) {
            print_str("  FAIL at "); print_dec(i);
            print_str(": got "); print_hex(dma_dst10[i], 8);
            print_str(" expected "); print_hex(expected, 8);
            print_chr('\n');
            pass10 = 0;
            break;
        }
    }
    print_str(pass10 ? "  PASS\n" : "  FAIL\n");

    // ================================================================
    // Test 11: NPU Simple Convolution
    //   Input:  4×4, 16ch, all 1s
    //   Weight: 3×3, 16 IC, 16 OC, all 1s
    //   Config: stride=1, bias=0, scale=1, shift=1, no ReLU, no pool
    //   Output: 2×2, 16ch, expected all 72 (psum=144, quant=144>>1=72)
    // ================================================================
    print_str("[Test11] NPU simple convolution (4x4x16, 3x3, s1, 16OC)\n");

    volatile int8_t  *act_ddr = (volatile int8_t  *)0x40011000;
    volatile int8_t  *wgt_ddr = (volatile int8_t  *)0x40011100;
    volatile int32_t *out_ddr = (volatile int32_t *)0x40012000;

    // Fill DDR: act all 1s (16 SRAM words × 16 bytes = 256 bytes)
    for (int i = 0; i < 256; i++) act_ddr[i] = 1;
    // Fill DDR: wgt all 1s (9 SRAM words × 16 bytes = 144 bytes)
    for (int i = 0; i < 144; i++) wgt_ddr[i] = 1;
    // Clear DDR output region (4 SRAM words × 4 sub-beats = 16 int32)
    for (int i = 0; i < 16; i++) out_ddr[i] = 0xDEADDEAD;

    // ---- DMA: DDR → Act SRAM (16 words, base=0) ----
    npu_wr(NPU_DMA_SRAM_SEL, 0);         // write target = Act SRAM
    npu_wr(NPU_DMA_PATH_CTL, 0x2);       // read source = Act SRAM
    npu_wr(NPU_DMA_RD_DDR_ADDR, 0x40011000);
    npu_wr(NPU_DMA_RD_LEN, 15);          // 16 beats
    npu_wr(NPU_DMA_RD_SRAM_BASE, 0);
    npu_wr(NPU_DMA_RD_TRIG, 1);

    timeout = DMA_RD_TIMEOUT;
    while (timeout-- > 0) {
        if (npu_rd(NPU_DMA_STATUS) & 0x1) break;
    }
    if (timeout <= 0) print_str("  DMA act read timeout!\n");

    // ---- DMA: DDR → Wgt SRAM (9 words, base=16) ----
    npu_wr(NPU_DMA_SRAM_SEL, 1);         // write target = Wgt SRAM
    npu_wr(NPU_DMA_PATH_CTL, 0x4);       // read source = Wgt SRAM (bit[2:1]=10)
    npu_wr(NPU_DMA_RD_DDR_ADDR, 0x40011100);
    npu_wr(NPU_DMA_RD_LEN, 8);           // 9 beats
    npu_wr(NPU_DMA_RD_SRAM_BASE, 16);
    npu_wr(NPU_DMA_RD_TRIG, 1);

    timeout = DMA_RD_TIMEOUT;
    while (timeout-- > 0) {
        if (npu_rd(NPU_DMA_STATUS) & 0x1) break;
    }
    if (timeout <= 0) print_str("  DMA wgt read timeout!\n");

    // ---- Configure NPU registers ----
    npu_wr(NPU_ACT_ADDR_A, 0);           // Act SRAM base = 0
    npu_wr(NPU_WGT_ADDR_A, 16);          // Wgt SRAM base = 16
    npu_wr(NPU_OUT_ADDR_A, 0);           // Out SRAM base = 0
    npu_wr(NPU_IN_W, 4);
    npu_wr(NPU_IN_H, 4);
    npu_wr(NPU_IC, 16);
    npu_wr(NPU_OC, 16);
    npu_wr(NPU_KERNEL, (3 << 8) | 3);    // KH=3, KW=3
    npu_wr(NPU_STRIDE, (1 << 8) | 1);    // SX=1, SY=1

    // Per-channel bias=0, scale_mul=1, scale_shift=1
    for (int ch = 0; ch < 16; ch++) {
        npu_wr(NPU_BIAS(ch),  0);
        npu_wr(NPU_SCALE(ch), 1);
        npu_wr(NPU_SHIFT(ch), 1);
    }

    // ---- Start NPU ----
    npu_wr(NPU_CTRL, NPU_CTRL_START);    // start=1, relu_en=0, pool_en=0

    // ---- Wait for NPU done ----
    timeout = 100000;
    while (timeout-- > 0) {
        if (npu_rd(NPU_STATUS) & 0x1) break;
    }
    if (timeout <= 0) print_str("  NPU timeout!\n");
    if (npu_rd(NPU_STATUS) & 0x4) print_str("  DMA rd_err!\n");
    if (npu_rd(NPU_STATUS) & 0x8) print_str("  DMA wr_err!\n");

    // ---- DMA: Out SRAM → DDR (4 words, base=0) ----
    npu_wr(NPU_DMA_SRAM_SEL, 0);
    npu_wr(NPU_DMA_PATH_CTL, 0x1);       // dma_out_rd_sel=1, dma_rd_sram_sel=0 (Out)
    npu_wr(NPU_DMA_WR_DDR_ADDR, 0x40012000);
    npu_wr(NPU_DMA_WR_LEN, 3);           // 4 beats
    npu_wr(NPU_DMA_WR_SRAM_BASE, 0);
    npu_wr(NPU_DMA_WR_TRIG, 1);

    timeout = DMA_WR_TIMEOUT;
    while (timeout-- > 0) {
        if (npu_rd(NPU_DMA_STATUS) & 0x2) break;
    }
    if (timeout <= 0) print_str("  DMA out write timeout!\n");

    // ---- Verify output: all 4 positions × 16 channels = 72 ----
    // Each 128-bit SRAM word → 4 × 32-bit DDR words.
    // Byte layout: ch0=byte0, ch1=byte1, ..., ch15=byte15.
    // All channels = 72 → each 32-bit word = 0x48484848.
    int pass11 = 1;
    for (int i = 0; i < 16; i++) {
        if (out_ddr[i] != (int32_t)0x48484848) {
            print_str("  FAIL at "); print_dec(i);
            print_str(": got "); print_hex(out_ddr[i], 8);
            print_str(" expected 48484848\n");
            pass11 = 0;
            break;
        }
    }
    print_str(pass11 ? "  PASS\n" : "  FAIL\n");

    // ---- 汇总 ----
    int total_pass = pass1 + pass2 + pass3 + pass4 +
                     pass5 + pass6 + pass7 + pass8 + pass9 + pass10 + pass11;
    print_str("\n=== Result: "); print_dec(total_pass);
    print_str("/11 passed ===\n");

    if (total_pass == 11)
        print_str("ALL TESTS PASSED.\n");
    else
        print_str("SOME TESTS FAILED.\n");
}
