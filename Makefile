# ============================================================
# SoC 仿真 Makefile
# PicoRV32 + NPU + AXI Shared Memory
#
# 用法:
#   make fw       — 编译固件
#   make compile  — ModelSim 编译
#   make sim      — 编译 + 仿真
#   make waves    — 编译 + 仿真 + 打开波形
#   make clean    — 清理仿真产物
#   make distclean — 清理所有产物（含固件）
# ============================================================

# ---- 目录 ----
ROOT_DIR    := $(shell pwd)
FW_DIR      := $(ROOT_DIR)/firmware
BUILD_DIR   := $(FW_DIR)/build
SIM_DIR     := $(ROOT_DIR)/sim

# ---- 工具链 ----
RISCV_PREFIX ?= E:/Riscv_Tools/xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-
PYTHON       ?= python
ISA          ?= rv32imc

# ---- ModelSim ----
MODELSIM_HOME ?= /e/modelsim/win64
MGC_LICENSE_FILE ?= E:/modelsim/LICENSE.TXT
export MGC_LICENSE_FILE

VLIB  = vlib $(SIM_DIR)/work
VLOG  = vlog -sv -timescale 1ns/1ps -work $(SIM_DIR)/work
VSIM  = vsim -lib $(SIM_DIR)/work/axi_sys_tb

FILELIST = $(ROOT_DIR)/axi_sys.f

# ============================================================
# 固件编译
# ============================================================
FW_C_SRCS   = $(FW_DIR)/irq.c $(FW_DIR)/print.c $(FW_DIR)/libgcc_stub.c $(FW_DIR)/deepnet_deploy.c
FW_S_SRCS   = $(FW_DIR)/start7.S
FW_C_OBJS   = $(patsubst $(FW_DIR)/%.c,$(BUILD_DIR)/%_c.o,$(FW_C_SRCS))
FW_S_OBJS   = $(patsubst $(FW_DIR)/%.S,$(BUILD_DIR)/%_s.o,$(FW_S_SRCS))
FW_OBJS     = $(FW_S_OBJS) $(FW_C_OBJS)
FW_HEX      = $(BUILD_DIR)/firmware7.hex

CFLAGS  = -mabi=ilp32 -march=$(ISA) -O2 --std=c99 \
          -Werror -Wall -Wextra -Wshadow -Wundef \
          -Wpointer-arith -Wcast-qual -Wcast-align \
          -Wwrite-strings -Wredundant-decls \
          -Wstrict-prototypes -Wmissing-prototypes \
          -pedantic -ffreestanding -nostdlib

ASFLAGS = -mabi=ilp32 -march=$(ISA)

LDFLAGS = -Os -mabi=ilp32 -march=$(ISA) -ffreestanding -nostdlib \
          -Wl,--build-id=none,-Bstatic,-T,$(FW_DIR)/sections.lds \
          -Wl,-Map=$(BUILD_DIR)/firmware7.map,--strip-debug

$(BUILD_DIR)/%_c.o: $(FW_DIR)/%.c | $(BUILD_DIR)
	$(RISCV_PREFIX)gcc -c $(CFLAGS) -o $@ $<

$(BUILD_DIR)/%_s.o: $(FW_DIR)/%.S | $(BUILD_DIR)
	$(RISCV_PREFIX)gcc -c $(ASFLAGS) -o $@ $<

$(BUILD_DIR)/firmware7.elf: $(FW_OBJS) $(FW_DIR)/sections.lds
	$(RISCV_PREFIX)gcc $(LDFLAGS) -o $@ $(FW_OBJS)
	chmod -x $@

$(BUILD_DIR)/firmware7.bin: $(BUILD_DIR)/firmware7.elf
	$(RISCV_PREFIX)objcopy -O binary $< $@
	chmod -x $@

$(FW_HEX): $(BUILD_DIR)/firmware7.bin $(FW_DIR)/makehex.py
	$(PYTHON) $(FW_DIR)/makehex.py $< 524288 > $@

$(BUILD_DIR):
	mkdir -p $@

fw: $(FW_HEX)

# ============================================================
# ModelSim 编译
# ============================================================
WORK_DIR = $(SIM_DIR)/work

$(WORK_DIR): $(FW_HEX) $(FILELIST) | $(SIM_DIR)
	$(VLIB)
	$(VLOG) -f $(FILELIST)
	@echo "=== ModelSim 编译完成 ==="

compile: $(WORK_DIR)

# ============================================================
# 仿真
# ============================================================
sim: $(WORK_DIR)
	vsim -c -lib $(WORK_DIR) axi_sys_tb \
		-do "run -all; quit -f"
	@echo "=== 仿真完成 ==="

# ============================================================
# 波形查看
# ============================================================
waves: $(WORK_DIR)
	vsim -lib $(WORK_DIR) axi_sys_tb \
		-do "vcd file axi_sys_tb.vcd; vcd add -r /*; run -all"
	@echo "=== 波形文件已生成: $(SIM_DIR)/axi_sys_tb.vcd ==="

# ============================================================
# 清理
# ============================================================
clean:
	rm -rf $(SIM_DIR)/work $(SIM_DIR)/transcript \
	       $(SIM_DIR)/vsim.wlf $(SIM_DIR)/axi_sys_tb.vcd \
	       $(SIM_DIR)/modelsim.ini $(SIM_DIR)/*.wlf \
	       $(SIM_DIR)/library.info

distclean: clean
	rm -rf $(BUILD_DIR)

.PHONY: fw compile sim waves clean distclean
