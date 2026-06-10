#!/bin/bash
# Helper: build firmware (manual, bypassing make's TMP-mangling issue), compile RTL, run sim.
#
# Usage:
#   bash run_sim.sh                 # build deepnet_deploy + run
#   bash run_sim.sh deepnet_run     # build the infra/Test11 baseline + run
#   bash run_sim.sh deepnet_deploy rtl    # also force-recompile RTL
#   bash run_sim.sh deepnet_run rtl DEBUG # recompile RTL with +define+DEBUG (verbose traces)
set -e
export PATH="/c/msys64/usr/bin:/e/Riscv_Tools/xpack-riscv-none-elf-gcc-15.2.0-1/bin:/e/modelsim/win64:$PATH"
export TMP='E:\code\6-10\soc\.tmp' TEMP='E:\code\6-10\soc\.tmp'
export MGC_LICENSE_FILE='E:/modelsim/LICENSE.TXT'
mkdir -p /e/code/6-10/soc/.tmp
cd /e/code/6-10/soc

APP="${1:-deepnet_deploy}"        # firmware C entry source (defines usercode7)
RTL="$2"                          # "rtl" to recompile RTL
DBG="$3"                          # "DEBUG" to add +define+DEBUG

PFX=riscv-none-elf-
CF="-mabi=ilp32 -march=rv32imc -O2 --std=c99 -ffreestanding -nostdlib"
AF="-mabi=ilp32 -march=rv32imc"
LD="-Os -mabi=ilp32 -march=rv32imc -ffreestanding -nostdlib -Wl,--build-id=none,-Bstatic,-T,firmware/sections.lds -Wl,-Map=firmware/build/firmware7.map,--strip-debug"

mkdir -p firmware/build
echo "=== Compiling firmware ($APP) ==="
${PFX}gcc -c $AF -o firmware/build/start7_s.o firmware/start7.S
for f in irq print libgcc_stub "$APP"; do
  ${PFX}gcc -c $CF -o firmware/build/${f}_c.o firmware/${f}.c
done
${PFX}gcc $LD -o firmware/build/firmware7.elf \
  firmware/build/start7_s.o firmware/build/irq_c.o firmware/build/print_c.o \
  firmware/build/libgcc_stub_c.o firmware/build/${APP}_c.o 2>&1 | grep -v "RWX permissions" || true
${PFX}objcopy -O binary firmware/build/firmware7.elf firmware/build/firmware7.bin
python firmware/makehex.py firmware/build/firmware7.bin 524288 > firmware/build/firmware7.hex

DEFINE=""
[ "$DBG" == "DEBUG" ] && DEFINE="+define+DEBUG"
if [ "$RTL" == "rtl" ] || [ ! -d sim/work ]; then
  echo "=== Compiling RTL $DEFINE ==="
  rm -rf sim/work
  vlib sim/work >/dev/null 2>&1
  vlog -sv $DEFINE -timescale 1ns/1ps -work sim/work -f axi_sys.f 2>&1 | tail -2
fi

echo "=== Running simulation ==="
vsim -c -lib sim/work axi_sys_tb -do "run -all; quit -f" 2>&1 | \
  grep -vE "^# //|^# \*\*|Mentor|Copyright|secrets|Section|disclosure|prohibited|exempt|ModelSim SE|Loading|^# $"
