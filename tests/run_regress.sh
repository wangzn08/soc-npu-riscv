#!/usr/bin/env bash
# Compile RTL (full filelist) + optionally build firmware and run a sim.
# Usage:
#   bash tests/run_regress.sh compile           # RTL syntax check only
#   bash tests/run_regress.sh sim <fw.c> [extra.c ...]  # build fw + run axi_sys_tb
set -e
cd "$(dirname "$0")/.."
export PATH="/e/modelsim/win64:/e/Riscv_Tools/xpack-riscv-none-elf-gcc-15.2.0-1/bin:$PATH"
export MGC_LICENSE_FILE='E:/modelsim/LICENSE.TXT'
export LM_LICENSE_FILE="$MGC_LICENSE_FILE"
export TMP="$PWD/.tmp" TEMP="$PWD/.tmp"
mkdir -p .tmp

MODE="${1:-compile}"

echo "=== Compiling RTL ==="
rm -rf sim/work
vlib sim/work >/dev/null 2>&1
vlog -sv -timescale 1ns/1ps -work sim/work -f axi_sys.f 2>&1 | grep -iE "error|\*\* " | head -30 || true
echo "=== RTL compile finished ==="

if [ "$MODE" = "compile" ]; then exit 0; fi

APP="${2:-deepnet_deploy}"
shift 2 || true
PFX=riscv-none-elf-
CF="-mabi=ilp32 -march=rv32imc -O2 --std=c99 -Werror -Wall -Wextra -Wshadow -Wundef -Wpointer-arith -Wcast-qual -Wcast-align -Wwrite-strings -Wredundant-decls -Wstrict-prototypes -Wmissing-prototypes -pedantic -ffreestanding -nostdlib"
AF="-mabi=ilp32 -march=rv32imc"
LD="-Os -mabi=ilp32 -march=rv32imc -ffreestanding -nostdlib -Wl,--build-id=none,-Bstatic,-T,firmware/sections.lds -Wl,-Map=firmware/build/firmware7.map,--strip-debug"
mkdir -p firmware/build
echo "=== Compiling firmware ($APP) extra: $* ==="
${PFX}gcc -c $AF -o firmware/build/start7_s.o firmware/start7.S
OBJS="firmware/build/start7_s.o"
for f in irq print libgcc_stub "$APP" "$@"; do
  b=$(basename "$f" .c)
  ${PFX}gcc -c $CF -o firmware/build/${b}_c.o firmware/${b}.c
  OBJS="$OBJS firmware/build/${b}_c.o"
done
${PFX}gcc $LD -o firmware/build/firmware7.elf $OBJS 2>&1 | grep -v "RWX permissions" || true
${PFX}objcopy -O binary firmware/build/firmware7.elf firmware/build/firmware7.bin
python firmware/makehex.py firmware/build/firmware7.bin 524288 > firmware/build/firmware7.hex
echo "=== Running sim ==="
vsim -c -lib sim/work axi_sys_tb -do "run -all; quit -f" 2>&1 | \
  grep -vE "^# //|Mentor|Copyright|secrets|Section|disclosure|prohibited|exempt|ModelSim|Loading|^# $"
