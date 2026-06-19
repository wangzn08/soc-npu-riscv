#!/usr/bin/env bash
# Compile + run one standalone directed testbench in an isolated work lib.
# Usage: bash tests/run_tb.sh <work_suffix> <top_module> <rtl_and_tb_files...>
set -e
cd "$(dirname "$0")/.."
export PATH="/e/modelsim/win64:$PATH"
export MGC_LICENSE_FILE='E:/modelsim/LICENSE.TXT'
export LM_LICENSE_FILE="$MGC_LICENSE_FILE"
export TMP="$PWD/.tmp" TEMP="$PWD/.tmp"
mkdir -p .tmp
SUFFIX="$1"; TOP="$2"; shift 2
LIB="sim/work_$SUFFIX"
rm -rf "$LIB"
vlib "$LIB" >/dev/null 2>&1
vlog -sv -timescale 1ns/1ps -work "$LIB" "$@" 2>&1 | tail -6
vsim -c -lib "$LIB" "$TOP" -do "run -all; quit -f" 2>&1 | \
  grep -vE "^# //|^# \*\*|Mentor|Copyright|secrets|Section|disclosure|prohibited|exempt|ModelSim|Loading|^# $"
