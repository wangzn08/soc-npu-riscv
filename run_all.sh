#!/usr/bin/env bash
# ============================================================
# SoC 全流程编译仿真脚本
# PicoRV32 + NPU + AXI Shared Memory
#
# 用法:
#   bash run_all.sh          — 编译固件 + 编译RTL + 仿真
#   bash run_all.sh fw       — 仅编译固件
#   bash run_all.sh compile  — 仅编译RTL
#   bash run_all.sh sim      — 编译固件 + RTL + 仿真
#   bash run_all.sh waves    — 编译 + 仿真 + 波形
#   bash run_all.sh clean    — 清理仿真产物
#   bash run_all.sh distclean— 清理所有产物
#
# 可选:
#   FW_USER_C=upsample2x_smoke.c bash run_all.sh sim
#   bash run_all.sh sim upsample2x_smoke.c
#       — 临时链接一个 CPU/NPU 协同 smoke 固件，默认仍为 deepnet_deploy.c
#   bash run_all.sh sim yolo_concat_smoke.c yolo_ops.c
#       — 用户入口后面的额外 C 文件会追加进固件链接
# ============================================================

set -e

# ---- 目录 ----
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
FW_DIR="$ROOT_DIR/firmware"
BUILD_DIR="$FW_DIR/build"
SIM_DIR="$ROOT_DIR/sim"

# ---- 工具链 ----
RISCV_PREFIX="${RISCV_PREFIX:-E:/Riscv_Tools/xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-}"
PYTHON="${PYTHON:-python}"
ISA="${ISA:-rv32imc}"

# ---- ModelSim ----
MODELSIM_HOME="${MODELSIM_HOME:-/e/modelsim/win64}"
MGC_LICENSE_FILE="${MGC_LICENSE_FILE:-E:/modelsim/LICENSE.TXT}"
export MGC_LICENSE_FILE
LM_LICENSE_FILE="${LM_LICENSE_FILE:-$MGC_LICENSE_FILE}"
export LM_LICENSE_FILE

# ---- 临时目录 (解决 Windows 权限问题) ----
export TEMP="$ROOT_DIR/tmp"
export TMP="$ROOT_DIR/tmp"
export TMPDIR="$ROOT_DIR/tmp"
mkdir -p "$TEMP"

# ---- 固件源文件 ----
# Optional second argument overrides the C file that defines usercode7().
FW_USER_C="${2:-${FW_USER_C:-deepnet_deploy.c}}"
FW_C_SRCS=(
    "$FW_DIR/irq.c"
    "$FW_DIR/print.c"
    "$FW_DIR/libgcc_stub.c"
    "$FW_DIR/npu_desc.c"
    "$FW_DIR/$FW_USER_C"
)
FW_EXTRA_C_SRCS=()
if [ "$#" -gt 2 ]; then
    for extra_src in "${@:3}"; do
        FW_EXTRA_C_SRCS+=("$extra_src")
        extra_path="$FW_DIR/$extra_src"
        already_added=0
        for src in "${FW_C_SRCS[@]}"; do
            if [ "$src" = "$extra_path" ]; then
                already_added=1
                break
            fi
        done
        if [ "$already_added" -eq 0 ]; then
            FW_C_SRCS+=("$extra_path")
        fi
    done
fi
FW_S_SRCS=(
    "$FW_DIR/start7.S"
)
FW_HEX="$BUILD_DIR/firmware7.hex"
FW_LINKSCRIPT="$FW_DIR/sections.lds"

# ---- ModelSim 命令 ----
FILELIST="$ROOT_DIR/axi_sys.f"

# ============================================================
# 颜色输出
# ============================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ============================================================
# 固件编译
# ============================================================
compile_fw() {
    info "========== 编译固件 =========="
    info "用户固件入口: $FW_USER_C"
    if [ "${#FW_EXTRA_C_SRCS[@]}" -gt 0 ]; then
        info "附加固件源: ${FW_EXTRA_C_SRCS[*]}"
    fi
    mkdir -p "$BUILD_DIR"

    # CFLAGS
    CFLAGS=(
        -mabi=ilp32 -march="$ISA" -O2 --std=c99
        -Werror -Wall -Wextra -Wshadow -Wundef
        -Wpointer-arith -Wcast-qual -Wcast-align
        -Wwrite-strings -Wredundant-decls
        -Wstrict-prototypes -Wmissing-prototypes
        -pedantic -ffreestanding -nostdlib
    )
    # 额外编译开关注入口 (例如: EXTRA_CFLAGS=-DNPU_OC_OVERLAP=0)
    if [ -n "${EXTRA_CFLAGS:-}" ]; then
        # shellcheck disable=SC2206
        CFLAGS+=(${EXTRA_CFLAGS})
    fi
    # DESC_RECORD builds the generator firmware (records into resident DDR slots).
    # Replay (descriptor image preloaded) is the default firmware path.
    if [ -n "${DESC_RECORD:-}" ]; then
        CFLAGS+=(-DDESC_RECORD)
    fi
    ASFLAGS=(-mabi=ilp32 -march="$ISA")
    # Windows 原生 GCC 链接器需要 Windows 路径格式
    FW_LINKSCRIPT_WIN=$(cygpath -w "$FW_LINKSCRIPT" 2>/dev/null || echo "$FW_LINKSCRIPT")
    MAPFILE_WIN=$(cygpath -w "$BUILD_DIR/firmware7.map" 2>/dev/null || echo "$BUILD_DIR/firmware7.map")
    LDFLAGS=(
        -Os -mabi=ilp32 -march="$ISA" -ffreestanding -nostdlib
        -Wl,--build-id=none,-Bstatic,-T,"$FW_LINKSCRIPT_WIN"
        -Wl,-Map="$MAPFILE_WIN",--strip-debug
    )

    # 编译 .S 文件
    for src in "${FW_S_SRCS[@]}"; do
        base=$(basename "$src" .S)
        obj="$BUILD_DIR/${base}_s.o"
        info "编译汇编: $src"
        "${RISCV_PREFIX}gcc" -c "${ASFLAGS[@]}" -o "$obj" "$src"
    done

    # 编译 .c 文件
    for src in "${FW_C_SRCS[@]}"; do
        base=$(basename "$src" .c)
        obj="$BUILD_DIR/${base}_c.o"
        info "编译C: $src"
        "${RISCV_PREFIX}gcc" -c "${CFLAGS[@]}" -o "$obj" "$src"
    done

    # 链接
    OBJS=()
    for src in "${FW_S_SRCS[@]}"; do
        base=$(basename "$src" .S)
        OBJS+=("$BUILD_DIR/${base}_s.o")
    done
    for src in "${FW_C_SRCS[@]}"; do
        base=$(basename "$src" .c)
        OBJS+=("$BUILD_DIR/${base}_c.o")
    done

    ELF="$BUILD_DIR/firmware7.elf"
    BIN="$BUILD_DIR/firmware7.bin"

    info "链接: $ELF"
    # -lgcc (after objects) provides soft-float routines (__addsf3 etc.) for the
    # YOLO CPU decode; only referenced symbols are pulled in, so float-free builds
    # (MNIST) are unaffected. -nostdlib otherwise omits libgcc.
    "${RISCV_PREFIX}gcc" "${LDFLAGS[@]}" -o "$ELF" "${OBJS[@]}" -lgcc
    chmod -x "$ELF"

    info "生成BIN: $BIN"
    "${RISCV_PREFIX}objcopy" -O binary "$ELF" "$BIN"
    chmod -x "$BIN"

    info "生成HEX: $FW_HEX"
    "$PYTHON" "$FW_DIR/makehex.py" "$BIN" 524288 > "$FW_HEX"

    ok "固件编译完成: $FW_HEX"
}

# ============================================================
# ModelSim 编译
# ============================================================
compile_rtl() {
    info "========== 编译 RTL =========="
    mkdir -p "$SIM_DIR"

    # 创建 work library
    if [ -d "$SIM_DIR/work" ]; then
        info "work 库已存在，跳过 vlib"
    else
        info "创建 work 库"
        vlib "$SIM_DIR/work"
    fi

    info "编译 RTL 文件列表: $FILELIST"
    # YOLO_DDR=1 preloads the 320x320 image into DDR (firmware/yolo_img_ddr.hex)
    # via +define+YOLO_DDR; default off keeps MNIST builds unchanged.
    VLOG_DEFS=""
    if [ -n "${YOLO_DDR:-}" ] || [ -f "$ROOT_DIR/.yolo_ddr" ]; then
        VLOG_DEFS="+define+YOLO_DDR"
        info "YOLO_DDR 预载启用 (+define+YOLO_DDR)"
    fi
    # Pre-compiled descriptor image: DESC_RECORD dumps the image, DESC_REPLAY loads it.
    if [ -n "${DESC_RECORD:-}" ]; then
        VLOG_DEFS="$VLOG_DEFS +define+DESC_RECORD"
        info "描述符录制启用 (+define+DESC_RECORD)"
    fi
    if [ -n "${DESC_REPLAY:-}" ]; then
        VLOG_DEFS="$VLOG_DEFS +define+DESC_REPLAY"
        info "描述符回放启用 (+define+DESC_REPLAY)"
    fi
    vlog -sv -timescale 1ns/1ps $VLOG_DEFS -f "$FILELIST" -work "$SIM_DIR/work"

    ok "RTL 编译完成"
}

# ============================================================
# 仿真
# ============================================================
run_sim() {
    info "========== 运行仿真 =========="

    # 确保 firmware hex 存在
    if [ ! -f "$FW_HEX" ]; then
        warn "固件 HEX 不存在，先编译固件"
        compile_fw
    fi

    # 确保 RTL 编译过
    if [ ! -d "$SIM_DIR/work" ]; then
        warn "RTL 未编译，先编译 RTL"
        compile_rtl
    fi

    info "启动 vsim 仿真 (headless)..."
    vsim -c -lib "$SIM_DIR/work" axi_sys_tb -do "run -all; quit -f"

    ok "仿真完成"
}

# ============================================================
# 仿真 + 波形
# ============================================================
run_waves() {
    info "========== 运行仿真 (波形) =========="

    if [ ! -f "$FW_HEX" ]; then
        compile_fw
    fi
    if [ ! -d "$SIM_DIR/work" ]; then
        compile_rtl
    fi

    info "启动 vsim 仿真 (GUI + VCD)..."
    vsim -lib "$SIM_DIR/work" axi_sys_tb \
        -do "vcd file $SIM_DIR/axi_sys_tb.vcd; vcd add -r /*; run -all"

    ok "波形文件已生成: $SIM_DIR/axi_sys_tb.vcd"
}

# ============================================================
# 清理
# ============================================================
do_clean() {
    info "清理仿真产物..."
    rm -rf "$SIM_DIR/work" "$SIM_DIR/transcript" \
           "$SIM_DIR/vsim.wlf" "$SIM_DIR/axi_sys_tb.vcd" \
           "$SIM_DIR/modelsim.ini" "$SIM_DIR"/*.wlf \
           "$SIM_DIR/library.info"
    ok "仿真产物已清理"
}

do_distclean() {
    do_clean
    info "清理固件产物..."
    rm -rf "$BUILD_DIR"
    rm -rf "$ROOT_DIR/tmp"
    ok "所有产物已清理"
}

# ============================================================
# 主入口
# ============================================================
case "${1:-all}" in
    fw)
        compile_fw
        ;;
    compile)
        compile_rtl
        ;;
    sim)
        compile_fw
        compile_rtl
        run_sim
        ;;
    waves)
        compile_fw
        compile_rtl
        run_waves
        ;;
    clean)
        do_clean
        ;;
    distclean)
        do_distclean
        ;;
    all)
        compile_fw
        compile_rtl
        run_sim
        ;;
    *)
        echo "用法: $0 {fw|compile|sim|waves|clean|distclean|all}"
        exit 1
        ;;
esac
