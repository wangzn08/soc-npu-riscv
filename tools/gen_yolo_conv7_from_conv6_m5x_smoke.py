#!/usr/bin/env python3
"""Generate fixtures for the M5x conv7 (model.4 cv1) smoke.

conv7 = 64->64, 1x1 pointwise (the cv1 of the second C2f block, model.4). Its
input is conv6's output (80x8x64), baked in as a fixture. Standalone single-layer
smoke exercising IC=64 (4 ic-groups) OC=64 oc_single pointwise on the shared NPU.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
OUT_PATH = ROOT / "firmware" / "yolo_conv7_from_conv6_m5x_data.h"
M5W_GEN = ROOT / "tools" / "gen_yolo_conv6_from_c2f_m5w_smoke.py"

IN_W = 80
IN_H = 8
IC = 64
OC = 64
IC_GROUPS = IC // 16
SP = IN_W * IN_H

# conv7 = yolo_act_quant[7].
IN_SCALE = 0.0334292874
IN_ZP = -120
OUT_SCALE = 0.0397584029
OUT_ZP = -121
Q_SHIFT = 20
REQUANT_SHIFT = 12
RTL_TOL = 8


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fmt_u32_arr(name, words):
    lines = [f"static const uint32_t {name}[{len(words)}][4] = {{"]
    for w in words:
        lines.append("    {" + ", ".join(f"0x{x:08X}u" for x in w) + "},")
    lines.append("};")
    return "\n".join(lines)


def fmt_i32(name, vals):
    return f"static const int32_t {name}[{len(vals)}] = {{\n    " + \
        ", ".join(str(x) for x in vals) + "\n};"


def fmt_u32(name, vals):
    return f"static const uint32_t {name}[{len(vals)}] = {{\n    " + \
        ", ".join(f"{x}u" for x in vals) + "\n};"


def fmt_u8_mat(name, vals):
    lines = [f"static const uint8_t {name}[{vals.shape[0]}][{vals.shape[1]}] = {{"]
    for row in vals:
        lines.append("    {" + ", ".join(f"0x{int(x):02X}u" for x in row) + "},")
    lines.append("};")
    return "\n".join(lines)


def main() -> None:
    m5w = load_module("m5w_gen", M5W_GEN)
    conv0, conv6_out = m5w.compute_conv6()  # [SP, 64] int8 at conv6 out scale
    lut = conv0.load_lut()
    act = conv6_out.reshape(IN_H, IN_W, 64).transpose(2, 0, 1)  # [64, 8, 80]

    w = np.fromfile(WEIGHT_DIR / "conv7_w.bin", dtype=np.int8).reshape(OC, IC, 1, 1)
    b = np.fromfile(WEIGHT_DIR / "conv7_b.bin", dtype=np.float32)
    s = np.fromfile(WEIGHT_DIR / "conv7_s.bin", dtype=np.float32)

    wsum = w.astype(np.int32).sum(axis=(1, 2, 3))
    bias_q, scale_mul = [], []
    for oc in range(OC):
        mul = int(round(IN_SCALE * float(s[oc]) * 16.0 * (1 << Q_SHIFT)))
        if mul == 0:
            raise SystemExit(f"conv7 oc{oc} zero scale mul")
        scale_mul.append(mul)
        bias_equiv = int(round(float(b[oc]) * 16.0 * (1 << Q_SHIFT) / mul))
        bias_q.append(int(bias_equiv - IN_ZP * int(wsum[oc])))
    scale_shift = [Q_SHIFT] * OC
    requant_mul = int(round((1 << REQUANT_SHIFT) / (16.0 * OUT_SCALE)))

    expected = np.zeros((SP, OC), dtype=np.int8)
    for pos in range(SP):
        y, x = pos // IN_W, pos % IN_W
        for oc in range(OC):
            acc = 0
            for ic in range(IC):
                acc += int(act[ic, y, x]) * int(w[oc, ic, 0, 0])
            q44 = ((acc + bias_q[oc]) * scale_mul[oc]) >> Q_SHIFT
            silu = conv0.rtl_silu_byte(q44, lut)
            rq = ((conv0.s8(silu) * requant_mul) >> REQUANT_SHIFT) + OUT_ZP
            expected[pos, oc] = np.int8(conv0.s8(conv0.clamp_s8(rq)))
    expected_u8 = expected.astype(np.uint8)

    # input act words: tile-major addr = group*SP + pos, 16 lanes/group.
    act_words = []
    for g in range(IC_GROUPS):
        for pos in range(SP):
            y, x = pos // IN_W, pos % IN_W
            act_words.append(conv0.pack_i8_word(act[g * 16:(g + 1) * 16, y, x]))

    # weights: per oc, per ic-group (1x1 => KO=1).
    wgt_words = []
    for oc in range(OC):
        for g in range(IC_GROUPS):
            wgt_words.append(conv0.pack_i8_word(w[oc, g * 16:(g + 1) * 16, 0, 0]))

    body = f"""#ifndef YOLO_CONV7_FROM_CONV6_M5X_DATA_H
#define YOLO_CONV7_FROM_CONV6_M5X_DATA_H

#include <stdint.h>

#define YOLO_CONV7_IN_W {IN_W}u
#define YOLO_CONV7_IN_H {IN_H}u
#define YOLO_CONV7_IC {IC}u
#define YOLO_CONV7_OC {OC}u
#define YOLO_CONV7_ACT_WORDS {len(act_words)}u
#define YOLO_CONV7_WGT_WORDS {len(wgt_words)}u
#define YOLO_CONV7_OUT_SPATIAL {SP}u
#define YOLO_CONV7_OUT_WORDS {SP * (OC // 16)}u
#define YOLO_CONV7_REQUANT_MUL {requant_mul}u
#define YOLO_CONV7_REQUANT_SHIFT {REQUANT_SHIFT}u
#define YOLO_CONV7_REQUANT_ZP {OUT_ZP}
#define YOLO_CONV7_RTL_TOL {RTL_TOL}u

{fmt_u32_arr("yolo_conv7_act_words", act_words)}

{fmt_u32_arr("yolo_conv7_wgt_words", wgt_words)}

{fmt_i32("yolo_conv7_bias_q", bias_q)}

{fmt_u32("yolo_conv7_scale_mul", scale_mul)}

{fmt_u32("yolo_conv7_scale_shift", scale_shift)}

{fmt_u8_mat("yolo_conv7_expected_rtl", expected_u8)}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
