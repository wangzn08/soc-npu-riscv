#!/usr/bin/env python3
"""Generate fixtures for the M5w conv6 (model.3) smoke.

conv6 = 32->64, 3x3, stride 2, pad 1. Its input is the closed C2f (model.2)
conv5 output strip (160x16x32), computed by the M5v generator. This is a
standalone single-layer smoke (conv5 output baked in as the conv6 input
fixture) exercising OC=64 oc_single 3x3 stride-2 on the shared NPU.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
OUT_PATH = ROOT / "firmware" / "yolo_conv6_from_c2f_m5w_data.h"
M5V_GEN = ROOT / "tools" / "gen_yolo_c2f_close_m5v_smoke.py"

IN_W = 160
IN_H = 16
IC = 32
OC = 64
KH = KW = 3
KO = KH * KW
IC_GROUPS = IC // 16
STRIDE = 2
PAD = 1
OUT_W = (IN_W + 2 * PAD - KW) // STRIDE + 1   # 80
OUT_H = (IN_H + 2 * PAD - KH) // STRIDE + 1   # 8
SP_OUT = OUT_W * OUT_H
SP_IN = IN_W * IN_H

# conv6 = yolo_act_quant[6].
IN_SCALE = 0.0763198882
IN_ZP = -124
OUT_SCALE = 0.0334292874
OUT_ZP = -120
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
    m5v = load_module("m5v_gen", M5V_GEN)
    conv0, _w, _b, _sm, _rq, conv5_out = m5v.compute_close()  # [SP_IN, 32] int8
    lut = conv0.load_lut()
    act = conv5_out.reshape(IN_H, IN_W, 32).transpose(2, 0, 1)  # [32, 16, 160]

    w = np.fromfile(WEIGHT_DIR / "conv6_w.bin", dtype=np.int8).reshape(OC, IC, KH, KW)
    b = np.fromfile(WEIGHT_DIR / "conv6_b.bin", dtype=np.float32)
    s = np.fromfile(WEIGHT_DIR / "conv6_s.bin", dtype=np.float32)

    wsum = w.astype(np.int32).sum(axis=(1, 2, 3))
    bias_q, scale_mul = [], []
    for oc in range(OC):
        mul = int(round(IN_SCALE * float(s[oc]) * 16.0 * (1 << Q_SHIFT)))
        if mul == 0:
            raise SystemExit(f"conv6 oc{oc} zero scale mul")
        scale_mul.append(mul)
        bias_equiv = int(round(float(b[oc]) * 16.0 * (1 << Q_SHIFT) / mul))
        bias_q.append(int(bias_equiv - IN_ZP * int(wsum[oc])))
    scale_shift = [Q_SHIFT] * OC
    requant_mul = int(round((1 << REQUANT_SHIFT) / (16.0 * OUT_SCALE)))

    # conv6 RTL-integer model.
    expected = np.zeros((SP_OUT, OC), dtype=np.int8)
    for oy in range(OUT_H):
        for ox in range(OUT_W):
            pos = oy * OUT_W + ox
            for oc in range(OC):
                acc = 0
                for ic in range(IC):
                    for ky in range(KH):
                        for kx in range(KW):
                            iy = oy * STRIDE + ky - PAD
                            ix = ox * STRIDE + kx - PAD
                            av = int(act[ic, iy, ix]) if 0 <= iy < IN_H and 0 <= ix < IN_W else IN_ZP
                            acc += av * int(w[oc, ic, ky, kx])
                q44 = ((acc + bias_q[oc]) * scale_mul[oc]) >> Q_SHIFT
                silu = conv0.rtl_silu_byte(q44, lut)
                rq = ((conv0.s8(silu) * requant_mul) >> REQUANT_SHIFT) + OUT_ZP
                expected[pos, oc] = np.int8(conv0.s8(conv0.clamp_s8(rq)))
    expected_u8 = expected.astype(np.uint8)

    # conv6 input act words: tile-major addr = group*SP_IN + pos, 16 lanes/group.
    act_words = []
    for g in range(IC_GROUPS):
        for pos in range(SP_IN):
            y, x = pos // IN_W, pos % IN_W
            act_words.append(conv0.pack_i8_word(act[g * 16:(g + 1) * 16, y, x]))

    # conv6 weights: per oc, per ic-group, per ko.
    wgt_words = []
    for oc in range(OC):
        for icg in range(IC_GROUPS):
            for ko in range(KO):
                ky, kx = ko // KW, ko % KW
                wgt_words.append(conv0.pack_i8_word(w[oc, icg * 16:(icg + 1) * 16, ky, kx]))

    body = f"""#ifndef YOLO_CONV6_FROM_C2F_M5W_DATA_H
#define YOLO_CONV6_FROM_C2F_M5W_DATA_H

#include <stdint.h>

#define YOLO_CONV6_IN_W {IN_W}u
#define YOLO_CONV6_IN_H {IN_H}u
#define YOLO_CONV6_IC {IC}u
#define YOLO_CONV6_OC {OC}u
#define YOLO_CONV6_KH {KH}u
#define YOLO_CONV6_KW {KW}u
#define YOLO_CONV6_STRIDE {STRIDE}u
#define YOLO_CONV6_PAD {PAD}u
#define YOLO_CONV6_ACT_WORDS {len(act_words)}u
#define YOLO_CONV6_WGT_WORDS {len(wgt_words)}u
#define YOLO_CONV6_OUT_W {OUT_W}u
#define YOLO_CONV6_OUT_H {OUT_H}u
#define YOLO_CONV6_OUT_SPATIAL {SP_OUT}u
#define YOLO_CONV6_OUT_WORDS {SP_OUT * (OC // 16)}u
#define YOLO_CONV6_REQUANT_MUL {requant_mul}u
#define YOLO_CONV6_REQUANT_SHIFT {REQUANT_SHIFT}u
#define YOLO_CONV6_REQUANT_ZP {OUT_ZP}
#define YOLO_CONV6_PAD_VALUE {IN_ZP}
#define YOLO_CONV6_RTL_TOL {RTL_TOL}u

{fmt_u32_arr("yolo_conv6_act_words", act_words)}

{fmt_u32_arr("yolo_conv6_wgt_words", wgt_words)}

{fmt_i32("yolo_conv6_bias_q", bias_q)}

{fmt_u32("yolo_conv6_scale_mul", scale_mul)}

{fmt_u32("yolo_conv6_scale_shift", scale_shift)}

{fmt_u8_mat("yolo_conv6_expected_rtl", expected_u8)}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
