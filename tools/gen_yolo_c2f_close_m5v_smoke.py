#!/usr/bin/env python3
"""Generate fixtures for the M5v C2f-close smoke.

Closes the first C2f block (model.2): concat(s0, s1, add_out) -> conv5 (cv2).
  * s0 = conv2 group-0 (@conv2 scale), s1 = conv2 group-1 (@conv2 scale)
  * add_out = M5u residual (@glue[0] scale)
Each piece is integer-requantized to conv5's in_scale/in_zp (= /model.2/Concat),
concatenated to 48 channels, then conv5 (1x1, 48->32) on the shared NPU.

Per-piece requant is fixed-point: out = clamp_s8(((q-in_zp)*mul >> SHIFT)+cat_zp).
"""

from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
OUT_PATH = ROOT / "firmware" / "yolo_c2f_close_m5v_data.h"
M5U_GEN = ROOT / "tools" / "gen_yolo_c2f_add_m5u_smoke.py"

IN_W = 160
IN_H = 16
SP_OUT = IN_W * IN_H

# conv5 = yolo_act_quant[5] / /model.2/Concat (concat target) = conv5 in.
CAT_SCALE = 0.1612435579
CAT_ZP = -125
CONV5_OUT_SCALE = 0.0763198882
CONV5_OUT_ZP = -124

# source scales/zps of the three concat pieces.
CONV2_SCALE = 0.1601515412   # s0, s1 (conv2 out)
CONV2_ZP = -126
GLUE_SCALE = 0.1549137533    # add_out (glue[0] /model.2/m.0/Add)
GLUE_ZP = -124

REQ_SHIFT = 16          # per-piece requant fixed-point shift
Q_SHIFT = 20            # conv accumulator scale shift
CONV5_REQUANT_SHIFT = 12
RTL_TOL = 192


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


def requant_mul(in_scale: float) -> int:
    return int(round((in_scale / CAT_SCALE) * (1 << REQ_SHIFT)))


def requant_piece(conv0, q: np.ndarray, in_zp: int, mul: int) -> np.ndarray:
    """Integer requant to (CAT_SCALE, CAT_ZP). Mirrors the firmware exactly."""
    out = np.zeros_like(q, dtype=np.int8)
    for idx in np.ndindex(q.shape):
        v = (((int(q[idx]) - in_zp) * mul) >> REQ_SHIFT) + CAT_ZP
        out[idx] = np.int8(conv0.s8(conv0.clamp_s8(v)))
    return out


def conv5_qparams(weights, biases, wscales):
    wsum = weights.astype(np.int32).sum(axis=(1, 2, 3))
    bias_q, scale_mul = [], []
    for oc in range(weights.shape[0]):
        mul = int(round(CAT_SCALE * float(wscales[oc]) * 16.0 * (1 << Q_SHIFT)))
        if mul == 0:
            raise SystemExit(f"conv5 oc{oc} zero scale mul")
        scale_mul.append(mul)
        bias_equiv = int(round(float(biases[oc]) * 16.0 * (1 << Q_SHIFT) / mul))
        bias_q.append(int(bias_equiv - CAT_ZP * int(wsum[oc])))
    return bias_q, scale_mul, [Q_SHIFT] * weights.shape[0]


def pack_pw(conv0, weights):
    oc_n, ic_n = weights.shape[0], weights.shape[1]
    words = []
    for oc in range(oc_n):
        for g in range(ic_n // 16):
            words.append(conv0.pack_i8_word(weights[oc, g * 16:g * 16 + 16, 0, 0]))
    return words


def compute_close():
    """Compute the closed C2f (model.2) conv5 output. Returns (conv0, weights,
    bias_q, scale_mul, c5_requant_mul, expected[SP_OUT,32] int8)."""
    m5u = load_module("m5u_gen", M5U_GEN)
    conv0, conv2_out, _s1_glue, _conv4_glue, add_out, _stage = m5u.compute_pieces()
    lut = conv0.load_lut()

    s0 = conv2_out[0:16, :, :].transpose(1, 2, 0).reshape(SP_OUT, 16)
    s1 = conv2_out[16:32, :, :].transpose(1, 2, 0).reshape(SP_OUT, 16)

    mul_s0s1 = requant_mul(CONV2_SCALE)
    mul_add = requant_mul(GLUE_SCALE)
    s0_cat = requant_piece(conv0, s0, CONV2_ZP, mul_s0s1)
    s1_cat = requant_piece(conv0, s1, CONV2_ZP, mul_s0s1)
    add_cat = requant_piece(conv0, add_out, GLUE_ZP, mul_add)
    concat = np.concatenate([s0_cat, s1_cat, add_cat], axis=1)  # [SP, 48]

    w = np.fromfile(WEIGHT_DIR / "conv5_w.bin", dtype=np.int8).reshape(32, 48, 1, 1)
    b = np.fromfile(WEIGHT_DIR / "conv5_b.bin", dtype=np.float32)
    s = np.fromfile(WEIGHT_DIR / "conv5_s.bin", dtype=np.float32)
    bias_q, scale_mul, _shift = conv5_qparams(w, b, s)
    c5_requant_mul = int(round((1 << CONV5_REQUANT_SHIFT) / (16.0 * CONV5_OUT_SCALE)))

    expected = np.zeros((SP_OUT, 32), dtype=np.int8)
    for pos in range(SP_OUT):
        for oc in range(32):
            acc = 0
            for ic in range(48):
                acc += int(concat[pos, ic]) * int(w[oc, ic, 0, 0])
            q44 = ((acc + bias_q[oc]) * scale_mul[oc]) >> Q_SHIFT
            silu = conv0.rtl_silu_byte(q44, lut)
            rq = ((conv0.s8(silu) * c5_requant_mul) >> CONV5_REQUANT_SHIFT) + CONV5_OUT_ZP
            expected[pos, oc] = np.int8(conv0.s8(conv0.clamp_s8(rq)))
    return conv0, w, bias_q, scale_mul, c5_requant_mul, expected


def main() -> None:
    conv0, w, bias_q, scale_mul, c5_requant_mul, expected = compute_close()
    mul_s0s1 = requant_mul(CONV2_SCALE)
    mul_add = requant_mul(GLUE_SCALE)
    scale_shift = [Q_SHIFT] * 32
    expected_u8 = expected.astype(np.uint8)

    body = f"""#ifndef YOLO_C2F_CLOSE_M5V_DATA_H
#define YOLO_C2F_CLOSE_M5V_DATA_H

#include <stdint.h>

#define YOLO_C2F_CLOSE_IN_W {IN_W}u
#define YOLO_C2F_CLOSE_IN_H {IN_H}u
#define YOLO_C2F_CLOSE_SPATIAL {SP_OUT}u
#define YOLO_C2F_CLOSE_OC 32u
#define YOLO_C2F_CLOSE_IC 48u
#define YOLO_C2F_CLOSE_RTL_TOL {RTL_TOL}u

/* per-piece requant to /model.2/Concat (conv5 in): out=clamp(((q-in_zp)*mul>>SHIFT)+cat_zp) */
#define YOLO_C2F_CAT_REQ_SHIFT {REQ_SHIFT}u
#define YOLO_C2F_CAT_ZP {CAT_ZP}
#define YOLO_C2F_CAT_MUL_S0S1 {mul_s0s1}u
#define YOLO_C2F_CAT_MUL_ADD {mul_add}u
#define YOLO_C2F_CAT_INZP_S0S1 {CONV2_ZP}
#define YOLO_C2F_CAT_INZP_ADD {GLUE_ZP}

/* conv5 (cv2): 1x1, 48->32, SiLU + requant to (out_scale {CONV5_OUT_SCALE}, zp {CONV5_OUT_ZP}). */
#define YOLO_C2F_CONV5_WGT_WORDS {32 * 3}u
#define YOLO_C2F_CONV5_REQUANT_MUL {c5_requant_mul}u
#define YOLO_C2F_CONV5_REQUANT_SHIFT {CONV5_REQUANT_SHIFT}u
#define YOLO_C2F_CONV5_REQUANT_ZP {CONV5_OUT_ZP}

{fmt_u32_arr("yolo_c2f_conv5_wgt_words", pack_pw(conv0, w))}

{fmt_i32("yolo_c2f_conv5_bias_q", bias_q)}

{fmt_u32("yolo_c2f_conv5_scale_mul", scale_mul)}

{fmt_u32("yolo_c2f_conv5_scale_shift", scale_shift)}

{fmt_u8_mat("yolo_c2f_close_expected_rtl", expected_u8)}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
