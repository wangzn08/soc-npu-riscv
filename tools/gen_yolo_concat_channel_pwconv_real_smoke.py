#!/usr/bin/env python3
"""Generate a real YOLO conv5 concat-channel pointwise C-reference fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
LUT_PATH = ROOT / "rtl" / "silu_lut_q4_4.hex"
OUT_PATH = ROOT / "firmware" / "yolo_concat_channel_real_data.h"

IN_W = 2
IN_H = 2
IC = 48
OC = 32
IC_GROUPS = IC // 16
OC_GROUPS = OC // 16
SPATIAL = IN_W * IN_H
Q_SHIFT = 20
IN_SCALE = 0.1612435579
IN_ZP = -125
OUT_SCALE = 0.0763198882
OUT_ZP = -124
REQUANT_SHIFT = 12
CREF_TOL = 10


def s8(byte: int) -> int:
    byte &= 0xFF
    return byte - 256 if byte & 0x80 else byte


def clamp_s8(v: int) -> int:
    if v < -128:
        return 0x80
    if v > 127:
        return 0x7F
    return v & 0xFF


def c_lround(v: float) -> int:
    if v >= 0.0:
        return int(np.floor(v + 0.5))
    return int(np.ceil(v - 0.5))


def pack_i8_word(vals: np.ndarray) -> list[int]:
    if vals.shape != (16,):
        raise ValueError(f"expected 16 lanes, got {vals.shape}")
    raw = vals.astype(np.int8).view(np.uint8)
    words: list[int] = []
    for group in range(4):
        word = 0
        for b in range(4):
            word |= int(raw[group * 4 + b]) << (8 * b)
        words.append(word)
    return words


def load_lut() -> list[int]:
    vals: list[int] = []
    for line in LUT_PATH.read_text(encoding="ascii").splitlines():
        text = line.strip()
        if text:
            vals.append(int(text, 16) & 0xFF)
    if len(vals) != 256:
        raise SystemExit(f"{LUT_PATH} has {len(vals)} entries, expected 256")
    return vals


def rtl_silu_byte(v: int, lut: list[int]) -> int:
    if v < -128:
        v = -128
    elif v > 127:
        v = 127
    return lut[v & 0xFF]


def make_activation() -> np.ndarray:
    act = np.zeros((IC, IN_H, IN_W), dtype=np.int8)
    for ic in range(IC):
        for y in range(IN_H):
            for x in range(IN_W):
                pos = y * IN_W + x
                val = IN_ZP + (pos % 5) + (ic % 11) - 5
                act[ic, y, x] = np.int8(s8(clamp_s8(val)))
    return act


def pack_act_words(act: np.ndarray) -> list[list[int]]:
    words: list[list[int]] = []
    for group in range(IC_GROUPS):
        for pos in range(SPATIAL):
            y = pos // IN_W
            x = pos % IN_W
            words.append(pack_i8_word(act[group * 16:(group + 1) * 16, y, x]))
    return words


def pack_wgt_words(weights: np.ndarray) -> list[list[int]]:
    words: list[list[int]] = []
    for oc in range(OC):
        for group in range(IC_GROUPS):
            words.append(pack_i8_word(weights[oc, group * 16:(group + 1) * 16]))
    return words


def format_u32_array(name: str, words: list[list[int]]) -> str:
    lines = [f"static const uint32_t {name}[{len(words)}][4] = {{"]
    for lanes in words:
        lines.append("    {" + ", ".join(f"0x{x:08X}u" for x in lanes) + "},")
    lines.append("};")
    return "\n".join(lines)


def format_i32_array(name: str, vals: list[int]) -> str:
    return f"""static const int32_t {name}[{len(vals)}] = {{
    {", ".join(str(x) for x in vals)}
}};"""


def format_u32_flat_array(name: str, vals: list[int]) -> str:
    return f"""static const uint32_t {name}[{len(vals)}] = {{
    {", ".join(f"{x}u" for x in vals)}
}};"""


def make_qparams(weights: np.ndarray,
                 biases: np.ndarray,
                 wscales: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    wsum = weights.astype(np.int32).sum(axis=1)
    bias_int: list[int] = []
    scale_mul: list[int] = []
    for oc in range(OC):
        real_to_q44 = float(IN_SCALE * float(wscales[oc]) * 16.0)
        mul = int(round(real_to_q44 * (1 << Q_SHIFT)))
        if mul == 0:
            raise SystemExit(f"conv5 oc{oc} produced zero scale multiplier")
        scale_mul.append(mul)
        bias_equiv = int(round(float(biases[oc]) * 16.0 * (1 << Q_SHIFT) / mul))
        bias_int.append(int(bias_equiv - IN_ZP * int(wsum[oc])))
    return bias_int, scale_mul, [Q_SHIFT for _ in range(OC)]


def conv_cref(act: np.ndarray,
              weights: np.ndarray,
              biases: np.ndarray,
              wscales: np.ndarray) -> np.ndarray:
    out = np.zeros((SPATIAL, OC), dtype=np.uint8)
    wsum = weights.astype(np.int32).sum(axis=1)
    for pos in range(SPATIAL):
        y = pos // IN_W
        x = pos % IN_W
        avec = act[:, y, x].astype(np.int32)
        for oc in range(OC):
            acc = int(np.sum(avec * weights[oc].astype(np.int32)))
            acc_corr = acc - IN_ZP * int(wsum[oc])
            preact = float(acc_corr) * IN_SCALE * float(wscales[oc]) + float(biases[oc])
            silu = preact / (1.0 + np.exp(-preact))
            out[pos, oc] = clamp_s8(c_lround(silu / OUT_SCALE + OUT_ZP))
    return out


def conv_rtl_model(act: np.ndarray,
                   weights: np.ndarray,
                   bias_q: list[int],
                   scale_mul: list[int],
                   lut: list[int]) -> np.ndarray:
    out = np.zeros((SPATIAL, OC), dtype=np.int8)
    requant_mul = int(round((1 << REQUANT_SHIFT) / (16.0 * OUT_SCALE)))
    for pos in range(SPATIAL):
        y = pos // IN_W
        x = pos % IN_W
        avec = act[:, y, x].astype(np.int32)
        for oc in range(OC):
            acc = int(np.sum(avec * weights[oc].astype(np.int32)))
            q44 = ((acc + bias_q[oc]) * scale_mul[oc]) >> Q_SHIFT
            silu_byte = rtl_silu_byte(q44, lut)
            rq = ((s8(silu_byte) * requant_mul) >> REQUANT_SHIFT) + OUT_ZP
            out[pos, oc] = np.int8(s8(clamp_s8(rq)))
    return out


def main() -> None:
    weights = np.fromfile(WEIGHT_DIR / "conv5_w.bin", dtype=np.int8)
    biases = np.fromfile(WEIGHT_DIR / "conv5_b.bin", dtype=np.float32)
    wscales = np.fromfile(WEIGHT_DIR / "conv5_s.bin", dtype=np.float32)
    if weights.size != OC * IC or biases.size < OC or wscales.size < OC:
        raise SystemExit("conv5 asset size mismatch")

    weights = weights.reshape(OC, IC)
    biases = biases[:OC]
    wscales = wscales[:OC]
    act = make_activation()
    lut = load_lut()
    bias_q, scale_mul, scale_shift = make_qparams(weights, biases, wscales)
    requant_mul = int(round((1 << REQUANT_SHIFT) / (16.0 * OUT_SCALE)))

    expected = conv_cref(act, weights, biases, wscales)
    rtl_expected = conv_rtl_model(act, weights, bias_q, scale_mul, lut)
    expected_s = expected.astype(np.uint8).view(np.int8).astype(np.int16)
    diff = np.abs(expected_s - rtl_expected.astype(np.int16))
    if int(diff.max()) > CREF_TOL:
        raise SystemExit(f"CREF_TOL={CREF_TOL} too small; max diff is {int(diff.max())}")

    body = f"""#ifndef YOLO_CONCAT_CHANNEL_REAL_DATA_H
#define YOLO_CONCAT_CHANNEL_REAL_DATA_H

#include <stdint.h>

#define YOLO_CONCAT_CH_IN_W {IN_W}u
#define YOLO_CONCAT_CH_IN_H {IN_H}u
#define YOLO_CONCAT_CH_IC {IC}u
#define YOLO_CONCAT_CH_OC {OC}u
#define YOLO_CONCAT_CH_ACT_WORDS {SPATIAL * IC_GROUPS}u
#define YOLO_CONCAT_CH_WGT_WORDS {OC * IC_GROUPS}u
#define YOLO_CONCAT_CH_OUT_SPATIAL {SPATIAL}u
#define YOLO_CONCAT_CH_OUT_WORDS {SPATIAL * OC_GROUPS}u
#define YOLO_CONCAT_CH_REQUANT_MUL {requant_mul}u
#define YOLO_CONCAT_CH_REQUANT_SHIFT {REQUANT_SHIFT}u
#define YOLO_CONCAT_CH_REQUANT_ZP {OUT_ZP}
#define YOLO_CONCAT_CH_CREF_TOL {CREF_TOL}u

{format_u32_array("yolo_concat_ch_act_words", pack_act_words(act))}

{format_u32_array("yolo_concat_ch_wgt_words", pack_wgt_words(weights))}

{format_i32_array("yolo_concat_ch_bias_q", bias_q)}

{format_u32_flat_array("yolo_concat_ch_scale_mul", scale_mul)}

{format_u32_flat_array("yolo_concat_ch_scale_shift", scale_shift)}

static const uint8_t yolo_concat_ch_expected_cref[{SPATIAL}][{OC}] = {{
"""
    for pos in range(SPATIAL):
        body += "    {" + ", ".join(f"0x{int(v):02X}u" for v in expected[pos]) + "},\n"
    body += """};

#endif
"""
    OUT_PATH.write_text(body, encoding="ascii")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
