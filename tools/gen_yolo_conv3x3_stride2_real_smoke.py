#!/usr/bin/env python3
"""Generate a real YOLO conv1 stride-2 3x3 C-reference smoke fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
LUT_PATH = ROOT / "rtl" / "silu_lut_q4_4.hex"
OUT_PATH = ROOT / "firmware" / "yolo_conv3x3_stride2_real_data.h"

IN_W = 6
IN_H = 6
IC = 16
OC = 16
KH = 3
KW = 3
STRIDE = 2
PAD = 1
OUT_W = ((IN_W + PAD * 2 - KW) // STRIDE) + 1
OUT_H = ((IN_H + PAD * 2 - KH) // STRIDE) + 1
SP_IN = IN_W * IN_H
SP_OUT = OUT_W * OUT_H
KO = KH * KW
Q_SHIFT = 20
IN_SCALE = 0.2352971584
IN_ZP = -127
OUT_SCALE = 0.6557820439
OUT_ZP = -128
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
                val = IN_ZP + (pos % 11) + (ic % 5) - 2
                act[ic, y, x] = np.int8(s8(clamp_s8(val)))
    return act


def pack_act_words(act: np.ndarray) -> list[list[int]]:
    words: list[list[int]] = []
    for pos in range(SP_IN):
        y = pos // IN_W
        x = pos % IN_W
        words.append(pack_i8_word(act[:, y, x]))
    return words


def pack_wgt_words(weights: np.ndarray) -> list[list[int]]:
    words: list[list[int]] = []
    for oc in range(OC):
        for ko in range(KO):
            ky = ko // KW
            kx = ko % KW
            words.append(pack_i8_word(weights[oc, :, ky, kx]))
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
    wsum = weights.astype(np.int32).sum(axis=(1, 2, 3))
    bias_int: list[int] = []
    scale_mul: list[int] = []
    for oc in range(OC):
        real_to_q44 = float(IN_SCALE * float(wscales[oc]) * 16.0)
        mul = int(round(real_to_q44 * (1 << Q_SHIFT)))
        if mul == 0:
            raise SystemExit(f"conv1 oc{oc} produced zero scale multiplier")
        scale_mul.append(mul)
        bias_equiv = int(round(float(biases[oc]) * 16.0 * (1 << Q_SHIFT) / mul))
        bias_int.append(int(bias_equiv - IN_ZP * int(wsum[oc])))
    return bias_int, scale_mul, [Q_SHIFT for _ in range(OC)]


def conv_cref(act: np.ndarray,
              weights: np.ndarray,
              biases: np.ndarray,
              wscales: np.ndarray) -> np.ndarray:
    out = np.zeros((SP_OUT, OC), dtype=np.uint8)
    wsum = weights.astype(np.int32).sum(axis=(1, 2, 3))
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
                            acc += av * int(weights[oc, ic, ky, kx])
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
    out = np.zeros((SP_OUT, OC), dtype=np.int8)
    requant_mul = int(round((1 << REQUANT_SHIFT) / (16.0 * OUT_SCALE)))
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
                            acc += av * int(weights[oc, ic, ky, kx])
                q44 = ((acc + bias_q[oc]) * scale_mul[oc]) >> Q_SHIFT
                silu_byte = rtl_silu_byte(q44, lut)
                rq = ((s8(silu_byte) * requant_mul) >> REQUANT_SHIFT) + OUT_ZP
                out[pos, oc] = np.int8(s8(clamp_s8(rq)))
    return out


def main() -> None:
    weights = np.fromfile(WEIGHT_DIR / "conv1_w.bin", dtype=np.int8)
    biases = np.fromfile(WEIGHT_DIR / "conv1_b.bin", dtype=np.float32)
    wscales = np.fromfile(WEIGHT_DIR / "conv1_s.bin", dtype=np.float32)
    if weights.size != 32 * 16 * KH * KW or biases.size < 32 or wscales.size < 32:
        raise SystemExit("conv1 asset size mismatch")

    weights = weights.reshape(32, 16, KH, KW)[:OC]
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

    body = f"""#ifndef YOLO_CONV3X3_STRIDE2_REAL_DATA_H
#define YOLO_CONV3X3_STRIDE2_REAL_DATA_H

#include <stdint.h>

#define YOLO_CONV3X3_S2_IN_W {IN_W}u
#define YOLO_CONV3X3_S2_IN_H {IN_H}u
#define YOLO_CONV3X3_S2_IC {IC}u
#define YOLO_CONV3X3_S2_OC {OC}u
#define YOLO_CONV3X3_S2_KH {KH}u
#define YOLO_CONV3X3_S2_KW {KW}u
#define YOLO_CONV3X3_S2_STRIDE {STRIDE}u
#define YOLO_CONV3X3_S2_PAD {PAD}u
#define YOLO_CONV3X3_S2_IN_ZP {IN_ZP}
#define YOLO_CONV3X3_S2_ACT_WORDS {SP_IN}u
#define YOLO_CONV3X3_S2_WGT_WORDS {OC * KO}u
#define YOLO_CONV3X3_S2_OUT_WORDS {SP_OUT}u
#define YOLO_CONV3X3_S2_REQUANT_MUL {requant_mul}u
#define YOLO_CONV3X3_S2_REQUANT_SHIFT {REQUANT_SHIFT}u
#define YOLO_CONV3X3_S2_REQUANT_ZP {OUT_ZP}
#define YOLO_CONV3X3_S2_CREF_TOL {CREF_TOL}u

{format_u32_array("yolo_conv3x3_s2_act_words", pack_act_words(act))}

{format_u32_array("yolo_conv3x3_s2_wgt_words", pack_wgt_words(weights))}

{format_i32_array("yolo_conv3x3_s2_bias_q", bias_q)}

{format_u32_flat_array("yolo_conv3x3_s2_scale_mul", scale_mul)}

{format_u32_flat_array("yolo_conv3x3_s2_scale_shift", scale_shift)}

static const uint8_t yolo_conv3x3_s2_expected_cref[{SP_OUT}][{OC}] = {{
"""
    for pos in range(SP_OUT):
        body += "    {" + ", ".join(f"0x{int(v):02X}u" for v in expected[pos]) + "},\n"
    body += """};

#endif
"""
    OUT_PATH.write_text(body, encoding="ascii")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
