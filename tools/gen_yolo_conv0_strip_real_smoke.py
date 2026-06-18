#!/usr/bin/env python3
"""Generate a real YOLO conv0 top-strip C-reference smoke fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
LUT_PATH = ROOT / "rtl" / "silu_lut_q4_4.hex"
OUT_PATH = ROOT / "firmware" / "yolo_conv0_strip_real_data.h"

IN_W = 640
IN_H = 16
IC = 3
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
IN_SCALE = 0.0039215689
IN_ZP = -128
OUT_SCALE = 0.2352971584
OUT_ZP = -127
REQUANT_SHIFT = 12
RTL_TOL = 40


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
    for c in range(IC):
        for y in range(IN_H):
            for x in range(IN_W):
                pixel = (x * 3 + y * 11 + c * 37) & 0xFF
                q = c_lround((pixel / 255.0) / IN_SCALE + IN_ZP)
                act[c, y, x] = np.int8(s8(clamp_s8(q)))
    return act


def pack_act_words(act: np.ndarray) -> list[list[int]]:
    words: list[list[int]] = []
    for pos in range(SP_IN):
        y = pos // IN_W
        x = pos % IN_W
        lanes = np.full((16,), IN_ZP, dtype=np.int8)
        lanes[:IC] = act[:, y, x]
        words.append(pack_i8_word(lanes))
    return words


def pack_wgt_words(weights: np.ndarray) -> list[list[int]]:
    words: list[list[int]] = []
    for oc in range(OC):
        for ko in range(KO):
            ky = ko // KW
            kx = ko % KW
            lanes = np.zeros((16,), dtype=np.int8)
            lanes[:IC] = weights[oc, :, ky, kx]
            words.append(pack_i8_word(lanes))
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


def format_u8_matrix(name: str, vals: np.ndarray) -> str:
    lines = [f"static const uint8_t {name}[{vals.shape[0]}][{vals.shape[1]}] = {{"]
    for row in vals:
        lines.append("    {" + ", ".join(f"0x{int(x):02X}u" for x in row) + "},")
    lines.append("};")
    return "\n".join(lines)


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
            raise SystemExit(f"conv0 oc{oc} produced zero scale multiplier")
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
    weights = np.fromfile(WEIGHT_DIR / "conv0_w.bin", dtype=np.int8)
    biases = np.fromfile(WEIGHT_DIR / "conv0_b.bin", dtype=np.float32)
    wscales = np.fromfile(WEIGHT_DIR / "conv0_s.bin", dtype=np.float32)
    if weights.size != OC * IC * KH * KW or biases.size < OC or wscales.size < OC:
        raise SystemExit("conv0 asset size mismatch")

    weights = weights.reshape(OC, IC, KH, KW)
    act = make_activation()
    lut = load_lut()
    bias_q, scale_mul, scale_shift = make_qparams(weights, biases, wscales)
    requant_mul = int(round((1 << REQUANT_SHIFT) / (16.0 * OUT_SCALE)))

    expected = conv_cref(act, weights, biases, wscales)
    rtl_expected = conv_rtl_model(act, weights, bias_q, scale_mul, lut)
    expected_s = expected.astype(np.uint8).view(np.int8).astype(np.int16)
    diff = np.abs(expected_s - rtl_expected.astype(np.int16))
    cref_max_diff = int(diff.max())
    expected_rtl_u8 = rtl_expected.astype(np.uint8)

    body = f"""#ifndef YOLO_CONV0_STRIP_REAL_DATA_H
#define YOLO_CONV0_STRIP_REAL_DATA_H

#include <stdint.h>

#define YOLO_CONV0_STRIP_IN_W {IN_W}u
#define YOLO_CONV0_STRIP_IN_H {IN_H}u
#define YOLO_CONV0_STRIP_IC {IC}u
#define YOLO_CONV0_STRIP_OC {OC}u
#define YOLO_CONV0_STRIP_KH {KH}u
#define YOLO_CONV0_STRIP_KW {KW}u
#define YOLO_CONV0_STRIP_STRIDE {STRIDE}u
#define YOLO_CONV0_STRIP_PAD {PAD}u
#define YOLO_CONV0_STRIP_IN_ZP {IN_ZP}
#define YOLO_CONV0_STRIP_ACT_WORDS {SP_IN}u
#define YOLO_CONV0_STRIP_WGT_WORDS {OC * KO}u
#define YOLO_CONV0_STRIP_OUT_W {OUT_W}u
#define YOLO_CONV0_STRIP_OUT_H {OUT_H}u
#define YOLO_CONV0_STRIP_OUT_WORDS {SP_OUT}u
#define YOLO_CONV0_STRIP_REQUANT_MUL {requant_mul}u
#define YOLO_CONV0_STRIP_REQUANT_SHIFT {REQUANT_SHIFT}u
#define YOLO_CONV0_STRIP_REQUANT_ZP {OUT_ZP}
#define YOLO_CONV0_STRIP_RTL_TOL {RTL_TOL}u
#define YOLO_CONV0_STRIP_FLOAT_CREF_MAX_DIFF {cref_max_diff}u

{format_u32_array("yolo_conv0_strip_act_words", pack_act_words(act))}

{format_u32_array("yolo_conv0_strip_wgt_words", pack_wgt_words(weights))}

{format_i32_array("yolo_conv0_strip_bias_q", bias_q)}

{format_u32_flat_array("yolo_conv0_strip_scale_mul", scale_mul)}

{format_u32_flat_array("yolo_conv0_strip_scale_shift", scale_shift)}

{format_u8_matrix("yolo_conv0_strip_expected_rtl", expected_rtl_u8)}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
