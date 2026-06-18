#!/usr/bin/env python3
"""Generate a real-weight pointwise-conv SiLU smoke fixture.

This is a quantization-alignment stepping stone.  It uses real YOLOv8n conv2
weights, float bias, per-channel weight scales, and input zero-point correction,
then folds them into the current RTL post-process form:

    q4_4 = ((acc + bias_int[oc]) * scale_mul[oc]) >>> scale_shift[oc]
    out  = silu_lut_q4_4[clamp_s8(q4_4)]

The final output is therefore exact against the RTL integer/SILU path.  It is
not yet the full YOLO output-scale/zero-point requantization contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
LUT_PATH = ROOT / "rtl" / "silu_lut_q4_4.hex"
OUT_PATH = ROOT / "firmware" / "yolo_pwconv_silu_real_data.h"

IN_W = 2
IN_H = 2
SPATIAL = IN_W * IN_H
IC = 32
OC = 16
IC_GROUPS = IC // 16
SHIFT = 20
IN_SCALE = 0.6557820439
IN_ZP = -128
OUT_SCALE = 0.1601515412
OUT_ZP = -126
REQUANT_SHIFT = 12


def pack_i8_word(vals: np.ndarray) -> list[int]:
    raw = vals.astype(np.int8).view(np.uint8)
    words: list[int] = []
    for group in range(4):
        word = 0
        for b in range(4):
            word |= int(raw[group * 4 + b]) << (8 * b)
        words.append(word)
    return words


def make_activation() -> np.ndarray:
    act = np.zeros((IC, IN_H, IN_W), dtype=np.int8)
    for ic in range(IC):
        for y in range(IN_H):
            for x in range(IN_W):
                pos = y * IN_W + x
                # Values sit near the YOLO input zero-point so correction matters
                # without immediately saturating every SiLU output.
                act[ic, y, x] = np.int8(-128 + 1 + pos + (ic % 7))
    return act


def pack_act_words(act: np.ndarray) -> list[list[int]]:
    words: list[list[int]] = []
    for icg in range(IC_GROUPS):
        for pos in range(SPATIAL):
            y = pos // IN_W
            x = pos % IN_W
            words.append(pack_i8_word(act[icg * 16:(icg + 1) * 16, y, x]))
    return words


def pack_wgt_words(weights: np.ndarray) -> list[list[int]]:
    words: list[list[int]] = []
    for oc in range(OC):
        for icg in range(IC_GROUPS):
            words.append(pack_i8_word(weights[oc, icg * 16:(icg + 1) * 16]))
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


def format_u32_array(name: str, words: list[list[int]]) -> str:
    lines = [f"static const uint32_t {name}[{len(words)}][4] = {{"]
    for lanes in words:
        lines.append("    {" + ", ".join(f"0x{x:08X}u" for x in lanes) + "},")
    lines.append("};")
    return "\n".join(lines)


def main() -> None:
    weights = np.fromfile(WEIGHT_DIR / "conv2_w.bin", dtype=np.int8)
    biases = np.fromfile(WEIGHT_DIR / "conv2_b.bin", dtype=np.float32)
    wscales = np.fromfile(WEIGHT_DIR / "conv2_s.bin", dtype=np.float32)
    if weights.size != 32 * 32 or biases.size != 32 or wscales.size != 32:
        raise SystemExit("conv2 asset size mismatch")

    weights = weights.reshape(32, 32)[:OC, :IC]
    biases = biases[:OC]
    wscales = wscales[:OC]
    lut = load_lut()
    act = make_activation()

    wsum = weights.astype(np.int32).sum(axis=1)
    scale_mul: list[int] = []
    bias_int: list[int] = []
    for oc in range(OC):
        real_to_q44 = float(IN_SCALE * float(wscales[oc]) * 16.0)
        mul = int(round(real_to_q44 * (1 << SHIFT)))
        if mul == 0:
            raise SystemExit(f"conv2 oc{oc} produced zero scale multiplier")
        scale_mul.append(mul)
        bias_equiv = int(round(float(biases[oc]) * 16.0 * (1 << SHIFT) / mul))
        bias_int.append(int(bias_equiv - IN_ZP * int(wsum[oc])))

    expected = np.zeros((SPATIAL, OC), dtype=np.uint8)
    expected_requant = np.zeros((SPATIAL, OC), dtype=np.uint8)
    expected_cref = np.zeros((SPATIAL, OC), dtype=np.uint8)
    requant_mul = int(round((1 << REQUANT_SHIFT) / (16.0 * OUT_SCALE)))
    for pos in range(SPATIAL):
        y = pos // IN_W
        x = pos % IN_W
        avec = act[:, y, x].astype(np.int32)
        for oc in range(OC):
            acc = int(np.sum(avec * weights[oc].astype(np.int32)))
            acc_corr = acc - IN_ZP * int(wsum[oc])
            preact = float(acc_corr) * IN_SCALE * float(wscales[oc]) + float(biases[oc])
            silu = preact / (1.0 + np.exp(-preact))
            expected_cref[pos, oc] = clamp_s8(c_lround(silu / OUT_SCALE + OUT_ZP))
            q44 = ((acc + bias_int[oc]) * scale_mul[oc]) >> SHIFT
            silu_byte = rtl_silu_byte(q44, lut)
            expected[pos, oc] = silu_byte
            rq = ((s8(silu_byte) * requant_mul) >> REQUANT_SHIFT) + OUT_ZP
            expected_requant[pos, oc] = clamp_s8(rq)

    body = f"""#ifndef YOLO_PWCONV_SILU_REAL_DATA_H
#define YOLO_PWCONV_SILU_REAL_DATA_H

#include <stdint.h>

#define YOLO_SILU_REAL_IN_W {IN_W}u
#define YOLO_SILU_REAL_IN_H {IN_H}u
#define YOLO_SILU_REAL_IC {IC}u
#define YOLO_SILU_REAL_OC {OC}u
#define YOLO_SILU_REAL_ACT_WORDS {SPATIAL * IC_GROUPS}u
#define YOLO_SILU_REAL_WGT_WORDS {OC * IC_GROUPS}u
#define YOLO_SILU_REAL_OUT_WORDS {SPATIAL}u
#define YOLO_SILU_REAL_REQUANT_MUL {requant_mul}u
#define YOLO_SILU_REAL_REQUANT_SHIFT {REQUANT_SHIFT}u
#define YOLO_SILU_REAL_REQUANT_ZP {OUT_ZP}

{format_u32_array("yolo_silu_real_act_words", pack_act_words(act))}

{format_u32_array("yolo_silu_real_wgt_words", pack_wgt_words(weights))}

static const int32_t yolo_silu_real_bias[{OC}] = {{
    {", ".join(str(x) for x in bias_int)}
}};

static const uint32_t yolo_silu_real_scale_mul[{OC}] = {{
    {", ".join(f"{x}u" for x in scale_mul)}
}};

static const uint32_t yolo_silu_real_scale_shift[{OC}] = {{
    {", ".join(f"{SHIFT}u" for _ in range(OC))}
}};

static const uint8_t yolo_silu_real_expected[{SPATIAL}][{OC}] = {{
"""
    for pos in range(SPATIAL):
        body += "    {" + ", ".join(f"0x{int(v):02X}u" for v in expected[pos]) + "},\n"
    body += "};\n\n"
    body += f"static const uint8_t yolo_silu_real_expected_requant[{SPATIAL}][{OC}] = {{\n"
    for pos in range(SPATIAL):
        body += "    {" + ", ".join(f"0x{int(v):02X}u" for v in expected_requant[pos]) + "},\n"
    body += "};\n\n"
    body += f"static const uint8_t yolo_silu_real_expected_cref[{SPATIAL}][{OC}] = {{\n"
    for pos in range(SPATIAL):
        body += "    {" + ", ".join(f"0x{int(v):02X}u" for v in expected_cref[pos]) + "},\n"
    body += """};

#endif
"""

    OUT_PATH.write_text(body, encoding="ascii")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
