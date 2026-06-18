#!/usr/bin/env python3
"""Generate data for a tiny real-weight YOLO subgraph smoke.

The fixture uses real YOLOv8n conv2 INT8 pointwise weights, but small
deterministic activation tensors.  Golden values follow the current RTL
integer post-process contract used by yolo_run_pw_conv1x1:

    out = ReLU((sum_i int8_act[i] * int8_w[i] + bias) >>> scale_shift)

This deliberately validates shared-hardware scheduling and real weight layout
before taking on full YOLO zero-point/SiLU calibration semantics.

The same fixture also emits a newer C-reference contract for:

    pointwise conv + input zp correction + float bias/scale + SiLU + requant
    -> upsample2x -> concat -> pointwise conv + SiLU + requant

That path is approximate in hardware because SiLU is LUT-based, so the final
subgraph smoke uses a small signed-INT8 tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
WEIGHT_PATH = WEIGHT_DIR / "conv2_w.bin"
LUT_PATH = ROOT / "rtl" / "silu_lut_q4_4.hex"
OUT_PATH = ROOT / "firmware" / "yolo_subgraph_real_data.h"

IN_W = 2
IN_H = 2
UP_W = IN_W * 2
UP_H = IN_H * 2
SP0 = IN_W * IN_H
SP1 = UP_W * UP_H
IC0 = 32
OC0 = 16
SKIP_C = 16
IC1 = OC0 + SKIP_C
OC1 = 16
IC0_GROUPS = IC0 // 16
OC0_GROUPS = OC0 // 16
SKIP_GROUPS = SKIP_C // 16
IC1_GROUPS = IC1 // 16
SCALE_MUL = 1
SCALE_SHIFT = 7
Q_SHIFT = 20
IN_SCALE = 0.6557820439
IN_ZP = -128
OUT_SCALE = 0.1601515412
OUT_ZP = -126
REQUANT_SHIFT = 12
CREF_TOL = 3


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


def relu_shift(v: int) -> np.uint8:
    q = int(v) >> SCALE_SHIFT
    if q < 0:
        q = 0
    if q > 127:
        q = 127
    return np.uint8(q)


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


def make_act() -> np.ndarray:
    act = np.zeros((IC0, IN_H, IN_W), dtype=np.int8)
    for ic in range(IC0):
        for y in range(IN_H):
            for x in range(IN_W):
                pos = y * IN_W + x
                act[ic, y, x] = np.int8(-128 + 1 + pos + (ic % 7))
    return act


def make_skip() -> np.ndarray:
    skip = np.zeros((SKIP_C, UP_H, UP_W), dtype=np.int8)
    for ch in range(SKIP_C):
        for y in range(UP_H):
            for x in range(UP_W):
                pos = y * UP_W + x
                skip[ch, y, x] = np.int8(-128 + 1 + (pos % 5) + (ch % 3))
    return skip


def pack_act_words(tensor: np.ndarray, h: int, w: int) -> list[list[int]]:
    channels = tensor.shape[0]
    groups = channels // 16
    words: list[list[int]] = []
    for group in range(groups):
        for pos in range(h * w):
            y = pos // w
            x = pos % w
            words.append(pack_i8_word(tensor[group * 16:(group + 1) * 16, y, x]))
    return words


def pack_wgt_words(weights: np.ndarray, oc_count: int, ic_count: int) -> list[list[int]]:
    groups = ic_count // 16
    words: list[list[int]] = []
    for oc in range(oc_count):
        for group in range(groups):
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
    scale_mul: list[int] = []
    bias_int: list[int] = []
    for oc in range(weights.shape[0]):
        real_to_q44 = float(IN_SCALE * float(wscales[oc]) * 16.0)
        mul = int(round(real_to_q44 * (1 << Q_SHIFT)))
        if mul == 0:
            raise SystemExit(f"conv2 oc{oc} produced zero scale multiplier")
        scale_mul.append(mul)
        bias_equiv = int(round(float(biases[oc]) * 16.0 * (1 << Q_SHIFT) / mul))
        bias_int.append(int(bias_equiv - IN_ZP * int(wsum[oc])))
    return bias_int, scale_mul, [Q_SHIFT for _ in range(weights.shape[0])]


def conv_cref(inp: np.ndarray,
              weights: np.ndarray,
              biases: np.ndarray,
              wscales: np.ndarray) -> np.ndarray:
    oc_count, ic_count = weights.shape
    _, h, w = inp.shape
    out = np.zeros((oc_count, h, w), dtype=np.int8)
    wsum = weights.astype(np.int32).sum(axis=1)
    for y in range(h):
        for x in range(w):
            avec = inp[:, y, x].astype(np.int32)
            for oc in range(oc_count):
                acc = int(np.sum(avec * weights[oc].astype(np.int32)))
                acc_corr = acc - IN_ZP * int(wsum[oc])
                preact = float(acc_corr) * IN_SCALE * float(wscales[oc]) + float(biases[oc])
                silu = preact / (1.0 + np.exp(-preact))
                q = c_lround(silu / OUT_SCALE + OUT_ZP)
                out[oc, y, x] = np.int8(s8(clamp_s8(q)))
    return out


def conv_rtl_requant(inp: np.ndarray,
                     weights: np.ndarray,
                     bias_int: list[int],
                     scale_mul: list[int],
                     lut: list[int]) -> np.ndarray:
    oc_count, _ = weights.shape
    _, h, w = inp.shape
    out = np.zeros((oc_count, h, w), dtype=np.int8)
    requant_mul = int(round((1 << REQUANT_SHIFT) / (16.0 * OUT_SCALE)))
    for y in range(h):
        for x in range(w):
            avec = inp[:, y, x].astype(np.int32)
            for oc in range(oc_count):
                acc = int(np.sum(avec * weights[oc].astype(np.int32)))
                q44 = ((acc + bias_int[oc]) * scale_mul[oc]) >> Q_SHIFT
                silu_byte = rtl_silu_byte(q44, lut)
                rq = ((s8(silu_byte) * requant_mul) >> REQUANT_SHIFT) + OUT_ZP
                out[oc, y, x] = np.int8(s8(clamp_s8(rq)))
    return out


def main() -> None:
    weights = np.fromfile(WEIGHT_PATH, dtype=np.int8)
    biases = np.fromfile(WEIGHT_DIR / "conv2_b.bin", dtype=np.float32)
    wscales = np.fromfile(WEIGHT_DIR / "conv2_s.bin", dtype=np.float32)
    if weights.size != 32 * 32 or biases.size != 32 or wscales.size != 32:
        raise SystemExit("conv2 asset size mismatch")
    weights = weights.reshape(32, 32)
    weights16 = weights[:OC0, :IC0]
    biases16 = biases[:OC0]
    wscales16 = wscales[:OC0]
    lut = load_lut()

    act = make_act()
    skip = make_skip()
    requant_mul = int(round((1 << REQUANT_SHIFT) / (16.0 * OUT_SCALE)))
    bias_q, scale_mul, scale_shift = make_qparams(weights16, biases16, wscales16)

    conv0 = np.zeros((OC0, IN_H, IN_W), dtype=np.uint8)
    for pos in range(SP0):
        y = pos // IN_W
        x = pos % IN_W
        avec = act[:, y, x].astype(np.int32)
        for oc in range(OC0):
            conv0[oc, y, x] = relu_shift(int(np.sum(avec * weights16[oc].astype(np.int32))))

    up = np.zeros((OC0, UP_H, UP_W), dtype=np.int8)
    for oc in range(OC0):
        for y in range(UP_H):
            for x in range(UP_W):
                up[oc, y, x] = np.int8(conv0[oc, y >> 1, x >> 1])

    concat = np.concatenate([up, skip], axis=0)
    expected = np.zeros((SP1, OC1), dtype=np.uint8)
    for pos in range(SP1):
        y = pos // UP_W
        x = pos % UP_W
        avec = concat[:, y, x].astype(np.int32)
        for oc in range(OC1):
            expected[pos, oc] = relu_shift(int(np.sum(avec * weights16[oc].astype(np.int32))))

    conv0_c = conv_cref(act, weights16, biases16, wscales16)
    up_c = np.repeat(np.repeat(conv0_c, 2, axis=1), 2, axis=2)
    concat_c = np.concatenate([up_c, skip], axis=0)
    expected_cref_tensor = conv_cref(concat_c, weights16, biases16, wscales16)
    expected_cref = np.zeros((SP1, OC1), dtype=np.uint8)
    for pos in range(SP1):
        y = pos // UP_W
        x = pos % UP_W
        expected_cref[pos] = expected_cref_tensor[:, y, x].astype(np.int8).view(np.uint8)

    conv0_rtl = conv_rtl_requant(act, weights16, bias_q, scale_mul, lut)
    up_rtl = np.repeat(np.repeat(conv0_rtl, 2, axis=1), 2, axis=2)
    concat_rtl = np.concatenate([up_rtl, skip], axis=0)
    expected_rtl_tensor = conv_rtl_requant(concat_rtl, weights16, bias_q, scale_mul, lut)
    diff = np.abs(expected_rtl_tensor.astype(np.int16) - expected_cref_tensor.astype(np.int16))
    if int(diff.max()) > CREF_TOL:
        raise SystemExit(f"CREF_TOL={CREF_TOL} too small; max diff is {int(diff.max())}")

    body = f"""#ifndef YOLO_SUBGRAPH_REAL_DATA_H
#define YOLO_SUBGRAPH_REAL_DATA_H

#include <stdint.h>

#define YOLO_REAL_SUB_IN_W {IN_W}u
#define YOLO_REAL_SUB_IN_H {IN_H}u
#define YOLO_REAL_SUB_UP_W {UP_W}u
#define YOLO_REAL_SUB_UP_H {UP_H}u
#define YOLO_REAL_SUB_UP_SPATIAL {SP1}u
#define YOLO_REAL_SUB_IC0 {IC0}u
#define YOLO_REAL_SUB_OC0 {OC0}u
#define YOLO_REAL_SUB_OC0_GROUPS {OC0_GROUPS}u
#define YOLO_REAL_SUB_SKIP_GROUPS {SKIP_GROUPS}u
#define YOLO_REAL_SUB_IC1 {IC1}u
#define YOLO_REAL_SUB_OC1 {OC1}u
#define YOLO_REAL_SUB_ACT_WORDS {SP0 * IC0_GROUPS}u
#define YOLO_REAL_SUB_SKIP_WORDS {SP1 * SKIP_GROUPS}u
#define YOLO_REAL_SUB_CONV0_WORDS {SP0 * OC0_GROUPS}u
#define YOLO_REAL_SUB_UP_WORDS {SP1 * OC0_GROUPS}u
#define YOLO_REAL_SUB_OUT_WORDS {SP1}u
#define YOLO_REAL_SUB_WGT0_WORDS {OC0 * IC0_GROUPS}u
#define YOLO_REAL_SUB_WGT1_WORDS {OC1 * IC1_GROUPS}u
#define YOLO_REAL_SUB_SCALE_MUL {SCALE_MUL}u
#define YOLO_REAL_SUB_SCALE_SHIFT {SCALE_SHIFT}u
#define YOLO_REAL_SUB_REQUANT_MUL {requant_mul}u
#define YOLO_REAL_SUB_REQUANT_SHIFT {REQUANT_SHIFT}u
#define YOLO_REAL_SUB_REQUANT_ZP {OUT_ZP}
#define YOLO_REAL_SUB_CREF_TOL {CREF_TOL}u

{format_u32_array("yolo_real_sub_act_words", pack_act_words(act, IN_H, IN_W))}

{format_u32_array("yolo_real_sub_skip_words", pack_act_words(skip, UP_H, UP_W))}

{format_u32_array("yolo_real_sub_wgt0_words", pack_wgt_words(weights16, OC0, IC0))}

{format_u32_array("yolo_real_sub_wgt1_words", pack_wgt_words(weights16, OC1, IC1))}

static const int32_t yolo_real_sub_bias0[{OC0}] = {{
    {", ".join("0" for _ in range(OC0))}
}};

static const int32_t yolo_real_sub_bias1[{OC1}] = {{
    {", ".join("0" for _ in range(OC1))}
}};

{format_i32_array("yolo_real_sub_bias0_q", bias_q)}

{format_i32_array("yolo_real_sub_bias1_q", bias_q)}

{format_u32_flat_array("yolo_real_sub_scale_mul0", scale_mul)}

{format_u32_flat_array("yolo_real_sub_scale_mul1", scale_mul)}

{format_u32_flat_array("yolo_real_sub_scale_shift0", scale_shift)}

{format_u32_flat_array("yolo_real_sub_scale_shift1", scale_shift)}

static const uint8_t yolo_real_sub_expected[{SP1}][{OC1}] = {{
"""
    for pos in range(SP1):
        body += "    {" + ", ".join(f"{int(v)}u" for v in expected[pos]) + "},\n"
    body += "};\n\n"
    body += f"static const uint8_t yolo_real_sub_expected_cref[{SP1}][{OC1}] = {{\n"
    for pos in range(SP1):
        body += "    {" + ", ".join(f"0x{int(v):02X}u" for v in expected_cref[pos]) + "},\n"
    body += """};

#endif
"""

    OUT_PATH.write_text(body, encoding="ascii")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
