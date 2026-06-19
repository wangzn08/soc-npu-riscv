#!/usr/bin/env python3
"""Generate conv2 fixture for a conv0->conv1->conv2 strip-chain smoke."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
OUT_PATH = ROOT / "firmware" / "yolo_conv2_from_conv1_chain_data.h"
CONV0_GEN = ROOT / "tools" / "gen_yolo_conv0_strip_real_smoke.py"
CONV1_GEN = ROOT / "tools" / "gen_yolo_conv1_from_conv0_chain_smoke.py"

IN_W = 160
IN_H = 16
IC = 32
OC = 32
KH = 1
KW = 1
STRIDE = 1
PAD = 0
OUT_W = IN_W
OUT_H = IN_H
SP_OUT = OUT_W * OUT_H
Q_SHIFT = 20
IN_SCALE = 0.6557820439
IN_ZP = -128
OUT_SCALE = 0.1601515412
OUT_ZP = -126
REQUANT_SHIFT = 12
# The golden uses the software RTL approximation of conv1 as conv2 input, while
# the smoke feeds conv2 with the actual RTL conv1 output. Allow one extra chained
# quantization step beyond the single-layer tolerance.
RTL_TOL = 64


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def pack_wgt_words(conv0, weights: np.ndarray) -> list[list[int]]:
    words: list[list[int]] = []
    ic_groups = IC // 16
    for oc in range(OC):
        for group in range(ic_groups):
            lo = group * 16
            words.append(conv0.pack_i8_word(weights[oc, lo:lo + 16, 0, 0]))
    return words


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
            raise SystemExit(f"conv2 oc{oc} produced zero scale multiplier")
        scale_mul.append(mul)
        bias_equiv = int(round(float(biases[oc]) * 16.0 * (1 << Q_SHIFT) / mul))
        bias_int.append(int(bias_equiv - IN_ZP * int(wsum[oc])))
    return bias_int, scale_mul, [Q_SHIFT for _ in range(OC)]


def conv2_rtl_model(conv0, act: np.ndarray, weights: np.ndarray,
                    bias_q: list[int], scale_mul: list[int],
                    lut: list[int]) -> np.ndarray:
    out = np.zeros((SP_OUT, OC), dtype=np.int8)
    requant_mul = int(round((1 << REQUANT_SHIFT) / (16.0 * OUT_SCALE)))
    for y in range(IN_H):
        for x in range(IN_W):
            pos = y * IN_W + x
            for oc in range(OC):
                acc = 0
                for ic in range(IC):
                    acc += int(act[ic, y, x]) * int(weights[oc, ic, 0, 0])
                q44 = ((acc + bias_q[oc]) * scale_mul[oc]) >> Q_SHIFT
                silu_byte = conv0.rtl_silu_byte(q44, lut)
                rq = ((conv0.s8(silu_byte) * requant_mul) >> REQUANT_SHIFT) + OUT_ZP
                out[pos, oc] = np.int8(conv0.s8(conv0.clamp_s8(rq)))
    return out


def make_conv1_output(conv0, conv1) -> np.ndarray:
    conv0_weights = np.fromfile(WEIGHT_DIR / "conv0_w.bin", dtype=np.int8).reshape(
        conv0.OC, conv0.IC, conv0.KH, conv0.KW
    )
    conv0_biases = np.fromfile(WEIGHT_DIR / "conv0_b.bin", dtype=np.float32)
    conv0_wscales = np.fromfile(WEIGHT_DIR / "conv0_s.bin", dtype=np.float32)
    conv0_act = conv0.make_activation()
    lut = conv0.load_lut()
    conv0_bias_q, conv0_scale_mul, _ = conv0.make_qparams(conv0_weights, conv0_biases, conv0_wscales)
    conv0_out_flat = conv0.conv_rtl_model(conv0_act, conv0_weights, conv0_bias_q, conv0_scale_mul, lut)
    conv0_out = conv0_out_flat.reshape(conv0.OUT_H, conv0.OUT_W, conv0.OC).transpose(2, 0, 1)
    conv1_act = conv0_out[:, :conv1.IN_H, :]

    conv1_weights = np.fromfile(WEIGHT_DIR / "conv1_w.bin", dtype=np.int8).reshape(
        conv1.OC, conv1.IC, conv1.KH, conv1.KW
    )
    conv1_biases = np.fromfile(WEIGHT_DIR / "conv1_b.bin", dtype=np.float32)
    conv1_wscales = np.fromfile(WEIGHT_DIR / "conv1_s.bin", dtype=np.float32)
    conv1_bias_q, conv1_scale_mul, _ = conv1.make_qparams(conv1_weights, conv1_biases, conv1_wscales)
    conv1_flat = conv1.conv1_rtl_model(conv0, conv1_act, conv1_weights,
                                       conv1_bias_q, conv1_scale_mul, lut)
    return conv1_flat.reshape(OUT_H, OUT_W, IC).transpose(2, 0, 1)


def main() -> None:
    conv0 = load_module("conv0_gen", CONV0_GEN)
    conv1 = load_module("conv1_gen", CONV1_GEN)
    lut = conv0.load_lut()
    act = make_conv1_output(conv0, conv1)

    weights = np.fromfile(WEIGHT_DIR / "conv2_w.bin", dtype=np.int8)
    biases = np.fromfile(WEIGHT_DIR / "conv2_b.bin", dtype=np.float32)
    wscales = np.fromfile(WEIGHT_DIR / "conv2_s.bin", dtype=np.float32)
    if weights.size != OC * IC * KH * KW or biases.size < OC or wscales.size < OC:
        raise SystemExit("conv2 asset size mismatch")
    weights = weights.reshape(OC, IC, KH, KW)
    bias_q, scale_mul, scale_shift = make_qparams(weights, biases, wscales)
    requant_mul = int(round((1 << REQUANT_SHIFT) / (16.0 * OUT_SCALE)))
    expected = conv2_rtl_model(conv0, act, weights, bias_q, scale_mul, lut).astype(np.uint8)

    body = f"""#ifndef YOLO_CONV2_FROM_CONV1_CHAIN_DATA_H
#define YOLO_CONV2_FROM_CONV1_CHAIN_DATA_H

#include <stdint.h>

#define YOLO_CONV2_CHAIN_IN_W {IN_W}u
#define YOLO_CONV2_CHAIN_IN_H {IN_H}u
#define YOLO_CONV2_CHAIN_IC {IC}u
#define YOLO_CONV2_CHAIN_OC {OC}u
#define YOLO_CONV2_CHAIN_ACT_WORDS {IN_W * IN_H * (IC // 16)}u
#define YOLO_CONV2_CHAIN_WGT_WORDS {OC * (IC // 16)}u
#define YOLO_CONV2_CHAIN_OUT_W {OUT_W}u
#define YOLO_CONV2_CHAIN_OUT_H {OUT_H}u
#define YOLO_CONV2_CHAIN_OUT_SPATIAL {SP_OUT}u
#define YOLO_CONV2_CHAIN_OUT_WORDS {SP_OUT * (OC // 16)}u
#define YOLO_CONV2_CHAIN_REQUANT_MUL {requant_mul}u
#define YOLO_CONV2_CHAIN_REQUANT_SHIFT {REQUANT_SHIFT}u
#define YOLO_CONV2_CHAIN_REQUANT_ZP {OUT_ZP}
#define YOLO_CONV2_CHAIN_PAD_VALUE {IN_ZP}
#define YOLO_CONV2_CHAIN_RTL_TOL {RTL_TOL}u

{format_u32_array("yolo_conv2_chain_wgt_words", pack_wgt_words(conv0, weights))}

{format_i32_array("yolo_conv2_chain_bias_q", bias_q)}

{format_u32_flat_array("yolo_conv2_chain_scale_mul", scale_mul)}

{format_u32_flat_array("yolo_conv2_chain_scale_shift", scale_shift)}

{format_u8_matrix("yolo_conv2_chain_expected_rtl", expected)}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
