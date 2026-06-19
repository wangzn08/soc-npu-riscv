#!/usr/bin/env python3
"""Generate conv1 fixture for a conv0->conv1 strip-chain smoke."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
OUT_PATH = ROOT / "firmware" / "yolo_conv1_from_conv0_chain_data.h"
CONV0_GEN = ROOT / "tools" / "gen_yolo_conv0_strip_real_smoke.py"

IN_W = 320
IN_H = 32
IC = 16
OC = 32
KH = 3
KW = 3
STRIDE = 2
PAD = 1
OUT_W = ((IN_W + PAD * 2 - KW) // STRIDE) + 1
OUT_H = ((IN_H + PAD * 2 - KH) // STRIDE) + 1
SP_OUT = OUT_W * OUT_H
KO = KH * KW
Q_SHIFT = 20
IN_SCALE = 0.2352971584
IN_ZP = -127
OUT_SCALE = 0.6557820439
OUT_ZP = -128
REQUANT_SHIFT = 12
RTL_TOL = 40


def load_conv0_module():
    spec = importlib.util.spec_from_file_location("conv0_gen", CONV0_GEN)
    if spec is None or spec.loader is None:
        raise SystemExit("failed to load conv0 generator")
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
    for oc in range(OC):
        for ko in range(KO):
            ky = ko // KW
            kx = ko % KW
            words.append(conv0.pack_i8_word(weights[oc, :, ky, kx]))
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
            raise SystemExit(f"conv1 oc{oc} produced zero scale multiplier")
        scale_mul.append(mul)
        bias_equiv = int(round(float(biases[oc]) * 16.0 * (1 << Q_SHIFT) / mul))
        bias_int.append(int(bias_equiv - IN_ZP * int(wsum[oc])))
    return bias_int, scale_mul, [Q_SHIFT for _ in range(OC)]


def conv1_rtl_model(conv0, act: np.ndarray, weights: np.ndarray,
                    bias_q: list[int], scale_mul: list[int],
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
                silu_byte = conv0.rtl_silu_byte(q44, lut)
                rq = ((conv0.s8(silu_byte) * requant_mul) >> REQUANT_SHIFT) + OUT_ZP
                out[pos, oc] = np.int8(conv0.s8(conv0.clamp_s8(rq)))
    return out


def main() -> None:
    conv0 = load_conv0_module()
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
    act = conv0_out[:, :IN_H, :]

    weights = np.fromfile(WEIGHT_DIR / "conv1_w.bin", dtype=np.int8)
    biases = np.fromfile(WEIGHT_DIR / "conv1_b.bin", dtype=np.float32)
    wscales = np.fromfile(WEIGHT_DIR / "conv1_s.bin", dtype=np.float32)
    if weights.size != OC * IC * KH * KW or biases.size < OC or wscales.size < OC:
        raise SystemExit("conv1 asset size mismatch")
    weights = weights.reshape(OC, IC, KH, KW)
    bias_q, scale_mul, scale_shift = make_qparams(weights, biases, wscales)
    requant_mul = int(round((1 << REQUANT_SHIFT) / (16.0 * OUT_SCALE)))
    expected = conv1_rtl_model(conv0, act, weights, bias_q, scale_mul, lut).astype(np.uint8)

    body = f"""#ifndef YOLO_CONV1_FROM_CONV0_CHAIN_DATA_H
#define YOLO_CONV1_FROM_CONV0_CHAIN_DATA_H

#include <stdint.h>

#define YOLO_CONV1_CHAIN_IN_W {IN_W}u
#define YOLO_CONV1_CHAIN_IN_H {IN_H}u
#define YOLO_CONV1_CHAIN_IC {IC}u
#define YOLO_CONV1_CHAIN_OC {OC}u
#define YOLO_CONV1_CHAIN_KH {KH}u
#define YOLO_CONV1_CHAIN_KW {KW}u
#define YOLO_CONV1_CHAIN_STRIDE {STRIDE}u
#define YOLO_CONV1_CHAIN_PAD {PAD}u
#define YOLO_CONV1_CHAIN_ACT_WORDS {IN_W * IN_H}u
#define YOLO_CONV1_CHAIN_WGT_WORDS {OC * KO}u
#define YOLO_CONV1_CHAIN_OUT_W {OUT_W}u
#define YOLO_CONV1_CHAIN_OUT_H {OUT_H}u
#define YOLO_CONV1_CHAIN_OUT_SPATIAL {SP_OUT}u
#define YOLO_CONV1_CHAIN_OUT_WORDS {SP_OUT * (OC // 16)}u
#define YOLO_CONV1_CHAIN_REQUANT_MUL {requant_mul}u
#define YOLO_CONV1_CHAIN_REQUANT_SHIFT {REQUANT_SHIFT}u
#define YOLO_CONV1_CHAIN_REQUANT_ZP {OUT_ZP}
#define YOLO_CONV1_CHAIN_RTL_TOL {RTL_TOL}u

{format_u32_array("yolo_conv1_chain_wgt_words", pack_wgt_words(conv0, weights))}

{format_i32_array("yolo_conv1_chain_bias_q", bias_q)}

{format_u32_flat_array("yolo_conv1_chain_scale_mul", scale_mul)}

{format_u32_flat_array("yolo_conv1_chain_scale_shift", scale_shift)}

{format_u8_matrix("yolo_conv1_chain_expected_rtl", expected)}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
