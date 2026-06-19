#!/usr/bin/env python3
"""Generate fixtures for the M5u C2f residual-add smoke.

Chain: conv0 -> conv1 -> conv2 -> slice(s1) -> conv3 -> conv4, then the first
C2f bottleneck shortcut add `/model.2/m.0/Add`: add_out = s1 + conv4_out.

Both operands are requantized to the glue scale/zp, so the HW signed eltwise
adder computes add_out_q = sat_s8(conv4_q + s1_q - glue_zp).

  * conv4_q(glue): conv4 conv -> SiLU -> requant to glue (instead of conv4 own).
  * s1_q(glue):    an extra conv2 group-1 pass (cv1, SiLU) requantized to glue,
                   staged resident in Out SRAM as the eltwise skip source.

This file emits only the NEW pieces (staging conv2-group1 params + glue requant +
expected add). conv4 conv weights/bias/scale are reused from the conv4 header.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
OUT_PATH = ROOT / "firmware" / "yolo_c2f_add_m5u_data.h"
CONV0_GEN = ROOT / "tools" / "gen_yolo_conv0_strip_real_smoke.py"
CONV1_GEN = ROOT / "tools" / "gen_yolo_conv1_from_conv0_chain_smoke.py"
CONV2_GEN = ROOT / "tools" / "gen_yolo_conv2_from_conv1_chain_smoke.py"
CONV3_GEN = ROOT / "tools" / "gen_yolo_conv3_from_conv2_chain_smoke.py"
CONV4_GEN = ROOT / "tools" / "gen_yolo_conv4_from_conv3_chain_smoke.py"

# /model.2/m.0/Add glue quant (yolo_glue_quant[0]).
GLUE_SCALE = 0.1549137533
GLUE_ZP = -124
REQUANT_SHIFT = 12

IN_W = 160
IN_H = 16
SP_OUT = IN_W * IN_H
RTL_TOL = 160


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
    return f"static const int32_t {name}[{len(vals)}] = {{\n    " + \
        ", ".join(str(x) for x in vals) + "\n};"


def format_u32_flat_array(name: str, vals: list[int]) -> str:
    return f"static const uint32_t {name}[{len(vals)}] = {{\n    " + \
        ", ".join(f"{x}u" for x in vals) + "\n};"


def format_u8_matrix(name: str, vals: np.ndarray) -> str:
    lines = [f"static const uint8_t {name}[{vals.shape[0]}][{vals.shape[1]}] = {{"]
    for row in vals:
        lines.append("    {" + ", ".join(f"0x{int(x):02X}u" for x in row) + "},")
    lines.append("};")
    return "\n".join(lines)


def requant_to_glue(conv0, silu_byte: int) -> int:
    """Requantize a Q4.4 SiLU byte to the glue scale/zp (RTL-integer model)."""
    mul = int(round((1 << REQUANT_SHIFT) / (16.0 * GLUE_SCALE)))
    rq = ((conv0.s8(silu_byte) * mul) >> REQUANT_SHIFT) + GLUE_ZP
    return int(conv0.s8(conv0.clamp_s8(rq)))


def conv_silu_to_glue(conv0, act, weights, bias_q, scale_mul, lut,
                      kh, kw, pad, in_h, in_w, in_zp):
    """3x3/1x1 conv (stride 1) -> Q4.4 -> SiLU -> requant to glue. Returns
    [SP_OUT, OC] int8 at the glue scale."""
    oc_n = weights.shape[0]
    ic_n = weights.shape[1]
    out = np.zeros((SP_OUT, oc_n), dtype=np.int8)
    q_shift = 20
    for oy in range(in_h):
        for ox in range(in_w):
            pos = oy * in_w + ox
            for oc in range(oc_n):
                acc = 0
                for ic in range(ic_n):
                    for ky in range(kh):
                        for kx in range(kw):
                            iy = oy + ky - pad
                            ix = ox + kx - pad
                            av = int(act[ic, iy, ix]) if 0 <= iy < in_h and 0 <= ix < in_w else in_zp
                            acc += av * int(weights[oc, ic, ky, kx])
                q44 = ((acc + bias_q[oc]) * scale_mul[oc]) >> q_shift
                silu_byte = conv0.rtl_silu_byte(q44, lut)
                out[pos, oc] = np.int8(requant_to_glue(conv0, silu_byte))
    return out


def pack_pw_wgt_words(conv0, weights: np.ndarray) -> list[list[int]]:
    """Pack 1x1 weights [OC, IC, 1, 1] as OC * (IC/16) tile-major words."""
    oc_n = weights.shape[0]
    ic_n = weights.shape[1]
    words: list[list[int]] = []
    for oc in range(oc_n):
        for group in range(ic_n // 16):
            lo = group * 16
            words.append(conv0.pack_i8_word(weights[oc, lo:lo + 16, 0, 0]))
    return words


def compute_pieces():
    """Recompute the C2f-branch tensors shared by M5u and M5v.

    Returns (conv0_mod, conv2_out[32ch @conv2 scale], s1_glue[SP,16],
             conv4_glue[SP,16], add_out[SP,16], stage params dict)."""
    conv0 = load_module("conv0_gen", CONV0_GEN)
    conv1 = load_module("conv1_gen", CONV1_GEN)
    conv2 = load_module("conv2_gen", CONV2_GEN)
    conv3 = load_module("conv3_gen", CONV3_GEN)
    conv4 = load_module("conv4_gen", CONV4_GEN)
    lut = conv0.load_lut()

    conv1_out = conv2.make_conv1_output(conv0, conv1)

    conv2_w = np.fromfile(WEIGHT_DIR / "conv2_w.bin", dtype=np.int8).reshape(
        conv2.OC, conv2.IC, conv2.KH, conv2.KW)
    conv2_b = np.fromfile(WEIGHT_DIR / "conv2_b.bin", dtype=np.float32)
    conv2_s = np.fromfile(WEIGHT_DIR / "conv2_s.bin", dtype=np.float32)
    conv2_bias_q, conv2_scale_mul, conv2_shift = conv2.make_qparams(conv2_w, conv2_b, conv2_s)
    conv2_out = conv2.conv2_rtl_model(conv0, conv1_out, conv2_w,
                                      conv2_bias_q, conv2_scale_mul, lut)
    conv2_out = conv2_out.reshape(IN_H, IN_W, 32).transpose(2, 0, 1)

    s1_act = conv2_out[16:32, :, :]
    conv3_w = np.fromfile(WEIGHT_DIR / "conv3_w.bin", dtype=np.int8).reshape(
        conv3.OC, conv3.IC, conv3.KH, conv3.KW)
    conv3_b = np.fromfile(WEIGHT_DIR / "conv3_b.bin", dtype=np.float32)
    conv3_s = np.fromfile(WEIGHT_DIR / "conv3_s.bin", dtype=np.float32)
    conv3_bias_q, conv3_scale_mul, _ = conv3.make_qparams(conv3_w, conv3_b, conv3_s)
    conv3_flat = conv3.conv3_rtl_model(conv0, s1_act, conv3_w,
                                       conv3_bias_q, conv3_scale_mul, lut)
    conv4_act = conv3_flat.reshape(IN_H, IN_W, 16).transpose(2, 0, 1)

    conv4_w = np.fromfile(WEIGHT_DIR / "conv4_w.bin", dtype=np.int8).reshape(
        conv4.OC, conv4.IC, conv4.KH, conv4.KW)
    conv4_b = np.fromfile(WEIGHT_DIR / "conv4_b.bin", dtype=np.float32)
    conv4_s = np.fromfile(WEIGHT_DIR / "conv4_s.bin", dtype=np.float32)
    conv4_bias_q, conv4_scale_mul, _ = conv4.make_qparams(conv4_w, conv4_b, conv4_s)
    conv4_glue = conv_silu_to_glue(conv0, conv4_act, conv4_w, conv4_bias_q,
                                   conv4_scale_mul, lut, 3, 3, 1, IN_H, IN_W,
                                   conv4.IN_ZP)

    stage_w = conv2_w[16:32, :, :, :]
    stage = {
        "weights": stage_w,
        "bias_q": conv2_bias_q[16:32],
        "scale_mul": conv2_scale_mul[16:32],
        "shift": conv2_shift[16:32],
        "in_zp": conv2.IN_ZP,
    }
    s1_glue = conv_silu_to_glue(conv0, conv1_out, stage_w, stage["bias_q"],
                                stage["scale_mul"], lut, 1, 1, 0, IN_H, IN_W,
                                conv2.IN_ZP)

    add_out = np.zeros((SP_OUT, 16), dtype=np.int8)
    for pos in range(SP_OUT):
        for oc in range(16):
            v = int(conv4_glue[pos, oc]) + int(s1_glue[pos, oc]) - GLUE_ZP
            add_out[pos, oc] = np.int8(max(-128, min(127, v)))
    return conv0, conv2_out, s1_glue, conv4_glue, add_out, stage


def main() -> None:
    conv0 = load_module("conv0_gen", CONV0_GEN)
    conv1 = load_module("conv1_gen", CONV1_GEN)
    conv2 = load_module("conv2_gen", CONV2_GEN)
    conv3 = load_module("conv3_gen", CONV3_GEN)
    conv4 = load_module("conv4_gen", CONV4_GEN)
    lut = conv0.load_lut()

    # conv1 output (32ch) = conv2 input act.
    conv1_out = conv2.make_conv1_output(conv0, conv1)

    # conv2 (cv1) qparams + full output (at conv2 scale) for the conv3 branch.
    conv2_w = np.fromfile(WEIGHT_DIR / "conv2_w.bin", dtype=np.int8).reshape(
        conv2.OC, conv2.IC, conv2.KH, conv2.KW)
    conv2_b = np.fromfile(WEIGHT_DIR / "conv2_b.bin", dtype=np.float32)
    conv2_s = np.fromfile(WEIGHT_DIR / "conv2_s.bin", dtype=np.float32)
    conv2_bias_q, conv2_scale_mul, conv2_shift = conv2.make_qparams(conv2_w, conv2_b, conv2_s)
    conv2_out = conv2.conv2_rtl_model(conv0, conv1_out, conv2_w,
                                      conv2_bias_q, conv2_scale_mul, lut)
    conv2_out = conv2_out.reshape(IN_H, IN_W, 32).transpose(2, 0, 1)

    # s1 = conv2 group-1 (channels 16..31) at conv2 scale -> conv3 input.
    s1_act = conv2_out[16:32, :, :]
    conv3_w = np.fromfile(WEIGHT_DIR / "conv3_w.bin", dtype=np.int8).reshape(
        conv3.OC, conv3.IC, conv3.KH, conv3.KW)
    conv3_b = np.fromfile(WEIGHT_DIR / "conv3_b.bin", dtype=np.float32)
    conv3_s = np.fromfile(WEIGHT_DIR / "conv3_s.bin", dtype=np.float32)
    conv3_bias_q, conv3_scale_mul, _ = conv3.make_qparams(conv3_w, conv3_b, conv3_s)
    conv3_flat = conv3.conv3_rtl_model(conv0, s1_act, conv3_w,
                                       conv3_bias_q, conv3_scale_mul, lut)
    conv4_act = conv3_flat.reshape(IN_H, IN_W, 16).transpose(2, 0, 1)

    # conv4 conv -> SiLU -> requant to GLUE (reuse conv4 conv qparams).
    conv4_w = np.fromfile(WEIGHT_DIR / "conv4_w.bin", dtype=np.int8).reshape(
        conv4.OC, conv4.IC, conv4.KH, conv4.KW)
    conv4_b = np.fromfile(WEIGHT_DIR / "conv4_b.bin", dtype=np.float32)
    conv4_s = np.fromfile(WEIGHT_DIR / "conv4_s.bin", dtype=np.float32)
    conv4_bias_q, conv4_scale_mul, _ = conv4.make_qparams(conv4_w, conv4_b, conv4_s)
    conv4_glue = conv_silu_to_glue(conv0, conv4_act, conv4_w, conv4_bias_q,
                                   conv4_scale_mul, lut, 3, 3, 1, IN_H, IN_W,
                                   conv4.IN_ZP)

    # staging: conv2 group-1 (cv1) -> SiLU -> requant to GLUE = s1(glue).
    stage_w = conv2_w[16:32, :, :, :]                # [16, 32, 1, 1]
    stage_bias_q = conv2_bias_q[16:32]
    stage_scale_mul = conv2_scale_mul[16:32]
    stage_shift = conv2_shift[16:32]
    s1_glue = conv_silu_to_glue(conv0, conv1_out, stage_w, stage_bias_q,
                                stage_scale_mul, lut, 1, 1, 0, IN_H, IN_W,
                                conv2.IN_ZP)

    # residual add through the HW signed eltwise: conv4_q + s1_q - glue_zp.
    add_out = np.zeros((SP_OUT, 16), dtype=np.int8)
    for pos in range(SP_OUT):
        for oc in range(16):
            v = int(conv4_glue[pos, oc]) + int(s1_glue[pos, oc]) - GLUE_ZP
            v = max(-128, min(127, v))
            add_out[pos, oc] = np.int8(v)
    expected = add_out.astype(np.uint8)

    glue_requant_mul = int(round((1 << REQUANT_SHIFT) / (16.0 * GLUE_SCALE)))

    body = f"""#ifndef YOLO_C2F_ADD_M5U_DATA_H
#define YOLO_C2F_ADD_M5U_DATA_H

#include <stdint.h>

/* /model.2/m.0/Add glue quant: out_scale={GLUE_SCALE}, out_zp={GLUE_ZP}. */
#define YOLO_C2F_ADD_GLUE_REQUANT_MUL {glue_requant_mul}u
#define YOLO_C2F_ADD_GLUE_REQUANT_SHIFT {REQUANT_SHIFT}u
#define YOLO_C2F_ADD_GLUE_ZP {GLUE_ZP}

#define YOLO_C2F_ADD_IN_W {IN_W}u
#define YOLO_C2F_ADD_IN_H {IN_H}u
#define YOLO_C2F_ADD_OUT_SPATIAL {SP_OUT}u
#define YOLO_C2F_ADD_OC 16u
#define YOLO_C2F_ADD_RTL_TOL {RTL_TOL}u

/* Staging pass = conv2 group-1 (cv1): IC=32, OC=16, 1x1, requant-to-glue. */
#define YOLO_C2F_STAGE_IN_W {IN_W}u
#define YOLO_C2F_STAGE_IN_H {IN_H}u
#define YOLO_C2F_STAGE_IC 32u
#define YOLO_C2F_STAGE_OC 16u
#define YOLO_C2F_STAGE_ACT_WORDS {SP_OUT * 2}u
#define YOLO_C2F_STAGE_WGT_WORDS {16 * 2}u
#define YOLO_C2F_STAGE_PAD_VALUE {conv2.IN_ZP}

{format_u32_array("yolo_c2f_stage_wgt_words", pack_pw_wgt_words(conv0, stage_w))}

{format_i32_array("yolo_c2f_stage_bias_q", stage_bias_q)}

{format_u32_flat_array("yolo_c2f_stage_scale_mul", stage_scale_mul)}

{format_u32_flat_array("yolo_c2f_stage_scale_shift", stage_shift)}

{format_u8_matrix("yolo_c2f_add_expected_rtl", expected)}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
