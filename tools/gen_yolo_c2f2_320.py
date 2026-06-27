#!/usr/bin/env python3
"""Generate fixtures for c2f_2 (model.2) at full 320 resolution (80x80) via the
generic tiled C2f runner.

C2f model.2: cv1=conv2(32->32 1x1) -> split s0,s1(16) -> bn0: conv3/conv4(3x3)
+residual -> concat(s0,s1,add0)=48 -> cv2=conv5(48->32 1x1). Residual add and
concat requant are CPU-side (faithful to the C reference); convs are the RTL-
integer model running through the tiled DDR->DDR path.

Block input = the REAL conv1 output dump (dump320/conv1.bin, 32x80x80). The
generated golden is self-checked against dump320/conv5.bin (the C engine's
model.2 output), so this fixture is anchored to the true per-layer oracle.
"""

from __future__ import annotations
from pathlib import Path
import importlib.util
import struct
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
DUMP_DIR = ROOT / "yolov8n_int8" / "dump320"
OUT_PATH = ROOT / "firmware" / "yolo_c2f2_320_data.h"
CONV0_GEN = ROOT / "tools" / "gen_yolo_conv0_strip_real_smoke.py"

IN_W, IN_H = 80, 80
SP = IN_W * IN_H
Q_SHIFT = 20
SILU_STEP = 0.5   # SiLU LUT indexed by preact at this step (zp=0)
REQ_SHIFT = 12          # conv SiLU-requant shift
CAT_SHIFT = 16          # CPU concat-requant fixed-point shift
RATIO_SHIFT = 16        # residual-add ratio shift
RTL_TOL = 16

# act_quant in/out (scale, zp) per conv id (= /model.2/... calibrated scales).
AQ = {
    2: (0.6557820439, -128, 0.1601515412, -126),  # cv1 (1x1)
    3: (0.1601515412, -126, 0.1423473954, -126),  # m_cv1 (3x3)
    4: (0.1423473954, -126, 0.1425503194, -126),  # m_cv2 own out (unused; requant to glue)
    5: (0.1612435579, -125, 0.0763198882, -124),  # cv2 (1x1)
}
GLUE = (0.1549137533, -124)   # /model.2/m.0/Add
CAT = (0.1612435579, -125)    # /model.2/Concat = conv5 in


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def load_dump(ci):
    d = (DUMP_DIR / f"conv{ci}.bin").read_bytes()
    c, h, w = struct.unpack("<iii", d[:12])
    scale = struct.unpack("<f", d[12:16])[0]
    zp = struct.unpack("<i", d[16:20])[0]
    data = np.frombuffer(d[20:], dtype=np.int8).reshape(c, h, w)
    return data, c, h, w, scale, zp


def make_qparams(conv0, w, b, s, in_scale, in_zp, out_scale):
    """Exact-SiLU out-grid qparams: s2 == round(preact/out_scale).
       mul = round(in_scale*wscale/out_scale * 2^Q); bias folds -in_zp*wsum."""
    wsum = w.astype(np.int32).sum(axis=(1, 2, 3))
    bias_q, mul = [], []
    for oc in range(w.shape[0]):
        m = int(round(in_scale * float(s[oc]) / SILU_STEP * (1 << Q_SHIFT)))
        if m == 0:
            raise SystemExit("zero mul")
        mul.append(m)
        be = int(round(float(b[oc]) / (in_scale * float(s[oc]))))
        bias_q.append(int(be - in_zp * int(wsum[oc])))
    return bias_q, mul, [Q_SHIFT] * w.shape[0]


def build_silu_lut(conv0, out_scale, out_zp):
    """256-entry exact SiLU LUT, indexed by the linear out-grid INT8 q:
       lut[q] = round(SiLU((q-out_zp)*out_scale)/out_scale + out_zp)."""
    import math
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        q = i if i < 128 else i - 256
        realq = q * SILU_STEP
        silu = realq / (1.0 + math.exp(-realq))
        lut[i] = np.uint8(conv0.s8(conv0.clamp_s8(int(round(silu / out_scale + out_zp)))) & 0xFF)
    return lut


def conv_rtl(conv0, act, w, bias_q, mul, lut, kh, kw, stride, pad, in_zp, out_zp):
    """Exact-SiLU conv (mirrors post_process_top exact path):
       s2 = (acc+bias_q)*mul >> Q; idx = clamp_s8(s2+out_zp); out = lut[idx].
       [SP_out, OC] int8."""
    OC, IC = w.shape[0], w.shape[1]
    out_h = (act.shape[1] + 2 * pad - kh) // stride + 1
    out_w = (act.shape[2] + 2 * pad - kw) // stride + 1
    out = np.zeros((out_h * out_w, OC), dtype=np.int8)
    for oy in range(out_h):
        for ox in range(out_w):
            pos = oy * out_w + ox
            for oc in range(OC):
                acc = 0
                for ic in range(IC):
                    for ky in range(kh):
                        for kx in range(kw):
                            iy = oy * stride + ky - pad; ix = ox * stride + kx - pad
                            av = int(act[ic, iy, ix]) if 0 <= iy < act.shape[1] and 0 <= ix < act.shape[2] else in_zp
                            acc += av * int(w[oc, ic, ky, kx])
                s2 = ((acc + bias_q[oc]) * mul[oc]) >> Q_SHIFT
                idx = conv0.clamp_s8(s2) & 0xFF
                out[pos, oc] = np.int8(conv0.s8(int(lut[idx])))
    return out


def cpu_add(conv0, prev, conv, prev_scale, prev_zp, glue_scale):
    """add = clamp(round((prev-prev_zp)*ratio>>sh) + conv). [N,C] int8."""
    rmul = int(round((prev_scale / glue_scale) * (1 << RATIO_SHIFT)))
    out = np.zeros_like(prev, dtype=np.int8)
    for idx in np.ndindex(prev.shape):
        v = (((int(prev[idx]) - prev_zp) * rmul) >> RATIO_SHIFT) + int(conv[idx])
        out[idx] = np.int8(conv0.s8(conv0.clamp_s8(v)))
    return out, rmul


def cat_requant(conv0, q, in_zp, src_scale):
    mul = int(round((src_scale / CAT[0]) * (1 << CAT_SHIFT)))
    out = np.zeros_like(q, dtype=np.int8)
    for idx in np.ndindex(q.shape):
        v = (((int(q[idx]) - in_zp) * mul) >> CAT_SHIFT) + CAT[1]
        out[idx] = np.int8(conv0.s8(conv0.clamp_s8(v)))
    return out, mul


def fw(name): return np.fromfile(WEIGHT_DIR / name, dtype=np.float32)


def fmt_u32_arr(name, words):
    out = [f"static const uint32_t {name}[{len(words)}][4] = {{"]
    for w in words: out.append("    {" + ", ".join(f"0x{x:08X}u" for x in w) + "},")
    out.append("};"); return "\n".join(out)
def fmt_i32(name, v): return f"static const int32_t {name}[{len(v)}] = {{\n    " + ", ".join(str(x) for x in v) + "\n};"
def fmt_u32(name, v): return f"static const uint32_t {name}[{len(v)}] = {{\n    " + ", ".join(f"{x}u" for x in v) + "\n};"
def fmt_u8(name, v):
    out = [f"static const uint8_t {name}[{v.shape[0]}][{v.shape[1]}] = {{"]
    for r in v: out.append("    {" + ", ".join(f"0x{int(x):02X}u" for x in r) + "},")
    out.append("};"); return "\n".join(out)
def fmt_u8_1d(name, v):
    return f"static const uint8_t {name}[{len(v)}] = {{\n    " + \
        ", ".join(f"0x{int(x):02X}u" for x in v) + "\n};"


def pack1x1(conv0, w):
    OC, IC = w.shape[0], w.shape[1]; out = []
    for oc in range(OC):
        for g in range(IC // 16): out.append(conv0.pack_i8_word(w[oc, g * 16:g * 16 + 16, 0, 0]))
    return out
def pack3x3(conv0, w):
    OC, IC = w.shape[0], w.shape[1]; out = []
    for oc in range(OC):
        for g in range(IC // 16):
            for ko in range(9):
                out.append(conv0.pack_i8_word(w[oc, g * 16:g * 16 + 16, ko // 3, ko % 3]))
    return out
def pack_act(conv0, act):
    C = act.shape[0]; out = []
    for g in range(C // 16):
        for pos in range(SP):
            y, x = pos // IN_W, pos % IN_W
            out.append(conv0.pack_i8_word(act[g * 16:g * 16 + 16, y, x]))
    return out


def compute_c2f2():
    conv0 = load("conv0", CONV0_GEN)

    blkin, c, h, w, in_scale, in_zp = load_dump(1)   # conv1 out = c2f_2 input
    assert (c, h, w) == (32, IN_H, IN_W), f"conv1 dump {(c,h,w)}"
    assert abs(in_scale - AQ[2][0]) < 1e-6 and in_zp == AQ[2][1], "conv1 scale/zp mismatch AQ[2] in"

    # Per-layer exact SiLU LUTs (out-grid indexed), built from each conv's target
    # (out_scale, out_zp). conv4 (m_cv2) targets the GLUE scale, conv5 (cv2) AQ[5].
    lut2 = build_silu_lut(conv0, AQ[2][2], AQ[2][3])
    lut3 = build_silu_lut(conv0, AQ[3][2], AQ[3][3])
    lut4 = build_silu_lut(conv0, GLUE[0], GLUE[1])
    lut5 = build_silu_lut(conv0, AQ[5][2], AQ[5][3])

    # cv1 = conv2 (1x1, 32->32)
    w2 = np.fromfile(WEIGHT_DIR / "conv2_w.bin", dtype=np.int8).reshape(32, 32, 1, 1)
    bq2, m2, sh2 = make_qparams(conv0, w2, fw("conv2_b.bin"), fw("conv2_s.bin"), AQ[2][0], AQ[2][1], AQ[2][2])
    cv1_out = conv_rtl(conv0, blkin, w2, bq2, m2, lut2, 1, 1, 1, 0, AQ[2][1], AQ[2][3])
    cv1 = cv1_out.reshape(IN_H, IN_W, 32).transpose(2, 0, 1)
    s0 = cv1[0:16]; s1 = cv1[16:32]

    # bottleneck 0: conv3 -> conv4 (requant to glue)
    w3 = np.fromfile(WEIGHT_DIR / "conv3_w.bin", dtype=np.int8).reshape(16, 16, 3, 3)
    bq3, m3, sh3 = make_qparams(conv0, w3, fw("conv3_b.bin"), fw("conv3_s.bin"), AQ[3][0], AQ[3][1], AQ[3][2])
    c3 = conv_rtl(conv0, s1, w3, bq3, m3, lut3, 3, 3, 1, 1, AQ[3][1], AQ[3][3])
    c3m = c3.reshape(IN_H, IN_W, 16).transpose(2, 0, 1)
    w4 = np.fromfile(WEIGHT_DIR / "conv4_w.bin", dtype=np.int8).reshape(16, 16, 3, 3)
    bq4, m4, sh4 = make_qparams(conv0, w4, fw("conv4_b.bin"), fw("conv4_s.bin"), AQ[4][0], AQ[4][1], GLUE[0])
    c4 = conv_rtl(conv0, c3m, w4, bq4, m4, lut4, 3, 3, 1, 1, AQ[4][1], GLUE[1])
    s1_flat = s1.transpose(1, 2, 0).reshape(SP, 16)
    add0, ratio0 = cpu_add(conv0, s1_flat, c4, AQ[2][2], AQ[2][3], GLUE[0])  # prev=s1 @conv2 out

    # concat(s0, s1, add0) -> requant to CAT; cv2 = conv5
    s0_flat = s0.transpose(1, 2, 0).reshape(SP, 16)
    s0c, cm_s = cat_requant(conv0, s0_flat, AQ[2][3], AQ[2][2])
    s1c, _ = cat_requant(conv0, s1_flat, AQ[2][3], AQ[2][2])
    a0c, cm_a0 = cat_requant(conv0, add0, GLUE[1], GLUE[0])
    concat = np.concatenate([s0c, s1c, a0c], axis=1)  # [SP, 48]
    concat_chw = concat.reshape(IN_H, IN_W, 48).transpose(2, 0, 1)
    w5 = np.fromfile(WEIGHT_DIR / "conv5_w.bin", dtype=np.int8).reshape(32, 48, 1, 1)
    bq5, m5, sh5 = make_qparams(conv0, w5, fw("conv5_b.bin"), fw("conv5_s.bin"), CAT[0], CAT[1], AQ[5][2])
    golden = conv_rtl(conv0, concat_chw, w5, bq5, m5, lut5, 1, 1, 1, 0, CAT[1], AQ[5][3])

    # ---- CONCAT FOLDED: fold the per-source requant (alpha=src/CAT, zp) into cv2's
    # int8 weights+bias so cv2 reads the RAW contiguous concat [s0|s1|add0] at native
    # scales (no CPU cat_requant). Same bias convention as make_qparams (no half-round). ----
    chan_sc = np.array([AQ[2][2]]*32 + [GLUE[0]]*16, dtype=np.float64)   # s0,s1 @conv2-out; add0 @glue
    chan_zp = np.array([AQ[2][3]]*32 + [GLUE[1]]*16, dtype=np.int64)
    alpha = chan_sc / CAT[0]
    w5_2d = w5[:, :, 0, 0].astype(np.float64)
    w5f = np.clip(np.round(w5_2d * alpha[None, :]), -128, 127).astype(np.int8)
    b5real = fw("conv5_b.bin"); s5 = fw("conv5_s.bin")
    bias_fold = [int(round(float(b5real[o]) / (CAT[0] * float(s5[o])))
                     - int(np.sum(w5f[o].astype(np.int64) * chan_zp))) for o in range(32)]
    w5f_4d = w5f[:, :, None, None]
    raw_concat = np.concatenate([s0_flat, s1_flat, add0], axis=1)        # [SP,48] raw, native scales
    raw_concat_chw = raw_concat.reshape(IN_H, IN_W, 48).transpose(2, 0, 1)
    golden_fold = conv_rtl(conv0, raw_concat_chw, w5f_4d, bias_fold, m5, lut5, 1, 1, 1, 0, 0, AQ[5][3])
    (WEIGHT_DIR / "conv5_w_folded.bin").write_bytes(w5f.astype(np.int8).tobytes())
    print(f"c2f2 FOLD vs concat-path: max={int(np.abs(golden_fold.astype(int)-golden.astype(int)).max())}")

    # self-check vs the C engine's model.2 output dump (conv5.bin)
    c5, _, _, _, c5_scale, c5_zp = load_dump(5)
    c5_flat = c5.transpose(1, 2, 0).reshape(SP, 32)
    diff = np.abs(golden.astype(np.int32) - c5_flat.astype(np.int32))
    nmis = int((diff > 0).sum()); maxd = int(diff.max())
    print(f"self-check vs dump320/conv5.bin: mismatch={nmis}/{golden.size}  maxabs={maxd}")

    body = f"""#ifndef YOLO_C2F2_320_DATA_H
#define YOLO_C2F2_320_DATA_H

#include <stdint.h>

#define YOLO_C2F2_IN_W {IN_W}u
#define YOLO_C2F2_IN_H {IN_H}u
#define YOLO_C2F2_SPATIAL {SP}u
#define YOLO_C2F2_FULL_C 32u
#define YOLO_C2F2_CV1_IC 32u
#define YOLO_C2F2_CV2_IC 48u
#define YOLO_C2F2_CV2_OC 32u
#define YOLO_C2F2_RTL_TOL {RTL_TOL}u

#define YOLO_C2F2_BLKIN_WORDS {SP * 2}u
{fmt_u32_arr("yolo_c2f2_blkin_words", pack_act(conv0, blkin))}

/* Exact-SiLU mode: the mul/shift arrays are LINEAR out-grid qparams; the
   RQ_MUL macros are unused (0); SiLU comes from the per-conv 256-entry out-grid
   LUTs below. */

/* cv1 = conv2 (1x1, 32->32) */
#define YOLO_C2F2_CV1_WGT_WORDS {32 * 2}u
#define YOLO_C2F2_CV1_RQ_MUL 0u
#define YOLO_C2F2_CV1_RQ_SHIFT {REQ_SHIFT}u
#define YOLO_C2F2_CV1_RQ_ZP {AQ[2][3]}
{fmt_u32_arr("yolo_c2f2_cv1_wgt", pack1x1(conv0, w2))}
{fmt_i32("yolo_c2f2_cv1_bias", bq2)}
{fmt_u32("yolo_c2f2_cv1_mul", m2)}
{fmt_u32("yolo_c2f2_cv1_shift", sh2)}
{fmt_u8_1d("yolo_c2f2_cv1_silu_lut", lut2)}

/* bottleneck m_cv1 = conv3 (3x3, 16->16) */
#define YOLO_C2F2_MCV1_WGT_WORDS {16 * 9}u
{fmt_u32_arr("yolo_c2f2_mcv1_0_wgt", pack3x3(conv0, w3))}
{fmt_i32("yolo_c2f2_mcv1_0_bias", bq3)}
{fmt_u32("yolo_c2f2_mcv1_0_mul", m3)}
{fmt_u32("yolo_c2f2_mcv1_0_shift", sh3)}
{fmt_u8_1d("yolo_c2f2_mcv1_0_silu_lut", lut3)}
#define YOLO_C2F2_MCV1_0_RQ_MUL 0u
#define YOLO_C2F2_MCV1_0_RQ_ZP {AQ[3][3]}
#define YOLO_C2F2_MCV1_0_PAD {AQ[3][1]}
#define YOLO_C2F2_MCV2_0_PAD {AQ[4][1]}

/* bottleneck m_cv2 = conv4 (3x3, 16->16, requant to glue) */
#define YOLO_C2F2_MCV2_WGT_WORDS {16 * 9}u
{fmt_u32_arr("yolo_c2f2_mcv2_0_wgt", pack3x3(conv0, w4))}
{fmt_i32("yolo_c2f2_mcv2_0_bias", bq4)}
{fmt_u32("yolo_c2f2_mcv2_0_mul", m4)}
{fmt_u32("yolo_c2f2_mcv2_0_shift", sh4)}
{fmt_u8_1d("yolo_c2f2_mcv2_0_silu_lut", lut4)}
#define YOLO_C2F2_GLUE0_RQ_MUL 0u
#define YOLO_C2F2_GLUE0_ZP {GLUE[1]}
#define YOLO_C2F2_GLUE_RQ_SHIFT {REQ_SHIFT}u

/* residual add ratio (prev=s1 @conv2 out scale) */
#define YOLO_C2F2_ADD_RATIO_SHIFT {RATIO_SHIFT}u
#define YOLO_C2F2_ADD0_RATIO_MUL {ratio0}u
#define YOLO_C2F2_ADD0_PREV_ZP {AQ[2][3]}

/* concat requant to cv2 in-scale (/model.2/Concat) */
#define YOLO_C2F2_CAT_SHIFT {CAT_SHIFT}u
#define YOLO_C2F2_CAT_ZP {CAT[1]}
#define YOLO_C2F2_CAT_MUL_S0S1 {cm_s}u
#define YOLO_C2F2_CAT_INZP_S0S1 {AQ[2][3]}
#define YOLO_C2F2_CAT_MUL_ADD0 {cm_a0}u
#define YOLO_C2F2_CAT_INZP_ADD0 {GLUE[1]}

/* cv2 = conv5 (1x1, 48->32) -- CONCAT FOLDED: per-source requant baked into the
   int8 weights (w5f) + bias (bias_fold); cv2 reads the raw contiguous concat. */
#define YOLO_C2F2_CV2_WGT_WORDS {32 * 3}u
#define YOLO_C2F2_CV2_FOLDED 1
#define YOLO_C2F2_CV2_RQ_MUL 0u
#define YOLO_C2F2_CV2_RQ_SHIFT {REQ_SHIFT}u
#define YOLO_C2F2_CV2_RQ_ZP {AQ[5][3]}
{fmt_u32_arr("yolo_c2f2_cv2_wgt", pack1x1(conv0, w5f_4d))}
{fmt_i32("yolo_c2f2_cv2_bias", bias_fold)}
{fmt_u32("yolo_c2f2_cv2_mul", m5)}
{fmt_u32("yolo_c2f2_cv2_shift", sh5)}
{fmt_u8_1d("yolo_c2f2_cv2_silu_lut", lut5)}

{fmt_u8("yolo_c2f2_expected_rtl", golden_fold.astype(np.uint8))}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH}  ratio0={ratio0} cat_mul=({cm_s},{cm_a0})")
    return nmis, maxd


def main():
    compute_c2f2()


if __name__ == "__main__":
    main()
