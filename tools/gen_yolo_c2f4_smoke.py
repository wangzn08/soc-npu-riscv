#!/usr/bin/env python3
"""Generate fixtures for c2f_4 (model.4) via the generic C2f runner.

C2f: cv1=conv7(64->64 1x1) -> split s0,s1(32) -> bn0: conv8/conv9(3x3) +residual
-> bn1: conv10/conv11(3x3) +residual -> concat(s0,s1,add0,add1)=128 -> cv2=conv12
(128->64 1x1). Residual adds and concat requant are CPU-side (faithful to the C
reference); convs are the RTL-integer model. Block input = conv6 output (80x8x64).
"""

from __future__ import annotations
from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
OUT_PATH = ROOT / "firmware" / "yolo_c2f4_data.h"
M5W_GEN = ROOT / "tools" / "gen_yolo_conv6_from_c2f_m5w_smoke.py"

IN_W, IN_H = 80, 8
SP = IN_W * IN_H
Q_SHIFT = 20
REQ_SHIFT = 12          # conv SiLU-requant shift
CAT_SHIFT = 16          # CPU requant fixed-point shift
RATIO_SHIFT = 16        # residual-add ratio shift
RTL_TOL = 16

# act_quant in/out scales (index = conv id).
AQ = {
    7:  (0.0334292874, -120, 0.0397584029, -121),  # cv1
    8:  (0.0397584029, -121, 0.0204943288, -114),
    9:  (0.0204943288, -114, 0.0271009300, -118),  # m_cv2[0] own out (unused; requant to glue2)
    10: (0.0365898795, -113, 0.0219360031, -115),
    11: (0.0219360031, -115, 0.0483939648, -122),  # m_cv2[1] own out (unused; requant to glue3)
    12: (0.0560306832, -113, 0.0337357186, -120),  # cv2
}
GLUE2 = (0.0365898795, -113)   # /model.4/m.0/Add
GLUE3 = (0.0560306832, -113)   # /model.4/m.1/Add
CAT   = (0.0560306832, -113)   # /model.4/Concat = conv12 in


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def make_qparams(conv0, w, b, s, in_scale, in_zp):
    wsum = w.astype(np.int32).sum(axis=(1, 2, 3))
    bias_q, mul = [], []
    for oc in range(w.shape[0]):
        m = int(round(in_scale * float(s[oc]) * 16.0 * (1 << Q_SHIFT)))
        if m == 0:
            raise SystemExit("zero mul")
        mul.append(m)
        be = int(round(float(b[oc]) * 16.0 * (1 << Q_SHIFT) / m))
        bias_q.append(int(be - in_zp * int(wsum[oc])))
    return bias_q, mul, [Q_SHIFT] * w.shape[0]


def conv_rtl(conv0, act, w, bias_q, mul, lut, kh, kw, stride, pad, in_zp,
             out_scale, out_zp):
    """conv -> q44 -> SiLU LUT -> requant to (out_scale,out_zp). [SP_out, OC] int8."""
    OC, IC = w.shape[0], w.shape[1]
    out_h = (act.shape[1] + 2*pad - kh)//stride + 1
    out_w = (act.shape[2] + 2*pad - kw)//stride + 1
    rqm = int(round((1 << REQ_SHIFT) / (16.0 * out_scale)))
    out = np.zeros((out_h*out_w, OC), dtype=np.int8)
    for oy in range(out_h):
        for ox in range(out_w):
            pos = oy*out_w + ox
            for oc in range(OC):
                acc = 0
                for ic in range(IC):
                    for ky in range(kh):
                        for kx in range(kw):
                            iy = oy*stride + ky - pad; ix = ox*stride + kx - pad
                            av = int(act[ic, iy, ix]) if 0 <= iy < act.shape[1] and 0 <= ix < act.shape[2] else in_zp
                            acc += av * int(w[oc, ic, ky, kx])
                q44 = ((acc + bias_q[oc]) * mul[oc]) >> Q_SHIFT
                sb = conv0.rtl_silu_byte(q44, lut)
                rq = ((conv0.s8(sb) * rqm) >> REQ_SHIFT) + out_zp
                out[pos, oc] = np.int8(conv0.s8(conv0.clamp_s8(rq)))
    return out, rqm


def cpu_add(conv0, prev, conv, prev_scale, prev_zp, glue_scale):
    """add = clamp(round((prev-prev_zp)*ratio>>sh) + conv). [N,C] int8."""
    rmul = int(round((prev_scale/glue_scale) * (1 << RATIO_SHIFT)))
    out = np.zeros_like(prev, dtype=np.int8)
    for idx in np.ndindex(prev.shape):
        v = (((int(prev[idx]) - prev_zp) * rmul) >> RATIO_SHIFT) + int(conv[idx])
        out[idx] = np.int8(conv0.s8(conv0.clamp_s8(v)))
    return out, rmul


def cat_requant(conv0, q, in_zp, src_scale):
    mul = int(round((src_scale/CAT[0]) * (1 << CAT_SHIFT)))
    out = np.zeros_like(q, dtype=np.int8)
    for idx in np.ndindex(q.shape):
        v = (((int(q[idx]) - in_zp) * mul) >> CAT_SHIFT) + CAT[1]
        out[idx] = np.int8(conv0.s8(conv0.clamp_s8(v)))
    return out, mul


def ld(name): return np.fromfile(WEIGHT_DIR / name, dtype=np.int8), 0
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


def pack1x1(conv0, w):
    OC, IC = w.shape[0], w.shape[1]; out = []
    for oc in range(OC):
        for g in range(IC//16): out.append(conv0.pack_i8_word(w[oc, g*16:g*16+16, 0, 0]))
    return out
def pack3x3(conv0, w):
    OC, IC = w.shape[0], w.shape[1]; out = []
    for oc in range(OC):
        for g in range(IC//16):
            for ko in range(9):
                out.append(conv0.pack_i8_word(w[oc, g*16:g*16+16, ko//3, ko%3]))
    return out
def pack_act(conv0, act):
    C = act.shape[0]; out = []
    for g in range(C//16):
        for pos in range(SP):
            y, x = pos//IN_W, pos%IN_W
            out.append(conv0.pack_i8_word(act[g*16:g*16+16, y, x]))
    return out


def compute_c2f4():
    m5w = load("m5w", M5W_GEN)
    conv0, conv6_out = m5w.compute_conv6()   # [640,64] at conv6 out
    lut = conv0.load_lut()
    blkin = conv6_out.reshape(IN_H, IN_W, 64).transpose(2, 0, 1)  # [64,8,80]

    # cv1 = conv7
    w7 = np.fromfile(WEIGHT_DIR/"conv7_w.bin", dtype=np.int8).reshape(64,64,1,1)
    bq7, m7, sh7 = make_qparams(conv0, w7, fw("conv7_b.bin"), fw("conv7_s.bin"), AQ[7][0], AQ[7][1])
    cv1_out, rq7 = conv_rtl(conv0, blkin, w7, bq7, m7, lut, 1,1,1,0, AQ[7][1], AQ[7][2], AQ[7][3])
    cv1 = cv1_out.reshape(IN_H, IN_W, 64).transpose(2,0,1)
    s0 = cv1[0:32]; s1 = cv1[32:64]

    # bottleneck 0: conv8 -> conv9 (requant to glue2)
    w8 = np.fromfile(WEIGHT_DIR/"conv8_w.bin", dtype=np.int8).reshape(32,32,3,3)
    bq8, m8, sh8 = make_qparams(conv0, w8, fw("conv8_b.bin"), fw("conv8_s.bin"), AQ[8][0], AQ[8][1])
    c8, rq8 = conv_rtl(conv0, s1, w8, bq8, m8, lut, 3,3,1,1, AQ[8][1], AQ[8][2], AQ[8][3])
    c8m = c8.reshape(IN_H, IN_W, 32).transpose(2,0,1)
    w9 = np.fromfile(WEIGHT_DIR/"conv9_w.bin", dtype=np.int8).reshape(32,32,3,3)
    bq9, m9, sh9 = make_qparams(conv0, w9, fw("conv9_b.bin"), fw("conv9_s.bin"), AQ[9][0], AQ[9][1])
    c9, rq9 = conv_rtl(conv0, c8m, w9, bq9, m9, lut, 3,3,1,1, AQ[9][1], GLUE2[0], GLUE2[1])
    s1_flat = s1.transpose(1,2,0).reshape(SP,32)
    add0, ratio0 = cpu_add(conv0, s1_flat, c9, AQ[7][2], AQ[7][3], GLUE2[0])  # prev=s1 @conv7out

    # bottleneck 1: conv10(add0) -> conv11 (requant to glue3)
    add0m = add0.reshape(IN_H, IN_W, 32).transpose(2,0,1)
    w10 = np.fromfile(WEIGHT_DIR/"conv10_w.bin", dtype=np.int8).reshape(32,32,3,3)
    bq10, m10, sh10 = make_qparams(conv0, w10, fw("conv10_b.bin"), fw("conv10_s.bin"), AQ[10][0], AQ[10][1])
    c10, rq10 = conv_rtl(conv0, add0m, w10, bq10, m10, lut, 3,3,1,1, AQ[10][1], AQ[10][2], AQ[10][3])
    c10m = c10.reshape(IN_H, IN_W, 32).transpose(2,0,1)
    w11 = np.fromfile(WEIGHT_DIR/"conv11_w.bin", dtype=np.int8).reshape(32,32,3,3)
    bq11, m11, sh11 = make_qparams(conv0, w11, fw("conv11_b.bin"), fw("conv11_s.bin"), AQ[11][0], AQ[11][1])
    c11, rq11 = conv_rtl(conv0, c10m, w11, bq11, m11, lut, 3,3,1,1, AQ[11][1], GLUE3[0], GLUE3[1])
    add1, ratio1 = cpu_add(conv0, add0, c11, GLUE2[0], GLUE2[1], GLUE3[0])    # prev=add0 @glue2

    # concat -> requant to CAT; cv2 = conv12
    s0_flat = s0.transpose(1,2,0).reshape(SP,32)
    s0c, cm_s = cat_requant(conv0, s0_flat, AQ[7][3], AQ[7][2])
    s1c, _    = cat_requant(conv0, s1_flat, AQ[7][3], AQ[7][2])
    a0c, cm_a0= cat_requant(conv0, add0, GLUE2[1], GLUE2[0])
    a1c, cm_a1= cat_requant(conv0, add1, GLUE3[1], GLUE3[0])
    concat = np.concatenate([s0c, s1c, a0c, a1c], axis=1)  # [SP,128]
    concat_chw = concat.reshape(IN_H, IN_W, 128).transpose(2,0,1)
    w12 = np.fromfile(WEIGHT_DIR/"conv12_w.bin", dtype=np.int8).reshape(64,128,1,1)
    bq12, m12, sh12 = make_qparams(conv0, w12, fw("conv12_b.bin"), fw("conv12_s.bin"), CAT[0], CAT[1])
    golden, rq12 = conv_rtl(conv0, concat_chw, w12, bq12, m12, lut, 1,1,1,0, CAT[1], AQ[12][2], AQ[12][3])

    ratio_mul = [ratio0, ratio1]
    body = f"""#ifndef YOLO_C2F4_DATA_H
#define YOLO_C2F4_DATA_H

#include <stdint.h>

#define YOLO_C2F4_IN_W {IN_W}u
#define YOLO_C2F4_IN_H {IN_H}u
#define YOLO_C2F4_SPATIAL {SP}u
#define YOLO_C2F4_FULL_C 64u
#define YOLO_C2F4_CV1_IC 64u
#define YOLO_C2F4_CV2_IC 128u
#define YOLO_C2F4_CV2_OC 64u
#define YOLO_C2F4_RTL_TOL {RTL_TOL}u

#define YOLO_C2F4_BLKIN_WORDS {SP*4}u
{fmt_u32_arr("yolo_c2f4_blkin_words", pack_act(conv0, blkin))}

/* cv1 = conv7 */
#define YOLO_C2F4_CV1_WGT_WORDS {64*4}u
#define YOLO_C2F4_CV1_RQ_MUL {rq7}u
#define YOLO_C2F4_CV1_RQ_SHIFT {REQ_SHIFT}u
#define YOLO_C2F4_CV1_RQ_ZP {AQ[7][3]}
{fmt_u32_arr("yolo_c2f4_cv1_wgt", pack1x1(conv0, w7))}
{fmt_i32("yolo_c2f4_cv1_bias", bq7)}
{fmt_u32("yolo_c2f4_cv1_mul", m7)}
{fmt_u32("yolo_c2f4_cv1_shift", sh7)}

/* bottleneck m_cv1: conv8, conv10 (3x3) */
#define YOLO_C2F4_MCV1_WGT_WORDS {32*2*9}u
{fmt_u32_arr("yolo_c2f4_mcv1_0_wgt", pack3x3(conv0, w8))}
{fmt_i32("yolo_c2f4_mcv1_0_bias", bq8)}
{fmt_u32("yolo_c2f4_mcv1_0_mul", m8)}
{fmt_u32("yolo_c2f4_mcv1_0_shift", sh8)}
#define YOLO_C2F4_MCV1_0_RQ_MUL {rq8}u
#define YOLO_C2F4_MCV1_0_RQ_ZP {AQ[8][3]}
#define YOLO_C2F4_MCV1_0_PAD {AQ[8][1]}
#define YOLO_C2F4_MCV1_1_PAD {AQ[10][1]}
#define YOLO_C2F4_MCV2_0_PAD {AQ[9][1]}
#define YOLO_C2F4_MCV2_1_PAD {AQ[11][1]}
{fmt_u32_arr("yolo_c2f4_mcv1_1_wgt", pack3x3(conv0, w10))}
{fmt_i32("yolo_c2f4_mcv1_1_bias", bq10)}
{fmt_u32("yolo_c2f4_mcv1_1_mul", m10)}
{fmt_u32("yolo_c2f4_mcv1_1_shift", sh10)}
#define YOLO_C2F4_MCV1_1_RQ_MUL {rq10}u
#define YOLO_C2F4_MCV1_1_RQ_ZP {AQ[10][3]}

/* bottleneck m_cv2: conv9, conv11 (3x3, requant to glue) */
#define YOLO_C2F4_MCV2_WGT_WORDS {32*2*9}u
{fmt_u32_arr("yolo_c2f4_mcv2_0_wgt", pack3x3(conv0, w9))}
{fmt_i32("yolo_c2f4_mcv2_0_bias", bq9)}
{fmt_u32("yolo_c2f4_mcv2_0_mul", m9)}
{fmt_u32("yolo_c2f4_mcv2_0_shift", sh9)}
{fmt_u32_arr("yolo_c2f4_mcv2_1_wgt", pack3x3(conv0, w11))}
{fmt_i32("yolo_c2f4_mcv2_1_bias", bq11)}
{fmt_u32("yolo_c2f4_mcv2_1_mul", m11)}
{fmt_u32("yolo_c2f4_mcv2_1_shift", sh11)}

/* glue requant (m_cv2 target) */
#define YOLO_C2F4_GLUE0_RQ_MUL {rq9}u
#define YOLO_C2F4_GLUE0_ZP {GLUE2[1]}
#define YOLO_C2F4_GLUE1_RQ_MUL {rq11}u
#define YOLO_C2F4_GLUE1_ZP {GLUE3[1]}
#define YOLO_C2F4_GLUE_RQ_SHIFT {REQ_SHIFT}u

/* residual add ratios (prev_scale/glue_scale) */
#define YOLO_C2F4_ADD_RATIO_SHIFT {RATIO_SHIFT}u
#define YOLO_C2F4_ADD0_RATIO_MUL {ratio0}u
#define YOLO_C2F4_ADD0_PREV_ZP {AQ[7][3]}
#define YOLO_C2F4_ADD1_RATIO_MUL {ratio1}u
#define YOLO_C2F4_ADD1_PREV_ZP {GLUE2[1]}

/* concat requant to cv2 in-scale */
#define YOLO_C2F4_CAT_SHIFT {CAT_SHIFT}u
#define YOLO_C2F4_CAT_ZP {CAT[1]}
#define YOLO_C2F4_CAT_MUL_S0S1 {cm_s}u
#define YOLO_C2F4_CAT_INZP_S0S1 {AQ[7][3]}
#define YOLO_C2F4_CAT_MUL_ADD0 {cm_a0}u
#define YOLO_C2F4_CAT_INZP_ADD0 {GLUE2[1]}
#define YOLO_C2F4_CAT_MUL_ADD1 {cm_a1}u
#define YOLO_C2F4_CAT_INZP_ADD1 {GLUE3[1]}

/* cv2 = conv12 */
#define YOLO_C2F4_CV2_WGT_WORDS {64*8}u
#define YOLO_C2F4_CV2_RQ_MUL {rq12}u
#define YOLO_C2F4_CV2_RQ_SHIFT {REQ_SHIFT}u
#define YOLO_C2F4_CV2_RQ_ZP {AQ[12][3]}
{fmt_u32_arr("yolo_c2f4_cv2_wgt", pack1x1(conv0, w12))}
{fmt_i32("yolo_c2f4_cv2_bias", bq12)}
{fmt_u32("yolo_c2f4_cv2_mul", m12)}
{fmt_u32("yolo_c2f4_cv2_shift", sh12)}

{fmt_u8("yolo_c2f4_expected_rtl", golden.astype(np.uint8))}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH}  ratios={ratio_mul} cat_mul=({cm_s},{cm_a0},{cm_a1})")
    return conv0, golden   # golden = c2f_4 output [SP,64] int8


def main():
    compute_c2f4()


if __name__ == "__main__":
    main()
