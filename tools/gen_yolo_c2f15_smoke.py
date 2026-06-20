#!/usr/bin/env python3
"""c2f_15 (model.15, PAN p3, n=1, shortcut=0) via the generic C2f runner.
Input = cat2 = concat(upsample2x(fpn_mid=c2f_12 out), p4=c2f_4 out) requant to
conv31 in, baked (80x8 spatial). cv1=conv31(192->64), bn=conv32/33(32->32 3x3,
no residual), cv2=conv34(96->64). Output = pan_p3."""

from __future__ import annotations
from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
OUT_PATH = ROOT / "firmware" / "yolo_c2f15_data.h"
C2F4_GEN = ROOT / "tools" / "gen_yolo_c2f4_smoke.py"
C2F12_GEN = ROOT / "tools" / "gen_yolo_c2f12_smoke.py"

IN_W, IN_H = 80, 8
SP = IN_W * IN_H
CAT_SHIFT = 16

AQ = {
    31: (0.0337357186, -120, 0.0244056843, -117),  # cv1 (in=/model.14/Concat)
    32: (0.0244056843, -117, 0.0180846192, -113),
    33: (0.0180846192, -113, 0.0324782133, -119),  # m_cv2 own out (shortcut=0)
    34: (0.0324782133, -119, 0.0308109522, -119),  # cv2 (in=/model.15/Concat)
}
FPN_MID = (0.0303480383, -119)   # c2f_12 out (conv30 out)
P4 = (0.0337357186, -120)        # c2f_4 out (conv12 out)
CAT1 = (0.0337357186, -120)      # conv31 in
CAT2 = (0.0324782133, -119)      # conv34 in


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def upsample2x(x): return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def requant(conv0, q, in_zp, src, dst, dst_zp):
    mul = int(round((src/dst) * (1 << CAT_SHIFT)))
    out = np.zeros_like(q, dtype=np.int8)
    for idx in np.ndindex(q.shape):
        v = (((int(q[idx]) - in_zp) * mul) >> CAT_SHIFT) + dst_zp
        out[idx] = np.int8(conv0.s8(conv0.clamp_s8(v)))
    return out, mul


def pack_act(conv0, act):
    C = act.shape[0]; out = []
    for g in range(C//16):
        for pos in range(SP):
            y, x = pos//IN_W, pos%IN_W
            out.append(conv0.pack_i8_word(act[g*16:g*16+16, y, x]))
    return out


def Wt(name, shape): return np.fromfile(WEIGHT_DIR/name, dtype=np.int8).reshape(*shape)


def main():
    g = load("c2f4", C2F4_GEN)
    c2f12 = load("c2f12", C2F12_GEN)
    conv0, fpn_mid = c2f12.main()        # [160,128] @ conv30 out (40x4)
    _, p4 = g.compute_c2f4()             # [640,64] @ conv12 out (80x8)
    lut = conv0.load_lut()

    fpn_chw = fpn_mid.reshape(4, 40, 128).transpose(2, 0, 1)   # [128,4,40]
    up = upsample2x(fpn_chw)                                    # [128,8,80]
    upf = up.transpose(1, 2, 0).reshape(SP, 128)
    upc, _ = requant(conv0, upf, FPN_MID[1], FPN_MID[0], CAT1[0], CAT1[1])
    p4f = p4.reshape(SP, 64)
    p4c, _ = requant(conv0, p4f, P4[1], P4[0], CAT1[0], CAT1[1])
    cat = np.concatenate([upc, p4c], axis=1).reshape(IN_H, IN_W, 192).transpose(2, 0, 1)  # [192,8,80]

    w31 = Wt("conv31_w.bin", (64, 192, 1, 1))
    bq31, m31, sh31 = g.make_qparams(conv0, w31, g.fw("conv31_b.bin"), g.fw("conv31_s.bin"), AQ[31][0], AQ[31][1])
    cv1o, rq31 = g.conv_rtl(conv0, cat, w31, bq31, m31, lut, 1, 1, 1, 0, AQ[31][1], AQ[31][2], AQ[31][3])
    cv1 = cv1o.reshape(IN_H, IN_W, 64).transpose(2, 0, 1); s0 = cv1[0:32]; s1 = cv1[32:64]

    w32 = Wt("conv32_w.bin", (32, 32, 3, 3)); bq32, m32, sh32 = g.make_qparams(conv0, w32, g.fw("conv32_b.bin"), g.fw("conv32_s.bin"), AQ[32][0], AQ[32][1])
    c32, rq32 = g.conv_rtl(conv0, s1, w32, bq32, m32, lut, 3, 3, 1, 1, AQ[32][1], AQ[32][2], AQ[32][3]); c32m = c32.reshape(IN_H, IN_W, 32).transpose(2, 0, 1)
    w33 = Wt("conv33_w.bin", (32, 32, 3, 3)); bq33, m33, sh33 = g.make_qparams(conv0, w33, g.fw("conv33_b.bin"), g.fw("conv33_s.bin"), AQ[33][0], AQ[33][1])
    add0, rq33 = g.conv_rtl(conv0, c32m, w33, bq33, m33, lut, 3, 3, 1, 1, AQ[33][1], AQ[33][2], AQ[33][3])

    s0f = s0.transpose(1, 2, 0).reshape(SP, 32); s1f = s1.transpose(1, 2, 0).reshape(SP, 32)
    s0c, cm_s = requant(conv0, s0f, AQ[31][3], AQ[31][2], CAT2[0], CAT2[1])
    s1c, _ = requant(conv0, s1f, AQ[31][3], AQ[31][2], CAT2[0], CAT2[1])
    a0c, cm_a0 = requant(conv0, add0, AQ[33][3], AQ[33][2], CAT2[0], CAT2[1])
    concat = np.concatenate([s0c, s1c, a0c], axis=1).reshape(IN_H, IN_W, 96).transpose(2, 0, 1)
    w34 = Wt("conv34_w.bin", (64, 96, 1, 1)); bq34, m34, sh34 = g.make_qparams(conv0, w34, g.fw("conv34_b.bin"), g.fw("conv34_s.bin"), CAT2[0], CAT2[1])
    golden, rq34 = g.conv_rtl(conv0, concat, w34, bq34, m34, lut, 1, 1, 1, 0, CAT2[1], AQ[34][2], AQ[34][3])

    u32a = lambda n, ws: "\n".join([f"static const uint32_t {n}[{len(ws)}][4] = {{"]+["    {"+", ".join(f"0x{v:08X}u" for v in x)+"}," for x in ws]+["};"])
    body = f"""#ifndef YOLO_C2F15_DATA_H
#define YOLO_C2F15_DATA_H

#include <stdint.h>

#define YOLO_C2F15_IN_W {IN_W}u
#define YOLO_C2F15_IN_H {IN_H}u
#define YOLO_C2F15_SPATIAL {SP}u
#define YOLO_C2F15_FULL_C 64u
#define YOLO_C2F15_CV1_IC 192u
#define YOLO_C2F15_CV2_IC 96u
#define YOLO_C2F15_CV2_OC 64u
#define YOLO_C2F15_RTL_TOL 16u

#define YOLO_C2F15_BLKIN_WORDS {SP*12}u
{u32a("yolo_c2f15_blkin_words", pack_act(conv0, cat))}

#define YOLO_C2F15_CV1_WGT_WORDS {64*12}u
#define YOLO_C2F15_CV1_RQ_MUL {rq31}u
#define YOLO_C2F15_CV1_RQ_SHIFT {g.REQ_SHIFT}u
#define YOLO_C2F15_CV1_RQ_ZP {AQ[31][3]}
{u32a("yolo_c2f15_cv1_wgt", g.pack1x1(conv0, w31))}
{g.fmt_i32("yolo_c2f15_cv1_bias", bq31)}
{g.fmt_u32("yolo_c2f15_cv1_mul", m31)}
{g.fmt_u32("yolo_c2f15_cv1_shift", sh31)}

#define YOLO_C2F15_MCV1_WGT_WORDS {32*2*9}u
{u32a("yolo_c2f15_mcv1_0_wgt", g.pack3x3(conv0, w32))}
{g.fmt_i32("yolo_c2f15_mcv1_0_bias", bq32)}
{g.fmt_u32("yolo_c2f15_mcv1_0_mul", m32)}
{g.fmt_u32("yolo_c2f15_mcv1_0_shift", sh32)}
#define YOLO_C2F15_MCV1_0_RQ_MUL {rq32}u
#define YOLO_C2F15_MCV1_0_RQ_ZP {AQ[32][3]}
#define YOLO_C2F15_MCV1_0_PAD {AQ[32][1]}

#define YOLO_C2F15_MCV2_WGT_WORDS {32*2*9}u
{u32a("yolo_c2f15_mcv2_0_wgt", g.pack3x3(conv0, w33))}
{g.fmt_i32("yolo_c2f15_mcv2_0_bias", bq33)}
{g.fmt_u32("yolo_c2f15_mcv2_0_mul", m33)}
{g.fmt_u32("yolo_c2f15_mcv2_0_shift", sh33)}
#define YOLO_C2F15_MCV2_0_PAD {AQ[33][1]}
#define YOLO_C2F15_GLUE0_RQ_MUL {rq33}u
#define YOLO_C2F15_GLUE0_ZP {AQ[33][3]}
#define YOLO_C2F15_GLUE_RQ_SHIFT {g.REQ_SHIFT}u

#define YOLO_C2F15_CAT_SHIFT {CAT_SHIFT}u
#define YOLO_C2F15_CAT_ZP {CAT2[1]}
#define YOLO_C2F15_CAT_MUL_S0S1 {cm_s}u
#define YOLO_C2F15_CAT_INZP_S0S1 {AQ[31][3]}
#define YOLO_C2F15_CAT_MUL_ADD0 {cm_a0}u
#define YOLO_C2F15_CAT_INZP_ADD0 {AQ[33][3]}

#define YOLO_C2F15_CV2_WGT_WORDS {64*6}u
#define YOLO_C2F15_CV2_RQ_MUL {rq34}u
#define YOLO_C2F15_CV2_RQ_SHIFT {g.REQ_SHIFT}u
#define YOLO_C2F15_CV2_RQ_ZP {AQ[34][3]}
{u32a("yolo_c2f15_cv2_wgt", g.pack1x1(conv0, w34))}
{g.fmt_i32("yolo_c2f15_cv2_bias", bq34)}
{g.fmt_u32("yolo_c2f15_cv2_mul", m34)}
{g.fmt_u32("yolo_c2f15_cv2_shift", sh34)}

{g.fmt_u8("yolo_c2f15_expected_rtl", golden.astype(np.uint8))}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return conv0, golden


if __name__ == "__main__":
    main()
