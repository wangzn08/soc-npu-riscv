#!/usr/bin/env python3
"""c2f_12 (model.12, FPN, n=1, shortcut=0) via the generic C2f runner. Input =
cat1 = concat(upsample2x(SPPF out), p5) requantized to conv27 in-scale, baked in.
cv1=conv27(384->128), bn0=conv28/29(64->64 3x3, NO residual), cv2=conv30(192->128).
Validates the runner's shortcut=0 path. SPPF out (model.9) is p5-level 20x2 ->
upsample 40x4; p5 = c2f_6 output (40x4x128)."""

from __future__ import annotations
from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
OUT_PATH = ROOT / "firmware" / "yolo_c2f12_data.h"
C2F4_GEN = ROOT / "tools" / "gen_yolo_c2f4_smoke.py"
C2F6_GEN = ROOT / "tools" / "gen_yolo_c2f6_smoke.py"
SPPF_GEN = ROOT / "tools" / "gen_yolo_sppf_smoke.py"

IN_W, IN_H = 40, 4           # cat1 spatial (after upsample)
SP = IN_W * IN_H
CAT_SHIFT = 16

# conv indices for c2f_12
AQ = {
    27: (0.0468565077, -122, 0.0443715937, -122),  # cv1 (in = /model.11/Concat)
    28: (0.0443715937, -122, 0.0349353999, -120),
    29: (0.0349353999, -120, 0.0368834250, -120),  # m_cv2 own out (shortcut=0 keeps it)
    30: (0.0443715937, -122, 0.0303480383, -119),  # cv2 (in = /model.12/Concat)
}
SPPF_OUT = (0.0468565077, -122)   # conv26 out
P5 = (0.0331330635, -120)         # c2f_6 out (conv19 out)
CAT1 = (0.0468565077, -122)       # conv27 in (/model.11/Concat)
CAT2 = (0.0443715937, -122)       # conv30 in (/model.12/Concat)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def upsample2x(x):  # [C,H,W] -> [C,2H,2W] nearest
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def requant(conv0, q, in_zp, src_scale, dst_scale, dst_zp):
    mul = int(round((src_scale/dst_scale) * (1 << CAT_SHIFT)))
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
    c2f6 = load("c2f6", C2F6_GEN)
    sppf = load("sppf", SPPF_GEN)
    conv0, sppf_out = sppf.main()       # [40,256] @ SPPF out (20x2 spatial)
    _, p5_out = c2f6.compute_c2f6()     # [160,128] @ c2f_6 out (40x4 spatial)
    lut = conv0.load_lut()

    # up1 = upsample2x(SPPF out 20x2 -> 40x4); SPPF out scale == conv27 in => identity
    sppf_chw = sppf_out.reshape(2, 20, 256).transpose(2, 0, 1)   # [256,2,20]
    up1 = upsample2x(sppf_chw)                                    # [256,4,40]
    up1f = up1.transpose(1, 2, 0).reshape(SP, 256)
    up1c, cm_up = requant(conv0, up1f, SPPF_OUT[1], SPPF_OUT[0], CAT1[0], CAT1[1])
    # p5 (c2f_6 out 40x4x128) requant to conv27 in
    p5f = p5_out.reshape(IN_H, IN_W, 128).transpose(2, 0, 1).transpose(1, 2, 0).reshape(SP, 128)
    p5c, cm_p5 = requant(conv0, p5f, P5[1], P5[0], CAT1[0], CAT1[1])
    cat1 = np.concatenate([up1c, p5c], axis=1).reshape(IN_H, IN_W, 384).transpose(2, 0, 1)  # [384,4,40]

    # cv1 = conv27 (384->128)
    w27 = Wt("conv27_w.bin", (128, 384, 1, 1))
    bq27, m27, sh27 = g.make_qparams(conv0, w27, g.fw("conv27_b.bin"), g.fw("conv27_s.bin"), AQ[27][0], AQ[27][1])
    cv1o, rq27 = g.conv_rtl(conv0, cat1, w27, bq27, m27, lut, 1, 1, 1, 0, AQ[27][1], AQ[27][2], AQ[27][3])
    cv1 = cv1o.reshape(IN_H, IN_W, 128).transpose(2, 0, 1); s0 = cv1[0:64]; s1 = cv1[64:128]

    # bn0: conv28 -> conv29 (shortcut=0 => add0 = conv29 out @ its own scale)
    w28 = Wt("conv28_w.bin", (64, 64, 3, 3)); bq28, m28, sh28 = g.make_qparams(conv0, w28, g.fw("conv28_b.bin"), g.fw("conv28_s.bin"), AQ[28][0], AQ[28][1])
    c28, rq28 = g.conv_rtl(conv0, s1, w28, bq28, m28, lut, 3, 3, 1, 1, AQ[28][1], AQ[28][2], AQ[28][3]); c28m = c28.reshape(IN_H, IN_W, 64).transpose(2, 0, 1)
    w29 = Wt("conv29_w.bin", (64, 64, 3, 3)); bq29, m29, sh29 = g.make_qparams(conv0, w29, g.fw("conv29_b.bin"), g.fw("conv29_s.bin"), AQ[29][0], AQ[29][1])
    add0, rq29 = g.conv_rtl(conv0, c28m, w29, bq29, m29, lut, 3, 3, 1, 1, AQ[29][1], AQ[29][2], AQ[29][3])  # [SP,64] @ conv29 out

    # concat(s0,s1,add0) -> requant to conv30 in (CAT2); cv2 = conv30
    s0f = s0.transpose(1, 2, 0).reshape(SP, 64); s1f = s1.transpose(1, 2, 0).reshape(SP, 64)
    s0c, cm_s = requant(conv0, s0f, AQ[27][3], AQ[27][2], CAT2[0], CAT2[1])  # conv27 out -> conv30 in
    s1c, _ = requant(conv0, s1f, AQ[27][3], AQ[27][2], CAT2[0], CAT2[1])
    a0c, cm_a0 = requant(conv0, add0, AQ[29][3], AQ[29][2], CAT2[0], CAT2[1])  # conv29 out -> conv30 in
    concat = np.concatenate([s0c, s1c, a0c], axis=1).reshape(IN_H, IN_W, 192).transpose(2, 0, 1)
    w30 = Wt("conv30_w.bin", (128, 192, 1, 1)); bq30, m30, sh30 = g.make_qparams(conv0, w30, g.fw("conv30_b.bin"), g.fw("conv30_s.bin"), CAT2[0], CAT2[1])
    golden, rq30 = g.conv_rtl(conv0, concat, w30, bq30, m30, lut, 1, 1, 1, 0, CAT2[1], AQ[30][2], AQ[30][3])

    u32a = lambda n, ws: "\n".join([f"static const uint32_t {n}[{len(ws)}][4] = {{"]+["    {"+", ".join(f"0x{v:08X}u" for v in x)+"}," for x in ws]+["};"])
    body = f"""#ifndef YOLO_C2F12_DATA_H
#define YOLO_C2F12_DATA_H

#include <stdint.h>

#define YOLO_C2F12_IN_W {IN_W}u
#define YOLO_C2F12_IN_H {IN_H}u
#define YOLO_C2F12_SPATIAL {SP}u
#define YOLO_C2F12_FULL_C 128u
#define YOLO_C2F12_CV1_IC 384u
#define YOLO_C2F12_CV2_IC 192u
#define YOLO_C2F12_CV2_OC 128u
#define YOLO_C2F12_RTL_TOL 16u

#define YOLO_C2F12_BLKIN_WORDS {SP*24}u
{u32a("yolo_c2f12_blkin_words", pack_act(conv0, cat1))}

#define YOLO_C2F12_CV1_WGT_WORDS {128*24}u
#define YOLO_C2F12_CV1_RQ_MUL {rq27}u
#define YOLO_C2F12_CV1_RQ_SHIFT {g.REQ_SHIFT}u
#define YOLO_C2F12_CV1_RQ_ZP {AQ[27][3]}
{u32a("yolo_c2f12_cv1_wgt", g.pack1x1(conv0, w27))}
{g.fmt_i32("yolo_c2f12_cv1_bias", bq27)}
{g.fmt_u32("yolo_c2f12_cv1_mul", m27)}
{g.fmt_u32("yolo_c2f12_cv1_shift", sh27)}

#define YOLO_C2F12_MCV1_WGT_WORDS {64*4*9}u
{u32a("yolo_c2f12_mcv1_0_wgt", g.pack3x3(conv0, w28))}
{g.fmt_i32("yolo_c2f12_mcv1_0_bias", bq28)}
{g.fmt_u32("yolo_c2f12_mcv1_0_mul", m28)}
{g.fmt_u32("yolo_c2f12_mcv1_0_shift", sh28)}
#define YOLO_C2F12_MCV1_0_RQ_MUL {rq28}u
#define YOLO_C2F12_MCV1_0_RQ_ZP {AQ[28][3]}
#define YOLO_C2F12_MCV1_0_PAD {AQ[28][1]}

#define YOLO_C2F12_MCV2_WGT_WORDS {64*4*9}u
{u32a("yolo_c2f12_mcv2_0_wgt", g.pack3x3(conv0, w29))}
{g.fmt_i32("yolo_c2f12_mcv2_0_bias", bq29)}
{g.fmt_u32("yolo_c2f12_mcv2_0_mul", m29)}
{g.fmt_u32("yolo_c2f12_mcv2_0_shift", sh29)}
#define YOLO_C2F12_MCV2_0_PAD {AQ[29][1]}
/* shortcut=0: m_cv2 requant target = conv29 own out scale */
#define YOLO_C2F12_GLUE0_RQ_MUL {rq29}u
#define YOLO_C2F12_GLUE0_ZP {AQ[29][3]}
#define YOLO_C2F12_GLUE_RQ_SHIFT {g.REQ_SHIFT}u

#define YOLO_C2F12_CAT_SHIFT {CAT_SHIFT}u
#define YOLO_C2F12_CAT_ZP {CAT2[1]}
#define YOLO_C2F12_CAT_MUL_S0S1 {cm_s}u
#define YOLO_C2F12_CAT_INZP_S0S1 {AQ[27][3]}
#define YOLO_C2F12_CAT_MUL_ADD0 {cm_a0}u
#define YOLO_C2F12_CAT_INZP_ADD0 {AQ[29][3]}

#define YOLO_C2F12_CV2_WGT_WORDS {128*12}u
#define YOLO_C2F12_CV2_RQ_MUL {rq30}u
#define YOLO_C2F12_CV2_RQ_SHIFT {g.REQ_SHIFT}u
#define YOLO_C2F12_CV2_RQ_ZP {AQ[30][3]}
{u32a("yolo_c2f12_cv2_wgt", g.pack1x1(conv0, w30))}
{g.fmt_i32("yolo_c2f12_cv2_bias", bq30)}
{g.fmt_u32("yolo_c2f12_cv2_mul", m30)}
{g.fmt_u32("yolo_c2f12_cv2_shift", sh30)}

{g.fmt_u8("yolo_c2f12_expected_rtl", golden.astype(np.uint8))}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return conv0, golden


if __name__ == "__main__":
    main()
