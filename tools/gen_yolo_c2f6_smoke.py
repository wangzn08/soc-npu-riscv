#!/usr/bin/env python3
"""c2f_6 (model.6, n=2, 128ch) via the generic C2f runner. Input = conv13 output
(40x4x128). cv1=conv14(128->128), bn0=conv15/16, bn1=conv17/18 (64->64 3x3),
cv2=conv19(256->128). Exercises the runner's OC>64 cv1/cv2 chunked path."""

from __future__ import annotations
from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
OUT_PATH = ROOT / "firmware" / "yolo_c2f6_data.h"
C2F4_GEN = ROOT / "tools" / "gen_yolo_c2f4_smoke.py"
CONV13_GEN = ROOT / "tools" / "gen_yolo_conv13_m6b_smoke.py"

IN_W, IN_H = 40, 4
SP = IN_W * IN_H
CAT_SHIFT = 16

AQ = {
    14: (0.0319866650, -119, 0.0365007035, -120),  # cv1
    15: (0.0365007035, -120, 0.0307744816, -119),
    16: (0.0307744816, -119, 0.0272509996, -118),  # own out unused (->glue5)
    17: (0.0315765068, -110, 0.0313216858, -119),
    18: (0.0313216858, -119, 0.0618021861, -123),  # own out unused (->glue6)
    19: (0.0707620457, -116, 0.0331330635, -120),  # cv2
}
GLUE5 = (0.0315765068, -110)
GLUE6 = (0.0707620457, -116)
CAT   = (0.0707620457, -116)   # /model.6/Concat = conv19 in


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def pack_act(conv0, act):
    C = act.shape[0]; out = []
    for g in range(C//16):
        for pos in range(SP):
            y, x = pos//IN_W, pos%IN_W
            out.append(conv0.pack_i8_word(act[g*16:g*16+16, y, x]))
    return out


def cat_req(conv0, q, in_zp, src_scale):
    mul = int(round((src_scale/CAT[0]) * (1 << CAT_SHIFT)))
    out = np.zeros_like(q, dtype=np.int8)
    for idx in np.ndindex(q.shape):
        v = (((int(q[idx]) - in_zp) * mul) >> CAT_SHIFT) + CAT[1]
        out[idx] = np.int8(conv0.s8(conv0.clamp_s8(v)))
    return out, mul


def W(name, shape): return np.fromfile(WEIGHT_DIR/name, dtype=np.int8).reshape(*shape)


def main():
    g = load("c2f4", C2F4_GEN)
    c13 = load("conv13", CONV13_GEN)
    conv0, conv13_out = c13.compute_conv13()      # [160,128] @ conv13 out = conv14 in
    lut = conv0.load_lut()
    blkin = conv13_out.reshape(IN_H, IN_W, 128).transpose(2,0,1)  # [128,4,40]

    # cv1 = conv14 (1x1 128->128)
    w14 = W("conv14_w.bin", (128,128,1,1))
    bq14,m14,sh14 = g.make_qparams(conv0,w14,g.fw("conv14_b.bin"),g.fw("conv14_s.bin"),AQ[14][0],AQ[14][1])
    cv1o,rq14 = g.conv_rtl(conv0,blkin,w14,bq14,m14,lut,1,1,1,0,AQ[14][1],AQ[14][2],AQ[14][3])
    cv1 = cv1o.reshape(IN_H,IN_W,128).transpose(2,0,1); s0=cv1[0:64]; s1=cv1[64:128]

    # bn0: conv15 -> conv16 (->glue5)
    w15=W("conv15_w.bin",(64,64,3,3)); bq15,m15,sh15=g.make_qparams(conv0,w15,g.fw("conv15_b.bin"),g.fw("conv15_s.bin"),AQ[15][0],AQ[15][1])
    c15,rq15=g.conv_rtl(conv0,s1,w15,bq15,m15,lut,3,3,1,1,AQ[15][1],AQ[15][2],AQ[15][3]); c15m=c15.reshape(IN_H,IN_W,64).transpose(2,0,1)
    w16=W("conv16_w.bin",(64,64,3,3)); bq16,m16,sh16=g.make_qparams(conv0,w16,g.fw("conv16_b.bin"),g.fw("conv16_s.bin"),AQ[16][0],AQ[16][1])
    c16,rq16=g.conv_rtl(conv0,c15m,w16,bq16,m16,lut,3,3,1,1,AQ[16][1],GLUE5[0],GLUE5[1])
    s1f=s1.transpose(1,2,0).reshape(SP,64); add0,ratio0=g.cpu_add(conv0,s1f,c16,AQ[14][2],AQ[14][3],GLUE5[0])

    # bn1: conv17(add0) -> conv18 (->glue6)
    add0m=add0.reshape(IN_H,IN_W,64).transpose(2,0,1)
    w17=W("conv17_w.bin",(64,64,3,3)); bq17,m17,sh17=g.make_qparams(conv0,w17,g.fw("conv17_b.bin"),g.fw("conv17_s.bin"),AQ[17][0],AQ[17][1])
    c17,rq17=g.conv_rtl(conv0,add0m,w17,bq17,m17,lut,3,3,1,1,AQ[17][1],AQ[17][2],AQ[17][3]); c17m=c17.reshape(IN_H,IN_W,64).transpose(2,0,1)
    w18=W("conv18_w.bin",(64,64,3,3)); bq18,m18,sh18=g.make_qparams(conv0,w18,g.fw("conv18_b.bin"),g.fw("conv18_s.bin"),AQ[18][0],AQ[18][1])
    c18,rq18=g.conv_rtl(conv0,c17m,w18,bq18,m18,lut,3,3,1,1,AQ[18][1],GLUE6[0],GLUE6[1])
    add1,ratio1=g.cpu_add(conv0,add0,c18,GLUE5[0],GLUE5[1],GLUE6[0])

    # concat -> cv2 = conv19 (1x1 256->128)
    s0f=s0.transpose(1,2,0).reshape(SP,64)
    s0c,cm_s=cat_req(conv0,s0f,AQ[14][3],AQ[14][2]); s1c,_=cat_req(conv0,s1f,AQ[14][3],AQ[14][2])
    a0c,cm_a0=cat_req(conv0,add0,GLUE5[1],GLUE5[0]); a1c,cm_a1=cat_req(conv0,add1,GLUE6[1],GLUE6[0])
    concat=np.concatenate([s0c,s1c,a0c,a1c],axis=1).reshape(IN_H,IN_W,256).transpose(2,0,1)
    w19=W("conv19_w.bin",(128,256,1,1)); bq19,m19,sh19=g.make_qparams(conv0,w19,g.fw("conv19_b.bin"),g.fw("conv19_s.bin"),CAT[0],CAT[1])
    golden,rq19=g.conv_rtl(conv0,concat,w19,bq19,m19,lut,1,1,1,0,CAT[1],AQ[19][2],AQ[19][3])

    u32a=lambda n,ws:"\n".join([f"static const uint32_t {n}[{len(ws)}][4] = {{"]+["    {"+", ".join(f"0x{v:08X}u" for v in x)+"},"for x in ws]+["};"])
    body=f"""#ifndef YOLO_C2F6_DATA_H
#define YOLO_C2F6_DATA_H

#include <stdint.h>

#define YOLO_C2F6_IN_W {IN_W}u
#define YOLO_C2F6_IN_H {IN_H}u
#define YOLO_C2F6_SPATIAL {SP}u
#define YOLO_C2F6_FULL_C 128u
#define YOLO_C2F6_CV1_IC 128u
#define YOLO_C2F6_CV2_IC 256u
#define YOLO_C2F6_CV2_OC 128u
#define YOLO_C2F6_RTL_TOL 16u

#define YOLO_C2F6_BLKIN_WORDS {SP*8}u
{u32a("yolo_c2f6_blkin_words", pack_act(conv0, blkin))}

#define YOLO_C2F6_CV1_WGT_WORDS {128*8}u
#define YOLO_C2F6_CV1_RQ_MUL {rq14}u
#define YOLO_C2F6_CV1_RQ_SHIFT {g.REQ_SHIFT}u
#define YOLO_C2F6_CV1_RQ_ZP {AQ[14][3]}
{u32a("yolo_c2f6_cv1_wgt", g.pack1x1(conv0,w14))}
{g.fmt_i32("yolo_c2f6_cv1_bias",bq14)}
{g.fmt_u32("yolo_c2f6_cv1_mul",m14)}
{g.fmt_u32("yolo_c2f6_cv1_shift",sh14)}

#define YOLO_C2F6_MCV1_WGT_WORDS {64*4*9}u
{u32a("yolo_c2f6_mcv1_0_wgt", g.pack3x3(conv0,w15))}
{g.fmt_i32("yolo_c2f6_mcv1_0_bias",bq15)}
{g.fmt_u32("yolo_c2f6_mcv1_0_mul",m15)}
{g.fmt_u32("yolo_c2f6_mcv1_0_shift",sh15)}
#define YOLO_C2F6_MCV1_0_RQ_MUL {rq15}u
#define YOLO_C2F6_MCV1_0_RQ_ZP {AQ[15][3]}
#define YOLO_C2F6_MCV1_0_PAD {AQ[15][1]}
{u32a("yolo_c2f6_mcv1_1_wgt", g.pack3x3(conv0,w17))}
{g.fmt_i32("yolo_c2f6_mcv1_1_bias",bq17)}
{g.fmt_u32("yolo_c2f6_mcv1_1_mul",m17)}
{g.fmt_u32("yolo_c2f6_mcv1_1_shift",sh17)}
#define YOLO_C2F6_MCV1_1_RQ_MUL {rq17}u
#define YOLO_C2F6_MCV1_1_RQ_ZP {AQ[17][3]}
#define YOLO_C2F6_MCV1_1_PAD {AQ[17][1]}

#define YOLO_C2F6_MCV2_WGT_WORDS {64*4*9}u
{u32a("yolo_c2f6_mcv2_0_wgt", g.pack3x3(conv0,w16))}
{g.fmt_i32("yolo_c2f6_mcv2_0_bias",bq16)}
{g.fmt_u32("yolo_c2f6_mcv2_0_mul",m16)}
{g.fmt_u32("yolo_c2f6_mcv2_0_shift",sh16)}
#define YOLO_C2F6_MCV2_0_PAD {AQ[16][1]}
{u32a("yolo_c2f6_mcv2_1_wgt", g.pack3x3(conv0,w18))}
{g.fmt_i32("yolo_c2f6_mcv2_1_bias",bq18)}
{g.fmt_u32("yolo_c2f6_mcv2_1_mul",m18)}
{g.fmt_u32("yolo_c2f6_mcv2_1_shift",sh18)}
#define YOLO_C2F6_MCV2_1_PAD {AQ[18][1]}

#define YOLO_C2F6_GLUE0_RQ_MUL {rq16}u
#define YOLO_C2F6_GLUE0_ZP {GLUE5[1]}
#define YOLO_C2F6_GLUE1_RQ_MUL {rq18}u
#define YOLO_C2F6_GLUE1_ZP {GLUE6[1]}
#define YOLO_C2F6_GLUE_RQ_SHIFT {g.REQ_SHIFT}u

#define YOLO_C2F6_ADD_RATIO_SHIFT {g.RATIO_SHIFT}u
#define YOLO_C2F6_ADD0_RATIO_MUL {ratio0}u
#define YOLO_C2F6_ADD0_PREV_ZP {AQ[14][3]}
#define YOLO_C2F6_ADD1_RATIO_MUL {ratio1}u
#define YOLO_C2F6_ADD1_PREV_ZP {GLUE5[1]}

#define YOLO_C2F6_CAT_SHIFT {CAT_SHIFT}u
#define YOLO_C2F6_CAT_ZP {CAT[1]}
#define YOLO_C2F6_CAT_MUL_S0S1 {cm_s}u
#define YOLO_C2F6_CAT_INZP_S0S1 {AQ[14][3]}
#define YOLO_C2F6_CAT_MUL_ADD0 {cm_a0}u
#define YOLO_C2F6_CAT_INZP_ADD0 {GLUE5[1]}
#define YOLO_C2F6_CAT_MUL_ADD1 {cm_a1}u
#define YOLO_C2F6_CAT_INZP_ADD1 {GLUE6[1]}

#define YOLO_C2F6_CV2_WGT_WORDS {128*16}u
#define YOLO_C2F6_CV2_RQ_MUL {rq19}u
#define YOLO_C2F6_CV2_RQ_SHIFT {g.REQ_SHIFT}u
#define YOLO_C2F6_CV2_RQ_ZP {AQ[19][3]}
{u32a("yolo_c2f6_cv2_wgt", g.pack1x1(conv0,w19))}
{g.fmt_i32("yolo_c2f6_cv2_bias",bq19)}
{g.fmt_u32("yolo_c2f6_cv2_mul",m19)}
{g.fmt_u32("yolo_c2f6_cv2_shift",sh19)}

{g.fmt_u8("yolo_c2f6_expected_rtl", golden.astype(np.uint8))}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH} ratios=({ratio0},{ratio1}) cat=({cm_s},{cm_a0},{cm_a1})")


if __name__ == "__main__":
    main()
