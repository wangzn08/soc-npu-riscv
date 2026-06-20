#!/usr/bin/env python3
"""c2f_8 (model.8, n=1, 256ch) via the generic C2f runner. Input = conv20 output
(20x2x256). cv1=conv21(256->256), bn0=conv22/23(128->128 3x3), cv2=conv24(384->256).
Exercises the runner with half_c=128 (bottleneck OC>64 chunked) + cv1/cv2 OC=256."""

from __future__ import annotations
from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
OUT_PATH = ROOT / "firmware" / "yolo_c2f8_data.h"
C2F4_GEN = ROOT / "tools" / "gen_yolo_c2f4_smoke.py"
CONV20_GEN = ROOT / "tools" / "gen_yolo_conv20_smoke.py"

IN_W, IN_H = 20, 2
SP = IN_W * IN_H
CAT_SHIFT = 16

AQ = {
    21: (0.0388779789, -121, 0.0546551012, -123),  # cv1
    22: (0.0546551012, -123, 0.0524734072, -123),
    23: (0.0524734072, -123, 0.0760652423, -124),  # own out unused (->glue8)
    24: (0.0779053494, -121, 0.0433438867, -122),  # cv2
}
GLUE8 = (0.0779053494, -121)
CAT   = (0.0779053494, -121)   # /model.8/Concat = conv24 in


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
    c20 = load("conv20", CONV20_GEN)
    conv0, conv20_out = c20.main()    # [40,256] @ conv20 out = conv21 in
    lut = conv0.load_lut()
    blkin = conv20_out.reshape(IN_H, IN_W, 256).transpose(2,0,1)  # [256,2,20]

    # cv1 = conv21 (1x1 256->256)
    w21 = W("conv21_w.bin", (256,256,1,1))
    bq21,m21,sh21 = g.make_qparams(conv0,w21,g.fw("conv21_b.bin"),g.fw("conv21_s.bin"),AQ[21][0],AQ[21][1])
    cv1o,rq21 = g.conv_rtl(conv0,blkin,w21,bq21,m21,lut,1,1,1,0,AQ[21][1],AQ[21][2],AQ[21][3])
    cv1 = cv1o.reshape(IN_H,IN_W,256).transpose(2,0,1); s0=cv1[0:128]; s1=cv1[128:256]

    # bn0: conv22 -> conv23 (->glue8)
    w22=W("conv22_w.bin",(128,128,3,3)); bq22,m22,sh22=g.make_qparams(conv0,w22,g.fw("conv22_b.bin"),g.fw("conv22_s.bin"),AQ[22][0],AQ[22][1])
    c22,rq22=g.conv_rtl(conv0,s1,w22,bq22,m22,lut,3,3,1,1,AQ[22][1],AQ[22][2],AQ[22][3]); c22m=c22.reshape(IN_H,IN_W,128).transpose(2,0,1)
    w23=W("conv23_w.bin",(128,128,3,3)); bq23,m23,sh23=g.make_qparams(conv0,w23,g.fw("conv23_b.bin"),g.fw("conv23_s.bin"),AQ[23][0],AQ[23][1])
    c23,rq23=g.conv_rtl(conv0,c22m,w23,bq23,m23,lut,3,3,1,1,AQ[23][1],GLUE8[0],GLUE8[1])
    s1f=s1.transpose(1,2,0).reshape(SP,128); add0,ratio0=g.cpu_add(conv0,s1f,c23,AQ[21][2],AQ[21][3],GLUE8[0])

    # concat -> cv2 = conv24 (1x1 384->256)
    s0f=s0.transpose(1,2,0).reshape(SP,128)
    s0c,cm_s=cat_req(conv0,s0f,AQ[21][3],AQ[21][2]); s1c,_=cat_req(conv0,s1f,AQ[21][3],AQ[21][2])
    a0c,cm_a0=cat_req(conv0,add0,GLUE8[1],GLUE8[0])
    concat=np.concatenate([s0c,s1c,a0c],axis=1).reshape(IN_H,IN_W,384).transpose(2,0,1)
    w24=W("conv24_w.bin",(256,384,1,1)); bq24,m24,sh24=g.make_qparams(conv0,w24,g.fw("conv24_b.bin"),g.fw("conv24_s.bin"),CAT[0],CAT[1])
    golden,rq24=g.conv_rtl(conv0,concat,w24,bq24,m24,lut,1,1,1,0,CAT[1],AQ[24][2],AQ[24][3])

    u32a=lambda n,ws:"\n".join([f"static const uint32_t {n}[{len(ws)}][4] = {{"]+["    {"+", ".join(f"0x{v:08X}u" for v in x)+"},"for x in ws]+["};"])
    body=f"""#ifndef YOLO_C2F8_DATA_H
#define YOLO_C2F8_DATA_H

#include <stdint.h>

#define YOLO_C2F8_IN_W {IN_W}u
#define YOLO_C2F8_IN_H {IN_H}u
#define YOLO_C2F8_SPATIAL {SP}u
#define YOLO_C2F8_FULL_C 256u
#define YOLO_C2F8_CV1_IC 256u
#define YOLO_C2F8_CV2_IC 384u
#define YOLO_C2F8_CV2_OC 256u
#define YOLO_C2F8_RTL_TOL 16u

#define YOLO_C2F8_BLKIN_WORDS {SP*16}u
{u32a("yolo_c2f8_blkin_words", pack_act(conv0, blkin))}

#define YOLO_C2F8_CV1_WGT_WORDS {256*16}u
#define YOLO_C2F8_CV1_RQ_MUL {rq21}u
#define YOLO_C2F8_CV1_RQ_SHIFT {g.REQ_SHIFT}u
#define YOLO_C2F8_CV1_RQ_ZP {AQ[21][3]}
{u32a("yolo_c2f8_cv1_wgt", g.pack1x1(conv0,w21))}
{g.fmt_i32("yolo_c2f8_cv1_bias",bq21)}
{g.fmt_u32("yolo_c2f8_cv1_mul",m21)}
{g.fmt_u32("yolo_c2f8_cv1_shift",sh21)}

#define YOLO_C2F8_MCV1_WGT_WORDS {128*8*9}u
{u32a("yolo_c2f8_mcv1_0_wgt", g.pack3x3(conv0,w22))}
{g.fmt_i32("yolo_c2f8_mcv1_0_bias",bq22)}
{g.fmt_u32("yolo_c2f8_mcv1_0_mul",m22)}
{g.fmt_u32("yolo_c2f8_mcv1_0_shift",sh22)}
#define YOLO_C2F8_MCV1_0_RQ_MUL {rq22}u
#define YOLO_C2F8_MCV1_0_RQ_ZP {AQ[22][3]}
#define YOLO_C2F8_MCV1_0_PAD {AQ[22][1]}

#define YOLO_C2F8_MCV2_WGT_WORDS {128*8*9}u
{u32a("yolo_c2f8_mcv2_0_wgt", g.pack3x3(conv0,w23))}
{g.fmt_i32("yolo_c2f8_mcv2_0_bias",bq23)}
{g.fmt_u32("yolo_c2f8_mcv2_0_mul",m23)}
{g.fmt_u32("yolo_c2f8_mcv2_0_shift",sh23)}
#define YOLO_C2F8_MCV2_0_PAD {AQ[23][1]}

#define YOLO_C2F8_GLUE0_RQ_MUL {rq23}u
#define YOLO_C2F8_GLUE0_ZP {GLUE8[1]}
#define YOLO_C2F8_GLUE_RQ_SHIFT {g.REQ_SHIFT}u

#define YOLO_C2F8_ADD_RATIO_SHIFT {g.RATIO_SHIFT}u
#define YOLO_C2F8_ADD0_RATIO_MUL {ratio0}u
#define YOLO_C2F8_ADD0_PREV_ZP {AQ[21][3]}

#define YOLO_C2F8_CAT_SHIFT {CAT_SHIFT}u
#define YOLO_C2F8_CAT_ZP {CAT[1]}
#define YOLO_C2F8_CAT_MUL_S0S1 {cm_s}u
#define YOLO_C2F8_CAT_INZP_S0S1 {AQ[21][3]}
#define YOLO_C2F8_CAT_MUL_ADD0 {cm_a0}u
#define YOLO_C2F8_CAT_INZP_ADD0 {GLUE8[1]}

#define YOLO_C2F8_CV2_WGT_WORDS {256*24}u
#define YOLO_C2F8_CV2_RQ_MUL {rq24}u
#define YOLO_C2F8_CV2_RQ_SHIFT {g.REQ_SHIFT}u
#define YOLO_C2F8_CV2_RQ_ZP {AQ[24][3]}
{u32a("yolo_c2f8_cv2_wgt", g.pack1x1(conv0,w24))}
{g.fmt_i32("yolo_c2f8_cv2_bias",bq24)}
{g.fmt_u32("yolo_c2f8_cv2_mul",m24)}
{g.fmt_u32("yolo_c2f8_cv2_shift",sh24)}

{g.fmt_u8("yolo_c2f8_expected_rtl", golden.astype(np.uint8))}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH} ratio0={ratio0} cat=({cm_s},{cm_a0})")
    return conv0, golden


if __name__ == "__main__":
    main()
