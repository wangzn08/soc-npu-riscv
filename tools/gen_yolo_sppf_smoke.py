#!/usr/bin/env python3
"""SPPF (model.9): cv1=conv25(256->128 1x1) -> 3x MaxPool5x5(s1,p2,int8) ->
concat(cv1,m0,m1,m2)=512 -> cv2=conv26(512->256 1x1). Input = c2f_8 output
(20x2x256). MaxPool + concat are CPU-side (same scale/zp => concat is identity)."""

from __future__ import annotations
from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
OUT_PATH = ROOT / "firmware" / "yolo_sppf_data.h"
C2F4_GEN = ROOT / "tools" / "gen_yolo_c2f4_smoke.py"
C2F8_GEN = ROOT / "tools" / "gen_yolo_c2f8_smoke.py"

IN_W, IN_H = 20, 2
SP = IN_W * IN_H
# conv25 (cv1): in=conv24 out; out used by maxpool/concat/conv26-in.
C25 = (0.0433438867, -122, 0.0334274098, -120)
C26 = (0.0334274098, -120, 0.0468565077, -122)   # cv2; in = concat scale


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def maxpool5(x):  # x [C,H,W] int8 -> [C,H,W] int8, k=5 s1 p2, pad=-128
    C, H, W = x.shape
    out = np.full_like(x, -128)
    for c in range(C):
        for oh in range(H):
            for ow in range(W):
                mx = -128
                for kh in range(5):
                    for kw in range(5):
                        ih, iw = oh-2+kh, ow-2+kw
                        if 0 <= ih < H and 0 <= iw < W:
                            v = int(x[c, ih, iw])
                            if v > mx: mx = v
                out[c, oh, ow] = mx
    return out


def pack_act(conv0, act):
    C = act.shape[0]; out = []
    for g in range(C//16):
        for pos in range(SP):
            y, x = pos//IN_W, pos%IN_W
            out.append(conv0.pack_i8_word(act[g*16:g*16+16, y, x]))
    return out


def W(name, shape): return np.fromfile(WEIGHT_DIR/name, dtype=np.int8).reshape(*shape)


def main():
    g = load("c2f4", C2F4_GEN)
    c2f8 = load("c2f8", C2F8_GEN)
    conv0, c2f8_out = c2f8.main()      # [40,256] @ conv24 out = conv25 in
    lut = conv0.load_lut()
    blkin = c2f8_out.reshape(IN_H, IN_W, 256).transpose(2, 0, 1)  # [256,2,20]

    # cv1 = conv25 (1x1 256->128)
    w25 = W("conv25_w.bin", (128, 256, 1, 1))
    bq25, m25, sh25 = g.make_qparams(conv0, w25, g.fw("conv25_b.bin"), g.fw("conv25_s.bin"), C25[0], C25[1])
    cv1o, rq25 = g.conv_rtl(conv0, blkin, w25, bq25, m25, lut, 1, 1, 1, 0, C25[1], C25[2], C25[3])
    cv1 = cv1o.reshape(IN_H, IN_W, 128).transpose(2, 0, 1)  # [128,2,20]

    m0 = maxpool5(cv1); m1 = maxpool5(m0); m2 = maxpool5(m1)
    # concat (cv1, m0, m1, m2) — all at C25 out scale = C26 in scale (identity requant)
    concat = np.concatenate([cv1, m0, m1, m2], axis=0)  # [512,2,20]

    w26 = W("conv26_w.bin", (256, 512, 1, 1))
    bq26, m26, sh26 = g.make_qparams(conv0, w26, g.fw("conv26_b.bin"), g.fw("conv26_s.bin"), C26[0], C26[1])
    golden, rq26 = g.conv_rtl(conv0, concat, w26, bq26, m26, lut, 1, 1, 1, 0, C26[1], C26[2], C26[3])

    u32a = lambda n, ws: "\n".join([f"static const uint32_t {n}[{len(ws)}][4] = {{"]+["    {"+", ".join(f"0x{v:08X}u" for v in x)+"}," for x in ws]+["};"])

    body = f"""#ifndef YOLO_SPPF_DATA_H
#define YOLO_SPPF_DATA_H

#include <stdint.h>

#define YOLO_SPPF_IN_W {IN_W}u
#define YOLO_SPPF_IN_H {IN_H}u
#define YOLO_SPPF_SPATIAL {SP}u
#define YOLO_SPPF_C25_IC 256u
#define YOLO_SPPF_C25_OC 128u
#define YOLO_SPPF_C26_IC 512u
#define YOLO_SPPF_C26_OC 256u
#define YOLO_SPPF_RTL_TOL 16u

#define YOLO_SPPF_BLKIN_WORDS {SP*16}u
{u32a("yolo_sppf_blkin_words", pack_act(conv0, blkin))}

#define YOLO_SPPF_C25_WGT_WORDS {128*16}u
#define YOLO_SPPF_C25_RQ_MUL {rq25}u
#define YOLO_SPPF_C25_RQ_SHIFT {g.REQ_SHIFT}u
#define YOLO_SPPF_C25_RQ_ZP {C25[3]}
{u32a("yolo_sppf_c25_wgt", g.pack1x1(conv0, w25))}
{g.fmt_i32("yolo_sppf_c25_bias", bq25)}
{g.fmt_u32("yolo_sppf_c25_mul", m25)}
{g.fmt_u32("yolo_sppf_c25_shift", sh25)}

#define YOLO_SPPF_C26_WGT_WORDS {256*32}u
#define YOLO_SPPF_C26_RQ_MUL {rq26}u
#define YOLO_SPPF_C26_RQ_SHIFT {g.REQ_SHIFT}u
#define YOLO_SPPF_C26_RQ_ZP {C26[3]}
{u32a("yolo_sppf_c26_wgt", g.pack1x1(conv0, w26))}
{g.fmt_i32("yolo_sppf_c26_bias", bq26)}
{g.fmt_u32("yolo_sppf_c26_mul", m26)}
{g.fmt_u32("yolo_sppf_c26_shift", sh26)}

{g.fmt_u8("yolo_sppf_expected_rtl", golden.astype(np.uint8))}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return conv0, golden


if __name__ == "__main__":
    main()
