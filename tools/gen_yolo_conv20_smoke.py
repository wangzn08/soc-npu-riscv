#!/usr/bin/env python3
"""conv20 (model.7): 128->256 3x3 stride2 pad1. Input = c2f_6 output (40x4x128).
OC=256 => 4-chunk OC>64 conv. Output 20x2x256."""

from __future__ import annotations
from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
OUT_PATH = ROOT / "firmware" / "yolo_conv20_data.h"
C2F4_GEN = ROOT / "tools" / "gen_yolo_c2f4_smoke.py"
C2F6_GEN = ROOT / "tools" / "gen_yolo_c2f6_smoke.py"

IN_W, IN_H, IC, OC = 40, 4, 128, 256
KH = KW = 3
STRIDE, PAD = 2, 1
OUT_W = (IN_W + 2*PAD - KW)//STRIDE + 1   # 20
OUT_H = (IN_H + 2*PAD - KH)//STRIDE + 1   # 2
SP_OUT = OUT_W * OUT_H
IN_SCALE, IN_ZP = 0.0331330635, -120
OUT_SCALE, OUT_ZP = 0.0388779789, -121
RTL_TOL = 16


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def pack_act(conv0, act):
    C = act.shape[0]; out = []
    for g in range(C//16):
        for pos in range(SP_in()):
            y, x = pos//IN_W, pos%IN_W
            out.append(conv0.pack_i8_word(act[g*16:g*16+16, y, x]))
    return out
def SP_in(): return IN_W * IN_H


def main():
    g = load("c2f4", C2F4_GEN)
    c2f6 = load("c2f6", C2F6_GEN)
    conv0, c2f6_out = c2f6.compute_c2f6()    # [160,128] @ conv19 out = conv20 in
    lut = conv0.load_lut()
    act = c2f6_out.reshape(IN_H, IN_W, IC).transpose(2, 0, 1)  # [128,4,40]

    w = np.fromfile(WEIGHT_DIR/"conv20_w.bin", dtype=np.int8).reshape(OC, IC, KH, KW)
    bq, mul, sh = g.make_qparams(conv0, w, g.fw("conv20_b.bin"), g.fw("conv20_s.bin"), IN_SCALE, IN_ZP)
    golden, rqm = g.conv_rtl(conv0, act, w, bq, mul, lut, KH, KW, STRIDE, PAD, IN_ZP, OUT_SCALE, OUT_ZP)

    act_words = pack_act(conv0, act)
    wgt_words = g.pack3x3(conv0, w)

    def u32a(name, words):
        out=[f"static const uint32_t {name}[{len(words)}][4] = {{"]
        for x in words: out.append("    {"+", ".join(f"0x{v:08X}u" for v in x)+"},")
        out.append("};"); return "\n".join(out)

    body = f"""#ifndef YOLO_CONV20_DATA_H
#define YOLO_CONV20_DATA_H

#include <stdint.h>

#define YOLO_CONV20_IN_W {IN_W}u
#define YOLO_CONV20_IN_H {IN_H}u
#define YOLO_CONV20_IC {IC}u
#define YOLO_CONV20_OC {OC}u
#define YOLO_CONV20_KH {KH}u
#define YOLO_CONV20_KW {KW}u
#define YOLO_CONV20_STRIDE {STRIDE}u
#define YOLO_CONV20_PAD {PAD}u
#define YOLO_CONV20_ACT_WORDS {len(act_words)}u
#define YOLO_CONV20_WGT_WORDS {len(wgt_words)}u
#define YOLO_CONV20_WGT_PER_OC {(IC//16)*9}u
#define YOLO_CONV20_OUT_SPATIAL {SP_OUT}u
#define YOLO_CONV20_REQUANT_MUL {rqm}u
#define YOLO_CONV20_REQUANT_SHIFT {g.REQ_SHIFT}u
#define YOLO_CONV20_REQUANT_ZP {OUT_ZP}
#define YOLO_CONV20_PAD_VALUE {IN_ZP}
#define YOLO_CONV20_RTL_TOL {RTL_TOL}u

{u32a("yolo_conv20_act_words", act_words)}

{u32a("yolo_conv20_wgt_words", wgt_words)}

{g.fmt_i32("yolo_conv20_bias_q", bq)}

{g.fmt_u32("yolo_conv20_scale_mul", mul)}

{g.fmt_u32("yolo_conv20_scale_shift", sh)}

{g.fmt_u8("yolo_conv20_expected_rtl", golden.astype(np.uint8))}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return conv0, golden   # conv20 output [SP_OUT,256] int8 (20x2x256)


if __name__ == "__main__":
    main()
