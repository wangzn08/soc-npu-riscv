#!/usr/bin/env python3
"""conv13 (model.5): 64->128 3x3 stride2 pad1. Input = c2f_4 (model.4) output
(80x8x64), baked in. First OC=128 layer => exercises the OC>64 chunked conv path
(two 64-OC oc_single passes). Output 40x4x128."""

from __future__ import annotations
from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
OUT_PATH = ROOT / "firmware" / "yolo_conv13_m6b_data.h"
C2F4_GEN = ROOT / "tools" / "gen_yolo_c2f4_smoke.py"

IN_W, IN_H, IC, OC = 80, 8, 64, 128
KH = KW = 3
STRIDE, PAD = 2, 1
OUT_W = (IN_W + 2*PAD - KW)//STRIDE + 1   # 40
OUT_H = (IN_H + 2*PAD - KH)//STRIDE + 1   # 4
SP_OUT = OUT_W * OUT_H
IN_SCALE, IN_ZP = 0.0337357186, -120     # act_quant[13] in
OUT_SCALE, OUT_ZP = 0.0319866650, -119   # act_quant[13] out
RTL_TOL = 16


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    g = load("c2f4", C2F4_GEN)
    conv0, c2f4_out = g.compute_c2f4()    # [640,64] at conv12 out = conv13 in
    lut = conv0.load_lut()
    act = c2f4_out.reshape(IN_H, IN_W, IC).transpose(2, 0, 1)  # [64,8,80]

    w = np.fromfile(WEIGHT_DIR/"conv13_w.bin", dtype=np.int8).reshape(OC, IC, KH, KW)
    bq, mul, sh = g.make_qparams(conv0, w, g.fw("conv13_b.bin"), g.fw("conv13_s.bin"), IN_SCALE, IN_ZP)
    golden, rqm = g.conv_rtl(conv0, act, w, bq, mul, lut, KH, KW, STRIDE, PAD, IN_ZP, OUT_SCALE, OUT_ZP)

    act_words = g.pack_act(conv0, act)        # IN_W/IN_H from c2f4 module = 80/8
    wgt_words = g.pack3x3(conv0, w)           # OC*(IC/16)*9 = 128*4*9

    def u32a(name, words):
        out=[f"static const uint32_t {name}[{len(words)}][4] = {{"]
        for x in words: out.append("    {"+", ".join(f"0x{v:08X}u" for v in x)+"},")
        out.append("};"); return "\n".join(out)

    body = f"""#ifndef YOLO_CONV13_M6B_DATA_H
#define YOLO_CONV13_M6B_DATA_H

#include <stdint.h>

#define YOLO_CONV13_IN_W {IN_W}u
#define YOLO_CONV13_IN_H {IN_H}u
#define YOLO_CONV13_IC {IC}u
#define YOLO_CONV13_OC {OC}u
#define YOLO_CONV13_KH {KH}u
#define YOLO_CONV13_KW {KW}u
#define YOLO_CONV13_STRIDE {STRIDE}u
#define YOLO_CONV13_PAD {PAD}u
#define YOLO_CONV13_ACT_WORDS {len(act_words)}u
#define YOLO_CONV13_WGT_WORDS {len(wgt_words)}u
#define YOLO_CONV13_WGT_PER_OC {(IC//16)*9}u
#define YOLO_CONV13_OUT_W {OUT_W}u
#define YOLO_CONV13_OUT_H {OUT_H}u
#define YOLO_CONV13_OUT_SPATIAL {SP_OUT}u
#define YOLO_CONV13_REQUANT_MUL {rqm}u
#define YOLO_CONV13_REQUANT_SHIFT {g.REQ_SHIFT}u
#define YOLO_CONV13_REQUANT_ZP {OUT_ZP}
#define YOLO_CONV13_PAD_VALUE {IN_ZP}
#define YOLO_CONV13_RTL_TOL {RTL_TOL}u

{u32a("yolo_conv13_act_words", act_words)}

{u32a("yolo_conv13_wgt_words", wgt_words)}

{g.fmt_i32("yolo_conv13_bias_q", bq)}

{g.fmt_u32("yolo_conv13_scale_mul", mul)}

{g.fmt_u32("yolo_conv13_scale_shift", sh)}

{g.fmt_u8("yolo_conv13_expected_rtl", golden.astype(np.uint8))}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
