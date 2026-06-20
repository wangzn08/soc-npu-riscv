#!/usr/bin/env python3
"""Detect-head scale-0 bbox branch: pan_p3(64) -> conv36(3x3 SiLU) ->
conv38(3x3 SiLU) -> conv41(1x1 LINEAR, has_silu=0). Validates the head conv chain
incl. the linear output conv (silu_requant_en && !silu_en path). Input = c2f_15
output (pan_p3, 80x8x64)."""

from __future__ import annotations
from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "yolov8n_int8" / "weights"
OUT_PATH = ROOT / "firmware" / "yolo_head_bbox0_data.h"
C2F4_GEN = ROOT / "tools" / "gen_yolo_c2f4_smoke.py"
C2F15_GEN = ROOT / "tools" / "gen_yolo_c2f15_smoke.py"

IN_W, IN_H = 80, 8
SP = IN_W * IN_H
Q_SHIFT = 20
AQ36 = (0.0308109522, -119, 0.0312871225, -119)
AQ38 = (0.0312871225, -119, 0.1906583309, -127)
AQ41 = (0.1906583309, -127, 0.1402078271, -60)   # has_silu=0 (LINEAR)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def lin_qparams(conv0, w, b, s, in_scale, in_zp, out_scale):
    """LINEAR conv qparams so s2 = round(real_preact/out_scale): scale_mul=
    round(in*w/out * 2^Q), bias_q = round(bias/(in*w)) - in_zp*wsum."""
    wsum = w.astype(np.int32).sum(axis=(1, 2, 3))
    bias_q, mul = [], []
    for oc in range(w.shape[0]):
        ratio = in_scale * float(s[oc]) / out_scale
        mul.append(int(round(ratio * (1 << Q_SHIFT))))
        bias_q.append(int(round(float(b[oc]) / (in_scale * float(s[oc]))) - in_zp * int(wsum[oc])))
    return bias_q, mul, [Q_SHIFT] * w.shape[0]


def conv_lin(conv0, act, w, bias_q, mul, in_zp, out_zp):
    """1x1 LINEAR conv: s2=(acc+bias)*mul>>Q; out=clamp_s8(s2+out_zp). [SP,OC]."""
    OC, IC = w.shape[0], w.shape[1]
    out = np.zeros((SP, OC), dtype=np.int8)
    for pos in range(SP):
        y, x = pos // IN_W, pos % IN_W
        for oc in range(OC):
            acc = 0
            for ic in range(IC):
                acc += int(act[ic, y, x]) * int(w[oc, ic, 0, 0])
            s2 = ((acc + bias_q[oc]) * mul[oc]) >> Q_SHIFT
            out[pos, oc] = np.int8(conv0.s8(conv0.clamp_s8(s2 + out_zp)))
    return out


def pack_act(conv0, act):
    C = act.shape[0]; out = []
    for g in range(C // 16):
        for pos in range(SP):
            y, x = pos // IN_W, pos % IN_W
            out.append(conv0.pack_i8_word(act[g*16:g*16+16, y, x]))
    return out


def Wt(name, shape): return np.fromfile(WEIGHT_DIR/name, dtype=np.int8).reshape(*shape)


def main():
    g = load("c2f4", C2F4_GEN)
    c2f15 = load("c2f15", C2F15_GEN)
    conv0, pan_p3 = c2f15.main()        # [640,64] @ c2f_15 out (80x8)
    lut = conv0.load_lut()
    act = pan_p3.reshape(IN_H, IN_W, 64).transpose(2, 0, 1)   # [64,8,80]

    w36 = Wt("conv36_w.bin", (64, 64, 3, 3))
    bq36, m36, sh36 = g.make_qparams(conv0, w36, g.fw("conv36_b.bin"), g.fw("conv36_s.bin"), AQ36[0], AQ36[1])
    c36, rq36 = g.conv_rtl(conv0, act, w36, bq36, m36, lut, 3, 3, 1, 1, AQ36[1], AQ36[2], AQ36[3])
    c36m = c36.reshape(IN_H, IN_W, 64).transpose(2, 0, 1)

    w38 = Wt("conv38_w.bin", (64, 64, 3, 3))
    bq38, m38, sh38 = g.make_qparams(conv0, w38, g.fw("conv38_b.bin"), g.fw("conv38_s.bin"), AQ38[0], AQ38[1])
    c38, rq38 = g.conv_rtl(conv0, c36m, w38, bq38, m38, lut, 3, 3, 1, 1, AQ38[1], AQ38[2], AQ38[3])
    c38m = c38.reshape(IN_H, IN_W, 64).transpose(2, 0, 1)

    w41 = Wt("conv41_w.bin", (64, 64, 1, 1))
    bq41, m41, sh41 = lin_qparams(conv0, w41, g.fw("conv41_b.bin"), g.fw("conv41_s.bin"), AQ41[0], AQ41[1], AQ41[2])
    golden = conv_lin(conv0, c38m, w41, bq41, m41, AQ41[1], AQ41[3])

    u32a = lambda n, ws: "\n".join([f"static const uint32_t {n}[{len(ws)}][4] = {{"]+["    {"+", ".join(f"0x{v:08X}u" for v in x)+"}," for x in ws]+["};"])
    body = f"""#ifndef YOLO_HEAD_BBOX0_DATA_H
#define YOLO_HEAD_BBOX0_DATA_H

#include <stdint.h>

#define YOLO_HB0_IN_W {IN_W}u
#define YOLO_HB0_IN_H {IN_H}u
#define YOLO_HB0_SPATIAL {SP}u
#define YOLO_HB0_RTL_TOL 16u

#define YOLO_HB0_BLKIN_WORDS {SP*4}u
{u32a("yolo_hb0_blkin_words", pack_act(conv0, act))}

/* conv36 (3x3 SiLU 64->64) */
#define YOLO_HB0_C36_WGT_WORDS {64*4*9}u
#define YOLO_HB0_C36_RQ_MUL {rq36}u
#define YOLO_HB0_C36_RQ_ZP {AQ36[3]}
#define YOLO_HB0_C36_PAD {AQ36[1]}
{u32a("yolo_hb0_c36_wgt", g.pack3x3(conv0, w36))}
{g.fmt_i32("yolo_hb0_c36_bias", bq36)}
{g.fmt_u32("yolo_hb0_c36_mul", m36)}
{g.fmt_u32("yolo_hb0_c36_shift", sh36)}

/* conv38 (3x3 SiLU 64->64) */
#define YOLO_HB0_C38_WGT_WORDS {64*4*9}u
#define YOLO_HB0_C38_RQ_MUL {rq38}u
#define YOLO_HB0_C38_RQ_ZP {AQ38[3]}
#define YOLO_HB0_C38_PAD {AQ38[1]}
{u32a("yolo_hb0_c38_wgt", g.pack3x3(conv0, w38))}
{g.fmt_i32("yolo_hb0_c38_bias", bq38)}
{g.fmt_u32("yolo_hb0_c38_mul", m38)}
{g.fmt_u32("yolo_hb0_c38_shift", sh38)}

/* conv41 (1x1 LINEAR 64->64): lin qparams; out = clamp_s8(s2 + out_zp) */
#define YOLO_HB0_C41_WGT_WORDS {64*4}u
#define YOLO_HB0_C41_OUT_ZP {AQ41[3]}
#define YOLO_HB0_C41_RQ_SHIFT {Q_SHIFT}u
{u32a("yolo_hb0_c41_wgt", g.pack1x1(conv0, w41))}
{g.fmt_i32("yolo_hb0_c41_bias", bq41)}
{g.fmt_u32("yolo_hb0_c41_mul", m41)}
{g.fmt_u32("yolo_hb0_c41_shift", sh41)}

{g.fmt_u8("yolo_hb0_expected_rtl", golden.astype(np.uint8))}

#endif
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return conv0, golden


if __name__ == "__main__":
    main()
