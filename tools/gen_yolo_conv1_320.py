#!/usr/bin/env python3
"""conv1 (model.1, 16->32 3x3 s2 pad1) at FULL 320-resolution: input = real conv0
output (160x160x16) from the @320 golden dump, run through the tiled primitive.
First real on-SoC 320 per-layer cycle/accuracy datapoint.

Prereq: yolov8n_int8/dump320/conv0.bin and conv1.bin (see docs/notes/yolov8n-320-golden.md).
Golden = RTL-model (SiLU LUT) at full 160x160, compared to dump320/conv1.bin scales.
"""
from __future__ import annotations
from pathlib import Path
import struct
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WDIR = ROOT / "yolov8n_int8" / "weights"
DUMP = ROOT / "yolov8n_int8" / "dump320"
LUT_PATH = ROOT / "rtl" / "silu_lut_q4_4.hex"
OUT_PATH = ROOT / "firmware" / "yolo_conv1_320_data.h"

KH = KW = 3
STRIDE, PAD = 2, 1
Q_SHIFT = 20
REQ_SHIFT = 12
RTL_TOL = 16


def s8(b): b &= 0xFF; return b - 256 if b & 0x80 else b
def clamp_s8(v): return 0x80 if v < -128 else (0x7F if v > 127 else v & 0xFF)
def load_lut():
    v = [int(x, 16) & 0xFF for x in LUT_PATH.read_text().split()]
    assert len(v) == 256
    return v
def rtl_silu(v, lut): v = -128 if v < -128 else (127 if v > 127 else v); return lut[v & 0xFF]
def pack_i8_word(vals):
    raw = vals.astype(np.int8).view(np.uint8)
    return [sum(int(raw[g*4+b]) << (8*b) for b in range(4)) for g in range(4)]


def read_dump(name):
    with open(DUMP / name, "rb") as f:
        c, h, w = struct.unpack("<3i", f.read(12))
        sc = struct.unpack("<f", f.read(4))[0]
        zp = struct.unpack("<i", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.int8).reshape(c, h, w).copy()
    return data, sc, zp


def main():
    act, in_scale, in_zp = read_dump("conv0.bin")        # [16,160,160]
    _, out_scale, out_zp = read_dump("conv1.bin")        # scales for requant
    IC, IN_H, IN_W = act.shape
    OC = 32
    OUT_H = (IN_H + 2*PAD - KH)//STRIDE + 1               # 80
    OUT_W = (IN_W + 2*PAD - KW)//STRIDE + 1               # 80
    SP_OUT = OUT_H * OUT_W

    w = np.fromfile(WDIR/"conv1_w.bin", dtype=np.int8).reshape(OC, IC, KH, KW)
    b = np.fromfile(WDIR/"conv1_b.bin", dtype=np.float32)[:OC]
    sc = np.fromfile(WDIR/"conv1_s.bin", dtype=np.float32)[:OC]
    lut = load_lut()
    wsum = w.astype(np.int32).sum(axis=(1, 2, 3))

    bias_q, mul = [], []
    for oc in range(OC):
        m = int(round(in_scale * float(sc[oc]) * 16.0 * (1 << Q_SHIFT)))
        mul.append(m)
        be = int(round(float(b[oc]) * 16.0 * (1 << Q_SHIFT) / m))
        bias_q.append(int(be - in_zp * int(wsum[oc])))
    requant_mul = int(round((1 << REQ_SHIFT) / (16.0 * out_scale)))

    # RTL-model golden at full resolution (vectorized conv via im2col)
    golden = np.zeros((SP_OUT, OC), dtype=np.int8)
    ap = np.full((IC, IN_H + 2*PAD, IN_W + 2*PAD), in_zp, dtype=np.int32)
    ap[:, PAD:PAD+IN_H, PAD:PAD+IN_W] = act
    for oy in range(OUT_H):
        for ox in range(OUT_W):
            iy, ix = oy*STRIDE, ox*STRIDE
            win = ap[:, iy:iy+KH, ix:ix+KW]                # [IC,3,3]
            for oc in range(OC):
                acc = int(np.sum(win * w[oc].astype(np.int32)))
                q44 = ((acc + bias_q[oc]) * mul[oc]) >> Q_SHIFT
                rq = ((s8(rtl_silu(q44, lut)) * requant_mul) >> REQ_SHIFT) + out_zp
                golden[oy*OUT_W+ox, oc] = np.int8(s8(clamp_s8(rq)))

    # pack act tile-major [ic_group][row][col], ic_group = IC/16
    act_words = []
    for g in range(IC//16):
        for y in range(IN_H):
            for x in range(IN_W):
                act_words.append(pack_i8_word(act[g*16:g*16+16, y, x]))
    wgt_words = []
    for oc in range(OC):
        for ko in range(9):
            wgt_words.append(pack_i8_word(w[oc, :, ko//3, ko%3]))

    def u32a(name, words):
        L = [f"static const uint32_t {name}[{len(words)}][4] = {{"]
        L += ["    {"+", ".join(f"0x{v:08X}u" for v in x)+"}," for x in words]
        L.append("};"); return "\n".join(L)
    def i32(name, v): return f"static const int32_t {name}[{len(v)}] = {{ {', '.join(map(str,v))} }};"
    def u32(name, v): return f"static const uint32_t {name}[{len(v)}] = {{ {', '.join(f'{x}u' for x in v)} }};"

    g8 = [f"static const uint8_t yolo_conv1_320_golden[{SP_OUT}][{OC}] = {{"]
    g8 += ["    {"+", ".join(f"0x{int(x)&0xFF:02X}u" for x in golden[p])+"}," for p in range(SP_OUT)]
    g8.append("};")

    body = f"""#ifndef YOLO_CONV1_320_DATA_H
#define YOLO_CONV1_320_DATA_H
#include <stdint.h>
#define C1_IN_W {IN_W}u
#define C1_IN_H {IN_H}u
#define C1_IC {IC}u
#define C1_OC {OC}u
#define C1_OUT_W {OUT_W}u
#define C1_OUT_H {OUT_H}u
#define C1_OUT_SPATIAL {SP_OUT}u
#define C1_WGT_PER_OC {(IC//16)*9}u
#define C1_ACT_WORDS {len(act_words)}u
#define C1_WGT_WORDS {len(wgt_words)}u
#define C1_PAD_VALUE {in_zp}
#define C1_REQUANT_MUL {requant_mul}u
#define C1_REQUANT_SHIFT {REQ_SHIFT}u
#define C1_REQUANT_ZP {out_zp}
#define C1_RTL_TOL {RTL_TOL}u

{u32a("yolo_conv1_320_act_words", act_words)}

{u32a("yolo_conv1_320_wgt_words", wgt_words)}

{i32("yolo_conv1_320_bias_q", bias_q)}

{u32("yolo_conv1_320_scale_mul", mul)}

{u32("yolo_conv1_320_scale_shift", [Q_SHIFT]*OC)}

{chr(10).join(g8)}

#endif
"""
    OUT_PATH.write_text(body, encoding="ascii")
    print(f"wrote {OUT_PATH}: {IN_W}x{IN_H}x{IC} -> {OUT_W}x{OUT_H}x{OC}, "
          f"{len(act_words)} act words, golden {SP_OUT}x{OC}")


if __name__ == "__main__":
    main()
