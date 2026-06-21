#!/usr/bin/env python3
"""conv1 (model.1, 16->32 3x3 s2 pad1) at FULL 320 resolution with the EXACT
per-layer SiLU LUT (CTRL[22]). Input = real conv0 dump (160x160x16). The golden
is the exact-SiLU RTL model and is self-checked against dump320/conv1.bin (the C
float oracle) -- conv1 is the worst LUT-saturator (~20% with the legacy +-8 LUT),
so this is the key proof that exact SiLU fixes a standalone tiled conv.
"""
from __future__ import annotations
from pathlib import Path
import struct, math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WDIR = ROOT / "yolov8n_int8" / "weights"
DUMP = ROOT / "yolov8n_int8" / "dump320"
OUT_PATH = ROOT / "firmware" / "yolo_conv1_320_exact_data.h"

KH = KW = 3
STRIDE, PAD = 2, 1
Q_SHIFT = 20
RTL_TOL = 16


def s8(b): b &= 0xFF; return b - 256 if b & 0x80 else b
def clamp_s8(v): return 0x80 if v < -128 else (0x7F if v > 127 else v & 0xFF)
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


def build_silu_lut(out_scale, out_zp):
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        q = i if i < 128 else i - 256
        realq = (q - out_zp) * out_scale
        silu = realq / (1.0 + math.exp(-realq))
        lut[i] = clamp_s8(int(round(silu / out_scale + out_zp)))
    return lut


def main():
    act, in_scale, in_zp = read_dump("conv0.bin")        # [16,160,160]
    c1, out_scale, out_zp = read_dump("conv1.bin")        # golden + out scales
    IC, IN_H, IN_W = act.shape
    OC = 32
    OUT_H = (IN_H + 2*PAD - KH)//STRIDE + 1               # 80
    OUT_W = (IN_W + 2*PAD - KW)//STRIDE + 1               # 80
    SP_OUT = OUT_H * OUT_W

    w = np.fromfile(WDIR/"conv1_w.bin", dtype=np.int8).reshape(OC, IC, KH, KW)
    b = np.fromfile(WDIR/"conv1_b.bin", dtype=np.float32)[:OC]
    sc = np.fromfile(WDIR/"conv1_s.bin", dtype=np.float32)[:OC]
    wsum = w.astype(np.int32).sum(axis=(1, 2, 3))
    lut = build_silu_lut(out_scale, out_zp)

    # exact out-grid qparams: s2 == round(preact/out_scale)
    bias_q, mul = [], []
    for oc in range(OC):
        m = int(round(in_scale * float(sc[oc]) / out_scale * (1 << Q_SHIFT)))
        mul.append(m)
        be = int(round(float(b[oc]) / (in_scale * float(sc[oc]))))
        bias_q.append(int(be - in_zp * int(wsum[oc])))

    # exact-SiLU golden at full resolution
    golden = np.zeros((SP_OUT, OC), dtype=np.int8)
    ap = np.full((IC, IN_H + 2*PAD, IN_W + 2*PAD), in_zp, dtype=np.int32)
    ap[:, PAD:PAD+IN_H, PAD:PAD+IN_W] = act
    for oy in range(OUT_H):
        for ox in range(OUT_W):
            iy, ix = oy*STRIDE, ox*STRIDE
            win = ap[:, iy:iy+KH, ix:ix+KW]
            for oc in range(OC):
                acc = int(np.sum(win * w[oc].astype(np.int32)))
                s2 = ((acc + bias_q[oc]) * mul[oc]) >> Q_SHIFT
                idx = clamp_s8(s2 + out_zp)
                golden[oy*OUT_W+ox, oc] = np.int8(s8(int(lut[idx])))

    # self-check vs the C float dump (conv1.bin)
    cflat = c1.transpose(1, 2, 0).reshape(SP_OUT, OC)
    df = np.abs(golden.astype(int) - cflat.astype(int))
    print(f"exact conv1 vs dump320/conv1.bin: mism={int((df>0).sum())}/{golden.size} max={int(df.max())}")

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
    def u8(name, v): return f"static const uint8_t {name}[{len(v)}] = {{ {', '.join(f'0x{int(x):02X}u' for x in v)} }};"

    g8 = [f"static const uint8_t yolo_conv1_320e_golden[{SP_OUT}][{OC}] = {{"]
    g8 += ["    {"+", ".join(f"0x{int(x)&0xFF:02X}u" for x in golden[p])+"}," for p in range(SP_OUT)]
    g8.append("};")

    body = f"""#ifndef YOLO_CONV1_320_EXACT_DATA_H
#define YOLO_CONV1_320_EXACT_DATA_H
#include <stdint.h>
#define C1E_IN_W {IN_W}u
#define C1E_IN_H {IN_H}u
#define C1E_IC {IC}u
#define C1E_OC {OC}u
#define C1E_OUT_W {OUT_W}u
#define C1E_OUT_H {OUT_H}u
#define C1E_OUT_SPATIAL {SP_OUT}u
#define C1E_WGT_PER_OC {(IC//16)*9}u
#define C1E_ACT_WORDS {len(act_words)}u
#define C1E_WGT_WORDS {len(wgt_words)}u
#define C1E_PAD_VALUE {in_zp}
#define C1E_OUT_ZP {out_zp}
#define C1E_RTL_TOL {RTL_TOL}u

{u32a("yolo_conv1_320e_act_words", act_words)}

{u32a("yolo_conv1_320e_wgt_words", wgt_words)}

{i32("yolo_conv1_320e_bias_q", bias_q)}

{u32("yolo_conv1_320e_scale_mul", mul)}

{u32("yolo_conv1_320e_scale_shift", [Q_SHIFT]*OC)}

{u8("yolo_conv1_320e_silu_lut", lut)}

{chr(10).join(g8)}

#endif
"""
    OUT_PATH.write_text(body, encoding="ascii")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
