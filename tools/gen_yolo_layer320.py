#!/usr/bin/env python3
"""Generic @320 single-conv tiled measurement fixture.
Usage: python tools/gen_yolo_layer320.py <ci>
Measures a trunk conv (fed by a single dumped conv golden) at full 320 resolution
through the tiled primitive. Emits firmware/yolo_layer320_data.h (LYR_* macros).

Config: ci -> (feeder conv index, oc, ic, kh, kw, stride, pad). All listed convs
have SiLU (backbone). Input golden = dump320/conv<feeder>.bin; output validated
against an RTL-model golden (SiLU LUT) at full resolution.
"""
from __future__ import annotations
from pathlib import Path
import struct, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WDIR = ROOT / "yolov8n_int8" / "weights"
DUMP = ROOT / "yolov8n_int8" / "dump320"
LUT_PATH = ROOT / "rtl" / "silu_lut_q4_4.hex"
OUT_PATH = ROOT / "firmware" / "yolo_layer320_data.h"

Q_SHIFT, REQ_SHIFT, RTL_TOL = 20, 12, 16

# ci -> (feeder_ci, oc, ic, kh, kw, stride, pad)
CFG = {
    2:  (1,   32,  32, 1, 1, 1, 0),   # c2f_2 cv1 (1x1) on 160x160x32 -> tiled pointwise
    1:  (0,   32,  16, 3, 3, 2, 1),
    6:  (5,   64,  32, 3, 3, 2, 1),
    13: (12,  128, 64, 3, 3, 2, 1),
    20: (19,  256, 128, 3, 3, 2, 1),
}


def s8(b): b &= 0xFF; return b - 256 if b & 0x80 else b
def clamp_s8(v): return 0x80 if v < -128 else (0x7F if v > 127 else v & 0xFF)
def load_lut():
    v = [int(x, 16) & 0xFF for x in LUT_PATH.read_text().split()]; assert len(v) == 256; return v
def rtl_silu(v, lut): v = -128 if v < -128 else (127 if v > 127 else v); return lut[v & 0xFF]
def pack_i8_word(vals):
    raw = vals.astype(np.int8).view(np.uint8)
    return [sum(int(raw[g*4+b]) << (8*b) for b in range(4)) for g in range(4)]
def read_dump(name):
    with open(DUMP / name, "rb") as f:
        c, h, w = struct.unpack("<3i", f.read(12)); sc = struct.unpack("<f", f.read(4))[0]
        zp = struct.unpack("<i", f.read(4))[0]
        d = np.frombuffer(f.read(), dtype=np.int8).reshape(c, h, w).copy()
    return d, sc, zp


def main():
    ci = int(sys.argv[1])
    feeder, OC, IC, KH, KW, STRIDE, PAD = CFG[ci]
    act, in_scale, in_zp = read_dump(f"conv{feeder}.bin")
    _, out_scale, out_zp = read_dump(f"conv{ci}.bin")
    assert act.shape[0] == IC, f"feeder ic {act.shape[0]} != {IC}"
    _, IN_H, IN_W = act.shape
    OUT_H = (IN_H + 2*PAD - KH)//STRIDE + 1
    OUT_W = (IN_W + 2*PAD - KW)//STRIDE + 1
    SP_OUT = OUT_H * OUT_W

    w = np.fromfile(WDIR/f"conv{ci}_w.bin", dtype=np.int8).reshape(OC, IC, KH, KW)
    b = np.fromfile(WDIR/f"conv{ci}_b.bin", dtype=np.float32)[:OC]
    sc = np.fromfile(WDIR/f"conv{ci}_s.bin", dtype=np.float32)[:OC]
    lut = load_lut(); wsum = w.astype(np.int32).sum(axis=(1, 2, 3))

    bias_q, mul = [], []
    for oc in range(OC):
        m = int(round(in_scale * float(sc[oc]) * 16.0 * (1 << Q_SHIFT))); mul.append(m)
        be = int(round(float(b[oc]) * 16.0 * (1 << Q_SHIFT) / m))
        bias_q.append(int(be - in_zp * int(wsum[oc])))
    requant_mul = int(round((1 << REQ_SHIFT) / (16.0 * out_scale)))

    ap = np.full((IC, IN_H+2*PAD, IN_W+2*PAD), in_zp, dtype=np.int32)
    ap[:, PAD:PAD+IN_H, PAD:PAD+IN_W] = act
    golden = np.zeros((SP_OUT, OC), dtype=np.int8)
    for oy in range(OUT_H):
        for ox in range(OUT_W):
            win = ap[:, oy*STRIDE:oy*STRIDE+KH, ox*STRIDE:ox*STRIDE+KW]
            for oc in range(OC):
                acc = int(np.sum(win * w[oc].astype(np.int32)))
                q44 = ((acc + bias_q[oc]) * mul[oc]) >> Q_SHIFT
                rq = ((s8(rtl_silu(q44, lut)) * requant_mul) >> REQ_SHIFT) + out_zp
                golden[oy*OUT_W+ox, oc] = np.int8(s8(clamp_s8(rq)))

    act_words = [pack_i8_word(act[g*16:g*16+16, y, x])
                 for g in range(IC//16) for y in range(IN_H) for x in range(IN_W)]
    wgt_words = [pack_i8_word(w[oc, g*16:g*16+16, ko//KW, ko%KW])
                 for oc in range(OC) for g in range(IC//16) for ko in range(KH*KW)]

    def u32a(name, words):
        L = [f"static const uint32_t {name}[{len(words)}][4] = {{"]
        L += ["    {"+", ".join(f"0x{v:08X}u" for v in x)+"}," for x in words]; L.append("};"); return "\n".join(L)
    g8 = [f"static const uint8_t yolo_layer320_golden[{SP_OUT}][{OC}] = {{"]
    g8 += ["    {"+", ".join(f"0x{int(x)&0xFF:02X}u" for x in golden[p])+"}," for p in range(SP_OUT)]; g8.append("};")

    body = f"""#ifndef YOLO_LAYER320_DATA_H
#define YOLO_LAYER320_DATA_H
#include <stdint.h>
#define LYR_CI {ci}
#define LYR_IN_W {IN_W}u
#define LYR_IN_H {IN_H}u
#define LYR_IC {IC}u
#define LYR_OC {OC}u
#define LYR_KH {KH}u
#define LYR_KW {KW}u
#define LYR_STRIDE {STRIDE}u
#define LYR_PAD {PAD}u
#define LYR_OUT_W {OUT_W}u
#define LYR_OUT_H {OUT_H}u
#define LYR_OUT_SPATIAL {SP_OUT}u
#define LYR_WGT_PER_OC {(IC//16)*KH*KW}u
#define LYR_ACT_WORDS {len(act_words)}u
#define LYR_WGT_WORDS {len(wgt_words)}u
#define LYR_PAD_VALUE {in_zp}
#define LYR_REQUANT_MUL {requant_mul}u
#define LYR_REQUANT_SHIFT {REQ_SHIFT}u
#define LYR_REQUANT_ZP {out_zp}
#define LYR_RTL_TOL {RTL_TOL}u

{u32a("yolo_layer320_act_words", act_words)}

{u32a("yolo_layer320_wgt_words", wgt_words)}

static const int32_t yolo_layer320_bias_q[{OC}] = {{ {", ".join(map(str, bias_q))} }};
static const uint32_t yolo_layer320_scale_mul[{OC}] = {{ {", ".join(f"{x}u" for x in mul)} }};
static const uint32_t yolo_layer320_scale_shift[{OC}] = {{ {", ".join(f"{Q_SHIFT}u" for _ in range(OC))} }};

{chr(10).join(g8)}

#endif
"""
    OUT_PATH.write_text(body, encoding="ascii")
    print(f"wrote {OUT_PATH}: conv{ci} {IN_W}x{IN_H}x{IC} -> {OUT_W}x{OUT_H}x{OC}, {len(act_words)} act words")


if __name__ == "__main__":
    main()
