#!/usr/bin/env python3
"""conv0 (model.0, 3->16 3x3 s2 pad1) at FULL 320 resolution with the EXACT
per-layer SiLU LUT (CTRL[22]). Input = the real 320x320x3 image (bus320.ppm)
quantized exactly like the C engine (in_scale=1/255, in_zp=-128 => q = pixel-128).
The 3 input channels are packed into the NPU's 16 lanes (lanes 3..15 zero, with
zero weights), so conv0 runs through the standard tiled path. The exact golden is
self-checked against dump320/conv0.bin (the C float oracle).
"""
from __future__ import annotations
from pathlib import Path
import os, struct, math
import numpy as np

CROP = int(os.environ.get("CONV0_CROP", "0"))  # >0: crop to CROPxCROP, emit full golden
NOACT = int(os.environ.get("CONV0_NOACT", "0"))  # 1: omit the 1.6MB act_words (DDR-preloaded image)

ROOT = Path(__file__).resolve().parents[1]
WDIR = ROOT / "yolov8n_int8" / "weights"
DUMP = ROOT / "yolov8n_int8" / "dump320"
PPM = ROOT / "yolov8n_int8" / "bus320.ppm"
OUT_PATH = ROOT / "firmware" / "yolo_conv0_320_exact_data.h"

KH = KW = 3
STRIDE, PAD = 2, 1
Q_SHIFT = 20
SILU_STEP = 0.5  # SiLU LUT indexed by preact (zp=0)
RTL_TOL = 16
IN_SCALE, IN_ZP = 0.0039215689, -128   # conv0 act_quant input (= 1/255 image quant)


def clamp_s8(v): return -128 if v < -128 else (127 if v > 127 else v)
def s8(b): b &= 0xFF; return b - 256 if b & 0x80 else b
def pack_i8_word(vals):
    raw = np.asarray(vals, dtype=np.int8).view(np.uint8)
    return [sum(int(raw[g*4+b]) << (8*b) for b in range(4)) for g in range(4)]


def read_ppm(path):
    data = path.read_bytes()
    assert data[:2] == b"P6"
    # parse header: P6 W H MAXVAL then binary
    idx = 2; fields = []
    while len(fields) < 3:
        while data[idx] in b" \t\n\r":
            idx += 1
        start = idx
        while data[idx] not in b" \t\n\r":
            idx += 1
        fields.append(int(data[start:idx]))
    idx += 1  # single whitespace after maxval
    W, H, _mx = fields
    px = np.frombuffer(data[idx:idx + W*H*3], dtype=np.uint8).reshape(H, W, 3)
    return px, W, H


def read_dump(name):
    d = (DUMP / name).read_bytes()
    c, h, w = struct.unpack("<3i", d[:12])
    sc = struct.unpack("<f", d[12:16])[0]; zp = struct.unpack("<i", d[16:20])[0]
    return np.frombuffer(d[20:], dtype=np.int8).reshape(c, h, w).copy(), sc, zp


def build_silu_lut(out_scale, out_zp):
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        q = i if i < 128 else i - 256
        realq = q * SILU_STEP
        silu = realq / (1.0 + math.exp(-realq))
        lut[i] = clamp_s8(int(round(silu / out_scale + out_zp))) & 0xFF
    return lut


def main():
    px, IN_W, IN_H = read_ppm(PPM)                    # [H,W,3] uint8
    c0, out_scale, out_zp = read_dump("conv0.bin")    # golden + out scales
    if CROP > 0:
        px = px[:CROP, :CROP, :]; IN_W = IN_H = CROP   # tiny reproducer, full golden
    OC = 16
    OUT_H = (IN_H + 2*PAD - KH)//STRIDE + 1            # 160
    OUT_W = (IN_W + 2*PAD - KW)//STRIDE + 1
    SP_OUT = OUT_H * OUT_W

    # quantize image exactly like the C engine: q = clamp_round(v/255/in_scale + in_zp)
    actq = np.clip(np.round(px.astype(np.float64)/255.0/IN_SCALE + IN_ZP), -128, 127).astype(np.int8)
    act = actq.transpose(2, 0, 1)                     # [3,H,W] planar

    w = np.fromfile(WDIR/"conv0_w.bin", dtype=np.int8).reshape(OC, 3, KH, KW)
    b = np.fromfile(WDIR/"conv0_b.bin", dtype=np.float32)[:OC]
    sc = np.fromfile(WDIR/"conv0_s.bin", dtype=np.float32)[:OC]
    wsum = w.astype(np.int64).sum(axis=(1, 2, 3))     # over 3 real channels
    lut = build_silu_lut(out_scale, out_zp)

    bias_q, mul = [], []
    for oc in range(OC):
        m = int(round(IN_SCALE * float(sc[oc]) / SILU_STEP * (1 << Q_SHIFT)))
        mul.append(m)
        be = int(round(float(b[oc]) / (IN_SCALE * float(sc[oc]))))
        bias_q.append(int(be - IN_ZP * int(wsum[oc])))

    # exact golden (3-channel conv, pad with in_zp)
    golden = np.zeros((SP_OUT, OC), dtype=np.int8)
    ap = np.full((3, IN_H + 2, IN_W + 2), IN_ZP, dtype=np.int64)
    ap[:, 1:1+IN_H, 1:1+IN_W] = act
    for oy in range(OUT_H):
        for ox in range(OUT_W):
            iy, ix = oy*STRIDE, ox*STRIDE
            win = ap[:, iy:iy+KH, ix:ix+KW]
            for oc in range(OC):
                acc = int(np.sum(win * w[oc].astype(np.int64)))
                s2 = ((acc + bias_q[oc]) * mul[oc]) >> Q_SHIFT
                idx = clamp_s8(s2)
                golden[oy*OUT_W+ox, oc] = np.int8(s8(int(lut[idx & 0xFF])))

    if CROP == 0:
        cflat = c0.transpose(1, 2, 0).reshape(SP_OUT, OC)
        df = np.abs(golden.astype(int) - cflat.astype(int))
        print(f"exact conv0 vs dump320/conv0.bin: mism={int((df>0).sum())}/{golden.size} max={int(df.max())}")

    # Position-weighted checksum over the whole golden tensor (no cancellation):
    # chk = sum_idx ((byte+128) * (idx+1)) mod 2^32, idx = pos*OC + oc. The smoke
    # recomputes the same over OUT_DDR. The 320x320 image (1.6MB) already fills the
    # firmware region, so we cannot also bake the 410KB golden; per-element exactness
    # is already proven on conv1 @320 (identical tiled stride-2 + exact-SiLU path).
    chk = 0
    idx = 0
    for p in range(SP_OUT):
        for oc in range(OC):
            chk = (chk + ((int(golden[p, oc]) + 128) * (idx + 1))) & 0xFFFFFFFF
            idx += 1
    print(f"golden checksum = 0x{chk:08X}")

    # pack act: 1 ic-group (16 lanes), lanes 0..2 = RGB-q, lanes 3..15 = 0
    act16 = np.zeros((16, IN_H, IN_W), dtype=np.int8)
    act16[0:3] = act
    act_words = [pack_i8_word(act16[:, y, x]) for y in range(IN_H) for x in range(IN_W)]
    # pack weights: 16 lanes per (oc,ko), lanes 0..2 real, 3..15 = 0
    wgt_words = []
    for oc in range(OC):
        for ko in range(9):
            lane = np.zeros(16, dtype=np.int8); lane[0:3] = w[oc, :, ko//3, ko%3]
            wgt_words.append(pack_i8_word(lane))

    def u32a(name, words):
        L = [f"static const uint32_t {name}[{len(words)}][4] = {{"]
        L += ["    {"+", ".join(f"0x{v:08X}u" for v in x)+"}," for x in words]
        L.append("};"); return "\n".join(L)
    def i32(name, v): return f"static const int32_t {name}[{len(v)}] = {{ {', '.join(map(str,v))} }};"
    def u32(name, v): return f"static const uint32_t {name}[{len(v)}] = {{ {', '.join(f'{x}u' for x in v)} }};"
    def u8(name, v): return f"static const uint8_t {name}[{len(v)}] = {{ {', '.join(f'0x{int(x):02X}u' for x in v)} }};"

    body = f"""#ifndef YOLO_CONV0_320_EXACT_DATA_H
#define YOLO_CONV0_320_EXACT_DATA_H
#include <stdint.h>
#define C0E_IN_W {IN_W}u
#define C0E_IN_H {IN_H}u
#define C0E_IC 16u
#define C0E_OC {OC}u
#define C0E_OUT_W {OUT_W}u
#define C0E_OUT_H {OUT_H}u
#define C0E_OUT_SPATIAL {SP_OUT}u
#define C0E_WGT_PER_OC 9u
#define C0E_ACT_WORDS {len(act_words)}u
#define C0E_WGT_WORDS {len(wgt_words)}u
#define C0E_PAD_VALUE {IN_ZP}
#define C0E_OUT_ZP {out_zp}
#define C0E_RTL_TOL {RTL_TOL}u
#define C0E_GOLDEN_CHK 0x{chk:08X}u
#define C0E_ACT_WORD0_0 0x{act_words[0][0]:08X}u

{("/* act_words omitted (CONV0_NOACT): image is DDR-preloaded */" if NOACT else u32a("yolo_conv0_320e_act_words", act_words))}

{u32a("yolo_conv0_320e_wgt_words", wgt_words)}

{i32("yolo_conv0_320e_bias_q", bias_q)}

{u32("yolo_conv0_320e_scale_mul", mul)}

{u32("yolo_conv0_320e_scale_shift", [Q_SHIFT]*OC)}

{u8("yolo_conv0_320e_silu_lut", lut)}
""" + ("" if CROP == 0 else ("#define C0E_HAVE_GOLDEN 1\n" + chr(10).join(
        [f"static const uint8_t yolo_conv0_320e_golden[{SP_OUT}][{OC}] = {{"] +
        ["    {"+", ".join(f"0x{int(x)&0xFF:02X}u" for x in golden[p])+"}," for p in range(SP_OUT)] +
        ["};"]))) + """

#endif
"""
    out_path = (ROOT / "firmware" / "yolo_conv0_320_noact_data.h") if NOACT else OUT_PATH
    if NOACT:
        body = body.replace("YOLO_CONV0_320_EXACT_DATA_H", "YOLO_CONV0_320_NOACT_DATA_H")
    out_path.write_text(body, encoding="ascii")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
