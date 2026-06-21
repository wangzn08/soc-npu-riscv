#!/usr/bin/env python3
"""Tiny synthetic probe to root-cause the conv0 <16-real-IC issue. A 4x4x16 input
with only lanes 0..2 non-zero (lanes 3..15 = 0), 3x3 stride1 pad0 conv, OC=16,
bias=0, scale_mul=1, scale_shift=0, INT32_OUT => the NPU writes the RAW int32
accumulator (acc) per OC. Compare to the pure 3-channel MAC golden: if they
differ, the NPU corrupts acc for zero input lanes."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "firmware" / "conv0_accprobe_data.h"
IN_W = IN_H = 4
OC = 16
KH = KW = 3
OUT_W = IN_W - KW + 1  # 2 (stride1 pad0)
OUT_H = IN_H - KH + 1
SP_OUT = OUT_W * OUT_H

rng = np.random.default_rng(7)
# input: 16 lanes, only 0..2 non-zero
act = np.zeros((16, IN_H, IN_W), dtype=np.int8)
act[0:3] = rng.integers(-40, 40, size=(3, IN_H, IN_W)).astype(np.int8)
# weights: 16 oc x 16 ic x 3 x 3, only ic 0..2 non-zero
w = np.zeros((OC, 16, KH, KW), dtype=np.int8)
w[:, 0:3] = rng.integers(-30, 30, size=(OC, 3, KH, KW)).astype(np.int8)


def pack(vals):
    raw = np.asarray(vals, dtype=np.int8).view(np.uint8)
    return [sum(int(raw[g*4+b]) << (8*b) for b in range(4)) for g in range(4)]


# golden: pure MAC (acc) per output pixel, per oc
acc = np.zeros((SP_OUT, OC), dtype=np.int64)
for oy in range(OUT_H):
    for ox in range(OUT_W):
        win = act[:, oy:oy+KH, ox:ox+KW]
        for oc in range(OC):
            acc[oy*OUT_W+ox, oc] = int(np.sum(win.astype(np.int64) * w[oc].astype(np.int64)))

act_words = [pack(act[:, y, x]) for y in range(IN_H) for x in range(IN_W)]
wgt_words = []
for oc in range(OC):
    for ko in range(9):
        wgt_words.append(pack(w[oc, :, ko//3, ko%3]))


def u32a(name, words):
    L = [f"static const uint32_t {name}[{len(words)}][4] = {{"]
    L += ["    {"+", ".join(f"0x{v:08X}u" for v in x)+"}," for x in words]
    L.append("};"); return "\n".join(L)
def i32a(name, m):
    L = [f"static const int32_t {name}[{m.shape[0]}][{m.shape[1]}] = {{"]
    L += ["    {"+", ".join(str(int(v)) for v in m[p])+"}," for p in range(m.shape[0])]
    L.append("};"); return "\n".join(L)


body = f"""#ifndef CONV0_ACCPROBE_DATA_H
#define CONV0_ACCPROBE_DATA_H
#include <stdint.h>
#define AP_IN_W {IN_W}u
#define AP_IN_H {IN_H}u
#define AP_OC {OC}u
#define AP_OUT_W {OUT_W}u
#define AP_OUT_H {OUT_H}u
#define AP_OUT_SPATIAL {SP_OUT}u
#define AP_ACT_WORDS {len(act_words)}u
#define AP_WGT_WORDS {len(wgt_words)}u
#define AP_WGT_PER_OC 9u

{u32a("ap_act_words", act_words)}

{u32a("ap_wgt_words", wgt_words)}

{i32a("ap_acc_golden", acc)}

#endif
"""
OUT.write_text(body, encoding="ascii")
print(f"wrote {OUT}; SP_OUT={SP_OUT} acc[0]={acc[0].tolist()}")
