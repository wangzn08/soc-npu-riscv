#!/usr/bin/env python3
"""Pack ALL 64 conv weights into one DDR blob (firmware/yolo_weights_ddr.hex) so
the full net is not limited by the 1.96MB firmware region. Each conv is packed
tile-major (1x1: oc*(ic/16); 3x3: oc*(ic/16)*9), with IC zero-padded to a 16-lane
multiple (conv0's IC=3 -> 16, lanes 3..15 zero). A small offset map
(firmware/yolo_weight_map.h) gives each layer's DDR word base + word count + dims.
The shared-memory model $readmemh-loads the blob at DDR word base 0x80000 (CPU
addr 0x4080_0000) under +define+YOLO_DDR. Weights are activation-independent, so
this blob is identical for any input image."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WDIR = ROOT / "yolov8n_int8" / "weights"
META = ROOT / "yolov8n_int8" / "act_quant_meta.json"
OUT_HEX = ROOT / "firmware" / "yolo_weights_ddr.hex"
OUT_MAP = ROOT / "firmware" / "yolo_weight_map.h"
DDR_WORD_BASE = 0x80000   # CPU addr 0x4080_0000


def pack_i8_word(vals):
    raw = np.asarray(vals, dtype=np.int8).view(np.uint8)
    return [sum(int(raw[g*4+b]) << (8*b) for b in range(4)) for g in range(4)]


def main():
    meta = json.load(open(META))
    convs = meta["convs"] if isinstance(meta, dict) else meta
    assert len(convs) == 64

    words = []          # list of [4]uint32 (128-bit) words
    offsets, counts, dims = [], [], []
    for i, c in enumerate(convs):
        oc, ic, kh, kw = c["oc"], c["ic"], c["kh"], c["kw"]
        pic = ((ic + 15) // 16) * 16
        # Prefer a folded weight override if a C2f generator emitted one (concat
        # per-source requant folded into this cv2's int8 weights). Same shape/order.
        wfold = WDIR / f"conv{i}_w_folded.bin"
        wsrc = wfold if wfold.exists() else (WDIR / f"conv{i}_w.bin")
        w = np.fromfile(wsrc, dtype=np.int8).reshape(oc, ic, kh, kw)
        wp = np.zeros((oc, pic, kh, kw), dtype=np.int8)
        wp[:, 0:ic] = w
        off = len(words)
        for o in range(oc):
            for g in range(pic // 16):
                for ko in range(kh * kw):
                    words.append(pack_i8_word(wp[o, g*16:g*16+16, ko // kw, ko % kw]))
        offsets.append(off)
        counts.append(len(words) - off)
        dims.append((oc, ic, kh, kw))

    # write hex (128-bit MSB-first per line: lane15..lane0 == word[3]..word[0])
    lines = []
    for wd in words:
        lines.append("".join(f"{wd[3-k]:08X}" for k in range(4)))
    OUT_HEX.write_text("\n".join(lines) + "\n", encoding="ascii")

    # write map header
    hl = ["#ifndef YOLO_WEIGHT_MAP_H", "#define YOLO_WEIGHT_MAP_H", "#include <stdint.h>", ""]
    hl.append(f"#define YOLO_WGT_DDR_BASE 0x{0x40000000 + DDR_WORD_BASE*16:08X}u")
    hl.append(f"#define YOLO_WGT_TOTAL_WORDS {len(words)}u")
    hl.append("// per-conv: {ddr_word_off, wgt_words, oc, ic, kh, kw}")
    hl.append("typedef struct { uint32_t off, words, oc, ic, kh, kw; } yolo_wgt_ent_t;")
    hl.append("static const yolo_wgt_ent_t yolo_wgt_map[64] = {")
    for i in range(64):
        oc, ic, kh, kw = dims[i]
        hl.append(f"    {{{offsets[i]}u, {counts[i]}u, {oc}u, {ic}u, {kh}u, {kw}u}},  // conv{i}")
    hl.append("};")
    hl.append("#endif")
    OUT_MAP.write_text("\n".join(hl) + "\n", encoding="ascii")

    print(f"wrote {OUT_HEX}: {len(words)} words ({len(words)*16//1024} KB), base 0x40800000")
    print(f"wrote {OUT_MAP}")


if __name__ == "__main__":
    main()
