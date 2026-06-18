#!/usr/bin/env python3
"""Generate a tiny real-weight YOLO pointwise-conv smoke header.

The generated fixture uses YOLOv8n conv2 real INT8 weights, but a tiny 2x2
synthetic activation block.  Golden values follow the RTL post-process path:
    q = (psum + bias) * scale_mul >>> scale_shift
    out = ReLU(q), clipped to 127
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_PATH = ROOT / "yolov8n_int8" / "weights" / "conv2_w.bin"
OUT_PATH = ROOT / "firmware" / "yolo_pwconv_real_data.h"

IN_W = 2
IN_H = 2
SPATIAL = IN_W * IN_H
IC = 32
OC = 16
IC_GROUPS = IC // 16
SCALE_MUL = 1
SCALE_SHIFT = 7


def pack_i8_word(vals: np.ndarray) -> list[int]:
    if vals.shape != (16,):
        raise ValueError(f"expected 16 lanes, got {vals.shape}")
    lanes: list[int] = []
    raw = vals.astype(np.int8).view(np.uint8)
    for group in range(4):
        word = 0
        for b in range(4):
            word |= int(raw[group * 4 + b]) << (8 * b)
        lanes.append(word)
    return lanes


def make_activation() -> np.ndarray:
    act = np.zeros((IC, IN_H, IN_W), dtype=np.int8)
    for ic in range(IC):
        for y in range(IN_H):
            for x in range(IN_W):
                pos = y * IN_W + x
                act[ic, y, x] = np.int8(1 + pos + (ic % 5))
    return act


def format_u32_array(name: str, words: list[list[int]]) -> str:
    lines = [f"static const uint32_t {name}[{len(words)}][4] = {{"]
    for lanes in words:
        lines.append("    {" + ", ".join(f"0x{x:08X}u" for x in lanes) + "},")
    lines.append("};")
    return "\n".join(lines)


def main() -> None:
    weights = np.fromfile(WEIGHT_PATH, dtype=np.int8)
    expected_size = 32 * 32
    if weights.size != expected_size:
        raise SystemExit(f"{WEIGHT_PATH} has {weights.size} bytes, expected {expected_size}")
    weights = weights.reshape(32, 32)

    act = make_activation()

    act_words: list[list[int]] = []
    for icg in range(IC_GROUPS):
        for pos in range(SPATIAL):
            y = pos // IN_W
            x = pos % IN_W
            act_words.append(pack_i8_word(act[icg * 16:(icg + 1) * 16, y, x]))

    wgt_words: list[list[int]] = []
    for oc in range(OC):
        for icg in range(IC_GROUPS):
            wgt_words.append(pack_i8_word(weights[oc, icg * 16:(icg + 1) * 16]))

    expected = np.zeros((SPATIAL, OC), dtype=np.uint8)
    for pos in range(SPATIAL):
        y = pos // IN_W
        x = pos % IN_W
        avec = act[:, y, x].astype(np.int32)
        for oc in range(OC):
            psum = int(np.sum(avec * weights[oc, :].astype(np.int32)))
            q = psum >> SCALE_SHIFT
            if q < 0:
                q = 0
            if q > 127:
                q = 127
            expected[pos, oc] = q

    bias = [0] * OC

    body = f"""#ifndef YOLO_PWCONV_REAL_DATA_H
#define YOLO_PWCONV_REAL_DATA_H

#include <stdint.h>

#define YOLO_REAL_PW_IN_W {IN_W}u
#define YOLO_REAL_PW_IN_H {IN_H}u
#define YOLO_REAL_PW_IC {IC}u
#define YOLO_REAL_PW_OC {OC}u
#define YOLO_REAL_PW_ACT_WORDS {len(act_words)}u
#define YOLO_REAL_PW_WGT_WORDS {len(wgt_words)}u
#define YOLO_REAL_PW_OUT_WORDS {SPATIAL}u
#define YOLO_REAL_PW_SCALE_MUL {SCALE_MUL}u
#define YOLO_REAL_PW_SCALE_SHIFT {SCALE_SHIFT}u

{format_u32_array("yolo_real_pw_act_words", act_words)}

{format_u32_array("yolo_real_pw_wgt_words", wgt_words)}

static const int32_t yolo_real_pw_bias[{OC}] = {{
    {", ".join(f"{x}" for x in bias)}
}};

static const uint8_t yolo_real_pw_expected[{SPATIAL}][{OC}] = {{
"""
    for pos in range(SPATIAL):
        body += "    {" + ", ".join(f"{int(x)}u" for x in expected[pos]) + "},\n"
    body += """};

#endif
"""

    OUT_PATH.write_text(body, encoding="ascii")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
