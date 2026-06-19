#!/usr/bin/env python3
"""Bake the c2f_2 (model.2) block input = conv1 output (32ch, 160x16) as a
tile-major fixture, so the generic-C2f-runner smoke can feed the block directly
without re-running the conv0->conv1 prefix."""

from __future__ import annotations
from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "firmware" / "yolo_c2f2_blockin_data.h"
CONV0_GEN = ROOT / "tools" / "gen_yolo_conv0_strip_real_smoke.py"
CONV1_GEN = ROOT / "tools" / "gen_yolo_conv1_from_conv0_chain_smoke.py"
CONV2_GEN = ROOT / "tools" / "gen_yolo_conv2_from_conv1_chain_smoke.py"

IN_W, IN_H, IC = 160, 16, 32
SP = IN_W * IN_H


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    conv0 = load("c0", CONV0_GEN)
    conv1 = load("c1", CONV1_GEN)
    conv2 = load("c2", CONV2_GEN)
    act = conv2.make_conv1_output(conv0, conv1)  # [32,16,160]
    words = []
    for g in range(IC // 16):
        for pos in range(SP):
            y, x = pos // IN_W, pos % IN_W
            words.append(conv0.pack_i8_word(act[g*16:(g+1)*16, y, x]))
    lines = [f"static const uint32_t yolo_c2f2_blockin_words[{len(words)}][4] = {{"]
    for w in words:
        lines.append("    {" + ", ".join(f"0x{x:08X}u" for x in w) + "},")
    lines.append("};")
    body = ("#ifndef YOLO_C2F2_BLOCKIN_DATA_H\n#define YOLO_C2F2_BLOCKIN_DATA_H\n\n"
            "#include <stdint.h>\n\n"
            f"#define YOLO_C2F2_BLOCKIN_WORDS {len(words)}u\n\n"
            + "\n".join(lines) + "\n\n#endif\n")
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
