#!/usr/bin/env python3
"""Emit firmware/yolo_img_ddr.hex: the 320x320x3 image (bus320.ppm) quantized
like the C engine (q = pixel-128) and packed tile-major into the NPU's 16-lane
words (lanes 0..2 = R,G,B; lanes 3..15 = 0), row-major. The shared-memory model
$readmemh-loads this into DDR at word base 0x40000 (CPU addr 0x4040_0000) under
+define+YOLO_DDR, so conv0 reads the image from DDR without baking 1.6MB into the
firmware region. One 128-bit word per line, MSB-first (lane15..lane0)."""
from pathlib import Path
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
PPM = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "yolov8n_int8" / "bus320.ppm"
OUT = ROOT / "firmware" / "yolo_img_ddr.hex"
IN_SCALE, IN_ZP = 0.0039215689, -128


def read_ppm(path):
    data = path.read_bytes()
    assert data[:2] == b"P6"
    idx = 2; fields = []
    while len(fields) < 3:
        while data[idx] in b" \t\n\r": idx += 1
        start = idx
        while data[idx] not in b" \t\n\r": idx += 1
        fields.append(int(data[start:idx]))
    idx += 1
    W, H, _ = fields
    px = np.frombuffer(data[idx:idx + W*H*3], dtype=np.uint8).reshape(H, W, 3)
    return px, W, H


def main():
    px, W, H = read_ppm(PPM)
    q = np.clip(np.round(px.astype(np.float64)/255.0/IN_SCALE + IN_ZP), -128, 127).astype(np.int8)
    # 16-lane word per pixel: lanes 0..2 = R,G,B, lanes 3..15 = 0
    lines = []
    for y in range(H):
        for x in range(W):
            lane = np.zeros(16, dtype=np.int8)
            lane[0:3] = q[y, x, :]
            b = lane.view(np.uint8)              # 16 bytes, lane0..lane15
            # 128-bit word MSB-first: lane15 is the high byte
            word = "".join(f"{int(b[15-k]):02X}" for k in range(16))
            lines.append(word)
    OUT.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"wrote {OUT}: {len(lines)} words ({H}x{W}), DDR word base 0x40000 (0x4040_0000)")


if __name__ == "__main__":
    main()
