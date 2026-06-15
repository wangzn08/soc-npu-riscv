# -*- coding: utf-8 -*-
# Offline RAW image packer: produces the un-expanded MNIST images as a $readmemh
# image of the DDR activation region (models a camera/sensor writing raw bytes to
# SDRAM). 784 pixels/image are byte-packed, 16 pixels per 128-bit word (byte b of
# word w = pixel[w*16+b], little-endian within the word) -> 49 words/image.
#
# At runtime the firmware DMAs these 49 raw words into an Act-SRAM scratch region,
# then the HW img_expand engine expands each byte into a zero-extended 16-channel
# tile-major word (pixel in ch0) IN SRAM -- no offline pad, no CPU pixel scatter.
# img_expand's output is bit-identical to the old offline tile-major format.
#
#   DDR layout: image d occupies words [d*49 .. d*49+48]  (10*49 = 490 words)
import re, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

FW_DIR = Path(__file__).resolve().parent
SRC = FW_DIR / 'mnist_test_images.h'
OUT = FW_DIR / 'act_ddr.hex'

src = SRC.read_text(encoding='utf-8', errors='replace')

def get_array(name):
    m = re.search(r'int8_t\s+' + name + r'\s*\[[^=]*=\s*\{(.*?)\};', src, re.DOTALL)
    if not m:
        raise RuntimeError('array not found: ' + name)
    return [int(x) for x in re.findall(r'-?\d+', m.group(1))]

def pack_word(px16):           # px16[b] -> byte b (LSB-first within 128-bit word)
    w = 0
    for b in range(16):
        w |= (px16[b] & 0xFF) << (8 * b)
    return '%032x' % w

WORDS_PER_IMG = 49             # ceil(784 / 16)
words = []
for d in range(10):
    img = get_array('mnist_img_%d' % d)
    assert len(img) == 784, (d, len(img))
    for sw in range(WORDS_PER_IMG):
        chunk = img[sw * 16: sw * 16 + 16]
        words.append(pack_word(chunk))   # last chunk (sw=48) is pixels 768..783

with OUT.open('w') as f:
    for h in words:
        f.write(h + '\n')

print('images: 10  words/img: %d  total: %d' % (WORDS_PER_IMG, len(words)))
img0 = get_array('mnist_img_0')
nz = next((i for i in range(784) if img0[i] != 0), -1)
print('img0 first nonzero pixel at i=%d val=%d (word %d byte %d)'
      % (nz, img0[nz], nz // 16, nz % 16))
print('written:', OUT)
