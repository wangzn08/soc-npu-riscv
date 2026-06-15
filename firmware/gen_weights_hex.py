# -*- coding: utf-8 -*-
# Offline weight packer: reproduces the EXACT tile-major layout that the firmware
# load_conv_weights()/pack_fc_tile() produce, and emits a $readmemh image of the
# DDR weight region (128-bit words, lane0 = LSB) so the CPU no longer has to pack
# weights byte-by-byte at runtime.
#
#   DDR layout (contiguous):
#     words [0     .. 4607]  conv1..conv6 packed (-> DMA to Wgt SRAM PING base 0)
#     words [4608  .. 8767]  FC1(4 tiles)+FC2  (-> DMA to Wgt SRAM PONG base 0)
import re, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

FW_DIR = Path(__file__).resolve().parent
SRC = FW_DIR / 'deepnet_weights.h'
OUT = FW_DIR / 'weights_ddr.hex'

src = SRC.read_text(encoding='utf-8', errors='replace')

def get_array(name):
    m = re.search(r'int8_t\s+' + name + r'\s*\[[^=]*=\s*\{(.*?)\};', src, re.DOTALL)
    if not m:
        raise RuntimeError('array not found: ' + name)
    return [int(x) for x in re.findall(r'-?\d+', m.group(1))]

# ---- conv packing: mirror load_conv_weights() ----
def pack_conv(W, oc_total, ic, kh=3, kw=3):
    khkw = kh * kw
    icg = (ic + 15) // 16
    words = []
    for p in range(oc_total // 16):
        for ocl in range(16):
            oc = p * 16 + ocl
            for g in range(icg):
                for ko in range(khkw):
                    word = []
                    for lane in range(16):
                        ich = g * 16 + lane
                        v = W[oc * ic * khkw + ich * khkw + ko] if ich < ic else 0
                        word.append(v & 0xFF)
                    words.append(word)
    return words

# ---- FC1 packing in CONV-OUTPUT order: lets the GEMM read the Pool3 output
# directly, eliminating the runtime CPU Pool3->FC1 transpose (reorder).
# Pool3 word w (0..63) = (tile=w//16, pos=w%16), 16 channels {tile*16..+15} at
# position pos.  GEMM IC-group w lane ch_in -> input (channel=tile*16+ch_in,
# position=pos).  Dot product is order-independent, so we just align the weight
# column to that input order: weight lane ch_in = trained_W[oc][channel*16+pos].
def pack_fc1_convorder(W, out_dim):
    words = []
    for t in range((out_dim + 15) // 16):
        for o in range(16):
            oc = t * 16 + o
            for w in range(64):           # IC-groups = Pool3 conv-output words
                tile = w // 16
                pos = w % 16
                word = []
                for ch_in in range(16):
                    ch = tile * 16 + ch_in
                    col = ch * 16 + pos    # trained channel-major column
                    v = W[oc * 1024 + col] if oc < out_dim else 0
                    word.append(v & 0xFF)
                words.append(word)
    return words

# ---- FC packing: mirror pack_fc_tile() over all OC tiles ----
def pack_fc(W, in_dim, out_dim):
    icg = (in_dim + 15) // 16
    words = []
    for t in range((out_dim + 15) // 16):
        for o in range(16):
            oc = t * 16 + o
            for g in range(icg):
                word = []
                for b in range(16):
                    ic = g * 16 + b
                    v = W[oc * in_dim + ic] if (oc < out_dim and ic < in_dim) else 0
                    word.append(v & 0xFF)
                words.append(word)
    return words

conv = []
conv += pack_conv(get_array('conv1_W'), 16, 1)
conv += pack_conv(get_array('conv2_W'), 16, 16)
conv += pack_conv(get_array('conv3_W'), 32, 16)
conv += pack_conv(get_array('conv4_W'), 32, 32)
conv += pack_conv(get_array('conv5_W'), 64, 32)
conv += pack_conv(get_array('conv6_W'), 64, 64)
fc = pack_fc1_convorder(get_array('affine1_W'), 50) + pack_fc(get_array('affine2_W'), 50, 10)

assert len(conv) == 4608, len(conv)
assert len(fc) == 4160, len(fc)
words = conv + fc

def word_hex(w):           # 128-bit hex, lane0 = LSB (rightmost)
    return ''.join('%02x' % w[15 - i] for i in range(16))

with OUT.open('w') as f:
    for w in words:
        f.write(word_hex(w) + '\n')

print('conv words:', len(conv), ' fc words:', len(fc), ' total:', len(words))
print('word[0]    (conv1 ocl0 g0 ko0):', word_hex(words[0]),
      '  lane0 should be conv1_W[0][0]=%d (0x%02x)' % (get_array('conv1_W')[0], get_array('conv1_W')[0] & 0xFF))
print('word[4608] (fc1 tile0 o0 g0):  ', word_hex(words[4608]))
print('written:', OUT)
