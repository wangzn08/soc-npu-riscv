#!/usr/bin/env python3
# Golden model: replicate deepnet_deploy.c integer arithmetic EXACTLY.
# Parses deepnet_weights.h + mnist_test_images.h, runs digit 0 & 1.
import re, sys, os

HDIR = os.path.dirname(os.path.abspath(__file__))

def load(fname):
    with open(os.path.join(HDIR, fname), 'r') as f:
        return f.read()

def parse_array(text, name):
    # match: <name> [..][..] = { ... };
    m = re.search(re.escape(name) + r'\s*(\[[^=]*?\])\s*=\s*\{(.*?)\};', text, re.S)
    if not m:
        raise RuntimeError("not found: " + name)
    dims = [int(d) for d in re.findall(r'\[(\d+)\]', m.group(1))]
    nums = [int(x) for x in re.findall(r'-?\d+', m.group(2))]
    return dims, nums

W = load('deepnet_weights.h')
IMG = load('mnist_test_images.h')

def arr(name):
    _, nums = parse_array(W, name)
    return nums

conv_W = {i: arr('conv%d_W' % i) for i in range(1, 7)}
conv_b = {i: arr('conv%d_b' % i) for i in range(1, 7)}
aff1_W = arr('affine1_W'); aff1_b = arr('affine1_b')
aff2_W = arr('affine2_W'); aff2_b = arr('affine2_b')

# scales from deepnet_deploy.c
SCALE = {1:8389,2:5381,3:5587,4:4959,5:5185,6:4119}
SCALE_AFF1 = 3907; SCALE_AFF2 = 7349; SHIFT = 20

def conv(inp, IC, IH, IW, Wt, b, OC, scale, pad=1):
    # inp: flat [IC][IH][IW] (channel-major here for golden simplicity)
    OH = IH + 2*pad - 3 + 1
    OW = IW + 2*pad - 3 + 1
    out = [0]*(OC*OH*OW)
    for oc in range(OC):
        for oy in range(OH):
            for ox in range(OW):
                acc = b[oc]
                for ic in range(IC):
                    for ky in range(3):
                        for kx in range(3):
                            iy = oy + ky - pad
                            ix = ox + kx - pad
                            if 0 <= iy < IH and 0 <= ix < IW:
                                a = inp[(ic*IH + iy)*IW + ix]
                                w = Wt[(oc*IC + ic)*9 + ky*3 + kx]
                                acc += a*w
                val = (acc * scale) >> SHIFT
                if val < 0: val = 0
                if val > 127: val = 127
                out[(oc*OH + oy)*OW + ox] = val
    return out, OC, OH, OW

def pool(inp, C, H, W_):
    OH, OW = H//2, W_//2
    out=[0]*(C*OH*OW)
    for c in range(C):
        for oy in range(OH):
            for ox in range(OW):
                m=0
                for dy in range(2):
                    for dx in range(2):
                        v=inp[(c*H+oy*2+dy)*W_+ox*2+dx]
                        if v>m: m=v
                out[(c*OH+oy)*OW+ox]=m
    return out, C, OH, OW

def stats(name, a):
    mx=max(abs(v) for v in a); nz=sum(1 for v in a if v!=0)
    print("  [%s] max=%d nz=%d/%d" % (name, mx, nz, len(a)))

def affine(inp, IN, OUT, Wt, b, scale, relu):
    out=[0]*OUT
    for oc in range(OUT):
        acc=b[oc]
        for ic in range(IN):
            acc += inp[ic]*Wt[oc*IN+ic]
        val=(acc*scale)>>SHIFT
        if relu and val<0: val=0
        if val>127: val=127
        if val< -128: val=-128
        out[oc]=val
    return out

def run(digit):
    _, img = parse_array(IMG, 'mnist_img_%d' % digit)  # 784 channel0
    print("Digit %d:" % digit)
    # conv1: IC=1
    x,C,H,Wd = conv(img, 1,28,28, conv_W[1], conv_b[1], 16, SCALE[1]); stats("Conv1",x)
    for (py,px) in [(0,0),(14,14),(7,10),(14,0)]:
        vals=[x[(oc*28+py)*28+px] for oc in range(16)]
        print("    conv1(%d,%d) ch0-15:"%(py,px), ' '.join('%02x'%v for v in vals))
    x,C,H,Wd = conv(x, 16,28,28, conv_W[2], conv_b[2], 16, SCALE[2]); stats("Conv2",x)
    x,C,H,Wd = pool(x,16,28,28); stats("Pool1",x)
    x,C,H,Wd = conv(x,16,14,14, conv_W[3], conv_b[3], 32, SCALE[3]); stats("Conv3",x)
    print("    conv3(7,7) ch0-15:", ' '.join('%02x'%(x[(oc*14+7)*14+7]&0xff) for oc in range(16)))
    print("    conv3(7,7) ch16-31:", ' '.join('%02x'%(x[(oc*14+7)*14+7]&0xff) for oc in range(16,32)))
    x,C,H,Wd = conv(x,32,14,14, conv_W[4], conv_b[4], 32, SCALE[4], pad=2); stats("Conv4",x)
    print("    conv4(8,8) ch0-15:", ' '.join('%02x'%(x[(oc*16+8)*16+8]&0xff) for oc in range(16)))
    print("    conv4(8,8) ch16-31:", ' '.join('%02x'%(x[(oc*16+8)*16+8]&0xff) for oc in range(16,32)))
    x,C,H,Wd = pool(x,32,16,16); stats("Pool2",x)
    x,C,H,Wd = conv(x,32,8,8, conv_W[5], conv_b[5], 64, SCALE[5]); stats("Conv5",x)
    x,C,H,Wd = conv(x,64,8,8, conv_W[6], conv_b[6], 64, SCALE[6]); stats("Conv6",x)
    x,C,H,Wd = pool(x,64,8,8); stats("Pool3",x)
    # affine1: flatten channel-major (matches golden layout C*H*W)
    a1 = affine(x, 1024, 50, aff1_W, aff1_b, SCALE_AFF1, True)
    scores = affine(a1, 50, 10, aff2_W, aff2_b, SCALE_AFF2, False)
    pred = max(range(10), key=lambda i: scores[i])
    print("  scores:", scores)
    print("  pred=%d  %s" % (pred, "OK" if pred==digit else "FAIL"))

for d in range(2):
    run(d)
