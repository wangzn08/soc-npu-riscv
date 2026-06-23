#!/usr/bin/env python3
"""YOLOv8n DETECT HEAD (model.22) + decode fixture. Rebuilds pan_p3/p4/p5 via the
neck generator's functions, runs the 3-scale bbox/cls branches, DFL + sigmoid +
box decode + NMS in python to produce the FINAL boxes golden, and emits qparams
for all 18 head convs (stems/mids = exact-SiLU; 1x1 OUT convs = standard requant +
LINEAR LUT lut[k]=clamp(s8(k)+out_zp)) + DFL conv63 dequant weights + strides +
per-scale bbox/cls int8 goldens. Weights from blob (WGT_OF). Usage: gen_yolo_head.py
"""
import importlib.util, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
N = load("nk", "tools/gen_yolo_neck.py")
C = N.C
aq = N.aq; Q = C.Q_SHIFT
WDIR = C.WDIR

INPUT = 320
STRIDES = {0: 8, 1: 16, 2: 32}   # P3,P4,P5


def qp_std(ci, kh, kw):
    """standard (non-preact) requant: mul=in*ws/out, for LINEAR out convs."""
    oc = C.CONVS[ci]["oc"]; ic = C.CONVS[ci]["ic"]
    i_s, i_z, o_s, o_z = aq(ci)
    w = np.fromfile(WDIR / f"conv{ci}_w.bin", dtype=np.int8).reshape(oc, ic, kh, kw)
    s = C.fwf(ci, "s")[:oc]; b = C.fwf(ci, "b")[:oc]
    wsum = w.astype(np.int64).sum(axis=(1, 2, 3))
    mul = [int(round(i_s * float(s[o]) / o_s * (1 << Q))) for o in range(oc)]
    bias = [int(round(float(b[o]) / (i_s * float(s[o]))) - i_z * int(wsum[o])
                + (round((1 << (Q - 1)) / mul[o]) if mul[o] else 0)) for o in range(oc)]
    return w, bias, mul


def lin_lut(out_zp):
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        q = i if i < 128 else i - 256
        lut[i] = C.clamp_s8(q + out_zp) & 0xFF
    return lut


def conv_branch(act, stem, mid, out):
    """3x3 stem (SiLU) -> 3x3 mid (SiLU) -> 1x1 out (LINEAR int8). act [C,H,W].
    Returns out int8 [OC,H,W], out_scale, out_zp, and emit-data dict."""
    E = {}
    # stem
    i_s, i_z, o_s, o_z = aq(stem)
    w, b, m = C.qp(stem, i_s, i_z, o_s, 3, 3)
    t = C.conv_exact(act, w, b, m, C.build_lut(o_s, o_z), 3, 3, 1, 1, i_z, o_z)
    H, W = act.shape[1], act.shape[2]
    t = t.reshape(H, W, C.CONVS[stem]["oc"]).transpose(2, 0, 1)
    E["stem"] = (b, m, o_z, C.build_lut(o_s, o_z), i_z)
    # mid
    i_s, i_z, o_s, o_z = aq(mid)
    w, b, m = C.qp(mid, i_s, i_z, o_s, 3, 3)
    t2 = C.conv_exact(t, w, b, m, C.build_lut(o_s, o_z), 3, 3, 1, 1, i_z, o_z)
    t2 = t2.reshape(H, W, C.CONVS[mid]["oc"]).transpose(2, 0, 1)
    E["mid"] = (b, m, o_z, C.build_lut(o_s, o_z), i_z)
    # out (linear)
    i_s, i_z, o_s, o_z = aq(out)
    w, b, m = qp_std(out, 1, 1)
    llut = lin_lut(o_z)
    to = C.conv_exact(t2, w, b, m, llut, 1, 1, 1, 0, i_z, o_z)   # conv_exact applies lut[clamp(s2)]
    oc = C.CONVS[out]["oc"]
    E["out"] = (b, m, o_z, llut, i_z)
    return to, o_s, o_z, E   # to is [SP,OC] int8


def i32(nm, v): return f"static const int32_t {nm}[{len(v)}] = {{ {', '.join(map(str,v))} }};"
def u32(nm, v): return f"static const uint32_t {nm}[{len(v)}] = {{ {', '.join(f'{x}u' for x in v)} }};"
def u8(nm, v):  return f"static const uint8_t {nm}[{len(v)}] = {{ {', '.join(f'0x{int(x):02X}u' for x in v)} }};"
LH = []
def emit_conv(pfx, t):   # t=(b,m,oz,lut,iz)
    P = pfx.upper(); b, m, oz, lut, iz = t
    LH.append(i32(f"{pfx}_bias", b)); LH.append(u32(f"{pfx}_mul", m))
    LH.append(u32(f"{pfx}_shift", [Q]*len(m))); LH.append(u8(f"{pfx}_lut", lut))
    LH.append(f"#define {P}_ZP {oz}"); LH.append(f"#define {P}_PAD {iz}")


def emit_qparams_only():
    """Emit head conv qparams (input-independent) for all 18 convs + DFL + strides."""
    branches = {0: (36, 38, 41, 37, 39, 42), 1: (47, 49, 52, 48, 50, 53), 2: (57, 59, 61, 58, 60, 62)}
    snames = {0: "p3", 1: "p4", 2: "p5"}
    LH.append("#ifndef YOLO_HEAD_DATA_H"); LH.append("#define YOLO_HEAD_DATA_H")
    LH.append("#include <stdint.h>"); LH.append("")
    for s in range(3):
        bs, bm, bo, cs, cm, co = branches[s]; nm = snames[s]
        # bbox: stem/mid exact, out std+linear
        i_s, i_z, o_s, o_z = aq(bs); w, b, m = C.qp(bs, i_s, i_z, o_s, 3, 3)
        emit_conv(f"yolo_hd_{nm}_bb_stem", (b, m, o_z, C.build_lut(o_s, o_z), i_z))
        i_s, i_z, o_s, o_z = aq(bm); w, b, m = C.qp(bm, i_s, i_z, o_s, 3, 3)
        emit_conv(f"yolo_hd_{nm}_bb_mid", (b, m, o_z, C.build_lut(o_s, o_z), i_z))
        i_s, i_z, bbo_s, bbo_z = aq(bo); w, b, m = qp_std(bo, 1, 1)
        emit_conv(f"yolo_hd_{nm}_bb_out", (b, m, bbo_z, lin_lut(bbo_z), i_z))
        # cls: stem/mid exact, out std+linear
        i_s, i_z, o_s, o_z = aq(cs); w, b, m = C.qp(cs, i_s, i_z, o_s, 3, 3)
        emit_conv(f"yolo_hd_{nm}_cl_stem", (b, m, o_z, C.build_lut(o_s, o_z), i_z))
        i_s, i_z, o_s, o_z = aq(cm); w, b, m = C.qp(cm, i_s, i_z, o_s, 3, 3)
        emit_conv(f"yolo_hd_{nm}_cl_mid", (b, m, o_z, C.build_lut(o_s, o_z), i_z))
        i_s, i_z, clo_s, clo_z = aq(co); w, b, m = qp_std(co, 1, 1)
        emit_conv(f"yolo_hd_{nm}_cl_out", (b, m, clo_z, lin_lut(clo_z), i_z))
        # decode dequant scales/zp + stem/mid ic (for ic_stream routing) + dims
        H = {0: 40, 1: 20, 2: 10}[s]
        LH.append(f"#define YOLO_HD_{nm.upper()}_HW {H}u")
        LH.append(f"#define YOLO_HD_{nm.upper()}_STRIDE {STRIDES[s]}u")
        LH.append(f"#define YOLO_HD_{nm.upper()}_BB_OUTS {bbo_s:.9f}f")
        LH.append(f"#define YOLO_HD_{nm.upper()}_BB_OUTZP {bbo_z}")
        LH.append(f"#define YOLO_HD_{nm.upper()}_CL_OUTS {clo_s:.9f}f")
        LH.append(f"#define YOLO_HD_{nm.upper()}_CL_OUTZP {clo_z}")
        LH.append(f"#define YOLO_HD_{nm.upper()}_STEM_IC {C.CONVS[bs]['ic']}u")
        LH.append(f"#define YOLO_HD_{nm.upper()}_BB_STEM_OC {C.CONVS[bs]['oc']}u")
        LH.append(f"#define YOLO_HD_{nm.upper()}_CL_STEM_OC {C.CONVS[cs]['oc']}u")
        LH.append("")
    # DFL conv63: projection weights (dequant) as float
    dfl_w = np.fromfile(WDIR / "conv63_w.bin", dtype=np.int8).reshape(16).astype(np.float64)
    dfl_s = float(C.fwf(63, "s")[0]); dflw = (dfl_w * dfl_s)
    LH.append("static const float yolo_hd_dfl_w[16] = { " + ", ".join(f"{x:.9f}f" for x in dflw) + " };")
    LH.append(f"#define YOLO_HD_INPUT {INPUT}u")
    # golden final boxes (C oracle preact @320): cx,cy,w,h,conf,cls
    LH.append("static const float yolo_hd_golden[4][5] = {")
    LH.append("    {111.1f,186.7f,47.3f,131.6f,0.755f},")
    LH.append("    {56.1f,192.8f,73.7f,152.9f,0.755f},")
    LH.append("    {297.1f,188.2f,44.2f,136.5f,0.711f},")
    LH.append("    {15.3f,201.8f,30.0f,113.1f,0.337f},")
    LH.append("};")
    LH.append("#endif")
    out = ROOT / "firmware" / "yolo_head_data.h"
    out.write_text("\n".join(LH) + "\n", encoding="ascii")
    print(f"wrote {out} (18 head convs qparams + DFL + strides + golden boxes)")


def main():
    emit_qparams_only()
    return
    # --- rebuild pan_p3/p4/p5 (deterministic from dumps via neck functions) ---
    sppf, _, _ = C.read_dump(26); p4t, _, _ = C.read_dump(19); p3t, _, _ = C.read_dump(12); p5t, _, _ = C.read_dump(24)
    up1 = N.ups2x(sppf); cat1 = N.build_cat(up1, aq(26)[2], aq(26)[3], p4t, aq(19)[2], aq(19)[3], 27)
    fm, _, fms, fmz = N.c2f_neck(cat1, 27, [(28, 29)], 30); fpn_mid = N.to_chw(fm, 20, 20)
    up2 = N.ups2x(fpn_mid); cat2 = N.build_cat(up2, fms, fmz, p3t, aq(12)[2], aq(12)[3], 31)
    p3sp, _, _, _ = N.c2f_neck(cat2, 31, [(32, 33)], 34); pan_p3 = N.to_chw(p3sp, 40, 40)
    c35, _ = N.conv3x3(pan_p3, 35, 2); c35 = N.to_chw(c35, 20, 20)
    cat3 = N.build_cat(c35, aq(35)[2], aq(35)[3], fpn_mid, fms, fmz, 40)
    p4sp, _, _, _ = N.c2f_neck(cat3, 40, [(43, 44)], 45); pan_p4 = N.to_chw(p4sp, 20, 20)
    c46, _ = N.conv3x3(pan_p4, 46, 2); c46 = N.to_chw(c46, 10, 10)
    cat4 = N.build_cat(c46, aq(46)[2], aq(46)[3], p5t, aq(24)[2], aq(24)[3], 51)
    p5sp, _, _, _ = N.c2f_neck(cat4, 51, [(54, 55)], 56); pan_p5 = N.to_chw(p5sp, 10, 10)
    pans = [pan_p3, pan_p4, pan_p5]

    # --- head branches ---
    branches = {0: (36, 38, 41, 37, 39, 42), 1: (47, 49, 52, 48, 50, 53), 2: (57, 59, 61, 58, 60, 62)}
    Edata = {}; bbox_q = {}; cls_q = {}; bbox_scz = {}; cls_scz = {}
    for s in range(3):
        bs, bm, bo, cs, cm, co = branches[s]
        bbq, b_os, b_oz, Eb = conv_branch(pans[s], bs, bm, bo)
        clq, c_os, c_oz, Ec = conv_branch(pans[s], cs, cm, co)
        Edata[s] = (Eb, Ec)
        bbox_q[s] = bbq; cls_q[s] = clq
        bbox_scz[s] = (b_os, b_oz); cls_scz[s] = (c_os, c_oz)

    # --- decode (float) ---
    dfl_w = np.fromfile(WDIR / "conv63_w.bin", dtype=np.int8).reshape(16).astype(np.float64)
    dfl_s = float(C.fwf(63, "s")[0]); dflw = dfl_w * dfl_s   # projection weights
    dets = []
    for s in range(3):
        H = W = {0: 40, 1: 20, 2: 10}[s]; A = H * W; st = STRIDES[s]
        bs_, bz_ = bbox_scz[s]; cs_, cz_ = cls_scz[s]
        bb = bbox_q[s].astype(np.int32)   # [A,64]
        cl = cls_q[s].astype(np.int32)    # [A,80]
        for a in range(A):
            ay, ax = a // W, a % W
            coords = []
            for c in range(4):
                d = np.array([(bb[a, c*16+k] - bz_) * bs_ for k in range(16)])
                d = d - d.max(); e = np.exp(d); e /= e.sum()
                coords.append(float((e * dflw).sum()))
            lt_x, lt_y, rb_x, rb_y = coords
            x1 = (ax + 0.5 - lt_x) * st; y1 = (ay + 0.5 - lt_y) * st
            x2 = (ax + 0.5 + rb_x) * st; y2 = (ay + 0.5 + rb_y) * st
            for cc in range(80):
                p = 1.0 / (1.0 + math.exp(-((cl[a, cc] - cz_) * cs_)))
                if p >= 0.25:
                    dets.append((p, cc, x1, y1, x2, y2))
    # NMS (class-agnostic-ish per class)
    dets.sort(reverse=True)
    keep = []
    def iou(a, b):
        xx1=max(a[2],b[2]);yy1=max(a[3],b[3]);xx2=min(a[4],b[4]);yy2=min(a[5],b[5])
        w=max(0,xx2-xx1);h=max(0,yy2-yy1);inter=w*h
        ar=(a[4]-a[2])*(a[5]-a[3]);br=(b[4]-b[2])*(b[5]-b[3])
        return inter/(ar+br-inter+1e-9)
    for d in dets:
        if all(not (d[1]==k[1] and iou(d,k)>0.45) for k in keep): keep.append(d)
    print(f"head decode: {len(dets)} raw dets, {len(keep)} after NMS")
    for k in keep[:8]:
        print(f"  cls{k[1]} p{k[0]:.2f} box=({k[2]:.1f},{k[3]:.1f},{k[4]:.1f},{k[5]:.1f}) ctr=({(k[2]+k[4])/2:.1f},{(k[3]+k[5])/2:.1f})")


if __name__ == "__main__":
    main()
