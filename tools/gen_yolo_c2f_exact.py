#!/usr/bin/env python3
"""General exact-SiLU C2f fixture generator at 320 res. Usage:
  gen_yolo_c2f_exact.py c2f4   (or c2f6 / c2f8)
Reads input from the C dump of the preceding layer, scales from act_quant_meta
(+ glue_ops by name), and emits firmware/yolo_<blk>_exact_data.h with exact out-
grid qparams, per-conv 256-entry SiLU LUTs, baked weights, the C2f-runner cfg
constants, and a golden self-checked vs the cv2-output dump. The CPU residual-add
and concat requant use the same fixed-point as the firmware C2f runner."""
import sys, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WDIR = ROOT / "yolov8n_int8" / "weights"
DUMP = ROOT / "yolov8n_int8" / "dump320"
import struct
META = json.load(open(ROOT / "yolov8n_int8" / "act_quant_meta.json"))
CONVS = META["convs"]; GLUE = {g["name"]: g for g in META["glue_ops"]}
Q_SHIFT = 20; REQ_SHIFT = 12; CAT_SHIFT = 16; RATIO_SHIFT = 16; RTL_TOL = 24

BLOCKS = {
    "c2f2": dict(in_dump=1, cv1=2, bns=[(3, 4, "/model.2/m.0/Add")], cv2=5, full_c=32),
    "c2f4": dict(in_dump=6, cv1=7, bns=[(8, 9, "/model.4/m.0/Add"), (10, 11, "/model.4/m.1/Add")], cv2=12, full_c=64),
    "c2f6": dict(in_dump=13, cv1=14, bns=[(15, 16, "/model.6/m.0/Add"), (17, 18, "/model.6/m.1/Add")], cv2=19, full_c=128),
    "c2f8": dict(in_dump=20, cv1=21, bns=[(22, 23, "/model.8/m.0/Add")], cv2=24, full_c=256),
}


def clamp_s8(v): return -128 if v < -128 else (127 if v > 127 else v)
def s8(b): b &= 0xFF; return b - 256 if b & 0x80 else b
def pack_i8_word(vals):
    raw = np.asarray(vals, dtype=np.int8).view(np.uint8)
    return [sum(int(raw[g*4+b]) << (8*b) for b in range(4)) for g in range(4)]
def read_dump(ci):
    d = (DUMP / f"conv{ci}.bin").read_bytes()
    c, h, w = struct.unpack("<3i", d[:12]); sc = struct.unpack("<f", d[12:16])[0]; zp = struct.unpack("<i", d[16:20])[0]
    return np.frombuffer(d[20:], dtype=np.int8).reshape(c, h, w).copy(), sc, zp
def fwf(ci, t): return np.fromfile(WDIR / f"conv{ci}_{t}.bin", dtype=np.float32)
def build_lut(out_scale, out_zp):
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        q = i if i < 128 else i - 256
        r = (q - out_zp) * out_scale
        lut[i] = clamp_s8(int(round((r / (1 + math.exp(-r))) / out_scale + out_zp))) & 0xFF
    return lut


def qp(ci, in_scale, in_zp, out_scale, kh, kw):
    """exact out-grid qparams for conv ci targeting (out_scale)."""
    oc = CONVS[ci]["oc"]; ic = CONVS[ci]["ic"]
    w = np.fromfile(WDIR / f"conv{ci}_w.bin", dtype=np.int8).reshape(oc, ic, kh, kw)
    s = fwf(ci, "s")[:oc]; b = fwf(ci, "b")[:oc]
    wsum = w.astype(np.int64).sum(axis=(1, 2, 3))
    mul = [int(round(in_scale * float(s[o]) / out_scale * (1 << Q_SHIFT))) for o in range(oc)]
    bias = [int(round(float(b[o]) / (in_scale * float(s[o]))) - in_zp * int(wsum[o])) for o in range(oc)]
    return w, bias, mul


def conv_exact(act, w, bias, mul, lut, kh, kw, stride, pad, in_zp, out_zp):
    OC, IC = w.shape[0], w.shape[1]
    oh = (act.shape[1] + 2*pad - kh)//stride + 1; ow = (act.shape[2] + 2*pad - kw)//stride + 1
    out = np.zeros((oh*ow, OC), dtype=np.int8)
    ap = np.full((IC, act.shape[1]+2*pad, act.shape[2]+2*pad), in_zp, dtype=np.int64)
    ap[:, pad:pad+act.shape[1], pad:pad+act.shape[2]] = act
    for oy in range(oh):
        for ox in range(ow):
            win = ap[:, oy*stride:oy*stride+kh, ox*stride:ox*stride+kw]
            for o in range(OC):
                acc = int(np.sum(win * w[o].astype(np.int64)))
                s2 = ((acc + bias[o]) * mul[o]) >> Q_SHIFT
                out[oy*ow+ox, o] = np.int8(s8(int(lut[clamp_s8(s2 + out_zp) & 0xFF])))
    return out


def main():
    blk = sys.argv[1]; cfg = BLOCKS[blk]; modeln = blk[3:]
    cv1, cv2 = cfg["cv1"], cfg["cv2"]; bns = cfg["bns"]; n = len(bns)
    full_c = cfg["full_c"]; half = full_c // 2
    act, in_sc, in_zp = read_dump(cfg["in_dump"])
    IC0 = act.shape[0]; H, W = act.shape[1], act.shape[2]; SP = H*W
    aq = lambda ci: (CONVS[ci]["in_scale"], CONVS[ci]["in_zp"], CONVS[ci]["out_scale"], CONVS[ci]["out_zp"])
    lut = lambda os, oz: build_lut(os, oz)

    # cv1 (1x1) -> split s0,s1
    i_s, i_z, o_s, o_z = aq(cv1)
    w1, b1, m1 = qp(cv1, i_s, i_z, o_s, 1, 1)
    cv1o = conv_exact(act, w1, b1, m1, lut(o_s, o_z), 1, 1, 1, 0, i_z, o_z)
    cv1c = cv1o.reshape(H, W, full_c).transpose(2, 0, 1)
    s0 = cv1c[:half]; s1 = cv1c[half:]
    cv1_scale, cv1_zp = o_s, o_z

    adds = []   # (data[SP,half], scale, zp)
    prev = s1.transpose(1, 2, 0).reshape(SP, half); prev_sc, prev_zp = cv1_scale, cv1_zp
    bn_data = []
    for (mc1, mc2, gname) in bns:
        g = GLUE[gname]; gsc, gzp = g["out_scale"], g["out_zp"]
        a_s, a_z, ao_s, ao_z = aq(mc1)
        w_a, b_a, m_a = qp(mc1, a_s, a_z, ao_s, 3, 3)
        prevc = prev.reshape(H, W, half).transpose(2, 0, 1)
        c_a = conv_exact(prevc, w_a, b_a, m_a, lut(ao_s, ao_z), 3, 3, 1, 1, a_z, ao_z)
        c_ac = c_a.reshape(H, W, half).transpose(2, 0, 1)
        b_s, b_z, _, _ = aq(mc2)
        w_b, b_b, m_b = qp(mc2, b_s, b_z, gsc, 3, 3)   # m_cv2 requants to glue
        c_b = conv_exact(c_ac, w_b, b_b, m_b, lut(gsc, gzp), 3, 3, 1, 1, b_z, gzp)
        # residual add: out = clamp((prev-prev_zp)*ratio>>sh + c_b) @ glue
        rmul = int(round((prev_sc / gsc) * (1 << RATIO_SHIFT)))
        add = np.zeros((SP, half), dtype=np.int8)
        for idx in np.ndindex(prev.shape):
            v = (((int(prev[idx]) - prev_zp) * rmul) >> RATIO_SHIFT) + int(c_b[idx])
            add[idx] = np.int8(s8(clamp_s8(v) & 0xFF))
        adds.append((add, gsc, gzp)); bn_data.append((w_a, b_a, m_a, ao_s, ao_z, w_b, b_b, m_b, gsc, gzp, rmul, prev_zp))
        prev = add; prev_sc, prev_zp = gsc, gzp

    # concat(s0,s1,add0..) requant to cv2 in (CAT)
    cat_s, cat_z = CONVS[cv2]["in_scale"], CONVS[cv2]["in_zp"]
    def cat_req(q, in_zp_, src_sc):
        mul = int(round((src_sc / cat_s) * (1 << CAT_SHIFT)))
        out = np.zeros_like(q, dtype=np.int8)
        for idx in np.ndindex(q.shape):
            out[idx] = np.int8(s8(clamp_s8((((int(q[idx]) - in_zp_) * mul) >> CAT_SHIFT) + cat_z) & 0xFF))
        return out, mul
    s0f = s0.transpose(1, 2, 0).reshape(SP, half); s1f = s1.transpose(1, 2, 0).reshape(SP, half)
    s0c, cm_s = cat_req(s0f, cv1_zp, cv1_scale); s1c, _ = cat_req(s1f, cv1_zp, cv1_scale)
    pieces = [s0c, s1c]; cat_muls = []
    for (add, gsc, gzp) in adds:
        ac, cm = cat_req(add, gzp, gsc); pieces.append(ac); cat_muls.append((cm, gzp))
    concat = np.concatenate(pieces, axis=1)
    cv2_ic = half*(2+n)
    concat_c = concat.reshape(H, W, cv2_ic).transpose(2, 0, 1)
    # cv2 (1x1)
    _, _, c2o_s, c2o_z = aq(cv2)
    w2, b2, m2 = qp(cv2, cat_s, cat_z, c2o_s, 1, 1)
    golden = conv_exact(concat_c, w2, b2, m2, lut(c2o_s, c2o_z), 1, 1, 1, 0, cat_z, c2o_z)

    outd, _, _ = read_dump(cv2)
    cflat = outd.transpose(1, 2, 0).reshape(SP, CONVS[cv2]["oc"])
    df = np.abs(golden.astype(int) - cflat.astype(int))
    print(f"{blk} exact vs dump{cv2}: mism={int((df>0).sum())}/{golden.size} max={int(df.max())}")

    # ---- emit ----
    def pack1(w): return [pack_i8_word(w[o, g*16:g*16+16, 0, 0]) for o in range(w.shape[0]) for g in range(w.shape[1]//16)]
    def pack3(w): return [pack_i8_word(w[o, g*16:g*16+16, k//3, k%3]) for o in range(w.shape[0]) for g in range(w.shape[1]//16) for k in range(9)]
    def packact(a):
        C = a.shape[0]; return [pack_i8_word(a[g*16:g*16+16, y, x]) for g in range(C//16) for y in range(H) for x in range(W)]
    def u32a(nm, ws): return f"static const uint32_t {nm}[{len(ws)}][4] = {{\n" + "\n".join("    {"+", ".join(f'0x{v:08X}u' for v in x)+"}," for x in ws) + "\n};"
    def i32(nm, v): return f"static const int32_t {nm}[{len(v)}] = {{ {', '.join(map(str,v))} }};"
    def u32(nm, v): return f"static const uint32_t {nm}[{len(v)}] = {{ {', '.join(f'{x}u' for x in v)} }};"
    def u8(nm, v): return f"static const uint8_t {nm}[{len(v)}] = {{ {', '.join(f'0x{int(x):02X}u' for x in v)} }};"

    U = blk.upper(); P = f"YOLO_{U}"
    L = [f"#ifndef {U}_EXACT_DATA_H", f"#define {U}_EXACT_DATA_H", "#include <stdint.h>", ""]
    L += [f"#define {P}_IN_W {W}u", f"#define {P}_IN_H {H}u", f"#define {P}_SPATIAL {SP}u",
          f"#define {P}_FULL_C {full_c}u", f"#define {P}_CV1_IC {IC0}u", f"#define {P}_CV2_IC {cv2_ic}u",
          f"#define {P}_CV2_OC {CONVS[cv2]['oc']}u", f"#define {P}_N {n}u", f"#define {P}_RTL_TOL {RTL_TOL}u", ""]
    L += [f"#define {P}_BLKIN_WORDS {SP*(IC0//16)}u", u32a(f"yolo_{blk}_blkin", packact(act)), ""]
    # cv1
    L += [f"#define {P}_CV1_WGT_WORDS {CONVS[cv1]['oc']*(IC0//16)}u",
          u32a(f"yolo_{blk}_cv1_wgt", pack1(w1)), i32(f"yolo_{blk}_cv1_bias", b1),
          u32(f"yolo_{blk}_cv1_mul", m1), u32(f"yolo_{blk}_cv1_shift", [Q_SHIFT]*len(m1)),
          u8(f"yolo_{blk}_cv1_lut", build_lut(cv1_scale, cv1_zp)),
          f"#define {P}_CV1_RQ_ZP {cv1_zp}", ""]
    # bottlenecks
    L += [f"#define {P}_MCV_WGT_WORDS {half*(half//16)*9}u",
          f"#define {P}_GLUE_RQ_SHIFT {REQ_SHIFT}u", f"#define {P}_ADD_RATIO_SHIFT {RATIO_SHIFT}u",
          f"#define {P}_CAT_SHIFT {CAT_SHIFT}u", f"#define {P}_CAT_ZP {cat_z}",
          f"#define {P}_CAT_MUL_S0S1 {cm_s}u", f"#define {P}_CAT_INZP_S0S1 {cv1_zp}", ""]
    for bi, (w_a, b_a, m_a, ao_s, ao_z, w_b, b_b, m_b, gsc, gzp, rmul, pzp) in enumerate(bn_data):
        L += [u32a(f"yolo_{blk}_mcv1_{bi}_wgt", pack3(w_a)), i32(f"yolo_{blk}_mcv1_{bi}_bias", b_a),
              u32(f"yolo_{blk}_mcv1_{bi}_mul", m_a), u32(f"yolo_{blk}_mcv1_{bi}_shift", [Q_SHIFT]*len(m_a)),
              u8(f"yolo_{blk}_mcv1_{bi}_lut", build_lut(ao_s, ao_z)),
              f"#define {P}_MCV1_{bi}_RQ_ZP {ao_z}", f"#define {P}_MCV1_{bi}_PAD {CONVS[bns[bi][0]]['in_zp']}",
              u32a(f"yolo_{blk}_mcv2_{bi}_wgt", pack3(w_b)), i32(f"yolo_{blk}_mcv2_{bi}_bias", b_b),
              u32(f"yolo_{blk}_mcv2_{bi}_mul", m_b), u32(f"yolo_{blk}_mcv2_{bi}_shift", [Q_SHIFT]*len(m_b)),
              u8(f"yolo_{blk}_mcv2_{bi}_lut", build_lut(gsc, gzp)),
              f"#define {P}_MCV2_{bi}_PAD {CONVS[bns[bi][1]]['in_zp']}",
              f"#define {P}_GLUE{bi}_ZP {gzp}", f"#define {P}_ADD{bi}_RATIO_MUL {rmul}u",
              f"#define {P}_ADD{bi}_PREV_ZP {pzp}",
              f"#define {P}_CAT_MUL_ADD{bi} {cat_muls[bi][0]}u", f"#define {P}_CAT_INZP_ADD{bi} {cat_muls[bi][1]}", ""]
    # cv2
    L += [f"#define {P}_CV2_WGT_WORDS {CONVS[cv2]['oc']*(cv2_ic//16)}u",
          u32a(f"yolo_{blk}_cv2_wgt", pack1(w2)), i32(f"yolo_{blk}_cv2_bias", b2),
          u32(f"yolo_{blk}_cv2_mul", m2), u32(f"yolo_{blk}_cv2_shift", [Q_SHIFT]*len(m2)),
          u8(f"yolo_{blk}_cv2_lut", build_lut(c2o_s, c2o_z)), f"#define {P}_CV2_RQ_ZP {c2o_z}", ""]
    # golden
    L += [f"static const uint8_t yolo_{blk}_golden[{SP}][{CONVS[cv2]['oc']}] = {{"]
    L += ["    {"+", ".join(f"0x{int(x)&0xFF:02X}u" for x in golden[p])+"}," for p in range(SP)]
    L += ["};", "#endif"]
    out = ROOT / "firmware" / f"yolo_{blk}_exact_data.h"
    out.write_text("\n".join(L) + "\n", encoding="ascii")
    print(f"wrote {out} (n={n}, {W}x{H}, cv2_ic={cv2_ic}, max vs dump above)")


if __name__ == "__main__":
    main()
