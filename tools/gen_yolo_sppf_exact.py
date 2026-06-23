#!/usr/bin/env python3
"""Exact-SiLU SPPF (model.9) fixture at the REAL 10x10 spatial (not the legacy
20x2 shape-hack). conv25 (1x1 256->128) -> 3x MaxPool5x5(s1,p2) -> concat(512)
-> conv26 (1x1 512->256). MaxPool/concat are scale-preserving (int8 max + same-
scale concat = identity requant), so only conv25/26 carry exact qparams + LUTs.
Weights are read from the DDR blob at runtime (WGT_OF(25/26)); this header emits
only qparams/LUTs + the conv26 golden. Reuses gen_yolo_c2f_exact infrastructure
(preact-scale LUT, Q_SHIFT=20). Usage: gen_yolo_sppf_exact.py"""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c2fx", ROOT / "tools" / "gen_yolo_c2f_exact.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)

IN_DUMP = 24   # c2f_8 cv2 output = SPPF input
CV1_CI, CV2_CI = 25, 26


def maxpool5(x):   # [C,H,W] int8 -> same, k5 s1 p2, pad = -128 (int8 min)
    Cn, H, W = x.shape
    out = np.full_like(x, -128)
    for c in range(Cn):
        for oh in range(H):
            for ow in range(W):
                mx = -128
                for kh in range(5):
                    for kw in range(5):
                        ih, iw = oh - 2 + kh, ow - 2 + kw
                        if 0 <= ih < H and 0 <= iw < W:
                            v = int(x[c, ih, iw])
                            if v > mx: mx = v
                out[c, oh, ow] = mx
    return out


def i32(nm, v): return f"static const int32_t {nm}[{len(v)}] = {{ {', '.join(map(str,v))} }};"
def u32(nm, v): return f"static const uint32_t {nm}[{len(v)}] = {{ {', '.join(f'{x}u' for x in v)} }};"
def u8(nm, v): return f"static const uint8_t {nm}[{len(v)}] = {{ {', '.join(f'0x{int(x):02X}u' for x in v)} }};"


def main():
    act, in_sc, in_zp = C.read_dump(IN_DUMP)
    H, W = act.shape[1], act.shape[2]; SP = H * W
    aq = lambda ci: (C.CONVS[ci]["in_scale"], C.CONVS[ci]["in_zp"],
                     C.CONVS[ci]["out_scale"], C.CONVS[ci]["out_zp"])

    # conv25 (1x1) exact
    i_s, i_z, o_s, o_z = aq(CV1_CI)
    w25, b25, m25 = C.qp(CV1_CI, i_s, i_z, o_s, 1, 1)
    cv1 = C.conv_exact(act, w25, b25, m25, C.build_lut(o_s, o_z), 1, 1, 1, 0, i_z, o_z)
    cv1c = cv1.reshape(H, W, w25.shape[0]).transpose(2, 0, 1)     # [128,H,W]
    c25_zp = o_z

    # 3x serial maxpool + concat(cv1,m0,m1,m2) -- all at conv25 out scale
    m0 = maxpool5(cv1c); m1 = maxpool5(m0); m2 = maxpool5(m1)
    concat = np.concatenate([cv1c, m0, m1, m2], axis=0)           # [512,H,W]

    # conv26 (1x1) exact: in = concat scale (= conv25 out)
    i2_s, i2_z, o2_s, o2_z = aq(CV2_CI)
    w26, b26, m26 = C.qp(CV2_CI, i2_s, i2_z, o2_s, 1, 1)
    golden = C.conv_exact(concat, w26, b26, m26, C.build_lut(o2_s, o2_z), 1, 1, 1, 0, i2_z, o2_z)

    outd, _, _ = C.read_dump(CV2_CI)
    cflat = outd.transpose(1, 2, 0).reshape(SP, C.CONVS[CV2_CI]["oc"])
    df = np.abs(golden.astype(int) - cflat.astype(int))
    print(f"sppf exact vs dump{CV2_CI}: mism={int((df>0).sum())}/{golden.size} max={int(df.max())}")

    L = ["#ifndef YOLO_SPPF_EXACT_DATA_H", "#define YOLO_SPPF_EXACT_DATA_H",
         "#include <stdint.h>", ""]
    L += [f"#define SPPFE_IN_W {W}u", f"#define SPPFE_IN_H {H}u", f"#define SPPFE_SPATIAL {SP}u",
          f"#define SPPFE_C25_IC {C.CONVS[CV1_CI]['ic']}u", f"#define SPPFE_C25_OC {C.CONVS[CV1_CI]['oc']}u",
          f"#define SPPFE_C26_IC {C.CONVS[CV2_CI]['ic']}u", f"#define SPPFE_C26_OC {C.CONVS[CV2_CI]['oc']}u",
          f"#define SPPFE_C25_ZP {c25_zp}", f"#define SPPFE_RTL_TOL 24u", ""]
    L += [i32("yolo_sppf_e_c25_bias", b25), u32("yolo_sppf_e_c25_mul", m25),
          u32("yolo_sppf_e_c25_shift", [C.Q_SHIFT]*len(m25)),
          u8("yolo_sppf_e_c25_lut", C.build_lut(o_s, o_z)), ""]
    L += [i32("yolo_sppf_e_c26_bias", b26), u32("yolo_sppf_e_c26_mul", m26),
          u32("yolo_sppf_e_c26_shift", [C.Q_SHIFT]*len(m26)),
          u8("yolo_sppf_e_c26_lut", C.build_lut(o2_s, o2_z)), ""]
    L += [f"static const uint8_t yolo_sppf_e_golden[{SP}][{C.CONVS[CV2_CI]['oc']}] = {{"]
    L += ["    {"+", ".join(f"0x{int(x)&0xFF:02X}u" for x in golden[p])+"}," for p in range(SP)]
    L += ["};", "#endif"]
    out = ROOT / "firmware" / "yolo_sppf_exact_data.h"
    out.write_text("\n".join(L) + "\n", encoding="ascii")
    print(f"wrote {out} ({W}x{H}, c25 256->128, c26 512->256)")


if __name__ == "__main__":
    main()
