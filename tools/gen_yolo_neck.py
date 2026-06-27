#!/usr/bin/env python3
"""Exact-SiLU YOLOv8n NECK (FPN/PAN, model.10..21) fixture. Reuses
gen_yolo_c2f_exact infra (preact-scale LUT, qp/conv_exact/build_lut). Weights come
from the DDR blob at runtime (WGT_OF), so this emits ONLY qparams + cat/glue/concat
requant constants + pan_p3/p4/p5 goldens (for firmware stage checksums).

Recipe (C oracle yolov8n_infer.c), SoC @320 dims:
  SPPF=10x10x256(P5). FPN: up1=ups(SPPF); cat1=concat(up1,P4tap[c2f6 out, conv19]);
  c2f_12{27,m28,m29,30} -> fpn_mid 20x20x128. up2=ups(fpn_mid);
  cat2=concat(up2,P3tap[c2f4 out, conv12]); c2f_15{31,m32,m33,34} -> pan_p3 40x40x64.
  PAN: conv35(pan_p3 3x3 s2); cat3=concat(c35,fpn_mid); c2f_18{40,m43,m44,45} ->
  pan_p4 20x20x128. conv46(pan_p4 3x3 s2); cat4=concat(c46,P5tap[c2f8 out, conv24]);
  c2f_21{51,m54,m55,56} -> pan_p5 10x10x256. All neck C2f shortcut=0.
"""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c2fx", ROOT / "tools" / "gen_yolo_c2f_exact.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
Q, CAT = C.Q_SHIFT, C.CAT_SHIFT
aq = lambda ci: (C.CONVS[ci]["in_scale"], C.CONVS[ci]["in_zp"], C.CONVS[ci]["out_scale"], C.CONVS[ci]["out_zp"])
clamp = C.clamp_s8; s8 = C.s8


def ups2x(x):  # [C,H,W] int8 nearest 2x (scale/zp unchanged)
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def cat_req(q_sp_c, in_zp, src_sc, cat_s, cat_z):  # q [SP,C] -> int8 at cat scale
    mul = int(round((src_sc / cat_s) * (1 << CAT))); h = 1 << (CAT - 1)
    out = np.zeros_like(q_sp_c, dtype=np.int8)
    for idx in np.ndindex(q_sp_c.shape):
        out[idx] = np.int8(s8(clamp((((int(q_sp_c[idx]) - in_zp) * mul + h) >> CAT) + cat_z) & 0xFF))
    return out, mul


def conv1x1(act, ci, lin=False):  # act [C,H,W] -> [SP,OC] int8; lin=linear(no SiLU)
    i_s, i_z, o_s, o_z = aq(ci)
    w, b, m = C.qp(ci, i_s, i_z, o_s, 1, 1)
    lut = C.build_lut(o_s, o_z) if not lin else None
    if lin:   # standard requant (no SiLU): mul=in*ws/out, identity-ish path
        raise NotImplementedError
    return C.conv_exact(act, w, b, m, lut, 1, 1, 1, 0, i_z, o_z), (b, m, o_z, lut)


def conv3x3(act, ci, stride):  # act [C,H,W] -> [SP,OC] int8 (exact SiLU)
    i_s, i_z, o_s, o_z = aq(ci)
    w, b, m = C.qp(ci, i_s, i_z, o_s, 3, 3)
    out = C.conv_exact(act, w, b, m, C.build_lut(o_s, o_z), 3, 3, stride, 1, i_z, o_z)
    return out, (b, m, o_z, C.build_lut(o_s, o_z), i_z)


def c2f_neck(act, cv1, bns, cv2, fold_in=None):
    """shortcut=0 C2f. act [C,H,W]. Returns (out[SP,OC], emit_dict, out_scale,zp).
    fold_in=(up_ch, up_sc, up_z, tap_sc, tap_z): the input is the RAW [up|tap] concat
    at native scales; the FPN/PAN cat requant is folded into cv1's weights+bias (like
    the cv2 concat fold), so the firmware skips the per-lane concat2_rq."""
    n = len(bns); full_c = C.CONVS[cv1]["oc"]; half = full_c // 2
    H, W = act.shape[1], act.shape[2]; SP = H * W
    # cv1 1x1 -> split
    i_s, i_z, o_s, o_z = aq(cv1)
    w1, b1, m1 = C.qp(cv1, i_s, i_z, o_s, 1, 1)
    cv1_folded = 0
    if fold_in is not None:
        # fold the FPN/PAN cat requant into cv1 weights (cv1 reads raw [up|tap]).
        up_ch, up_sc, up_z, tap_sc, tap_z = fold_in
        cic = w1.shape[1]
        chan_sc = np.array([up_sc]*up_ch + [tap_sc]*(cic-up_ch), dtype=np.float64)
        chan_zp = np.array([up_z]*up_ch + [tap_z]*(cic-up_ch), dtype=np.int64)
        alpha = chan_sc / i_s          # i_s = cv1 in (cat) scale
        w1f = np.clip(np.round(w1[:, :, 0, 0].astype(np.float64) * alpha[None, :]), -128, 127).astype(np.int8)
        b1real = C.fwf(cv1, "b")[:full_c]; s1v = C.fwf(cv1, "s")[:full_c]
        b1 = [int(round(float(b1real[o]) / (i_s * float(s1v[o])))
                  - int(np.sum(w1f[o].astype(np.int64) * chan_zp))
                  + (round((1 << (C.Q_SHIFT - 1)) / m1[o]) if m1[o] else 0)) for o in range(full_c)]
        w1 = w1f[:, :, None, None]
        (C.WDIR / f"conv{cv1}_w_folded.bin").write_bytes(w1f.astype(np.int8).tobytes())
        cv1_folded = 1
    cv1o = C.conv_exact(act, w1, b1, m1, C.build_lut(o_s, o_z), 1, 1, 1, 0,
                        0 if cv1_folded else i_z, o_z)
    cv1c = cv1o.reshape(H, W, full_c).transpose(2, 0, 1)
    s0, s1 = cv1c[:half], cv1c[half:]
    cv1_s, cv1_z = o_s, o_z
    E = {"cv1": (b1, m1, o_z, C.build_lut(o_s, o_z)), "cv1_folded": cv1_folded}
    adds = []; prev = s1; pieces_scz = []
    for bi, (mc1, mc2) in enumerate(bns):
        a_s, a_z, ao_s, ao_z = aq(mc1)
        wa, ba, ma = C.qp(mc1, a_s, a_z, ao_s, 3, 3)
        ca = C.conv_exact(prev, wa, ba, ma, C.build_lut(ao_s, ao_z), 3, 3, 1, 1, a_z, ao_z)
        cac = ca.reshape(H, W, half).transpose(2, 0, 1)
        b_s, b_z, bo_s, bo_z = aq(mc2)
        wb, bb, mb = C.qp(mc2, b_s, b_z, bo_s, 3, 3)
        cb = C.conv_exact(cac, wb, bb, mb, C.build_lut(bo_s, bo_z), 3, 3, 1, 1, b_z, bo_z)
        # shortcut=0: add = cb (pass-through), scale=bo
        addc = cb.reshape(H, W, half).transpose(2, 0, 1)
        adds.append((addc, bo_s, bo_z))
        E[f"mcv1_{bi}"] = (ba, ma, ao_z, C.build_lut(ao_s, ao_z), C.CONVS[mc1]["in_zp"])
        E[f"mcv2_{bi}"] = (bb, mb, bo_z, C.build_lut(bo_s, bo_z), C.CONVS[mc2]["in_zp"])
        prev = addc
    # concat (s0,s1,add..) requant to cv2 in
    cat_s, cat_z = C.CONVS[cv2]["in_scale"], C.CONVS[cv2]["in_zp"]
    s0f = s0.transpose(1, 2, 0).reshape(SP, half); s1f = s1.transpose(1, 2, 0).reshape(SP, half)
    s0c, cm_s = cat_req(s0f, cv1_z, cv1_s, cat_s, cat_z)
    s1c, _ = cat_req(s1f, cv1_z, cv1_s, cat_s, cat_z)
    pieces = [s0c, s1c]; cat_add = []
    for (addc, gs, gz) in adds:
        af = addc.transpose(1, 2, 0).reshape(SP, half)
        ac, cm = cat_req(af, gz, gs, cat_s, cat_z); pieces.append(ac); cat_add.append((cm, gz))
    concat = np.concatenate(pieces, axis=1)
    cv2_ic = half * (2 + n)
    concat_c = concat.reshape(H, W, cv2_ic).transpose(2, 0, 1)
    _, _, c2o_s, c2o_z = aq(cv2)
    w2, b2, m2 = C.qp(cv2, cat_s, cat_z, c2o_s, 1, 1)
    # ---- CONCAT FOLDED: fold per-source requant (alpha=src/cat, zp) into cv2 int8
    # weights+bias so cv2 reads the RAW contiguous concat [s0|s1|add0..] at native
    # scales (no CPU cat_req). Same qp() bias convention (incl. the +half rounding). ----
    oc = w2.shape[0]
    b2real = C.fwf(cv2, "b")[:oc]; s2v = C.fwf(cv2, "s")[:oc]
    chan_sc = np.array([cv1_s]*full_c + sum([[gs]*half for (_a, gs, _gz) in adds], []), dtype=np.float64)
    chan_zp = np.array([cv1_z]*full_c + sum([[gz]*half for (_a, _gs, gz) in adds], []), dtype=np.int64)
    alpha = chan_sc / cat_s
    w2f = np.clip(np.round(w2[:, :, 0, 0].astype(np.float64) * alpha[None, :]), -128, 127).astype(np.int8)
    bias_fold = [int(round(float(b2real[o]) / (cat_s * float(s2v[o])))
                     - int(np.sum(w2f[o].astype(np.int64) * chan_zp))
                     + (round((1 << (C.Q_SHIFT - 1)) / m2[o]) if m2[o] else 0)) for o in range(oc)]
    raw_concat = np.concatenate([s0f, s1f] + [a.transpose(1, 2, 0).reshape(SP, half) for (a, _gs, _gz) in adds], axis=1)
    raw_concat_c = raw_concat.reshape(H, W, cv2_ic).transpose(2, 0, 1)
    out = C.conv_exact(raw_concat_c, w2f[:, :, None, None], bias_fold, m2, C.build_lut(c2o_s, c2o_z), 1, 1, 1, 0, 0, c2o_z)
    (C.WDIR / f"conv{cv2}_w_folded.bin").write_bytes(w2f.astype(np.int8).tobytes())
    E["cv2"] = (bias_fold, m2, c2o_z, C.build_lut(c2o_s, c2o_z))
    E["cat"] = (cm_s, cv1_z, cat_z, cat_add)   # cat_mul_s0s1, inzp_s0s1, cat_zp, [(mul,zp)..]
    E["dims"] = (W, H, SP, full_c, cv2_ic, C.CONVS[cv2]["oc"], n, C.CONVS[cv1]["ic"])
    return out, E, c2o_s, c2o_z


# ---- emit helpers ----
def i32(nm, v): return f"static const int32_t {nm}[{len(v)}] = {{ {', '.join(map(str,v))} }};"
def u32(nm, v): return f"static const uint32_t {nm}[{len(v)}] = {{ {', '.join(f'{x}u' for x in v)} }};"
def u8(nm, v):  return f"static const uint8_t {nm}[{len(v)}] = {{ {', '.join(f'0x{int(x):02X}u' for x in v)} }};"
L = []
def emit_c2f(pfx, E):
    P = pfx.upper()
    W, H, SP, full_c, cv2_ic, cv2_oc, n, cv1_ic = E["dims"]
    L.append(f"#define {P}_IN_W {W}u")
    L.append(f"#define {P}_IN_H {H}u")
    L.append(f"#define {P}_SPATIAL {SP}u")
    L.append(f"#define {P}_FULL_C {full_c}u")
    L.append(f"#define {P}_CV1_IC {cv1_ic}u")
    L.append(f"#define {P}_CV2_IC {cv2_ic}u")
    L.append(f"#define {P}_CV2_OC {cv2_oc}u")
    L.append(f"#define {P}_N {n}u")
    L.append(f"#define {P}_CV2_FOLDED 1")   # concat requant folded into cv2 weights (blob has w2f)
    L.append(f"#define {P}_CV1_FOLDED {E.get('cv1_folded', 0)}")  # FPN/PAN cat folded into cv1 weights
    def conv(name, t):
        b, m, z, lut = t[0], t[1], t[2], t[3]
        L.append(i32(f"{pfx}_{name}_bias", b)); L.append(u32(f"{pfx}_{name}_mul", m))
        L.append(u32(f"{pfx}_{name}_shift", [Q]*len(m))); L.append(u8(f"{pfx}_{name}_lut", lut))
        L.append(f"#define {P}_{name.upper()}_ZP {z}")
        if len(t) > 4: L.append(f"#define {P}_{name.upper()}_PAD {t[4]}")
    conv("cv1", E["cv1"])
    for bi in range(n):
        conv(f"mcv1_{bi}", E[f"mcv1_{bi}"]); conv(f"mcv2_{bi}", E[f"mcv2_{bi}"])
    conv("cv2", E["cv2"])
    cm_s, inzp_s, cat_z, cat_add = E["cat"]
    L.append(f"#define {P}_CAT_SHIFT {CAT}u")
    L.append(f"#define {P}_CAT_ZP {cat_z}")
    L.append(f"#define {P}_CAT_MUL_S0S1 {cm_s}u")
    L.append(f"#define {P}_CAT_INZP_S0S1 {inzp_s}")
    for bi, (cm, gz) in enumerate(cat_add):
        L.append(f"#define {P}_CAT_MUL_ADD{bi} {cm}u")
        L.append(f"#define {P}_CAT_INZP_ADD{bi} {gz}")
    L.append("")


def golden(nm, out_sp_oc):   # tile-major int8 golden [SP][OC]
    SP, OC = out_sp_oc.shape
    L.append(f"static const uint8_t {nm}[{SP}][{OC}] = {{")
    L.extend(["    {"+", ".join(f"0x{int(x)&0xFF:02X}u" for x in out_sp_oc[p])+"}," for p in range(SP)])
    L.append("};")


def concat2_defs(pfx, up_zp, up_sc, tap_zp, tap_sc, cat_ci):
    """upsample part already at its conv's out scale; tap at its out scale; both
    requant to cat_ci in scale."""
    cat_s, cat_z = C.CONVS[cat_ci]["in_scale"], C.CONVS[cat_ci]["in_zp"]
    mu = int(round((up_sc / cat_s) * (1 << CAT))); mt = int(round((tap_sc / cat_s) * (1 << CAT)))
    P = pfx.upper()
    L.append(f"#define {P}_MUL_UP {mu}u")
    L.append(f"#define {P}_INZP_UP {up_zp}")
    L.append(f"#define {P}_MUL_TAP {mt}u")
    L.append(f"#define {P}_INZP_TAP {tap_zp}")
    L.append(f"#define {P}_CAT_ZP {cat_z}")
    L.append(f"#define {P}_CAT_SHIFT {CAT}u")
    L.append("")


def to_chw(sp_oc, H, W):  # [SP,OC] -> [OC,H,W]
    OC = sp_oc.shape[1]; return sp_oc.reshape(H, W, OC).transpose(2, 0, 1)


def build_cat(up, up_sc, up_z, tap, tap_sc, tap_z, cat_ci):
    cat_s, cat_z = C.CONVS[cat_ci]["in_scale"], C.CONVS[cat_ci]["in_zp"]
    C0 = up.shape[0]; H, W = up.shape[1], up.shape[2]; SP = H * W
    upf = up.transpose(1, 2, 0).reshape(SP, C0); tapf = tap.transpose(1, 2, 0).reshape(SP, tap.shape[0])
    uc, _ = cat_req(upf, up_z, up_sc, cat_s, cat_z); tc, _ = cat_req(tapf, tap_z, tap_sc, cat_s, cat_z)
    cat = np.concatenate([uc, tc], axis=1)
    return cat.reshape(H, W, C0 + tap.shape[0]).transpose(2, 0, 1)


def build_cat_raw(up, tap):
    """Raw [up|tap] concat at NATIVE scales (no requant) -> cv1 reads this with the
    cat requant folded into its weights. Channel order [up..|tap..] matches cv1 fold."""
    return np.concatenate([up, tap], axis=0)   # [up_C+tap_C, H, W]


def emit_conv(pfx, t, kh):  # t=(b,m,oz,lut,izp); kh for doc only
    P = pfx.upper(); b, m, oz, lut, izp = t
    L.append(i32(f"{pfx}_bias", b)); L.append(u32(f"{pfx}_mul", m))
    L.append(u32(f"{pfx}_shift", [Q]*len(m))); L.append(u8(f"{pfx}_lut", lut))
    L.append(f"#define {P}_ZP {oz}"); L.append(f"#define {P}_PAD {izp}"); L.append("")


def main():
    L.append("#ifndef YOLO_NECK_DATA_H"); L.append("#define YOLO_NECK_DATA_H")
    L.append("#include <stdint.h>"); L.append("")

    sppf, _, _ = C.read_dump(26)      # SPPF out (P5) 10x10x256
    p4tap, _, _ = C.read_dump(19)     # c2f6 out (P4) 20x20x128
    p3tap, _, _ = C.read_dump(12)     # c2f4 out (P3) 40x40x64
    p5tap, _, _ = C.read_dump(24)     # c2f8 out 10x10x256
    sp_s, sp_z = aq(26)[2], aq(26)[3]
    p4_s, p4_z = aq(19)[2], aq(19)[3]
    p3_s, p3_z = aq(12)[2], aq(12)[3]
    p5_s, p5_z = aq(24)[2], aq(24)[3]

    # ---- FPN top-down ----
    up1 = ups2x(sppf)                                   # [256,20,20]
    cat1 = build_cat_raw(up1, p4tap)
    concat2_defs("yolo_nk_cat1", sp_z, sp_s, p4_z, p4_s, 27)
    fm_sp, E12, fm_s, fm_z = c2f_neck(cat1, 27, [(28, 29)], 30,
                                      fold_in=(up1.shape[0], sp_s, sp_z, p4_s, p4_z))
    emit_c2f("yolo_nk_c12", E12)
    fpn_mid = to_chw(fm_sp, 20, 20)                     # [128,20,20]

    up2 = ups2x(fpn_mid)                                # [128,40,40]
    cat2 = build_cat_raw(up2, p3tap)
    concat2_defs("yolo_nk_cat2", fm_z, fm_s, p3_z, p3_s, 31)
    p3_sp, E15, p3o_s, p3o_z = c2f_neck(cat2, 31, [(32, 33)], 34,
                                        fold_in=(up2.shape[0], fm_s, fm_z, p3_s, p3_z))
    emit_c2f("yolo_nk_c15", E15)
    pan_p3 = to_chw(p3_sp, 40, 40)                      # [64,40,40] HEAD P3

    # ---- PAN bottom-up ----
    c35_sp, t35 = conv3x3(pan_p3, 35, 2)               # [400,64] -> 20x20
    emit_conv("yolo_nk_c35", t35, 3)
    c35 = to_chw(c35_sp, 20, 20)
    cat3 = build_cat_raw(c35, fpn_mid)
    concat2_defs("yolo_nk_cat3", aq(35)[3], aq(35)[2], fm_z, fm_s, 40)
    p4_sp, E18, p4o_s, p4o_z = c2f_neck(cat3, 40, [(43, 44)], 45,
                                        fold_in=(c35.shape[0], aq(35)[2], aq(35)[3], fm_s, fm_z))
    emit_c2f("yolo_nk_c18", E18)
    pan_p4 = to_chw(p4_sp, 20, 20)                      # [128,20,20] HEAD P4

    c46_sp, t46 = conv3x3(pan_p4, 46, 2)               # [100,128] -> 10x10
    emit_conv("yolo_nk_c46", t46, 3)
    c46 = to_chw(c46_sp, 10, 10)
    cat4 = build_cat_raw(c46, p5tap)
    concat2_defs("yolo_nk_cat4", aq(46)[3], aq(46)[2], p5_z, p5_s, 51)
    p5_sp, E21, p5o_s, p5o_z = c2f_neck(cat4, 51, [(54, 55)], 56,
                                        fold_in=(c46.shape[0], aq(46)[2], aq(46)[3], p5_s, p5_z))
    emit_c2f("yolo_nk_c21", E21)
    pan_p5 = p5_sp                                      # [100,256] HEAD P5

    # ---- goldens (tile-major [SP][OC]) for firmware stage checksums ----
    golden("yolo_nk_pan_p3_golden", p3_sp)
    golden("yolo_nk_pan_p4_golden", p4_sp)
    golden("yolo_nk_pan_p5_golden", p5_sp)
    L.append(f"#define YOLO_NK_P3_SP {p3_sp.shape[0]}u")
    L.append(f"#define YOLO_NK_P3_OC {p3_sp.shape[1]}u")
    L.append(f"#define YOLO_NK_P4_SP {p4_sp.shape[0]}u")
    L.append(f"#define YOLO_NK_P4_OC {p4_sp.shape[1]}u")
    L.append(f"#define YOLO_NK_P5_SP {p5_sp.shape[0]}u")
    L.append(f"#define YOLO_NK_P5_OC {p5_sp.shape[1]}u")
    L.append("#endif")
    out = ROOT / "firmware" / "yolo_neck_data.h"
    out.write_text("\n".join(L) + "\n", encoding="ascii")
    print(f"wrote {out}: pan_p3{p3_sp.shape} p4{p4_sp.shape} p5{p5_sp.shape}")


if __name__ == "__main__":
    main()
