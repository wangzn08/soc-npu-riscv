#!/usr/bin/env python3
"""General per-conv exact-SiLU fixture generator for backbone convs (esp. the
stride-2 downsamples conv6/13/20). Usage: gen_yolo_conv_exact.py <ci> <prev_ci>
where prev_ci is the dump that feeds conv<ci> (standalone validation input).

Reads in/out (scale,zp) from act_quant_meta.json; weights come from the DDR blob
at runtime (this only loads conv<ci>_w.bin to precompute wsum for the bias). Emits
firmware/yolo_conv<ci>_exact_data.h with out-grid qparams, the 256-entry exact
SiLU LUT, dims, pad_value(=in_zp), the baked input (dump<prev>) for a standalone
smoke, and a position-weighted golden checksum (golden self-checked vs dump<ci>)."""
import sys, json, struct, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WDIR = ROOT / "yolov8n_int8" / "weights"
DUMP = ROOT / "yolov8n_int8" / "dump320"
META = json.load(open(ROOT / "yolov8n_int8" / "act_quant_meta.json"))
CONVS = META["convs"] if isinstance(META, dict) else META
Q_SHIFT = 20
SILU_STEP = 0.5  # SiLU LUT indexed by preact (zp=0)


def clamp_s8(v): return -128 if v < -128 else (127 if v > 127 else v)
def s8(b): b &= 0xFF; return b - 256 if b & 0x80 else b
def pack_i8_word(vals):
    raw = np.asarray(vals, dtype=np.int8).view(np.uint8)
    return [sum(int(raw[g*4+b]) << (8*b) for b in range(4)) for g in range(4)]
def read_dump(ci):
    d = (DUMP / f"conv{ci}.bin").read_bytes()
    c, h, w = struct.unpack("<3i", d[:12]); sc = struct.unpack("<f", d[12:16])[0]; zp = struct.unpack("<i", d[16:20])[0]
    return np.frombuffer(d[20:], dtype=np.int8).reshape(c, h, w).copy(), sc, zp
def build_silu_lut(out_scale, out_zp):
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        q = i if i < 128 else i - 256
        realq = q * SILU_STEP
        silu = realq / (1.0 + math.exp(-realq))
        lut[i] = clamp_s8(int(round(silu / out_scale + out_zp))) & 0xFF
    return lut


def main():
    ci = int(sys.argv[1]); prev = int(sys.argv[2])
    c = CONVS[ci]
    OC, IC, KH, KW = c["oc"], c["ic"], c["kh"], c["kw"]
    in_scale, in_zp, out_scale, out_zp = c["in_scale"], c["in_zp"], c["out_scale"], c["out_zp"]
    assert c["has_silu"], f"conv{ci} has no SiLU (use the linear path)"

    act, dsc, dzp = read_dump(prev)
    assert act.shape[0] == IC, f"dump{prev} C={act.shape[0]} != conv{ci} IC={IC}"
    assert abs(dsc - in_scale) < 1e-6, f"dump{prev} scale {dsc} != conv{ci} in_scale {in_scale}"
    IN_H, IN_W = act.shape[1], act.shape[2]
    # stride: infer from output dump dims
    outd, _, _ = read_dump(ci)
    OUT_H, OUT_W = outd.shape[1], outd.shape[2]
    STRIDE = 2 if OUT_H < IN_H else 1
    PAD = 1
    SP_OUT = OUT_H * OUT_W

    w = np.fromfile(WDIR / f"conv{ci}_w.bin", dtype=np.int8).reshape(OC, IC, KH, KW)
    wsum = w.astype(np.int64).sum(axis=(1, 2, 3))
    sc = np.fromfile(WDIR / f"conv{ci}_s.bin", dtype=np.float32)[:OC]
    b = np.fromfile(WDIR / f"conv{ci}_b.bin", dtype=np.float32)[:OC]
    lut = build_silu_lut(out_scale, out_zp)

    mul, bias_q = [], []
    for oc in range(OC):
        m = int(round(in_scale * float(sc[oc]) / SILU_STEP * (1 << Q_SHIFT)))
        mul.append(m); bias_q.append(int(round(float(b[oc]) / (in_scale * float(sc[oc]))) - in_zp * int(wsum[oc])))

    golden = np.zeros((SP_OUT, OC), dtype=np.int8)
    ap = np.full((IC, IN_H + 2*PAD, IN_W + 2*PAD), in_zp, dtype=np.int64)
    ap[:, PAD:PAD+IN_H, PAD:PAD+IN_W] = act
    for oy in range(OUT_H):
        for ox in range(OUT_W):
            win = ap[:, oy*STRIDE:oy*STRIDE+KH, ox*STRIDE:ox*STRIDE+KW]
            for oc in range(OC):
                acc = int(np.sum(win * w[oc].astype(np.int64)))
                s2 = ((acc + bias_q[oc]) * mul[oc]) >> Q_SHIFT
                golden[oy*OUT_W+ox, oc] = np.int8(s8(int(lut[clamp_s8(s2) & 0xFF])))

    cflat = outd.transpose(1, 2, 0).reshape(SP_OUT, OC)
    df = np.abs(golden.astype(int) - cflat.astype(int))
    print(f"conv{ci} exact vs dump{ci}: mism={int((df>0).sum())}/{golden.size} max={int(df.max())}")

    # baked input (dump<prev>) tile-major; weights NOT emitted (DDR blob)
    pic = ((IC + 15) // 16) * 16
    a16 = np.zeros((pic, IN_H, IN_W), dtype=np.int8); a16[0:IC] = act
    act_words = [pack_i8_word(a16[g*16:g*16+16, y, x]) for g in range(pic//16) for y in range(IN_H) for x in range(IN_W)]
    chk = 0
    for idx, (p, o) in enumerate((p, o) for p in range(SP_OUT) for o in range(OC)):
        chk = (chk + ((int(golden[p, o]) + 128) * (idx + 1))) & 0xFFFFFFFF

    def u32a(name, ws):
        L = [f"static const uint32_t {name}[{len(ws)}][4] = {{"]
        L += ["    {"+", ".join(f"0x{v:08X}u" for v in x)+"}," for x in ws]; L.append("};"); return "\n".join(L)
    def i32(name, v): return f"static const int32_t {name}[{len(v)}] = {{ {', '.join(map(str,v))} }};"
    def u32(name, v): return f"static const uint32_t {name}[{len(v)}] = {{ {', '.join(f'{x}u' for x in v)} }};"
    def u8(name, v): return f"static const uint8_t {name}[{len(v)}] = {{ {', '.join(f'0x{int(x):02X}u' for x in v)} }};"

    P = f"C{ci}E"
    body = f"""#ifndef YOLO_CONV{ci}_EXACT_DATA_H
#define YOLO_CONV{ci}_EXACT_DATA_H
#include <stdint.h>
#define {P}_CI {ci}
#define {P}_IN_W {IN_W}u
#define {P}_IN_H {IN_H}u
#define {P}_IC {pic}u
#define {P}_OC {OC}u
#define {P}_OUT_W {OUT_W}u
#define {P}_OUT_H {OUT_H}u
#define {P}_OUT_SPATIAL {SP_OUT}u
#define {P}_STRIDE {STRIDE}u
#define {P}_WGT_PER_OC {(pic//16)*KH*KW}u
#define {P}_ACT_WORDS {len(act_words)}u
#define {P}_PAD_VALUE {in_zp}
#define {P}_OUT_ZP {out_zp}
#define {P}_GOLDEN_CHK 0x{chk:08X}u

{u32a(f"yolo_conv{ci}e_act_words", act_words)}

{i32(f"yolo_conv{ci}e_bias_q", bias_q)}

{u32(f"yolo_conv{ci}e_scale_mul", mul)}

{u32(f"yolo_conv{ci}e_scale_shift", [Q_SHIFT]*OC)}

{u8(f"yolo_conv{ci}e_silu_lut", lut)}

#endif
"""
    out = ROOT / "firmware" / f"yolo_conv{ci}_exact_data.h"
    out.write_text(body, encoding="ascii")
    print(f"wrote {out} (chk=0x{chk:08X}, {STRIDE=}, {IN_W}x{IN_H}->{OUT_W}x{OUT_H})")


if __name__ == "__main__":
    main()
