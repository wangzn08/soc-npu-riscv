# tools/gen_dfl_vectors.py
# Generate directed vectors + integer golden for rtl/dfl_unit.v (and the SoC smoke).
#
# Model (mirrors the RTL fixed-point datapath exactly):
#   one "coord word" = 16 INT8 logits z[0..15]
#   idx_k = int8_max - z_k          (0..255)
#   e_k   = EXP_LUT[idx_k]          (Q1.15 unsigned, exp(0)=32768)
#   Sden  = sum_k e_k               (unsigned)
#   Snum  = sum_k e_k * W_k         (W_k Q8.8 signed)
#   dist  = Snum // Sden            (Q8.8; the Q1.15 factors cancel)
#
# Output tile-major: each input word = one coord; the engine packs 4 consecutive
# coords (one anchor) into one output word (lanes 0..3 = 4 Q8.8 distances).
import math
import random

random.seed(1234)

SCALE = 0.05                              # example per-scale bbox dequant scale
NWORDS = 64                               # 16 anchors * 4 coords (multiple of 4)
Wk = [round(k * 256) for k in range(16)]  # Q8.8; standard DFL weights W_k = k
EXP = [min(32768, round(math.exp(-i * SCALE) * 32768)) for i in range(256)]


def dfl_word(z):
    mx = max(z)
    sden = 0
    snum = 0
    for k in range(16):
        e = EXP[mx - z[k]]                # mx - z[k] in 0..255
        sden += e
        snum += e * Wk[k]
    if sden == 0:
        return 0
    d = snum // sden
    if d > 32767:
        d = 32767
    if d < -32768:
        d = -32768
    return d & 0xFFFF


inputs = []
golden = []
for _ in range(NWORDS):
    z = [random.randint(-40, 40) for _ in range(16)]
    inputs.append(z)
    golden.append(dfl_word(z))


def pack_word(z):
    w = 0
    for k in range(16):
        w |= (z[k] & 0xFF) << (8 * k)
    return w


# --- ModelSim .mem for tb_dfl_unit.v ---
with open("tests/dfl_vectors.mem", "w") as f:
    f.write(f"{NWORDS}\n")
    for v in EXP:
        f.write(f"{v:04x}\n")
    for v in Wk:
        f.write(f"{v & 0xFFFF:04x}\n")
    for z in inputs:
        f.write(f"{pack_word(z):032x}\n")
    for g in golden:
        f.write(f"{g:04x}\n")

# --- C header for the SoC end-to-end smoke ---
with open("firmware/yolo_dfl_smoke_data.h", "w") as f:
    f.write("#ifndef YOLO_DFL_SMOKE_DATA_H\n#define YOLO_DFL_SMOKE_DATA_H\n")
    f.write("#include <stdint.h>\n")
    f.write(f"#define DFL_NWORDS {NWORDS}u\n")
    f.write("static const uint16_t DFL_EXP_LUT[256] = {\n")
    f.write(",".join(str(v) for v in EXP))
    f.write("\n};\n")
    f.write("static const int16_t DFL_WK[16] = {\n")
    f.write(",".join(str(v) for v in Wk))
    f.write("\n};\n")
    f.write(f"static const uint32_t DFL_ACT[{NWORDS}][4] = {{\n")
    for z in inputs:
        w = pack_word(z)
        lanes = [(w >> (32 * i)) & 0xFFFFFFFF for i in range(4)]
        f.write("  {" + ",".join(f"0x{l:08x}u" for l in lanes) + "},\n")
    f.write("};\n")
    f.write(f"static const uint16_t DFL_GOLD[{NWORDS}] = {{\n")
    f.write(",".join(str(g) for g in golden))
    f.write("\n};\n")
    f.write("#endif\n")

print("wrote tests/dfl_vectors.mem and firmware/yolo_dfl_smoke_data.h", NWORDS, "words")
