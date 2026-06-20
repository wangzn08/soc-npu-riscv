# tools/gen_sigmoid_lut.py
# Default sigmoid post-process LUT (256-entry INT8 -> Q0.8 unsigned probability):
#   prob = sigmoid((int8 - ZP) * SCALE),  q = round(prob * 255)
# This is the boot-time default; the table is also runtime-loadable per detect
# scale via NPU_SIGM_LOAD. SCALE/ZP here are an example (replace with the real
# cls-head dequant params at deploy).
import math

SCALE = 0.05
ZP = 0
with open("rtl/sigmoid_lut_q0_8.hex", "w") as f:
    for i in range(256):
        x = i - 256 if i >= 128 else i        # int8 value at index i
        v = 1.0 / (1.0 + math.exp(-((x - ZP) * SCALE)))
        q = min(255, max(0, round(v * 255)))  # Q0.8 unsigned
        f.write(f"{q:02x}\n")
print("wrote rtl/sigmoid_lut_q0_8.hex")
