# RESUME — YOLOv8n @320 on-SoC deploy (read this first in a new session)

Branch: `codex/yolov8n-rtl-m0`. Last commit before this doc: `08b78fd`.
Read also: `CLAUDE.md`, `docs/notes/yolov8n-320-golden.md` (full detail/history),
memory `soc-silu-lut-range-gap.md`.

## ONE-LINE STATUS
Core accuracy bottleneck SOLVED + proven (preact-scale SiLU -> 4 correct boxes in
C oracle, no RTL change). All infra + tooling done. Full STEM (conv0->conv1->c2f_2)
runs end-to-end on SoC. **Blocked at backbone extension by a c2f_4 (n=2 + exact-SiLU)
runtime bug.** MNIST baseline 10/10 / 941,155 cyc unchanged throughout.

## THE BREAKPOINT (start here)
`firmware/yolo_full_stem.c` chains conv0->conv1->c2f_2->conv6->c2f_4. Stages 0-3
PASS (prints "[stage2 c2f_2 done]" "[stage3 conv6 done]"); **c2f_4 fails**.
- Standalone `yolo_c2f4_320_exact_smoke.c` (baked weights, dump6 input): COMPLETES
  (61M cyc) but WRONG (errors=64 at pos 0, all OCs) vs its own python golden.
- In the chain (blob weights): HANGS (TIMEOUT).
Statically NARROWED: general-gen c2f_2 constants == known-good; exact-c2f4 bn[1]
constants == legacy c2f4 (passes on RTL, same runner). runner n=2 dataflow OK,
exact-SiLU OK for n=1 + OC=32 (conv1). Bug = **half_groups=2 + exact-SiLU runtime
interaction** (c2f_4 is the first exact C2f with half_groups=2; c2f_2 has 1).

### NEXT ACTION (resume here)
Runtime intermediate dump of c2f_4 exact: in a copy of yolo_c2f4_320_exact_smoke.c,
after the run, read+checksum (or per-pos compare) each stage buffer (CV1_OUT split
s0/s1, BN_OUT, MCV2_DDR, ADD0_DDR, ADD1_DDR, CONCAT_DDR) and diff vs a python dump
from gen_yolo_c2f_exact.py (add intermediate prints to compute_* there). Find the
FIRST stage that diverges. Suspects given the narrowing: the CPU residual-add /
concat with half_groups=2 reading EXACT-mode int8 (zp handling), or an OC-tile
interaction in the bottleneck 3x3 (OC=32) under exact. Build/run:
`bash run_all.sh sim yolo_c2f4_320_exact_smoke.c yolo_c2f.c yolo_ops.c` (no marker).

## HOW TO BUILD / RUN (Git Bash from repo root)
- MNIST regression (must stay 10/10): `bash run_all.sh sim`
- A YOLO smoke needing DDR preload (image+weights): `touch .yolo_ddr` first, then
  `bash run_all.sh sim <smoke>.c yolo_c2f.c yolo_ops.c`. `.yolo_ddr` is gitignored;
  it sets +define+YOLO_DDR (env vars don't survive the Bash tool, so use the file).
  REMOVE `.yolo_ddr` before MNIST runs.
- Bash tool note: it resets cwd; prefix every command with `builtin cd /e/code/6-10/soc;`.
  Long sims: run with run_in_background:true (each multi-block sim is 3-6 min).
- C oracle (fast, no RTL): `cd yolov8n_int8; gcc -O2 -o test_infer test_infer.c
  yolov8n_infer.c -lm; YOLO_SILU_PREACT=1 YOLO_SILU_STEP=0.5 ./test_infer bus320.ppm`
  -> 4 boxes (the proven-correct SoC SiLU). Plain `./test_infer` = float golden.

## THE KEY FIX (already applied everywhere)
SiLU LUT must be indexed by the PREACT scale, not the output scale. All exact-SiLU
generators use SILU_STEP=0.5: stage-2 mul = in_scale*wscale/SILU_STEP, LUT[i] =
round(SiLU(i*0.5)/out_scale + out_zp), index zp = 0. The CTRL[22] RTL path is
UNCHANGED (out = LUT[clamp(s2)]); silu_setup in yolo_c2f.c passes silu_requant_zp=0
for exact mode. Generators: gen_yolo_conv0_320_exact.py, gen_yolo_conv1_320_exact.py,
gen_yolo_conv_exact.py (<ci> <prev>), gen_yolo_c2f_exact.py (<blk>), gen_yolo_c2f2_320.py.

## INFRASTRUCTURE (done, reusable)
- DDR preload (under +define+YOLO_DDR): image at 0x4040_0000 (gen_yolo_img_hex.py ->
  firmware/yolo_img_ddr.hex), all-64-conv weight blob at 0x4080_0000..0x40B0_1000
  (gen_yolo_weights_blob.py -> yolo_weights_ddr.hex + yolo_weight_map.h, WGT_OF(ci)).
- DDR MAP HAZARD: written buffers MUST avoid the image (0x4040_0000..0x405A_0000)
  AND the weight blob (0x4080_0000..0x40B0_1000). Free: 0x4000_0000..0x4040_0000,
  0x405A_0000..0x4080_0000, 0x40C0_0000..0x4100_0000 (16MB total, 24-bit addr).
- C2f runner (firmware/yolo_c2f.c, yolo_c2f.h): cfg.silu_exact + per-conv LUTs;
  cfg.wgt_in_blob + cv*/mcv*_wgt_ddr (WGT_OF) read weights from blob (no push_wgt).
- RTL fixes landed: row_par stride-1-only (tiled stride-2 -> serial), MAX_WIDTH=512
  (im2col, covers conv0 320-wide), exact-SiLU CTRL[22] + NPU_SILU_LOAD 0x3F4.
  TB cycle timeout raised 200M->700M (rtl/axi_sys_tb.v).

## VERIFIED ON RTL (all preact-scale where exact)
conv0/conv1/c2f_2 standalone; full stem conv0->conv1->c2f_2 (all DDR weights +
preloaded image) PASS; conv6 standalone (blob weights). conv13/20 + c2f4/6/8 data
generated + python-validated (NOT yet RTL-clean for n>=2 -- see breakpoint).

## REMAINING TO RUN A FULL IMAGE (after c2f_4 fix)
1. Fix c2f_4 n=2-exact bug (breakpoint). Then c2f_6/c2f_8 likely share it (n=2 /
   half_groups>1) -- fix once.
2. Assemble full backbone conv0..SPPF (chain in yolo_full_stem.c style; SPPF needs
   conv25/26 exact + maxpools). 3. Neck (upsample2x HW + concat + P3/P4/P5). 4. Heads
   (conv36-62 exact + linear output convs + DFL conv63). 5. Decode on CPU (DFL/sigmoid
   HW exist) + NMS. Validate FINAL boxes vs C oracle (4 persons), NOT per-layer.
   Note: per-layer-vs-dump is the WRONG gate (over-constrains a LUT SiLU); judge at
   detection level. Full-net RTL sim will be long (~30-60 min); the result is already
   known-correct (C oracle), so firmware assembly proves the DATAPATH.

## GOLDEN (320, conf 0.25 / nms 0.45) — 4 persons
float: (111.5,185.5) (58.6,192.8) (294.1,187.4) (14.6,207.7)
preact-scale SoC: (111.1,186.7) (56.1,192.8) (297.1,188.2) (15.3,201.8)
