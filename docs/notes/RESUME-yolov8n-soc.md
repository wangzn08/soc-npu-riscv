# RESUME — YOLOv8n @320 on-SoC deploy (read this first in a new session)

Branch: `codex/yolov8n-rtl-m0`. Last commit before this doc: `08b78fd`.
Read also: `CLAUDE.md`, `docs/notes/yolov8n-320-golden.md` (full detail/history),
memory `soc-silu-lut-range-gap.md`.

## ONE-LINE STATUS
Core accuracy bottleneck SOLVED (preact-scale SiLU -> 4 boxes, C oracle). **c2f_4
SOLVED (2026-06-22): root cause was PW 1x1 weight-buffer overflow, NOT the doc's old
"half_groups=2 + exact-SiLU" guess.** Fixed via PW per-IC-group weight streaming
(icg>ICG_BUF). c2f_4 exact maxdiff=0, c2f_6 PASS, MNIST 10/10 941,155 cyc unchanged.
**Now blocked at c2f_8 by a SEPARATE pre-existing bug: 3x3 conv non-reuse (icg>4).**

## c2f_4 RESOLUTION (done — for history)
stage-checksum (generator bakes per-stage golden; smoke compares each DDR buffer)
proved CV1/ADD0/ADD1/CONCAT bit-exact; bug isolated to **cv2 (1x1, IC=128, icg=8)**.
Root cause: `top_controller_fsm.v:201` forced `reuse_mode=1` for PW unconditionally
(`|| i_pw_en`), and `wgt_reader.v:104 pf_all = prefetch_all || oc_single`, so cv2
tried to hold 8 IC groups in the 4-deep `wgt_buf` (ICG_BUF=4) -> groups 4-7 alias.
Unrelated to exact-SiLU (exact just removed the saturation that masked it).
Fix (user chose "PW weight streaming", no buffer growth):
- RTL: `wire reuse_mode = !i_gemm_en && (ic_groups <= ICG_BUF);` (drop `|| i_pw_en`).
  S_CALC_KERNEL already re-enters S_PREFETCH_WGT per IC tile for PW, so non-reuse
  streams one IC group at a time into `wgt_buf[*][*][0]`. No-op for MNIST (pw_en=0).
- FW (`yolo_run_conv2d_tiled`): 1x1 with `icg > YOLO_ICG_BUF(4)` tiles OC into <=16
  WITHOUT oc_single (oc_single forces pf_all=full-IC residency). IC<=4 keeps the
  fast resident oc_single path. Weight layout `[oc][icg][ko]` already matches.
Verified: c2f4 exact maxdiff=0; c2f6 legacy PASS (cv2 icg16); c2f8 cv1 icg16 OK;
MNIST 10/10 941,155 cyc. PW streaming covers icg<=16 (likely 24, not yet reached).

## THE BREAKPOINT (start here) — 3x3 conv large IC (im2col ICG_MAX=4)
`yolo_c2f8_320_exact_smoke.c` (stage-checksum, baked, n=1, ~27s sim): **CV1 OK** (cv1
icg16 PW streaming correct), **ADD0 MISMATCH = all 0x80 (-128, saturated)**. ADD0 =
residual_add(s1, mcv2), so the bottleneck **3x3 conv (m_cv1/m_cv2, half=128 => icg=8)**
output is garbage. ROOT CAUSE: **`rtl/im2col_line_buffer.v:29 parameter ICG_MAX = 4`**
— the im2col line buffer / window holds only 4 IC tiles (64 ch). A 3x3 conv with
icg>4 reads garbage for IC tiles 4+. PROOF it's the im2col, not weights: forcing the
3x3 onto the PW-style weight-streaming path (no oc_single) changed the cycle count
(27.2M->23.4M) but produced **byte-identical wrong output** — i.e. both the resident
and streamed weight paths fail identically, so the activation window (ICG_MAX), not
wgt_buf, is the limit. Also `i_ic_groups` is [3:0] (caps at ~16). This is a THIRD,
separate capacity limit (after wgt_buf ICG_BUF, fixed for PW). c2f_8 bottleneck is the
FIRST ever icg>4 3x3 conv (MNIST/conv0-1/c2f2-6 bottlenecks all icg<=4). NOT a
regression. Firmware IC-tiling can't fix 3x3 (int32_out is FC-single-position only).

### NEXT ACTION (resume here) — needs a DECISION
Two RTL options for 3x3 large-IC (icg>4); pick one:
1. Bump `ICG_MAX` in im2col_line_buffer.v (+widen `i_ic_groups` past [3:0]); also the
   3x3 conv then hits the SAME wgt_buf overflow cv2 had, so re-add the 3x3 arm of the
   conv2d_tiled streaming (was prototyped then reverted: stream when icg>ICG_BUF, tile
   OC<=16, drop oc_single + mask row_par). Covers icg up to the new ICG_MAX (<=16 unless
   counters widened); BRAM cost grows (lb_bank0-3 = 4 x MAX_WIDTH x ICG_MAX x 128b).
2. im2col IC-streaming: process IC tiles in chunks of <=ICG_MAX, re-load rows per chunk,
   accumulate in the array across chunks (FSM change, general, no BRAM blowup).
Affected layers: c2f8 bottleneck (icg8), downsample conv13/conv20 (icg8/16), SPPF, many
head 3x3. Build/run the isolation smoke (no marker, ~27s):
`bash run_all.sh sim yolo_c2f8_320_exact_smoke.c yolo_c2f.c yolo_ops.c`.

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
