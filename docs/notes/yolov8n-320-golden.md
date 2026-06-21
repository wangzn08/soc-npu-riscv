# YOLOv8n @320 golden (C reference) — SoC deploy baseline

The C engine (`yolov8n_int8/yolov8n_infer.c`) is now resolution-parametric
(`g_yolo_input`, default 640). Same INT8 weights + per-tensor quant scales are
reused at 320×320 (no retrain, no requant). Decode grid derives from head tensor
dims; strides stay 8/16/32.

Regenerate inputs: `python` resize `bus.jpg` -> `bus640.ppm`/`bus320.ppm` (P6).
Build: `gcc -O2 -o test_infer test_infer.c yolov8n_infer.c -lm`.

## Golden detections (conf_thr=0.25, nms_thr=0.45)

640×640 (regression, unchanged — 4 persons):
```
[0] conf=0.848 bbox=(584.5,374.9,108.0,289.2) class=0
[1] conf=0.814 bbox=(115.9,385.6,147.7,299.5) class=0
[2] conf=0.774 bbox=(225.9,373.9, 93.4,268.6) class=0
[3] conf=0.337 bbox=( 29.0,419.8, 57.4,191.8) class=0
```

320×320 (SoC deploy golden — 4 persons, coords ≈ half of 640):
```
[0] conf=0.829 bbox=(111.5,185.5,47.0,131.3) class=0
[1] conf=0.829 bbox=( 58.6,192.8,78.4,153.6) class=0
[2] conf=0.755 bbox=(294.1,187.4,50.2,141.0) class=0
[3] conf=0.337 bbox=( 14.6,207.7,29.2,101.7) class=0
```

At 320: P3/P4/P5 = 40/20/10, num_anchors = 1600+400+100 = 2100.

## Per-layer golden oracle (for yolo_full.c stage validation)

`dbg_dump_conv` writes full int8 tensors when both env vars are set:
```
cd yolov8n_int8 && mkdir -p dump320
YOLO_DEBUG_DUMP=1 YOLO_DUMP_DIR=dump320 ./test_infer bus320.ppm
```
Produces `dump320/conv<ci>.bin` for ci=0..62 (conv63 is the DFL projection, done in
decode). Format: `int32 c,h,w; float scale; int32 zp; int8 data[c*h*w]` (CHW).
Each yolo_full.c stage's DDR output is compared against the matching conv<ci>.bin.
Sanity: conv0=16x160x160 zp=-127, conv26(SPPF out)=256x10x10, conv61(head)=64x10x10.

## Measured @320 per-layer cycles (tiled, perf counters)

tools/gen_yolo_layer320.py <ci> + firmware/yolo_layer320_smoke.c (PERF around the
tiled call; excludes test DDR init). All PASS vs RTL-model golden (<=tol).

| layer | shape | cyc_total (no rp) | cyc_total (row_par) | note |
|-------|-------|-----------|----------|------|
| conv1  | 160x160x16 -> 80x80x32  (3x3 s2) | 809,452    | **441,118**   | ic=16, 5 strips |
| conv20 | 20x20x128 -> 10x10x256 (3x3 s2) | 59,716,861 | **6,468,621** | ic=128/oc=256, 1 strip |

**RESOLVED:** the tiled primitive now AUTO-enables row_par (CTRL[9]) when
strip_out_rows>=16. row_par is correct in the tiled path including multi-strip
(conv1 = 5 strips, PASS) and ~9x faster on deep layers. It only misorders strips
SMALLER than 16 rows (the conv13 strip=2 halo stress test), which stay on the
serial path. Firmware NPU/DMA timeouts raised (NPU 60M, DMA 4M) for big layers.

Note: conv20 still 6.47M (ic128/oc256 oc_single over 8 ic-groups is heavy). Full
net is a mix; summing measured layers is the path to the real total (next: chain
assembly). Naive full-net @320 still likely ~50-150M cycles -> sim is long; the
on-chip residency + per-layer measurement-then-sum is the practical route.

## 1x1 pointwise tiling — DONE

yolo_run_conv2d_tiled now routes 1x1 strips through the PW engine (kh==kw==1; no
halo, no pad, row_par masked off — PW path isn't row_par-aware). conv2 @320
(80x80x32 1x1, tiled) PASS, cyc_total=829,875.

Root cause of the earlier mismatch was a generator bug: gen_yolo_layer320.py
weight packing dropped the ic-group loop, so any IC>16 layer kept only the first
16 input channels. Fixed (pack per oc x ic-group x tap). NOTE: cycle counts are
data-independent so prior measurements stand; correctness now verified for ic>16.

Both conv types tile correctly -> backbone C2f assembly is unblocked.
Measured @320 (tiled, row_par where applicable): conv1=441K, conv2(1x1)=830K,
conv20=6.47M, all PASS.

## NEXT: tiled C2f runner (precise, resumable plan)

The existing `yolo_run_c2f_block` (firmware/yolo_c2f.c) is reusable EXCEPT its 4
conv calls use the non-tiled `*_oc_chunks` path (loads whole tensor to SRAM),
which overflows at 320 (c2f_2 cv1 out 80x80x32=204KB > 128KB Out SRAM; cv2 concat
48x80x80=307KB > 256KB Act SRAM). CPU residual-add/concat are already DDR-based
and fine. Scoped change:
  1. DONE — swapped the 4 conv calls (cv1/m_cv1/m_cv2/cv2) in yolo_run_c2f_block
     to yolo_run_conv2d_tiled (DDR->DDR, 1x1 + 3x3). Added cfg->pad_row_ddr +
     cfg->strip (default 16). The 6 c2f smokes (c2f4/6/8/12/15 + backbone_tail)
     set pad_row_ddr=0x40080000, strip=16. Re-verified RTL: c2f4 PASS (31.99M),
     c2f6 PASS (18.33M), c2f8 PASS (11.39M), all Errors:0. The tiled path no
     longer overflows Out/Act SRAM, so 320 spatial is unblocked.
  2. c2f_2 @320 scales+integer model ALREADY exist in
     gen_yolo_c2f_close_m5v_smoke.py (+ m5u chain): glue=/model.2/m.0/Add
     (0.1549137533,-124 = yolo_glue_quant[0]); CAT_SCALE=0.1612435579,-125;
     conv5_out=0.0763198882,-124; CONV2 s0/s1=0.1601515412,-126. Re-run that
     chain at 80x80 with the real conv1 dump (dump320/conv1.bin), self-check vs
     dump320/conv5.bin (c2f_2 output golden).
  3. Firmware c2f2_320 smoke -> PASS -> chain backbone -> neck/head -> on-chip
     decode (DFL HW + sigmoid HW + int argmax/geometry/NMS) -> full TRAP cycles.

### RESOLVED (2026-06-21): stride-2 row_par reorder bug -> serial fallback

FIXED. Bisected with strip=8 (row_par off) => conv1 exact PASS, strip=16
(row_par on) => FAIL: the bug is the row_par 16-row drain REORDER under stride>1,
NOT padding/staging (those are bit-exact). yolo_run_conv2d_tiled now auto-enables
row_par only for stride==1 (&& strip>=16); stride-2 convs (conv0/1, downsamples
13/20/35/46/57) use the bit-exact serial path. conv1 @320 exact now PASSES,
aligned to dump320/conv1.bin within +-1. Cost: conv1 946K cyc serial vs 441K
buggy-row_par -- a row_par+stride2 reorder fix is a deferred perf optimization.
Legacy conv1_320 + c2f4 re-verified PASS (no regression). History below.

### [HISTORY] stride-2 tiled boundary bug (exposed by exact SiLU)

conv1 @320 (16->32 3x3 **stride2** pad1) with exact SiLU FAILS the C-dump check at
edge rows (errors=17, diff ~20). Root cause is NOT exact SiLU — it is a stride-2
tiled vertical-padding/strip bug that the old saturating Q4.4 LUT MASKED:
- The exact-SiLU golden matches dump320/conv1.bin within +-1 (gen self-check).
- At pos=1 (top-pad row), RTL oc3 output (-108) exactly matches a model where the
  TOP VERTICAL PAD ROW contributes 0 instead of in_zp(-127); other OCs partially
  match, so the staged strip / pad content is wrong for stride-2 strip 0.
- c2f_2 @320 (all stride-1 convs) PASSES exact -> the bug is stride-2-specific.
- The legacy conv1 smoke "passed" only because its preacts at these edges
  saturate the +-8 LUT to the same clamped value regardless of the pad error.

This is the plan's flagged top risk (分块 halo/边界 padding; stride2 单独 TB).
Fix needs a directed stride-2 tiled-pad TB then a top_controller_fsm / axi_dma /
yolo_run_conv2d_tiled strip-staging fix. firmware/yolo_conv1_320_exact_smoke.c +
gen_yolo_conv1_320_exact.py are the harness (exact-SiLU is correct; they pin the
stride-2 bug). conv0 (also stride2) will hit the same.

### Exact-SiLU layer coverage (2026-06-21)

| layer | stride | exact-SiLU vs C dump | RTL smoke |
|-------|--------|----------------------|-----------|
| conv0 | 2 | max 2 (gen self-check) | **FAIL** (see below) |
| conv1 | 2 | max 1 | PASS (serial path) |
| c2f_2 (conv2/3/4/5) | 1 | block max 13 | PASS |

**conv0 RESOLVED (im2col line-buffer width cap):** conv0 @320 now PASSES. The
root cause was NOT zero-lanes (an int32_out acc probe, conv0_accprobe_smoke.c,
proved the <16-real-IC MAC is bit-exact -- and incidentally documented that
int32_out OUT-SRAM layout is oc-group-major, word=(oc/4)*SP+pos, and only the
first oc-group drains correctly for multi-position convs). Bisected by cropping:
conv0 @18x18 PASSES, @320 FAILS => the trigger is WIDTH. `im2col_line_buffer` had
MAX_WIDTH=256 (ADDR_W=8); conv0 input is 320 wide (322 padded), so the line-buffer
write/read pointers wrapped at 256 and corrupted wide rows. conv1 (160) fit.
Fix: npu_top MAX_WIDTH 256->512 and ADDR_W=$clog2(MAX_WIDTH) in the line buffer.
conv0 @320 exact PASSES (checksum match); MNIST 10/10 / 941,155 cyc unchanged.
NOTE: strip tiling tiles HEIGHT only; for 640-wide layers bump MAX_WIDTH to 1024
or add WIDTH (column) tiling -- the FPGA-scalable route per the spatial-tiling plan.

### RESOLVED (2026-06-21): exact per-layer SiLU LUT + c2f_2 @320 PASS

The SiLU LUT range gap below is FIXED. Added CTRL[22] NPU_CTRL_SILU_EXACT_EN +
NPU_SILU_LOAD (0x3F4): a per-layer runtime-loadable SiLU LUT indexed by the
linear output-quantized preact (out-grid), so no +-8 saturation. The c2f_2 @320
smoke (firmware/yolo_c2f2_320_smoke.c, gen_yolo_c2f2_320.py, exact mode) runs the
full model.2 block on RTL (80x80, tiled DDR->DDR) and PASSES; its golden aligns
to the C float dump320/conv5.bin within ~13 (was 69), with cv1 alone within ~2
(was 116). MNIST 10/10 baseline unchanged (SiLU path gated off). tb_silu_lut.v
extended for the exact path. Original diagnosis kept below for history.

### BLOCKER found at Step 2.2 (2026-06-21): SiLU LUT range gap [RESOLVED above]

tools/gen_yolo_c2f2_320.py runs the full c2f_2 from the REAL conv1 dump and
self-checks vs dump320/conv5.bin. Result: mismatch 152231/204800, maxabs=69 —
and even cv1 (conv2 alone) is maxabs=116 vs dump320/conv2.bin.

Root cause is ARCHITECTURAL, not a generator bug:
- The C float engine matches dump320/conv2.bin BIT-EXACTLY (mism 0).
- The NPU SiLU uses a FIXED Q4.4 LUT (rtl/silu_lut_q4_4.hex, boot $readmemh in
  post_process_top.v) — input range +-8, resolution 1/16. The integer datapath
  feeds q44 ~= 16*preact into it.
- conv2 preact spans [-71.7, +26.4]. 6.4% of elements have |16*preact|>127 and
  SATURATE the LUT -> max error 116. Non-saturated elements match within +-2
  (pure LUT rounding).
- Prior per-conv/m5v smokes never caught this: they validate firmware against the
  SAME saturating RTL model, never against the C float golden. Step 2's per-layer
  alignment-to-C is the first true-oracle check.

Fix direction (recommended): make the SiLU LUT runtime-loadable PER LAYER, mirror
the existing sigmoid LUT load path (i_sigm_load_en / yolo_load_sigmoid_lut in the
same RTL file). Offline gen builds each layer's 256-entry table from its
(silu_in_scale, out_scale, out_zp); index = round(preact/silu_in_scale)+128 with
silu_in_scale covering that layer's preact range -> SiLU exact to int8. Touches
RTL (LUT load port) + firmware + all yolo generators. Plan reference:
docs/superpowers/plans/2026-06-18-yolov8n-silu-lut-m1.md.

gen_yolo_c2f2_320.py + the C2f tiled swap (commit 5c45855) are ready; c2f2_320
will align once the LUT fix lands.

## Phase 4 (on-SoC full inference) status

Done: all per-op + tiled conv + DFL HW + sigmoid HW verified; C@320 golden above.
Next: `firmware/yolo_full.c` — assemble model.0..22 (tiled conv / C2f / SPPF /
upsample / concat) at 320 in DDR, validate each stage vs a C per-layer dump;
then on-SoC decode (DFL HW + sigmoid HW + integer argmax/geometry/NMS), compare
final boxes to the 320 golden above; read TRAP cycles -> inference time.
