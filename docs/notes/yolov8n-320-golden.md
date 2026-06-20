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

## Phase 4 (on-SoC full inference) status

Done: all per-op + tiled conv + DFL HW + sigmoid HW verified; C@320 golden above.
Next: `firmware/yolo_full.c` — assemble model.0..22 (tiled conv / C2f / SPPF /
upsample / concat) at 320 in DDR, validate each stage vs a C per-layer dump;
then on-SoC decode (DFL HW + sigmoid HW + integer argmax/geometry/NMS), compare
final boxes to the 320 golden above; read TRAP cycles -> inference time.
