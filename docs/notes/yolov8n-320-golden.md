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

| layer | shape | cyc_total | npu_busy | note |
|-------|-------|-----------|----------|------|
| conv1  | 160x160x16 -> 80x80x32  (3x3 s2) | 809,452    | 425,300    | ic=16 (1 grp) |
| conv20 | 20x20x128 -> 10x10x256 (3x3 s2) | 59,716,861 | 59,180,552 | ic=128 (8 grp), **no row_par** |

**KEY FINDING:** the tiled primitive runs WITHOUT row_par (CTRL[9]), so deep
(ic128/oc256) layers process output positions ~serially -> conv20 alone = 59.7M
cycles. The full net this way would be hundreds of M cycles (impractical sim).
Enabling row_par cuts this ~15x BUT the tiled strip-drain must match row_par's
reverse-order 16-deep reorder semantics (adding NPU_CTRL_ROW_PAR naively breaks
conv13 tiled golden). **Optimization TODO before full-net is practical: row_par-
aware strip drain in yolo_run_conv2d_tiled.** Firmware NPU/DMA timeouts were
raised (yolo_ops.c: NPU 60M, DMA 4M) so big layers complete.

## Phase 4 (on-SoC full inference) status

Done: all per-op + tiled conv + DFL HW + sigmoid HW verified; C@320 golden above.
Next: `firmware/yolo_full.c` — assemble model.0..22 (tiled conv / C2f / SPPF /
upsample / concat) at 320 in DDR, validate each stage vs a C per-layer dump;
then on-SoC decode (DFL HW + sigmoid HW + integer argmax/geometry/NMS), compare
final boxes to the 320 golden above; read TRAP cycles -> inference time.
