# SiLU LUT range gap — quantified evaluation (2026-06-21)

Decision support for "评估 SiLU LUT 缺陷影响再动手". Authoritative data: the C
INT8 golden engine (`yolov8n_int8/yolov8n_infer.c`, matches ONNX/dumps bit-exact)
instrumented with a per-conv pre-activation probe (`YOLO_PREACT_STATS=1`), run on
`bus320.ppm`. The probe reports, per SiLU conv, the float pre-activation min/max
and the fraction with `|preact| > 7.9375` — exactly the elements that SATURATE
the NPU's fixed Q4.4 SiLU LUT (`rtl/silu_lut_q4_4.hex`, range ±8, since the
datapath feeds `q44 ≈ 16·preact` and `|16·preact|>127 ⟺ |preact|>7.9375`).

## Result: saturation is concentrated in the accuracy-critical layers

56 SiLU convs total. Bucketed by saturated-element fraction:

| bucket | layers | which |
|--------|--------|-------|
| **≥5%** | 6 | conv1 (20.0%, ±95), conv0 (12.3%, ±39), conv2 (6.4%, −72..26), **conv39** (6.3%, −48..23), conv3 (6.0%, ±32), **conv50** (5.4%, −101..68) |
| 3–5% | 3 | conv49 (3.6%), conv4 (3.6%), **conv38** (3.0%) |
| 1–3% | 3 | **conv59** (2.9%), **conv60** (2.75%), conv5 (1.05%) |
| 0.5–1% | 1 | conv47 (0.6%) |
| 0.05–0.5% | ~13 | conv23/51/44/57/36/48/45/14/58/18/46/56/37 |
| <0.05% (fine) | ~30 | most of the mid backbone/neck (conv6..conv35, conv40, conv43, conv54/55) |

Two clusters dominate, and they are the layers that matter most:
1. **STEM** (conv0–conv5): early high-dynamic-range feature extraction. conv0/1
   are the worst in the whole net (12–20% saturated, preact to ±95).
2. **DETECTION HEADS** (bbox: conv38/49/59, cls: conv39/50/60, plus conv47): the
   cls heads are extreme — conv50 preact reaches **−101**. A ±8 LUT is hopeless
   there; these feed box/score decode directly, so error here moves final boxes.

The mid backbone/neck is essentially unaffected (calibrated scales keep those
preacts within ±8), which is why MNIST and the single-layer demos never showed a
problem.

## Why it stayed hidden

Every prior per-conv / m5v / c2f smoke validated the FIRMWARE against the same
saturating RTL model (the generators' `conv_rtl` + `rtl_silu_byte`), never
against the C float golden. Step 2's per-layer alignment-to-C (逐层对 C dump) is
the first true-oracle check and it surfaced the gap immediately. Lesson: any
"逐层 bit/±1 对齐" claim must compare to the C dump, not the RTL model.

## Fix options (with cost / risk)

| option | fidelity | RTL | firmware | gen | risk |
|--------|----------|-----|----------|-----|------|
| **A. per-layer runtime-loadable SiLU LUT** (recommended) | exact-to-int8 every layer | add load port mirroring the sigmoid LUT (`i_sigm_load_en` already in `post_process_top.v`); index by the layer's own out-quant grid | `yolo_load_silu_lut()` + call per conv (256 writes/layer, or DMA the table) | each gen builds its layer LUT from (out_scale,out_zp); `rtl_silu_byte` indexes by out-grid | LOW — SiLU path is YOLO-only, default-OFF (CTRL[18]); MNIST uses ReLU, baseline 10/10 untouched |
| B. one wider fixed LUT (e.g. Q7.1, ±64) | still clips conv1(±95)/conv50(±101); loses near-0 resolution where SiLU matters | swap hex + index scale | none | regen one hex | MED — partial fix, hurts the 30 "fine" layers' resolution |
| C. ship as-is, document loss | broken stem+heads | none | none | none | HIGH — final boxes wrong; contradicts "逐层对齐" deliverable |

**Recommendation: Option A.** It is the only one that makes the per-layer
alignment-to-C deliverable true, the precedent (sigmoid LUT) already exists in the
same RTL file, and MNIST regression risk is essentially zero because the SiLU path
is gated off for the MNIST build. The cost is spread across RTL + firmware + the
yolo generators, but each piece is small and mirrors existing code.

## Reproduce

```bash
cd yolov8n_int8
gcc -O2 -o test_infer test_infer.c yolov8n_infer.c -lm
YOLO_PREACT_STATS=1 ./test_infer bus320.ppm 2>preact.log >/dev/null
```
Probe lives in `conv_int8()` behind `YOLO_PREACT_STATS` (default off, no effect on
normal runs). See [yolov8n-320-golden.md](yolov8n-320-golden.md) blocker section.
