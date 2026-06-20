# YOLOv8n Detect Head + Decode (Milestone 6) — design & open issues

## Structure (from yolov8n_infer.c, model.22)

3 scales fork from the PAN outputs; each has a parallel bbox + cls branch:
- Scale 0 (80x80, pan_p3 64ch): bbox=conv36->38->41, cls=conv37->39->42
- Scale 1 (40x40, pan_p4 128ch): bbox=conv47->49->52, cls=conv48->50->53
- Scale 2 (20x20, pan_p5 256ch): bbox=conv57->59->61, cls=conv58->60->62
- DFL: conv63 (16->1 1x1), bias-free.

Each branch: stem(3x3 SiLU) -> mid(3x3 SiLU) -> out(1x1 **LINEAR**). bbox out =
64ch (4 coords x 16 DFL bins); cls out = 80ch (classes).

Tail (CPU/host, float): assemble [84, 8400], DFL softmax-expectation over 16 bins
(conv63 weights) -> 4 distances, anchor decode -> xyxy, sigmoid(cls), NMS.

## Status

- Inputs ready: the runner produces pan_p3 (c2f_15, verified) and would produce
  pan_p4/p5 (c2f_18/c2f_21 — same shortcut=0 runner path, not yet generated).
- The stem/mid convs (conv36/37/38/39 ...) are ordinary 3x3 SiLU convs — already
  a proven path (OC-chunked).

## Open issue 1: the LINEAR output convs (conv41/42/... has_silu=0)

The shared post-process only has a zero-point-aware signed requant **inside the
SiLU path** (it requants `LUT[q44]`). It has no "requant without SiLU":
- `silu_en=0` => output is `s2_val[7:0]` with only an unsigned upper clip, which
  mishandles negative values and applies no output zero-point.
- The SiLU path first clamps q44 to the Q4.4 range [-128,127] (=[-8,8]); a LINEAR
  conv's pre-output value is NOT bounded to ±8, so reusing the SiLU requant would
  clip it. (Verified: conv41 out_scale=0.140 zp=-60, conv42 out_scale=0.162
  zp=+109 — wide ranges.)

**Recommended:** run the linear output conv with `int32_out` (CTRL[13]) to emit
the raw INT32 scaled accumulation (`s2_quant`), then requant on the CPU:
`out_q = clamp_s8((s2 * req_mul >> req_shift) + out_zp)`, or — since the C tail
dequantizes logits to float immediately — just dequantize the INT32 directly and
feed the float decode. No new RTL. Needs: int32_out drain wired through the
OC-chunk path (the int32 serializer currently used for FC2/GEMM emits 16 INT32 /
position; for OC=64 heads that's 4 tiles — drain per chunk).

## Open issue 2: DFL/decode/NMS on a no-FPU CPU

PicoRV32 here is rv32imc (no hardware float). The C reference DFL uses
`softmax`/`exp` and float decode/NMS. On-chip that needs soft-float (libgcc) or a
fixed-point reimplementation — non-trivial. Per the roadmap, "CPU handles
DFL/decode/NMS first" can mean a **host-side checker** consuming the NPU-produced
INT8/INT32 logits, rather than running decode on the PicoRV32. Recommended for
first bring-up: NPU produces the head logits (drained to DDR), a host/Python
golden does DFL+decode+NMS and compares to `yolov8n_int8/dets_int8.json`.

## Suggested increments

1. Generate pan_p4 (c2f_18) and pan_p5 (c2f_21) via the runner (shortcut=0; same
   as c2f_12/15). Also conv35/46 (PAN downsamples, stride2, proven path).
2. Head stem/mid: run conv36/38 (and the cls/other scales) on the NPU (proven
   SiLU conv path) from the pan outputs; verify intermediate logits.
3. Linear output convs via int32_out -> CPU/host requant; verify bbox/cls logits
   (INT8 or dequantized float) against the C golden.
4. DFL + decode + NMS host-side; compare final boxes to dets_int8.json.

## Note on the remaining neck

c2f_18 (model.18) and c2f_21 (model.21) are the **same shortcut=0 runner path**
already verified by c2f_12/c2f_15; only their fixtures differ. conv35/46 are
stride-2 convs (proven path). These are mechanical to add when the head needs
pan_p4/pan_p5.
