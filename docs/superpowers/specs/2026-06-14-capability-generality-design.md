# Spec: Capability generality — output bit-width + depthwise conv

Two independent capability features (not cycle optimizations). They expand what
networks the NPU can run, not how fast it runs MNIST. Pick either; both are
mode-gated and must keep conv/GEMM byte-identical (SCORE_CHK=D30179DF) when off.

---

## Feature A: configurable output bit-width (let the final FC run on NPU)

### Problem
`post_process_top.v` stage 3 clamps the quantized result to **INT8** (`[0,127]`
relu / `[−128,127]`). This saturates final-classifier logits, so Affine2 (50→10)
and any wide-dynamic-range output layer **must run on CPU** (decision F). A general
NPU should be able to emit wider outputs so the whole network — including logits —
runs on the array.

### Design
Add an output-width mode (e.g. CTRL bit `out_wide` or a 2-bit `OUT_FMT` register:
INT8 / INT16 / INT32-passthrough):
- **post_process (`post_process_top.v`):** stage 3 clamp becomes width-aware —
  INT8 (current, byte-identical), INT16 (clamp to ±32767), or INT32 passthrough
  (skip clamp, emit the quantized 32-bit value, or even the raw pre-quant sum).
  The output bus `o_feat` widens or packs fewer channels per 128-bit word.
- **Out SRAM write (`npu_top.v`):** a 128-bit word holds 16×INT8, or 8×INT16, or
  4×INT32 — the write packing + the DMA/copy element stride must follow the width.
- **regfile + firmware:** `OUT_FMT` register; firmware reads back wider outputs.
- **Scope:** start with INT32 passthrough for FC2 (10 logits) — smallest, proves
  the path. Then INT16 for intermediate layers if useful.
- **General:** any layer needing >INT8 output; width is a runtime register.

### Verification
- Byte-identical with OUT_FMT=INT8 (SCORE_CHK `D30179DF`).
- New: run FC2 (Affine2) on the NPU in INT32 mode, compare argmax vs the CPU
  oracle (the `NPU_GEMM_PARITY` path) for all 10 images — must match.
- Win: removes the last CPU-side classifier layer; the network runs end-to-end
  on the NPU (architecture-completeness story).

### Risk
- Medium: post_process clamp + Out-SRAM packing + DMA stride must all agree on
  width. INT8 path must stay byte-identical (mode-gated).

---

## Feature B: depthwise / grouped convolution

### Problem
The 16×16 array does **dense** conv (every OC sums over all IC). Depthwise conv
(groups = IC, each output channel from ONE input channel) is the core of
MobileNet/MobileFaceNet (the repo has `MobileFaceNet_Tutorial_Pytorch`). On the
current array depthwise degrades to ~1 useful MAC/row → effectively unsupported.
This is the biggest *capability* gap for modern efficient nets.

### Design (sketch — needs its own brainstorming pass)
Add a `groups` / `depthwise` mode:
- **Depthwise mapping:** with groups=IC, OC=IC, each PE column computes one
  channel's 3×3 (over only its own input channel), not a full IC reduction. Map
  the 16 array rows to 16 channels (like decision M's row-IC idea but per-channel,
  no cross-channel sum), or map rows to spatial positions per channel.
- **Weights:** depthwise weight tensor is [C][kh][kw] (no IC dim) — `wgt_reader`
  packing changes; far fewer weights.
- **No IC reduction:** the column cascade-sum (used by conv/GEMM) is bypassed;
  each PE's result is its channel's output directly.
- **Pointwise (1×1) conv** pairs with depthwise (depthwise-separable); 1×1 already
  works via the GEMM path, so depthwise is the missing half.
- **Touchpoints:** `systolic_16x16`/`pe_core` (per-channel output, no cross-row
  sum in depthwise mode), `wgt_reader` (depthwise weight layout),
  `top_controller_fsm` (groups loop), regfile+firmware (`groups`/`depthwise` cfg).

### Verification
- Byte-identical when depthwise off (dense conv unchanged, SCORE_CHK `D30179DF`).
- A small depthwise test vector (e.g. one MobileFaceNet depthwise layer) compared
  to a CPU reference — bit-exact.

### Risk
- High: changes the array's reduction topology again (like decision M), plus a
  new weight layout and FSM groups loop. Do a brainstorming + Phase-0 (pick a real
  depthwise layer, define the bit-exact oracle) before implementing.

### Recommendation
- **Feature A (output width)** is smaller, lower-risk, and removes a concrete
  current limitation (CPU-only final FC) — good first capability win.
- **Feature B (depthwise)** is the bigger architecture story (runs MobileNet-class
  nets) but needs a design pass of its own.
