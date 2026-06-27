# Design — Eliminate YOLO C2f CPU glue (residual + concat) onto the NPU

## Context / problem
Phase-0 profiling (2026-06-24, `docs/notes/RESUME-yolov8n-soc.md`) showed the full
YOLOv8n@320 on-SoC run is 638.2M cyc, of which the NPU array is only **3.3%** and
**~59%** is CPU per-element post-processing inside conv/C2f. The convs themselves
(cv1/mcv1/mcv2/cv2) already run on the NPU; the CPU cost is two per-128-bit-word
**volatile-DDR** loops in `firmware/yolo_c2f.c`:
- **residual add** (`add_word`, lines ~180-190): `sat_s8((prev-prev_zp)*ratio_mul>>ratio_sh + mcv2_conv)`
- **concat requant** (`requant_word`, lines ~194-206): each source (s0,s1,add_i)
  rescaled to the cv2 input scale: `sat_s8((q-in_zp)*mul>>shift + cat_zp)`

The arithmetic is trivial; the cost is the per-word CPU→upsizer→arbiter→DDR round
trips. Goal: move BOTH onto the NPU so the CPU only schedules (matches the
"NPU covers all operators" competition story).

## Approach (chosen: fuse into the conv passes)

### A. Concat → fold per-source rescale into cv2 weights+bias (generator, ZERO new RTL)
cv2 is 1x1: `y_oc = Σ_c w[oc,c]·x_c + b_oc`. Concat rescales each input channel by
`α_c = cat_mul_c / 2^cat_shift` with zero-point `zp_c`. Folding is exact:
- `w'[oc,c] = round(w[oc,c]·α_c)` (kept in cv2's existing wscale; see below)
- `b'_oc   = b_oc − Σ_c w[oc,c]·α_c·zp_c`

So s0/s1/add all feed cv2 **at their own scales** and the requant disappears into
cv2's quantized weights+bias, computed offline by the generator. cv2 reads a
contiguous concat buffer (cv1_out already holds s0|s1 contiguous; residual writes
add_i into the following slots). The CPU `requant_word` loop is deleted.

Implementation detail: cv2 weights are INT8 with a per-conv wscale. Multiplying by
α_c (≈0.6..1.1) changes the effective weight scale per input channel. Since 1x1
conv quant uses one input scale + one weight scale, fold α_c into the **integer
weight values** before re-quantizing cv2's weight tensor, and recompute cv2's
`scale_mul/scale_shift` against the (now common) input scale = each source's own
scale chain. The generator already has every source's scale; it emits new
`cv2_wgt`/`cv2_bias`/`cv2_mul`/`cv2_shift`. Validate bit-exact vs the C reference.

### B. Residual → extend vector_alu with a ratio, fuse into the mcv2 pass (small RTL)
prev (bottleneck input) is NOT a conv input, so it cannot fold into weights — it
must be added. Reuse the existing eltwise path (`CTRL[3] eltwise_en`, `CTRL[20]
elt_signed`, `NPU_SKIP_BASE` 0x11C, `NPU_ELTWISE_ZP` 0x3D4). Today `vector_alu`
signed mode computes `sat_s8(conv + skip − zp)` (no ratio). Extend it to:
```
skip_scaled = (s8(skip) − s8(elt_zp)) * elt_ratio_mul >>> elt_ratio_shift   // arithmetic shift, round-half
out         = sat_s8( s8(conv) + skip_scaled )
```
- 16 lanes in parallel; one signed 8b×16b multiply + rounding shift per lane.
- New regs: `NPU_ELT_RATIO_MUL` (16b), `NPU_ELT_RATIO_SHIFT` (5-6b). Rounding =
  `+ (1<<(shift-1))` before the shift (matches CPU `add_word`).
- **Default identity**: `elt_ratio_mul=1, elt_ratio_shift=0` (or a dedicated
  `ratio_en` bit) ⇒ byte-identical to the current signed/unsigned eltwise, so the
  MNIST baseline is unaffected.
mcv2's NPU pass enables eltwise with skip=prev; its post-process result
`conv + ratio·(prev−zp)` is written straight into the concat buffer's add slot.
The CPU `add_word` loop is deleted.

### C. Integration: feeding the skip (prev) operand
mcv2 runs as a tiled DDR→DDR strip conv; the eltwise skip path reads Out SRAM Port
B at `skip_base + write_offset`. So for each output strip, burst-DMA the matching
`prev` strip into Out SRAM at `skip_base` BEFORE post-process drains it (burst, not
per-word). The strip loop in `yolo_run_conv2d_tiled`/`_ic_stream` gains an optional
"preload skip strip" step gated by an eltwise-active flag. Position/row alignment
must match the row_par drain order (mind the stride>1 reorder caveat — bottleneck
mcv1/mcv2 are stride-1, so row_par drain is in order).

## Components touched
- `rtl/vector_alu.v` — add ratio mul/shift to the signed eltwise (the only compute RTL change).
- `rtl/param_regfile.v` + `firmware/firmware.h` — `NPU_ELT_RATIO_MUL`/`NPU_ELT_RATIO_SHIFT` regs (both sides).
- `rtl/npu_top.v` — wire the two new cfg signals into `vector_alu`.
- `tests/tb_vector_alu_ratio.v` (new) — directed: ratio/zp combos vs C `add_word`; identity case == legacy.
- YOLO generators (`tools/gen_yolo_c2f_exact.py`, `gen_yolo_neck.py`, etc.) — fold
  concat scale into cv2 weights+bias; emit eltwise ratio/zp per bottleneck.
- `firmware/yolo_c2f.c` — delete `add_word`/`requant_word` loops; configure eltwise
  regs + per-strip skip preload for mcv2; cv2 reads contiguous concat buffer.
- `firmware/yolo_ops.c` — strip loop gains optional skip-strip DMA when eltwise active.

## Verification (in order; preserve baselines)
1. `rtl/vector_alu.v` directed TB: ratio/zp/round combos == C `add_word`; identity == legacy.
2. MNIST regression byte-identical: `bash run_all.sh sim` ⇒ 10/10, 941,155 cyc
   (eltwise ratio defaults to identity, so the MLP eltwise path is unchanged).
3. Generator self-check: folded-cv2 output bit-exact vs un-folded concat→cv2 (python).
4. C2f single-block RTL (c2f2, c2f4 n=2) golden vs C oracle — maxdiff unchanged.
5. Full net still 4 boxes (needs bus320 image hex regenerated first; current working
   tree has a swapped image — see RESUME). Re-profile: residual+concat buckets → ~0.

## UPDATE 2026-06-24 — superior design found: fold the residual CHAIN too (less RTL risk)

Profiling + the cv2-fold analysis revealed the residual is LINEAR and add_i is itself a
cv2 input, so the whole residual chain folds into cv2 weights — eliminating most of Part
B/C's risky skip-in-Out-SRAM work:

`add_i = r_i*(prev_i - z_i) + mcv2_i` (clamp dropped = more accurate). prev_0=s1,
prev_i=add_{i-1}. Expanding into cv2's 1x1 accumulate, cv2 reads `[s0, s1, mcv2_0..mcv2_{n-1}]`
(RAW mcv2 outputs, not the added values) with effective weights:
- W'_s0 = W_s0
- W'_s1 = W_s1 + Σ_i W_add_i · (Π_{j≤i} r_j)
- W'_mcv2_k = Σ_{i≥k} W_add_i · (Π_{j=k+1..i} r_j)
plus the source-scale α and zp folds already in Part A. Consequence:
- **n=1 blocks (c2f2, c2f8) + ALL neck C2f (shortcut=0, no residual)**: cv2 reads
  `[s0,s1,mcv2_0]`, residual VANISHES — no eltwise, no add, no skip-in-Out-SRAM.
- **n=2 blocks (c2f4, c2f6)**: only add_0 must still be materialized (it feeds mcv1_1's
  conv input); add_1 folds. So the WHOLE network keeps just **2 residual adds**.

This makes the vector_alu ratio HW (Part B, already built+verified) optional — only the 2
remaining add_0's need it (or keep them on CPU; they're tiny: 40x40 + 20x20). RECOMMENDED
implementation order is now: (1) Part-A/chain fold in generators (RTL-free), (2) keep the 2
add_0 residuals on CPU initially, (3) only if needed move them to the Part-B eltwise HW.
Validate the chain fold in python (drop-clamp vs dump) BEFORE firmware, same as fold-A.

IMPLEMENTATION STATUS (2026-06-24): Part-B RTL done+verified (MNIST byte-identical).
Part-A concat-only fold proven accuracy-safe in python (fold-A vs dump == concat vs dump).
NOT YET done: generator emit of folded cv2 weights (into the DDR weight blob via a
`conv{cv2}_w_folded.bin` side file that gen_yolo_weights_blob.py prefers) + bias_fold;
firmware contiguous concat buffer `[s0|s1|mcv2_0..]` + remove cat_req/add loops; regenerate;
restore bus320 (backed up to firmware/yolo_img_ddr.hex.userimg.bak); full-net detection sim.
The contiguous-concat firmware change ripples into the full_stem.c DDR buffer map (RESUME
flags it hazardous) — do it as a focused next increment with a c2f-block smoke gate first.

## Risk / notes
- Riskiest piece is the per-strip skip preload alignment in the tiled conv; gate it
  behind an eltwise-active flag so non-eltwise convs are untouched.
- Concat weight-folding changes cv2's effective per-channel weight precision; verify
  no extra INT8 rounding drift vs the CPU concat path at detection level (4 boxes).
- All new CTRL/reg defaults must keep MNIST byte-identical (hard gate before YOLO work).
