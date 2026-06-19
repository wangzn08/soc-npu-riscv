# BUG: stride-2 convolution is inaccurate (pervasive), found via conv6 (M5w)

**Status:** Open. Localized, not yet fixed. Discovered 2026-06-19 during the
YOLOv8n conv6 (model.3) bring-up.

## Symptom

`firmware/yolo_conv6_from_c2f_m5w_smoke.c` (conv6 = 32->64, 3x3, **stride 2**,
pad 1, OC=64 oc_single) disagrees with its golden on ~70-85% of output elements.
Worst channel oc=38 reaches a 161-LSB error (full sign flip).

## Evidence (this is a real compute defect, not a tolerance/SiLU artifact)

- The golden (`tools/gen_yolo_conv6_from_c2f_m5w_smoke.py`, RTL-integer model
  using the same `silu_lut_q4_4.hex` as RTL) matches a **C-float reference**
  within **4 LSB** for all 64 channels => the golden is correct.
- Independent C-float check at oc=38, output pos=171: true = -128, HW = +29
  (157-LSB error) => HW is genuinely wrong there, not a rounding/LUT artifact.
- Per-OC params (bias/scale) read back from the regfile match the golden exactly
  => not a parameter bug.
- Truncate-vs-round in the scale step: no material change => not a rounding bug.
- **Stride is the discriminator:**
  - Same data run at **stride 1**: ~83% of elements bit-exact (the remaining
    ~17% are benign ±1-q44 SiLU-LUT amplification, consistent with prior layers).
  - At **stride 2**: only ~15% bit-exact / ~30% within 2 LSB, and the error is
    **uniform across all output rows** (row 0 — which loads input rows 0,1,2
    correctly — is just as wrong as rows 1-7).
- Uniform-across-rows rules out the vertical row-load path; the defect is in the
  **horizontal stride-2 window selection**.

## Root-cause direction

`rtl/im2col_line_buffer.v` has **no stride input** — it shifts the window right
by 1 column per `i_win_advance` and emits a window every column (stride-1
internally). Stride>1 must therefore be realized in `rtl/top_controller_fsm.v`
by which windows are consumed / how `cur_in_col` and `group_size` advance
(`cur_in_col += group_size * i_stride_sx`, line ~604) and how `o_im2col_win_advance`
is gated (`load_tile == ic_groups-1`, line ~331). The stride-2 horizontal window
selection is wrong on correctly-loaded rows. The exact interaction with
`ic_groups`/`group_size` is the prime suspect.

## Scope / impact

- Affects **all YOLO stride-2 layers**: conv0, conv1, conv6, conv13, conv20, and
  the PAN downsamples conv35/conv46.
- Earlier conv0/conv1 strip smokes "passed" only under loose RTL tolerances; they
  are likely **masked false-passes** and should be re-audited once fixed.
- **MNIST is unaffected** (DeepConvNet uses only stride-1 convs), so the 10/10
  baseline does not exercise this path.

## Not yet done (the final nail)

A minimal `int32_out` (CTRL[13]) dump of the raw q44 (bypassing SiLU) compared
to the golden q44 would confirm the defect is in the accumulation/window stage
(vs any post-SiLU effect) at the bit level for all channels. The oc=38 vs
C-float check already strongly establishes a genuine compute error.

## Repro

```
bash tests/run_regress.sh sim yolo_conv6_from_c2f_m5w_smoke.c yolo_ops.c
# prints PER-OC maxerr, global error histogram, and per-row exact-rate.
# Toggle STRIDE=1 in the generator to see the error collapse to the benign band.
```

## Fix checklist (for whoever picks this up)

- [ ] Add a directed RTL tb for stride-2 conv (small dims, known golden), ideally
      reusing `tests/tb_npu_integ.v`, covering ic_groups=1 and ic_groups=2.
- [ ] Fix the stride-2 horizontal window selection in `top_controller_fsm.v`
      (and/or thread a stride into `im2col_line_buffer.v`).
- [ ] Re-verify: MNIST 10/10 byte-identical; re-audit conv0/conv1 strip smokes
      with a TIGHT tolerance; conv6 (M5w) within ~a few LSB of the golden.
