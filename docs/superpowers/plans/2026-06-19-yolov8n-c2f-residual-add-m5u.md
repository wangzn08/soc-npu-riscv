# YOLOv8n M5u C2f Residual-Add (signed eltwise) Bring-up

## Goal

Close the first C2f bottleneck's residual shortcut on shared hardware:
`add_out = s1 + conv4_out` (the `/model.2/m.0/Add` glue node), extending the
existing `conv0 -> conv1 -> conv2 -> slice(s1) -> conv3 -> conv4` chain.

This is the first YOLO use of an element-wise residual add through the shared
NPU post-process path. `concat(s0, s1, add_out) -> conv5` (the C2f close) is
deferred to M5v.

## Design Decision (residual add on shared HW)

The C reference C2f add dequantizes two operands at **different** quant scales
(`s1` at conv2 out-scale, `conv4_out` at conv4 out-scale), sums in real domain,
then requantizes to the glue scale `/model.2/m.0/Add`
(`out_scale=0.1549137533, out_zp=-124`).

Chosen approach (user-confirmed): **reuse the HW eltwise adder with both
operands pre-requantized to the glue scale/zp.** Then the add reduces to
`add_out_q = s1_q + conv4_q - glue_zp` (signed INT8, saturated).

### Completed (this milestone's hardware enablement) — VERIFIED

- `rtl/vector_alu.v`: added `i_signed_mode` + `i_elt_zp[7:0]`. Signed mode
  computes `sat_s8(s8(conv)+s8(skip)-s8(zp))` over [-128,127]; default mode is
  byte-identical to the legacy unsigned [0,127] MNIST residual.
- `tests/tb_vector_alu_signed.v`: directed test, 8/8 PASS (legacy + signed/zp +
  saturation + bypass).
- `rtl/param_regfile.v`: `CTRL[20] elt_signed` + `NPU_ELTWISE_ZP` (0x3D4).
- `rtl/npu_top.v`: `cfg_elt_signed`/`cfg_elt_zp` wired to `vector_alu`.
- `firmware/firmware.h`: `NPU_CTRL_ELT_SIGNED` (1<<20), `NPU_ELTWISE_ZP` (0x3D4).
- RTL compiles clean; **MNIST regression 10/10, TRAP 941,155 (== baseline),
  byte-identical** (new controls default OFF).
- `tests/run_tb.sh`, `tests/run_regress.sh`: harness scripts (work around the
  Bash-tool cwd; full license env).

## Staging mechanism (RESOLVED — no new hardware)

The HW skip source is read from **Out SRAM Port B** (`skip_rd_addr =
NPU_SKIP_BASE + out_wr_addr`). Only the post-process pipeline writes Out SRAM;
DMA cannot. And `post_process_top` has **no pure affine requant+zp path** — the
only zp-aware signed-saturating requant is the SiLU-requant path, which first
applies the SiLU LUT (verified by reading the RTL).

Resolution: `s1` is exactly conv2's (cv1, SiLU) group-1 output. So stage
`s1(glue)` by running **one extra conv2 group-1 pass** whose SiLU-requant target
is the glue scale/zp instead of conv2's own out-scale:
- OC tile = conv2 channels 16..31, weights `conv2_w[16:32]`, `SILU_EN |
  SILU_REQUANT_EN`, `NPU_SILU_REQUANT_CFG` set to the glue `mul/shift/zp`.
- Output written to a high Out-SRAM base `R_skip`, left resident (no drain).

This reuses only existing HW (per-pass programmable SiLU-requant target).

Then conv4 runs with `eltwise_en | elt_signed`, `NPU_ELTWISE_ZP = glue_zp`,
`NPU_SKIP_BASE = R_skip`, conv4's own SiLU-requant target set to the glue scale,
writing to Out base 0 (disjoint from `R_skip`). Output word =
`sat_s8(conv4_q + s1_q - glue_zp)`, drained to DDR and compared against
`yolo_glue_quant[0]` (`/model.2/m.0/Add`, scale 0.1549137533, zp -124).

## Implementation (after staging decided)

- `tools/gen_yolo_c2f_add_m5u_smoke.py`: reuse conv0..conv4 generators; model the
  staging requant + signed eltwise exactly (RTL-integer golden) plus a C-float
  tolerance check against `yolo_glue_quant[0]`.
- `firmware/yolo_c2f_add_m5u_smoke.c`: stage s1(glue) -> conv4 eltwise pass ->
  drain -> compare.
- Firmware helper in `firmware/yolo_ops.c` for the staged residual add.

## Verification — DONE

- `tb_vector_alu_signed`: 8/8 PASS.
- `tools/gen_yolo_c2f_add_m5u_smoke.py` -> `firmware/yolo_c2f_add_m5u_data.h`.
- `firmware/yolo_c2f_add_m5u_smoke.c` (+ `yolo_set_eltwise` helper in
  `yolo_ops.c`): stages s1(glue) to Out-SRAM R_SKIP=4096, runs conv4 eltwise
  pass adding it, drains and compares.
- `bash tests/run_regress.sh sim yolo_c2f_add_m5u_smoke.c yolo_ops.c yolo_plan.c`
  => `YOLO C2F RESIDUAL-ADD CPU SMOKE PASS`, TRAP 11,635,046.
- MNIST regression preserved: 10/10, TRAP 941,155 (== baseline).

## Next (M5v)

`concat(s0, s1, add_out) -> conv5` to fully close the first C2f block. s0 = conv2
group-0, s1 = conv2 group-1, add_out = this milestone's result; each requantized
to conv5's in-scale, concatenated to 48ch, then conv5 (1x1, 48->32).
