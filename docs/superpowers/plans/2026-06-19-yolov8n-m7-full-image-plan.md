# YOLOv8n M7: full-image RTL run — plan & handoff

Branch: `codex/yolov8n-rtl-m0`. Reference inference: `yolov8n_int8/` C engine
(`gcc -O2 test_infer.c yolov8n_infer.c -lm`; bus.jpg -> 4 person boxes).

## Where we are (all committed)

- stride-2 conv RTL bug fixed (`top_controller_fsm.v`).
- Generic C2f runner (`firmware/yolo_c2f.c/.h`, `yolo_run_c2f_block`): shortcut
  0/1, any n, per-bottleneck glue scales; CPU residual-add + concat requant.
- OC>64 chunking: `yolo_run_conv2d_oc_chunks` (3x3) and
  `yolo_run_pw_conv1x1_oc_chunks` (1x1) in `firmware/yolo_ops.c`.
- LINEAR conv HW path: `post_process_top.v` linear-requant (silu_requant_en &&
  !silu_en => clamp_s8(s2 + out_zp)); `tb_npu_integ` LIN_REQUANT + head bbox0.
- Verified bit-exact (RTL_TOL~8-16) per layer/block: whole backbone model.0..9
  (all convs + C2f 2/4/6/8 + SPPF), neck c2f_12 (FPN) + c2f_15 (PAN p3), detect
  head scale-0 bbox branch (conv36->38->conv41 LINEAR).
- First multi-block integrated RTL run: `yolo_backbone_tail_smoke.c`
  (conv20->c2f_8->SPPF, intermediates in DDR).
- MNIST 10/10, TRAP 941,155 byte-identical throughout.

## The ONE missing primitive: tiled conv (SRAM doesn't fit full maps)

Feature maps exceed Act SRAM (256KB): conv0 out = 320x320x16 = 1.6MB. So each
layer must run in output-row strips, with vertical halo for 3x3/stride-2.
Everything a strip calls (conv runners, C2f, pool, concat) already exists.

### `yolo_run_conv2d_tiled(...)` design (add to yolo_ops.c/.h)

Vertical pad handled by MATERIALIZING pad rows in SRAM (fill a DDR `pad_row_ddr`
with in_zp, DMA it for boundary rows); horizontal pad stays HW (pad_w). So each
strip runs with `yolo_run_conv2d_qparams_pads(..., pad_h=0, pad_w=pad, ...)`.

```
icg=in_c/16; out_w=(in_w+2*pad-kw)/stride+1; out_h=(in_h+2*pad-kh)/stride+1;
fill pad_row_ddr[0..in_w-1] = {in_zp x16}; yolo_set_pad_value(in_zp);
for oy0 in 0..out_h step strip_out_rows:
  so = min(strip_out_rows, out_h-oy0);
  strip_in_h = (so-1)*stride + kh;        // produces exactly 'so' rows w/ pad_h=0
  ir0 = oy0*stride - pad;                  // first input row (may be <0 => pad)
  for g in icg: for ri in 0..strip_in_h-1:
     r = ir0+ri; dst = (g*strip_in_h+ri)*in_w;   // Act SRAM tile-major within strip
     src = (0<=r<in_h) ? in_ddr+(g*in_h+r)*in_w*16 : pad_row_ddr;
     yolo_dma_ddr_to_act(src, dst, in_w);
  done=0;
  while done<out_c:                        // OC>64 chunks
     chunk=min(64,out_c-done);
     yolo_dma_ddr_to_wgt(wgt_ddr+done*wgt_per_oc*16, 0, chunk*wgt_per_oc);
     yolo_run_conv2d_qparams_pads(0,0,0, in_w, strip_in_h, in_c, chunk,
         kh,kw,stride, 0/*pad_h*/, pad/*pad_w*/, bias+done,mul+done,shift+done,
         ctrl|OC_SINGLE);
     for sg in 0..chunk/16-1:               // drain strip rows to full out tensor
        yolo_dma_out_to_ddr(out_ddr+(((done/16+sg)*out_h+oy0)*out_w)*16,
                            sg*so*out_w, so*out_w, 0);
     done+=chunk;
```
Notes: pointwise (1x1) layers: kh=kw=1,pad=0,stride=1 => strip_in_h=so, trivial
(no halo). The Act SRAM layout the conv FSM expects with HW_PAD is tile-major
unpadded height=strip_in_h, matching `dst` above.

### Validate (TDD)
Run an already-golden layer through tiled with FORCED multiple strips and compare
to its existing golden: `conv13` (yolo_conv13_m6b_data.h, 80x8x64 -> 40x4x128,
3x3 s2) with strip_out_rows=2 => exercises halo + boundary pad. Must match
golden bit-exact (RTL_TOL ~16). Also a 1x1 layer (e.g. conv7) tiled.

## After the primitive: assemble the full network

1. Lay out all feature maps + saved skips (p3=c2f_4? no: p4=c2f_4 out 80x8,
   p5=c2f_6 out 40x4, sppf 20x2; FPN/PAN reuse these) in DDR with fixed bases.
2. Firmware `yolo_full.c`: input = bus.jpg quantized conv0 input (640x640x3, from
   the C ref) baked or DMA'd from DDR; call tiled conv / C2f-runner / pool /
   upsample / concat per the model.0..22 graph (see yolov8n_infer.c order); save
   p4/p5/sppf for neck; run FPN/PAN; run detect head (stem/mid SiLU + LINEAR out).
3. Drain the 6 head logit tensors (bbox/cls x 3 scales) to DDR.
4. DFL + decode + NMS: host-side (PicoRV32 has no FPU). A host/Python checker
   reads the DDR logits and produces boxes; compare to the C ref / dets.
5. Golden: the C reference (`yolov8n_infer.c`) per-layer or final logits — NOT the
   pure-Python model (too slow at full res). Best: instrument the C engine to dump
   the input + a chosen intermediate/head-logit tensor, compare RTL vs that.

## Honest cost notes
- Real chip @200MHz: one image ~10s-100Ms of cycles (CPU-glue-bound) =>
  sub-second to a few seconds; FPGA is the practical target for a full run.
- RTL SIM of a full image: many hours (sim ~1e4 cyc/s); not for routine dev. Use
  per-layer/strip smokes for verification (as done); run the full thing on FPGA.
- Optimization levers (later): move concat/requant/residual off the scalar CPU
  (HW or DMA), keep inter-layer tensors resident, batch MMIO.

## Immediate next actions (fresh session)
1. Implement `yolo_run_conv2d_tiled` (+ a `_pads` OC-chunk inline) in
   firmware/yolo_ops.c/.h per the sketch above.
2. `firmware/yolo_conv13_tiled_smoke.c`: run conv13 via tiled (strip_out_rows=2),
   compare to `yolo_conv13_m6b_data.h` golden. `bash tests/run_regress.sh sim ...`.
3. Once tiled is proven, build `yolo_full.c` incrementally (backbone first,
   compare each stage's DDR output to a C-ref dump), then neck, head, host decode.
