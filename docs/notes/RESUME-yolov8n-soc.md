# RESUME — YOLOv8n @320 on-SoC deploy (read this first in a new session)

Branch: `codex/yolov8n-rtl-m0`. Last commit before this doc: `08b78fd`.
Read also: `CLAUDE.md`, `docs/notes/yolov8n-320-golden.md` (full detail/history),
memory `soc-silu-lut-range-gap.md`.

## OPTIMIZED (2026-06-24): 638.2M -> 162.9M cyc (-74.5%), still 4/4 boxes
Latency @200MHz: **3.19s -> 0.815s/img** (sub-second). NPU busy ~16.7M (~10%). Wins:
0. conv20/46 (icg8 stride2) switched from CPU ic_stream to resident conv2d_tiled (-4M).
5. **decode conf-gate without my_exp** (-16.6M): sigmoid monotonic -> compare raw logit
   `lg0 < ln(1/3)` instead of computing sigmoid(lg0)<0.25, skipping ~2100 soft-float exp.
6. **ICG_MAX/ICG_BUF 4->8** (267.8->167.0M): the im2col line buffer (ICG_MAX) + weight
   reuse buffer (ICG_BUF, in wgt_reader + top_controller_fsm) + firmware YOLO_ICG_BUF all
   4->8, so icg<=8 3x3 convs run the RESIDENT path instead of CPU INT32-psum ic_stream.
   This fixes the historical icg>4 3x3 breakpoint at its root (im2col window held only 4
   IC tiles). head cl_mid (conv39, icg5 @40x40) 41.8M -> 0.57M; whole head ic_stream gone.
   MNIST 941,155 byte-identical (icg<=4 unchanged). NPU busy 21.1->16.7M (resident runs
   fewer passes than ic_stream). conv20/46 still hardcode ic_stream (can switch to resident).
Earlier wins (1-4):
1. C2f concat -> cv2 weights; 2. SPPF separable maxpool; 3. ck gated; 4. neck cat -> cv1 weights.
Remaining buckets: c2f2/4/6 ~65M (CPU residual-add + conv orchestration), decode ~26.8M
(HW dfl_unit+sigmoid exist), SPPF maxpool 22.6M, conv20/46 ic_stream 11M (-> resident).
Architecture floor ~20-40M (~0.1-0.2s); NPU-busy floor ~16.7M.

## (older -55% milestone) 638.2M -> 284.4M, 1.42s/img. Adds to wins 1-3 below:
4. **neck FPN/PAN cat folded into c2f cv1 weights** (382.8->284.4M): same trick as #1
   but for the neck input concats (cat1-4 = upsample+tap). The cat per-source requant
   folds into the consuming c2f cv1's int8 weights+bias (conv27/31/40/51); cv1 reads the
   RAW [up|tap] native concat. Firmware replaces the 4 (upsample2+concat2_rq) with
   (upsample2 -> NK_CAT slot0 + copy_groups tap -> slot1); concat2_rq deleted. The neck
   cat stages collapsed 101M -> 3.7M (-96%; concat2_rq's per-lane requant was the cost).
   gen_yolo_neck.py: c2f_neck(fold_in=...), build_cat_raw, conv{cv1}_w_folded.bin, CV1_FOLDED.
(NOTE the old "680M ~= 3.4ms" further below is a typo — it's 3.4 SECONDS; YOLO is CPU-bound.) (NOTE: the old "680M ~= 3.4ms" line below is a
typo — 680M/200MHz = 3.4 SECONDS, not ms; YOLO is CPU-bound so it's seconds, while MNIST
941K = 4.7ms is correct). Wins, all bit-exact, 4 boxes preserved, ZERO new RTL, MNIST
untouched:
1. **C2f concat folded into cv2 weights** (638.2->470.7M): the concat per-source requant
   is a linear op before the 1x1 cv2, so it folds exactly into cv2's int8 weights+bias
   (w2f=round(w2*alpha_c), zp into bias). All 8 C2f (backbone c2f2/4/6/8 + neck
   c2f12/15/18/21). cv2 reads the raw contiguous concat; the CPU cat_req loop is deleted.
   Generators emit conv{cv2}_w_folded.bin (blob prefers it) + bias_fold + CV2_FOLDED; fw
   `yolo_c2f.c` cv2_folded mode (cv1 writes concat_ddr, residual->concat slot via add_slot,
   no cat_req). c2f4 standalone smoke maxdiff=0.
2. **SPPF maxpool separable** (470.7->416.9M): SPPF was CPU-arithmetic-bound; 5x5 -> 1x5
   then 5x1 (25->10 taps) + local int8 staging. SPPF stage 76.4M->22.6M (-70%).
3. **ck_stage gated** (416.9->382.8M): validation checksums moved behind YOLO_DEBUG_CK
   (off by default); the final 4-box check is the real gate.
Remaining big CPU buckets (next levers): neck upsample+concat ~100M (DDR-bound; fold the
FPN/PAN cat into the c2f cv1 weights, like #1 — needs a neck-gen refactor since cat-build
is decoupled from c2f_neck), head conv CPU orchestration ~88M (per-conv DMA/strip; batch or
descriptor-DMA), decode DFL/sigmoid/NMS ~44M (HW dfl_unit + sigmoid LUT exist, NMS stays CPU).
See spec `docs/superpowers/specs/2026-06-24-yolo-c2f-glue-to-npu-design.md`, memory
`soc-yolo-c2f-glue-npu.md`.

## PROFILE (2026-06-24): per-stage cycle attribution — NPU is only 3.3%!
Added Phase-0 `prof_mark` probes to `yolo_full_stem.c` (reads free-running RTL perf
counters NPU_PERF_CYC_TOTAL/BUSY/RD/WR_BEATS; dcyc=wall, dnpu=NPU-busy, dcyc-dnpu=
CPU/DMA glue). Probes are pure additive prints (git diff: 0 deletions, computation
== HEAD). Full-net RTL run = **638.2M cyc total**. Attribution:
- **NPU array actually busy: 21.1M = 3.3%** (the whole conv/GEMM compute)
- CPU pure data-movement (concat/upsample/maxpool, dnpu=0): 177.8M = **27.9%**
- CPU per-element post-proc glue inside conv/C2f (requant+SiLU LUT+residual+concat-
  requant, on exact-SiLU & ic_stream paths): **~378M = ~59%**
- CPU decode (DFL/sigmoid/NMS soft-float): 26.6M = 4.2%
- debug ck_stage (removable, not real inference): 34.1M = 5.3%
TOP single offenders: SPPF maxpool5x3+concat=**76.4M** (only 10x10x256! naive volatile
byte loop ~40cyc/op — easiest win); c2f2 glue=**81.1M** (80x80, biggest spatial);
c2f4=62.6M; neck up2+cat2=54.3M (40x40); head_P3=43.2M.
**Conclusion: YOLO bottleneck is NOT the array and NOT MMIO scheduling — it is that ALL
per-element post-processing + data-movement runs on the scalar PicoRV32.** Real next
levers (re-ranked by MEASUREMENT, overriding the old qualitative guesses):
(1) HW concat/upsample/maxpool engines (like sram_copy/img_expand) -> kills ~28%;
(2) move C2f/conv exact-SiLU+residual+concat-requant post-proc back onto NPU
post_process_top -> attacks the ~59% bucket. NOTE: ic_stream CPU psum-accumulate
(conv20/conv46) was *guessed* to be tier-1 but measured only 9M/4.8M glue — NOT heavy.
CAVEAT: this run scored 0/4 (not 4/4) because the working tree's `yolo_img_ddr.hex` +
`gen_yolo_img_hex.py` were pre-modified (image swapped off bus320 + generator takes
argv[1]); cycle attribution is unaffected (dims unchanged). To re-confirm 4/4, regen
bus320 image hex. Probe code lives in `yolo_full_stem.c` (`prof_reset`/`prof_mark`).

## DONE (2026-06-23): FULL YOLOv8n on-SoC -> 4 boxes match C oracle
`yolo_full_stem.c` runs the WHOLE net (conv0..head, model.0-22 + DFL/sigmoid/decode/
NMS on CPU) end-to-end on RTL with the DDR-preloaded bus320 image: `head dets=32,
golden-matched=4/4`, `YOLO FULL NET PASS`, 680M cyc (~3.4ms/img @200MHz, unoptimized).
KEY final fix: integer requant FLOOR bias accumulated over the deep chain and
collapsed detection confidence (P4 person 0.116). Folding rounding into the bias
(+round(2^(Q-1)/mul) in every exact generator) + concat/ratio requant +half restored
it (0.794 ~= oracle 0.755). RTL is UNCHANGED (still floors); only generated qparams +
CPU concat round. Soft-float decode links via `-lgcc` (run_all.sh). All datapaths
proven: large-IC PW (icg<=32) + 3x3 ic_stream (any stride) + neck FPN/PAN + head +
linear-out (exact path + linear LUT). MNIST baseline 10/10 / 941,155 cyc preserved.
Remaining = performance (on-chip residency, HW concat/upsample, fewer MMIO) + report.

## ONE-LINE STATUS
Core accuracy bottleneck SOLVED (preact-scale SiLU -> 4 boxes, C oracle). **c2f_4
SOLVED (2026-06-22): root cause was PW 1x1 weight-buffer overflow, NOT the doc's old
"half_groups=2 + exact-SiLU" guess.** Fixed via PW per-IC-group weight streaming
(icg>ICG_BUF). c2f_4 exact maxdiff=0, c2f_6 PASS, MNIST 10/10 941,155 cyc unchanged.
**3x3 large-IC also SOLVED** (CPU INT32-psum accumulate). **FULL CSPDarknet backbone
(model.0..8) now runs end-to-end on RTL** in `yolo_full_stem.c`: conv0->conv1->c2f2->
conv6->c2f4->conv13->c2f6->conv20->c2f8->SPPF, DDR-preloaded image + blob weights +
exact-SiLU, maxdiff=73 vs SPPF golden. **FULL feature extractor (model.0..9) runs on RTL.**
conv20 (large-IC stride2 3x3) + conv26 (icg32 PW) were the last unproven datapaths ->
validated in-chain. **Next: neck (FPN/PAN: upsample+concat+C2f x3) -> heads -> decode + NMS.**

## c2f_4 RESOLUTION (done — for history)
stage-checksum (generator bakes per-stage golden; smoke compares each DDR buffer)
proved CV1/ADD0/ADD1/CONCAT bit-exact; bug isolated to **cv2 (1x1, IC=128, icg=8)**.
Root cause: `top_controller_fsm.v:201` forced `reuse_mode=1` for PW unconditionally
(`|| i_pw_en`), and `wgt_reader.v:104 pf_all = prefetch_all || oc_single`, so cv2
tried to hold 8 IC groups in the 4-deep `wgt_buf` (ICG_BUF=4) -> groups 4-7 alias.
Unrelated to exact-SiLU (exact just removed the saturation that masked it).
Fix (user chose "PW weight streaming", no buffer growth):
- RTL: `wire reuse_mode = !i_gemm_en && (ic_groups <= ICG_BUF);` (drop `|| i_pw_en`).
  S_CALC_KERNEL already re-enters S_PREFETCH_WGT per IC tile for PW, so non-reuse
  streams one IC group at a time into `wgt_buf[*][*][0]`. No-op for MNIST (pw_en=0).
- FW (`yolo_run_conv2d_tiled`): 1x1 with `icg > YOLO_ICG_BUF(4)` tiles OC into <=16
  WITHOUT oc_single (oc_single forces pf_all=full-IC residency). IC<=4 keeps the
  fast resident oc_single path. Weight layout `[oc][icg][ko]` already matches.
Verified: c2f4 exact maxdiff=0; c2f6 legacy PASS (cv2 icg16); c2f8 cv1 icg16 OK;
MNIST 10/10 941,155 cyc. PW streaming covers icg<=16 (likely 24, not yet reached).

## 3x3 LARGE-IC RESOLUTION (done 2026-06-22 — for history)
c2f_8 bottleneck 3x3 (half_c=128, icg=8) overflowed the im2col line buffer
(ICG_MAX=4). SOLVED via **CPU INT32-psum accumulation** (not the HW-accumulate the
spec first proposed — the HW readback timing was high-risk; CPU is reliable and the
deep layers are small-spatial). `firmware/yolo_run_conv2d_ic_stream` (yolo_ops.c):
chunks IC into <=ICG_BUF groups, each chunk a raw-INT32 conv (bias0/mul1/shift0 +
`NPU_CTRL_INT32_OUT|NPU_CTRL_IC_STREAM` -> x4-spaced Out SRAM, drained to DDR), CPU
sums chunk partials, then `(acc+bias)*mul>>shift` + exact-SiLU LUT -> INT8. HW change
was tiny: `npu_top.v` int32 sequencer `i32_base` x4 when `cfg_ic_stream` (commit
dbe7fc2). c2f runner routes exact 3x3 with `icg>YOLO_ICG_BUF` to it. Scaffolding for
a future HW-accumulate optimization (CTRL[23] ic_stream FSM input, NPU_ACC_MODE/
NPU_PSUM_RD_BASE regs, post_process ACC_FIRST/ADD/FINAL + TB) is in place but unused.
VERIFIED: c2f8 exact maxdiff=0 (covers PW icg16/24 + 3x3 icg8), c2f4 maxdiff=0,
c2f6 PASS, MNIST 10/10 941,155 byte-identical.

## (OLD) THE BREAKPOINT — 3x3 conv large IC (im2col ICG_MAX=4) [RESOLVED above]
`yolo_c2f8_320_exact_smoke.c` (stage-checksum, baked, n=1, ~27s sim): **CV1 OK** (cv1
icg16 PW streaming correct), **ADD0 MISMATCH = all 0x80 (-128, saturated)**. ADD0 =
residual_add(s1, mcv2), so the bottleneck **3x3 conv (m_cv1/m_cv2, half=128 => icg=8)**
output is garbage. ROOT CAUSE: **`rtl/im2col_line_buffer.v:29 parameter ICG_MAX = 4`**
— the im2col line buffer / window holds only 4 IC tiles (64 ch). A 3x3 conv with
icg>4 reads garbage for IC tiles 4+. PROOF it's the im2col, not weights: forcing the
3x3 onto the PW-style weight-streaming path (no oc_single) changed the cycle count
(27.2M->23.4M) but produced **byte-identical wrong output** — i.e. both the resident
and streamed weight paths fail identically, so the activation window (ICG_MAX), not
wgt_buf, is the limit. Also `i_ic_groups` is [3:0] (caps at ~16). This is a THIRD,
separate capacity limit (after wgt_buf ICG_BUF, fixed for PW). c2f_8 bottleneck is the
FIRST ever icg>4 3x3 conv (MNIST/conv0-1/c2f2-6 bottlenecks all icg<=4). NOT a
regression. Firmware IC-tiling can't fix 3x3 (int32_out is FC-single-position only).

### FULL NETWORK RECIPE (from C oracle yolov8n_infer.c:585-760 — authoritative)
Conv indices (blob WGT_OF(ci)); qparams (bias/mul/shift/LUT) generate from meta+weights
via gen_yolo_c2f_exact.qp()+build_lut() (NO dump needed for qparams; dumps only for a
per-stage golden). SoC dims @320 = half the oracle's 640 comments (P3=40x40, P4=20x20,
P5=10x10). Neck C2f use cfg.shortcut=0. Neck concats are NOT scale-preserving -> requant
each part to the target conv's in_scale (concat2_to_conv): mul_part=(part_out_scale/
target_in_scale)*2^shift, like the C2f cat. Backbone taps to SAVE before overwrite:
P4tap=c2f6 out (conv19, =C2F6_OUT 0x40F90000), P3tap=c2f4 out (conv12, =C2F4_OUT
0x40E80000), P5tap=c2f8 out (conv24, =C2F8_OUT 0x406C0000).
```
FPN:  up1=upsample2x(SPPF_OUT 10x10x256); cat1=concat2_rq(up1, P4tap[c2f6], ->conv27.in)
      c2f_12 {cv1=27, m={28}, m2={29}, cv2=30, n=1, sc=0} -> fpn_mid 20x20x128
      up2=upsample2x(fpn_mid); cat2=concat2_rq(up2, P3tap[c2f4], ->conv31.in)
      c2f_15 {cv1=31, m={32}, m2={33}, cv2=34, n=1, sc=0} -> pan_p3 40x40x64  [HEAD P3]
PAN:  conv35(pan_p3, 3x3 s2)->20x20x64; cat3=concat2_rq(conv35, fpn_mid, ->conv40.in)
      c2f_18 {cv1=40, m={43}, m2={44}, cv2=45, n=1, sc=0} -> pan_p4 20x20x128 [HEAD P4]
      conv46(pan_p4, 3x3 s2)->10x10x128; cat4=concat2_rq(conv46, P5tap[c2f8], ->conv51.in)
      c2f_21 {cv1=51, m={54}, m2={55}, cv2=56, n=1, sc=0} -> pan_p5 10x10x256 [HEAD P5]
HEAD (model.22), per scale bbox(64ch)+cls(80ch), each = 3x3 stem -> 3x3 mid -> 1x1 out(linear):
      P3(pan_p3): bbox=36->38->41, cls=37->39->42
      P4(pan_p4): bbox=47->49->52, cls=48->50->53
      P5(pan_p5): bbox=57->59->61, cls=58->60->62   (1x1 out convs are LINEAR: no SiLU)
DECODE: concat 3 scales -> bbox[64,8400]/cls[80,8400]; DFL conv63 (16->1, bias-free,
      softmax-expectation over 16 bins, weighted by dequant conv63 w) -> [4,8400];
      cls sigmoid; box decode (anchor center +-dist, grid units * stride); NMS conf0.25/nms0.45.
      HW dfl_unit + sigmoid LUT exist; or CPU. Validate FINAL 4 person boxes vs C oracle.
```
conv35/46 are PAN downsamples (icg4/icg8 3x3 s2 -> conv2d_tiled resident / ic_stream).
Head 1x1 out convs (41/42/52/53/61/62/63) are LINEAR (no activation) -> use a linear/INT32
or no-SiLU requant path. cls/bbox stems (36-40,47-50,57-60) are 3x3 SiLU (some large-IC).

### STATUS 2026-06-22: model.0-21 (backbone+SPPF+NECK) ALL on RTL
`yolo_full_stem.c` chains conv0..pan_p3/p4/p5 (the 3 head input features) end-to-end:
ck SPPF=73, pan_p3=117, pan_p4=63, pan_p5=48 (all <120), 533M cyc, MNIST OK. Neck =
CPU upsample2/concat2_rq + NECK_C2F macro (c2f_12/15/18/21 shortcut=0) + conv35 (icg4
s2) + conv46 (icg8 s2 ic_stream). gen_yolo_neck.py emits qparams+goldens. ONLY the
HEAD (model.22) + decode + NMS remain.

### HEAD recipe + the LINEAR-OUT trick (resume here)
Head conv dims (ic,oc), all reading pan_p3[40x40x64]/pan_p4[20x20x128]/pan_p5[10x10x256]:
  bbox stem 36(64,64)/47(128,64)/57(256,64) 3x3; mid 38/49/59(64,64) 3x3; out 41/52/61(64,64) 1x1 LIN
  cls  stem 37(64,80)/48(128,80)/58(256,80) 3x3; mid 39/50/60(80,80) 3x3; out 42/53/62(80,80) 1x1 LIN
  DFL conv63(16,1) 1x1.
Large-IC 3x3 (ic_stream): stems 47/48 (icg8), 57/58 (icg16); cls mids 39/50/60 (icg5).
Small 3x3 (conv2d_tiled resident): stems 36/37 (icg4), bbox mids 38/49/59 (icg4).
**LINEAR out convs (41/42/52/53/61/62)**: reuse the EXACT conv path but with STANDARD
requant qparams (mul=round(in_scale*wscale/out_scale*2^20), bias=round(b/(in_scale*
wscale))-in_zp*wsum) AND a LINEAR LUT lut[k]=clamp_s8(s8(k)+out_zp). Output int8 then
dequants in decode as (int8-out_zp)*out_scale. (No int32 path needed.)
DECODE (CPU): assemble bbox[64,8400]/cls[80,8400] (dequant per scale, anchors P3 6400+
P4 1600+P5 400 @320 -> /4 = 1600+400+100=2100 anchors); DFL conv63 softmax-expectation
(4 coords x16 bins, weighted by dequant conv63 w, bias-free, no in_zp); cls sigmoid;
box decode (anchor center (ax+.5,ay+.5), x1=ax-lt,y1=ay-lt,x2=ax+rb,y2=ay+rb, *stride
{8,16,32}); NMS conf0.25/nms0.45. Validate FINAL 4 person boxes vs golden
(preact-scale SoC: (111.1,186.7)(56.1,192.8)(297.1,188.2)(15.3,201.8)). HW dfl_unit+
sigmoid LUT exist but CPU is simplest. Generator pattern: gen_yolo_neck.py (qparams via
qp/build_lut; add a qp_std for linear + emit conv63 wscale + strides).

### (older) NEXT ACTION — neck (FPN/PAN), then heads/decode
Full feature extractor (conv0..SPPF, model.0..9) is DONE in yolo_full_stem.c (stages 0-9).
SPPF output = SPPF_OUT (0x40780000, 10x10x256). Remaining for a full image:
1. **Neck (model.10..21, FPN/PAN top-down + bottom-up)**: upsample2x (HW yolo_run_upsample2x
   exists) + concat (CPU concat helper, same-scale or requant) + C2f blocks (yolo_run_c2f_block,
   no shortcut in neck C2f -- set cfg.shortcut=0). Needs exact data for each neck conv/C2f
   (extend gen_yolo_c2f_exact.py BLOCKS with the neck c2f indices, + a conv-exact gen for the
   neck 1x1/concat-feeding convs). Concat sources are P3/P4/P5 backbone taps (save c2f4/c2f6
   outputs = the P3/P4 skip tensors before they're overwritten -- mind DDR buffer reuse).
2. Heads (model.22): per-scale (P3/P4/P5) detect branch = 2x (3x3 conv exact) + 1x1 cls conv
   + 1x1 reg conv; DFL conv63 (1x1 16->1). 3. CPU decode: DFL (HW dfl_unit exists) + sigmoid
   (HW exists) + box decode + NMS. Validate FINAL boxes vs C oracle (4 persons @ conf0.25/
   nms0.45), NOT per-layer.
Reusable: yolo_run_conv2d_tiled (small-IC + large-IC 1x1 PW stream), yolo_run_conv2d_ic_stream
(large-IC 3x3, any stride), yolo_run_c2f_block (cfg.shortcut=0 for neck), yolo_run_upsample2x,
maxpool5/concat4 (in yolo_full_stem.c), DFL/sigmoid HW. Large-IC 3x3 needs psum_ddr scratch =
(out_c/16)*out_w*out_h*4*2 128-bit words; keep DDR clear of image (0x40400000..0x405A0000) +
blob (0x40800000..0x40B01000). gen_yolo_sppf_exact.py shows the pattern for a non-C2f exact gen.

## HOW TO BUILD / RUN (Git Bash from repo root)
- MNIST regression (must stay 10/10): `bash run_all.sh sim`
- A YOLO smoke needing DDR preload (image+weights): `touch .yolo_ddr` first, then
  `bash run_all.sh sim <smoke>.c yolo_c2f.c yolo_ops.c`. `.yolo_ddr` is gitignored;
  it sets +define+YOLO_DDR (env vars don't survive the Bash tool, so use the file).
  REMOVE `.yolo_ddr` before MNIST runs.
- Bash tool note: it resets cwd; prefix every command with `builtin cd /e/code/6-10/soc;`.
  Long sims: run with run_in_background:true (each multi-block sim is 3-6 min).
- C oracle (fast, no RTL): `cd yolov8n_int8; gcc -O2 -o test_infer test_infer.c
  yolov8n_infer.c -lm; YOLO_SILU_PREACT=1 YOLO_SILU_STEP=0.5 ./test_infer bus320.ppm`
  -> 4 boxes (the proven-correct SoC SiLU). Plain `./test_infer` = float golden.

## THE KEY FIX (already applied everywhere)
SiLU LUT must be indexed by the PREACT scale, not the output scale. All exact-SiLU
generators use SILU_STEP=0.5: stage-2 mul = in_scale*wscale/SILU_STEP, LUT[i] =
round(SiLU(i*0.5)/out_scale + out_zp), index zp = 0. The CTRL[22] RTL path is
UNCHANGED (out = LUT[clamp(s2)]); silu_setup in yolo_c2f.c passes silu_requant_zp=0
for exact mode. Generators: gen_yolo_conv0_320_exact.py, gen_yolo_conv1_320_exact.py,
gen_yolo_conv_exact.py (<ci> <prev>), gen_yolo_c2f_exact.py (<blk>), gen_yolo_c2f2_320.py.

## INFRASTRUCTURE (done, reusable)
- DDR preload (under +define+YOLO_DDR): image at 0x4040_0000 (gen_yolo_img_hex.py ->
  firmware/yolo_img_ddr.hex), all-64-conv weight blob at 0x4080_0000..0x40B0_1000
  (gen_yolo_weights_blob.py -> yolo_weights_ddr.hex + yolo_weight_map.h, WGT_OF(ci)).
- DDR MAP HAZARD: written buffers MUST avoid the image (0x4040_0000..0x405A_0000)
  AND the weight blob (0x4080_0000..0x40B0_1000). Free: 0x4000_0000..0x4040_0000,
  0x405A_0000..0x4080_0000, 0x40C0_0000..0x4100_0000 (16MB total, 24-bit addr).
- C2f runner (firmware/yolo_c2f.c, yolo_c2f.h): cfg.silu_exact + per-conv LUTs;
  cfg.wgt_in_blob + cv*/mcv*_wgt_ddr (WGT_OF) read weights from blob (no push_wgt).
- RTL fixes landed: row_par stride-1-only (tiled stride-2 -> serial), MAX_WIDTH=512
  (im2col, covers conv0 320-wide), exact-SiLU CTRL[22] + NPU_SILU_LOAD 0x3F4.
  TB cycle timeout raised 200M->700M (rtl/axi_sys_tb.v).

## VERIFIED ON RTL (all preact-scale where exact)
conv0/conv1/c2f_2 standalone; full stem conv0->conv1->c2f_2 (all DDR weights +
preloaded image) PASS; conv6 standalone (blob weights). conv13/20 + c2f4/6/8 data
generated + python-validated (NOT yet RTL-clean for n>=2 -- see breakpoint).

## REMAINING TO RUN A FULL IMAGE (after c2f_4 fix)
1. Fix c2f_4 n=2-exact bug (breakpoint). Then c2f_6/c2f_8 likely share it (n=2 /
   half_groups>1) -- fix once.
2. Assemble full backbone conv0..SPPF (chain in yolo_full_stem.c style; SPPF needs
   conv25/26 exact + maxpools). 3. Neck (upsample2x HW + concat + P3/P4/P5). 4. Heads
   (conv36-62 exact + linear output convs + DFL conv63). 5. Decode on CPU (DFL/sigmoid
   HW exist) + NMS. Validate FINAL boxes vs C oracle (4 persons), NOT per-layer.
   Note: per-layer-vs-dump is the WRONG gate (over-constrains a LUT SiLU); judge at
   detection level. Full-net RTL sim will be long (~30-60 min); the result is already
   known-correct (C oracle), so firmware assembly proves the DATAPATH.

## GOLDEN (320, conf 0.25 / nms 0.45) — 4 persons
float: (111.5,185.5) (58.6,192.8) (294.1,187.4) (14.6,207.7)
preact-scale SoC: (111.1,186.7) (56.1,192.8) (297.1,188.2) (15.3,201.8)
