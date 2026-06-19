# YOLOv8n Generic C2f Block Runner (route 2)

## Goal

Replace per-block hand-written C2f firmware with one parameterized runner so
model.4/6/8 (and the neck C2f's) are cheap to bring up. Prove it by reproducing
the already-verified c2f_2 (model.2) result, then extend to n=2 bottlenecks.

## C2f structure (from yolov8n_infer.c)

`cv1(1x1) -> split into s0,s1(half_c each) -> for i in 0..n-1: bottleneck_i(s1 or
prev) [+ shortcut add] -> concat(s0, s1, add_0..add_{n-1}) requant to cv2 in-scale
-> cv2(1x1)`. Backbone C2f's have shortcut=1; neck C2f's shortcut=0.

Proven pieces (c2f_2, n=1): M5u residual add via HW signed eltwise (stage s1 to
glue scale in Out SRAM by re-running cv1 group-1, then conv4 eltwise pass); M5v
CPU integer-requant of {s0,s1,add} to cv2 in-scale, concat tile-major, cv2 on NPU.

## Firmware runner (firmware/yolo_c2f.c)

`yolo_c2f_cfg_t` carries: dims (in_w/h, half_c groups, spatial), DDR bases, and
per-stage qparam/weight pointers for cv1, each bottleneck conv (m_cv1/m_cv2, 3x3),
cv2 (1x1), plus glue (add) and concat requant params, n_bottleneck, shortcut.

`yolo_run_c2f_block(cfg)`:
1. cv1 (1x1) over the block input -> DDR (split halves s0,s1).
2. prev = s1. For i in 0..n-1: m_cv1_i(3x3) -> m_cv2_i(3x3); if shortcut, stage
   prev(glue) to Out SRAM and run m_cv2_i with HW signed eltwise (add prev),
   producing add_i at the glue scale; prev = add_i. (no-shortcut: add_i = m_cv2_i out.)
3. CPU integer-requant s0, s1, add_0..add_{n-1} to cv2 in-scale; concat tile-major.
4. cv2 (1x1) on NPU -> block output.

## Increments

- [x] m6a-1: `yolo_run_c2f_block` (firmware/yolo_c2f.c/.h) for n=1 shortcut;
  `yolo_c2f_block_smoke.c` runs c2f_2 via the runner (block input = baked conv1
  output, cfg wired from existing c2f_2 fixtures) and matches the M5v golden.
  PASS (TRAP 38,600,132). The i>0 residual staging path returns 0 (TODO m6a-2).
- [ ] m6a-2: extend to n=2 (two bottlenecks, two adds, concat 4 pieces); bring up
  c2f_4 (model.4) end-to-end from conv7 output. Generate conv8..conv12 fixtures.
- [ ] m6a-3: c2f_6 (model.6) and c2f_8 (model.8) via the same runner.

## Verification

Each increment: `bash tests/run_regress.sh sim <smoke>.c yolo_ops.c yolo_c2f.c ...`,
bit-exact (RTL_TOL ~8) vs RTL-integer golden; MNIST 10/10 preserved.
