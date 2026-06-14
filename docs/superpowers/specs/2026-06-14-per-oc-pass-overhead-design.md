# Spec: Reduce per-OC-pass NPU-start overhead

## Context / problem
Layers with OC>16 run as multiple OC-passes (decision D), each a **separate NPU
start**: firmware writes config registers, triggers `NPU_CTRL_START`, waits for
the done IRQ. Phase-0 measurement (decision N work, 2026-06-14):
- Conv5/6 firmware-measured "busy" 23.2K/img vs RTL `fsm_busy` 9.7K/img →
  **~58% (~13.5K/img) is CPU-side per-start overhead**: MMIO config + IRQ
  round-trip, ≈ **1,683 cyc/start**, 8 starts/img (4 OC-passes × Conv5/6).
- This overhead is **general**: every multi-OC-pass layer pays it (Conv3=2,
  Conv4=2, Conv5=4, Conv6=4 passes; FC1=4 GEMM passes).

Per OC-pass, `npu_conv_pass` writes 58 MMIO (verified `firmware/deepnet_deploy.c`
~L438-459): 7 dims (IN_W/H/IC/OC/KERNEL/STRIDE/PAD) + 3 addrs (ACT/WGT/OUT) + 48
bias/scale/shift (16×3) + CTRL. **Only WGT_ADDR and the 16 BIAS values change
between passes** — dims, ACT/OUT addrs, scale, shift are layer-invariant but
re-written every pass.

## Two levels (do the cheap one first; Phase-0 the expensive one)

### Level 1 — firmware MMIO hoist (zero hardware risk, DO FIRST)
Hoist the layer-invariant config OUT of the per-pass loop in `npu_conv_pass`
(and `npu_gemm_pass`). The regfile holds these values across NPU starts, so write
them once per layer:
- Once/layer (before the pass loop): IN_W, IN_H, IC, OC, KERNEL, STRIDE, PAD,
  ACT_ADDR_A, OUT_ADDR_A, and the 16 SCALE + 16 SHIFT (all constant per layer).
- Per pass (keep in loop): WGT_ADDR_A (= wgt_base + pass*tile_words), 16 BIAS
  (= biases[oc_base+ch]), CTRL start.
- Per-pass MMIO 58 → 18 (WGT + 16 bias + CTRL); one-time/layer setup 41.
  Conv6 (4 passes): 232 → 113 MMIO/layer. Helps Conv3/4/5/6 + FC1.
- **Safety:** the FSM latches config at S_IDLE on `i_start`; the regfile values
  persist between starts, so they're valid at each pass's start. bias differs per
  pass → stays in the loop. **Must verify bit-identical SCORE_CHK=D30179DF.**
- **Measure:** clean full-run A/B (hoisted vs not). MMIO writes are CPU-side
  AXI-lite (handshake-stretched ~5-15 cyc each); expect a few K/img across all
  layers. **Per the #4 lesson, A/B the real change — don't trust a proxy.**

### Level 2 — OC-tiling inside one NPU start (big, bumps decision D)
Do all OC-tiles in ONE start: one spatial sweep, the FSM iterates OC-tiles
internally (reusing im2col LOAD + amortizing the IRQ round-trip 4×→1×). This is
the bigger win (could reclaim most of the ~1,683 cyc/start × 3 saved starts/layer)
**but conflicts with decision D**: hardware has 16 bias/scale/shift regs and "one
start = 16 OC". Needs either (a) >16 bias/scale/shift regs (e.g. 64) + the FSM
looping OC-tiles in S_POST/weight-prefetch, or (b) mid-sweep bias/scale reload
per OC-tile. Touchpoints: `param_regfile.v` (more bias regs or a reload path),
`top_controller_fsm.v` (OC-tile loop within a start: re-prefetch weights +
re-feed bias per OC-tile, im2col window reused), `wgt_reader.v` (per-OC-tile
weight base), firmware (one start, write all OC-tiles' bias up-front).
- **Gate (Phase 0):** after Level 1, re-measure the per-start overhead split
  (MMIO vs IRQ-latency). If Level 1 already removed most MMIO, the residual is
  IRQ latency — then Level 2's value = (#saved starts) × (IRQ-latency portion).
  Confirm that's worth the decision-D-bumping change before committing.

## Verification
- `bash run_all.sh clean && bash run_all.sh sim` → 10/10, Errors 0 each step.
- SCORE_CHK gate `D30179DF` (firmware checksum, already in deepnet_deploy.c).
- Clean full-run A/B per level (mode/flag on vs off).

## Risk
- Level 1: low (firmware only; regfile persistence is the only assumption — gated
  by SCORE_CHK). Level 2: high (decision-D restructure; FSM OC-loop + bias regs).

## Relationship
- Independent of decisions M/N (utilization). This attacks per-start *overhead*,
  the dominant Conv5/6 cost found in #4's Phase 0.
- Branch from the current head (`feat/npu-row-block`, contains M+N) or master
  after merge.
