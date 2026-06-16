# CLAUDE.md

Handoff note for the SoC/NPU MNIST deployment. Read before changing code.

## Project Snapshot

Workspace: `E:\code\6-10\soc`

This is a small RTL CPU+NPU SoC for the Phytium enterprise problem 3 track:

```text
Intelligent core fusion, low-power strong compute:
heterogeneous processor design based on CPU and NPU.
```

Current architecture:

- PicoRV32 RISC-V AXI CPU with private firmware RAM at `0x0000_0000`.
- UART MMIO `0x1000_0000`, TEST MMIO `0x2000_0000`.
- NPU register MMIO `0x3000_0000..0x3000_0FFF`.
- Shared DDR/memory model `0x4000_0000+`, now exposed as a 128-bit AXI4 data
  path behind the arbiter.
- NPU = AXI DMA + Act/Wgt/Out SRAM ping-pong banks + im2col + weight reader +
  16x16 systolic array + post-process (bias/quant/ReLU/maxpool) + vector ALU.
- Current accelerators/features include resident conv weights, weight-prefetch
  reuse, Conv+Pool fusion, hardware padding, GEMM/FC mode, DMA/NPU IRQs,
  OC-pass Out-SRAM overlap, 128-bit AXI DMA/shared memory, CPU 32->128 upsizer,
  RTL performance counters, and two on-chip data-movement engines: `img_expand`
  (camera raw bytes -> tile-major Conv1 input, in SRAM) and `sram_copy`
  (Out SRAM -> Act SRAM inter-layer residency, no DDR round-trip).
- NPU compute-core optimizations (opt-in CTRL bits, ported 2026-06-15): `row_par`
  CTRL[9] = 16-row spatial parallelism (task E); `gemm_reduce` CTRL[10] = GEMM
  16-row IC-reduction (decision M); `row_block` CTRL[11] = row-block packing for
  narrow layers (#4); `oc_single` CTRL[12] = compute ALL OC tiles in ONE NPU
  start with one shared im2col load (decision O); `int32_out` CTRL[13] = raw
  un-clamped INT32 output (decision Q), used to run FC2 on the NPU. Each is OFF
  by default and the FSM is byte-identical when off. Together they cut `npu_busy`
  by ~82% and put the whole MLP (FC1+FC2) on the NPU.

Simulator flow is ModelSim/Questa, not VCS. Use `run_all.sh` for normal
verification.

## Competition Requirements And Assessment

Important contest requirements and the current project status:

- CPU choice: Cortex-M0 or RISC-V. This project uses PicoRV32, so it matches the
  RISC-V option.
- NPU integration: at least a 4x4 systolic array integrated into the CPU SoC.
  This project uses a 16x16 array with 16 INT8 lanes per PE.
- AXI communication: support AXI-Lite single-beat control and AXI Burst data
  movement. This project has MMIO register control and DMA burst movement.
- Functional verification: must prove AXI Burst increment transfers, CPU/NPU
  cooperation, and NPU compute correctness. The current 10-image MNIST run is
  good system evidence, and directed AXI read/upsizer tests now exist; coverage
  and more NPU/DMA directed tests are still needed.
- Performance target: evaluated at 200 MHz. The design advertises a theoretical
  peak of about 1.64 TOPS at INT8. Effective utilization is now measurable by
  RTL counters, but current end-to-end utilization is still low.
- Bus target: AXI Burst bandwidth utilization should be at least 60%, with 80%
  as a stronger target. Measured NPU DMA read utilization is 100%. Write
  utilization now reads 66% only because on-chip residency cut DDR writes to 40
  beats total (10 tiny FC1-output bursts), so per-burst startup dominates the
  ratio -- the design barely writes DDR. If the report needs a high write-burst
  number, measure it on a write-heavy micro-benchmark rather than the residency
  deploy run.
- Low-power target: the report should show idle clock gating or better:
  array/operand gating, SRAM bank enable gating, and counters showing idle time.
  Current RTL has enables and counters, but not a strong low-power architecture
  story yet.
- Deliverables: design document, RTL source, RTL simulation report or FPGA
  validation report. FPGA validation can add points.

Current architectural assessment:

- The architecture is now a credible competition prototype, not only a functional
  demo. It has a complete CPU+NPU flow, IRQ-driven cooperation, conv/pool/FC
  acceleration, and hardware counters.
- The biggest remaining gap is not correctness. The 10-image deployment passes.
  AXI bandwidth now has strong counter evidence. The remaining proof gap is
  array utilization, setup/preload accounting, low-power evidence, and coverage.
- The former 32-bit shared-memory bottleneck has been removed. The NPU DMA,
  arbiter, and shared memory model are now 128-bit. The CPU remains a 32-bit
  single-beat master and reaches the 128-bit fabric through `axi_upsizer_32_128`.
- The current AXI path is good enough that bus bandwidth is no longer the main
  bottleneck. Remaining cycle cost is now dominated by one-time setup/weight
  preload and per-layer CPU MMIO scheduling. The CPU no longer does image
  scatter/pad, Pool3->FC1 reorder, or FC arithmetic in the default deploy path.
- `docs/superpowers` contains implementation plans/specs for GEMM/FC,
  hardware padding, and weight reuse. The code has already implemented most of
  those plans, but the plan checkboxes are stale and should be treated as design
  history unless re-audited.

If redesigning from scratch for a higher-scoring submission, the target would be:

```text
32-bit RISC-V CPU
+ AXI-Lite control bus
+ 128-bit AXI4 data crossbar / shared memory path
+ descriptor-based DMA
+ Act/Wgt/Psum/Out scratchpads
+ 16x16x16 INT8 systolic array
+ Conv/GEMM dual mode
+ hardware performance counters
+ clock gating, operand gating, and SRAM bank gating
```

The key principle is: CPU schedules, NPU covers Conv/Pool/FC, the data path stays
wide enough to feed the array, and performance/power claims are backed by
counters rather than architecture diagrams alone.

## Environment & Build

All commands run from an MSYS/Git Bash shell on Windows (not PowerShell/cmd).

```bash
bash run_all.sh              # full flow: fw + RTL compile + simulation
bash run_all.sh fw           # firmware only
bash run_all.sh compile      # RTL compile only (vlib + vlog)
bash run_all.sh sim          # full flow, headless
bash run_all.sh waves        # full flow + GUI waveform
bash run_all.sh clean        # remove sim artifacts
bash run_all.sh distclean    # also remove firmware/build
```

The Makefile is deprecated; use `run_all.sh` instead.

Tool paths (override via environment variables):
- `RISCV_PREFIX` — RISC-V GCC toolchain (default: `E:/Riscv_Tools/xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-`)
- `MGC_LICENSE_FILE` — ModelSim license (default: `E:/modelsim/LICENSE.TXT`)
- `PYTHON` — Python interpreter (default: `python`)

### How testing works

There is no unit-test framework. "The test" is one simulation run:
- The testbench (`rtl/axi_sys_tb.v`) prints `ALL TESTS PASSED.` when firmware writes `123456789` to MMIO `0x2000_0000`, or `ERROR!`/`TIMEOUT` otherwise.
- Firmware `print_*` output appears on the simulator console via UART MMIO.
- The current firmware (`deepnet_deploy.c`) runs 10 MNIST images and prints per-image predictions and an overall accuracy count.
- `vsim` must run from the repo root — firmware hex path is resolved by `$readmemh` relative to the simulation working directory.
- After editing RTL, run `bash run_all.sh clean` before recompiling (or vlog the changed file manually).

## Current Result

Latest normal regression verified on 2026-06-15 (NPU compute-core port:
row_par + oc_single + gemm_reduce + FC2-on-NPU INT32):

```text
Command: bash run_all.sh sim
Result:  10/10 correct, DEPLOY SUCCESS
TRAP:    941,155 clock cycles  (~4.7 ms for 10 images @ 200 MHz, ~0.47 ms/image)
Sim:     Errors: 0, Warnings: 18
Log:     sim/modelsim/direct_batch.log
```

Latency journey: 10,778,593 (original) -> 2,378,377 (img_expand + sram_copy +
weight preload + FC1 fold) -> 941,155 (compute-core port). -91% vs original,
-60% vs the data-movement merge.

Current-version changes to highlight:

- **FC2 hardwareization:** FC2 now runs as an NPU GEMM pass. `int32_out`
  (CTRL[13]) writes raw scaled INT32 logits, so the final layer no longer needs
  the old CPU dot-product and also avoids INT8 logit saturation.
- **NPU core update:** `row_par` (CTRL[9]) computes up to 16 output positions per
  sweep, `oc_single` (CTRL[12]) lets one start cover all OC tiles with a shared
  im2col load, `row_block` (CTRL[11]) improves narrow layers, and
  `gemm_reduce` (CTRL[10]) accelerates FC/GEMM IC reduction.
- **Image preload moved toward hardware:** `gen_act_hex.py` now stores raw
  byte-packed images in DDR. Runtime only DMAs 49 words/image into Act SRAM;
  `img_expand` builds the 16-lane Conv1 activation in SRAM, and spatial conv
  padding is injected by the NPU FSM through `NPU_CTRL_HW_PAD`.
- **Layer residency:** Conv outputs use `sram_copy` to move Out SRAM -> Act SRAM
  and ping-pong between Act regions. Conv1..Conv6 no longer round-trip through
  DDR.
- **FC1 input reorder removed:** Pool3->FC1 runtime transpose is gone. FC1
  weights are packed in Conv-output order by `gen_weights_hex.py`.
- **Done/IRQ semantics fixed for INT32 output:** in `int32_out` mode, the visible
  NPU done/IRQ waits until all 4 INT32 Out-SRAM words have been written.

Current RTL hardware counters printed by firmware:

```text
cyc_total = 818,640
npu_busy  = 225,360
arr_active=  27,170
rd_beats  =     530
wr_beats  =      80
rd_busy   =     530
wr_busy   =     120

array_util(arr/total)  = 3%
array_eff(arr/busy)    = 12%
npu_active(busy/total) = 27%
rd_bw_util             = 100%
wr_bw_util             = 66%
```

IMPORTANT - reading `array_util`/`arr_active` after the compute-core port:
`arr_active` collapsed 409,360 -> 27,170 NOT because the array does less work,
but because `row_par` (task E) packs 16 output pixels per array sweep and
`oc_single` (decision O) reuses one im2col load across all OC tiles, so the SAME
conv MACs finish in ~15x fewer array-active cycles. The honest headline metric is
`npu_busy`: 1,227,360 -> 225,360 (-82%). `array_util = arr_active/cyc_total` now
under-reads because `arr_active` is tiny; per active cycle the array does far more
work, so effective TOPS during compute is much higher than the 3% figure implies.
For the report, quote `npu_busy` reduction (or a per-conv MAC/cycle figure), not
`array_util`, for the compute-core story.

`TRAP - cyc_total ~= 0.12M` cycles is still just the one-time conv/FC weight
preload DMA. On-chip residency keeps inter-layer DDR traffic tiny (`rd_beats`
530, `wr_beats` 80: raw image in + FC1 staging for FC2 + final FC2 INT32 logits
out). `wr_bw_util` reads 66% only because there are so few write beats that
per-burst startup dominates -- not an efficiency regression. Do not confuse AXI
bandwidth utilization with cycle share.

Historical `NPU_PROFILE=1` measurement on 2026-06-13, summed over 10 images.
This is useful only as pre-compute-core / pre-current-dataflow history; do not
use it as the current bottleneck breakdown:

```text
infer_total: 4,405,440
  npu      : 2,075,310   (207,531/image)
  pad      : 1,050,840   (105,084/image)
  load     :   299,600   ( 29,960/image)
  reorder  :   389,330   ( 38,933/image)
  affine   :   577,150   ( 57,715/image)
argmax     :     5,118   (    512/image)

npu_per_layer (Conv1..Conv6):
  Conv1: 532,930
  Conv2: 364,030
  Conv3: 274,050
  Conv4: 307,850
  Conv5: 265,660
  Conv6: 330,790
```

Old profile meanings:

- `npu` is software time around `npu_conv_pass()` and includes NPU register
  setup, wait-for-IRQ, and output DMA drain/overlap. It is not the same as RTL
  `npu_busy`.
- `load` is input DDR -> Act SRAM DMA in `dma_ddr_to_act()`, not conv weight
  preload.
- `affine` in that old build was NPU FC1 plus CPU FC2. In the current default
  deploy build, FC1 and FC2 both run on the NPU.
- `pad` in that old build was mainly Conv1 input formatting into tile-major
  16-byte words. In the current build, raw image expansion is handled by
  `img_expand`, and spatial conv border padding is handled by the NPU FSM.

Success criteria:

- Real result: `=== Result: 10/10 correct ===` and `DEPLOY SUCCESS.`
- `ALL TESTS PASSED.` only means `start7.S` wrote `123456789` to TEST MMIO after
  `usercode7()` returned. It is useful, but it is not the MNIST success marker.
- `Errors: 0` from ModelSim only means the simulation finished without simulator
  errors.

## What Runs Where

NPU:

```text
Conv1..Conv6:
  HW padding when pad != 0
  im2col
  weight-prefetch reuse for ic_groups <= 4
  16x16 systolic array
  bias/scale/shift/ReLU

Conv2+Pool1, Conv4+Pool2, Conv6+Pool3:
  2x2 maxpool folded into the same NPU pass

FC1 + FC2 (whole MLP on NPU):
  GEMM/FC mode via CTRL[7] gemm_en, gemm_reduce CTRL[10] (16-row IC reduction)
  FC1 (1024->50, ReLU, INT8): input vector in Act PONG, weights in Wgt PONG
  FC2 (50->10): int32_out CTRL[13] emits raw un-clamped INT32 logits; CPU only
    reads the 10 logits for argmax (no CPU dot-product)

Data movement:
  Weights RESIDENT in DDR (packed by gen_weights_hex.py, loaded by $readmemh =
    flash/boot image); CPU DMAs them once into Wgt SRAM (resident all images)
  Raw images RESIDENT in DDR (gen_act_hex.py byte-packs them, 49 words/image =
    models a camera writing raw bytes to SDRAM); CPU DMAs the 49 raw words into
    an Act-SRAM scratch, then HW img_expand expands them into the tile-major
    Conv1 input IN SRAM -- no offline pad, no CPU pixel scatter
  Inter-layer conv activations stay ON-CHIP: HW sram_copy moves each conv's Out
    SRAM output to an Act-SRAM residency region (R0=word 0 / R1=word 1024 ping-
    pong). Conv1..Conv6 never round-trip through DDR; Conv6/Pool3 lands straight
    in Act PONG[0] where the FC1 GEMM reads it.
  Only remaining DDR traffic: raw image in + FC1 output staging for FC2 input
    + final FC2 INT32 logits out
  DDR <-> Act/Wgt/Out SRAM via NPU AXI DMA; SRAM<->SRAM via img_expand/sram_copy
  NPU done IRQ on CPU irq bit 3
  DMA read/write done IRQs on CPU irq bits 4/5
```

CPU/firmware:

```text
Layer scheduling and NPU/DMA register programming (one start per conv via oc_single)
Argmax over the NPU FC2 INT32 logits and result checking
(CPU no longer runs any conv/FC arithmetic -- the whole CNN+MLP is on the NPU)
```

Notes:

- Weights and input images BOTH enter the system from DDR. `gen_weights_hex.py`
  emits a packed weight image; `gen_act_hex.py` now emits RAW byte-packed images
  (49 words/image, the un-expanded camera bytes). The shared-memory model
  `$readmemh`-loads both into DDR at boot (models flash + sensor-resident data).
  Weights: CPU DMAs them DDR->Wgt SRAM. Images: CPU DMAs the 49 raw words
  DDR->Act SRAM scratch, then the HW `img_expand` engine expands them in SRAM
  (its output is bit-identical to the old offline tile-major pad). The firmware
  never packs weights or scatters image pixels on the CPU; the old CPU-pack/
  CPU-pad code and the build-time preload toggle were deleted.
- Inter-layer conv activations are kept on-chip by the HW `sram_copy` engine
  (Out SRAM -> Act SRAM), ping-ponging between two Act-SRAM regions R0/R1. No
  conv output round-trips through DDR. `npu_conv_pass(... act_in, act_dst,
  act_dst_pong)` carries the residency bases; `act_dst < 0` falls back to the old
  DDR-drain path. Conv6 copies Pool3 straight into Act PONG[0] so FC1 reads it
  with `in_resident=1` (no DDR load).
- FC1 is on NPU GEMM, and its weights are pre-packed in CONV-OUTPUT order
  (`pack_fc1_convorder`) so the GEMM reads the Pool3 output directly -- the old
  CPU Pool3->FC1 transpose (reorder) is gone.
- FC2 now runs on the NPU via `int32_out` (CTRL[13], decision Q): the GEMM emits
  scaled but un-clamped INT32 logits (4 Out words = 16xINT32), so final-logit/
  argmax fidelity is preserved without the old CPU dot-product. The CPU only reads
  the 10 INT32 logits from `FC2_OUT_DDR` and runs argmax.
- Conv3..Conv6 use `oc_single` (CTRL[12], decision O): one NPU start computes all
  OC tiles with a single shared im2col load. All layers use `row_par` (CTRL[9],
  task E) for 16-row spatial parallelism; narrow (out_w==8) layers also engage
  `row_block` (CTRL[11]). bias/scale/shift for all 64 OCs are resident in the
  param regfile (0x160..0x39C) so the FSM switches OC tiles without CPU reloads.
- The old CPU-side `pad_activation()` helper has been removed. Hardware padding
  is driven by `NPU_PAD` and `NPU_CTRL_HW_PAD`.
- DMA helpers wait on IRQ flags (`dma_rd_irq_flag`, `dma_wr_irq_flag`) rather
  than tight-polling `NPU_DMA_STATUS`. The IRQ handler reads `NPU_DMA_STATUS` to
  acknowledge the level-sensitive DMA-done latches.

Image preload and HW padding notes:

- The important change is not "padding itself"; it is that image formatting moved
  off the CPU/offline deploy path. `gen_act_hex.py` emits raw byte-packed 28x28
  MNIST images (49 128-bit words/image), then runtime `img_expand` converts each
  byte into one 16-lane activation word inside Act SRAM.
- Spatial convolution padding is handled by the NPU FSM, not materialized in DDR
  or SRAM. When `NPU_CTRL_HW_PAD` is set, the FSM injects border zero words while
  reading the unpadded tile-major activation. Current conv pads are Conv1=1,
  Conv2=1, Conv3=1, Conv4=2, Conv5=1, Conv6=1.
- 16-lane/channel zero-fill still exists because the datapath word is 128-bit:
  Conv1 uses only lane0, FC1 has 50 real outputs in a 64-slot tile layout, and
  FC2 writes 16 INT32 logits but only scores[0..9] are real classes.

## Current Optimization State

Implemented and verified in the current tree:

- 128-bit AXI/DDR data path: NPU DMA, arbiter, and shared memory model are
  128-bit. `axi_full_slave_v1_0_S00_AXI` now uses a width-correct `ADDR_LSB`
  and registered read pipeline.
- CPU 32->128 upsizer: `axi_upsizer_32_128` adapts the PicoRV32/bridge 32-bit
  single-beat CPU path to the 128-bit fabric with lane-select `WSTRB`. The latest
  version has a fast path for AW+W arriving in the same cycle, saving one cycle
  on common CPU shared-memory writes.
- Conv weight residency: `preload_conv_weights()` loads all conv weights once
  into Wgt SRAM PING bank. Conv passes select per-layer/per-tile bases.
- Weight-prefetch reuse: `wgt_reader` has an `ICG_BUF=4` buffer and dual-mode
  prefetch. FSM prefetches once per OC tile for conv layers with `ic_groups<=4`.
  GEMM and `ic_groups>4` use the fallback path.
- NPU pooling: Conv2/4/6 fuse 2x2 maxpool into the post-process path.
- NPU and DMA interrupts: NPU done uses bit 3; DMA read/write done use bits 4/5.
- OC-pass output overlap: firmware can drain the previous Out SRAM bank while
  the current OC pass computes (`NPU_OC_OVERLAP=1` by default). Resident layers
  bypass this (they copy on-chip serially instead of draining to DDR).
- `img_expand` engine (`rtl/img_expand.v`): reads raw byte-packed pixels from Act
  SRAM Port B and writes one zero-extended 16-channel tile-major word per pixel
  back to Act SRAM. Triggered by `NPU_DMA_EXPAND_TRIG` (0x158), done in
  `NPU_DMA_STATUS[3]`. Builds the Conv1 input in SRAM; output bit-identical to
  the old offline pad.
- `sram_copy` engine (`rtl/sram_copy.v`): copies N 128-bit words Out SRAM Port B
  -> Act SRAM Port B (1 word/cycle, no DDR). Triggered by `NPU_DMA_COPY_TRIG`
  (0x154), done in `NPU_DMA_STATUS[2]`. Drives inter-layer SRAM residency
  (R0/R1 ping-pong, and Conv6 -> Act PONG[0] for FC1). Both engines time-share
  SRAM Port B with `axi_dma` and are mux'd in `npu_top` by `expand_busy` >
  `copy_busy` > DMA priority; firmware never overlaps them with DMA.
- Hardware padding: FSM reads tile-major unpadded input and injects border zeros.
- GEMM/FC mode: `CTRL[7]` bypasses im2col, uses KH=KW=1, and supports
  `gemm_reduce` (CTRL[10]) for 16-row IC reduction. FC1 and FC2 deploy paths use
  NPU GEMM. FC2 uses `int32_out` (CTRL[13]) so final logits are scaled INT32, not
  clamped INT8.
- INT32 output completion: the externally visible NPU done/IRQ is delayed until
  the 4-word `int32_out` serializer has committed all 16 INT32 lanes to Out SRAM.
  This makes `done` mean "output is fully readable", not just "compute finished".
- Runtime Pool3->FC1 transpose hardware was removed. FC1 weights are pre-packed in
  Conv-output order by `gen_weights_hex.py`, so FC1 reads Pool3 directly.
- RTL performance counters: exposed at `0x3000_03A4..0x3000_03BC`, cleared by
  `NPU_PERF_CLR` at `0x3000_03A0`, printed by `print_perf()`.
- Debug output is gated behind `DEBUG_VERBOSE`; `NPU_PROFILE` defaults to 0.

Important caveats:

- `axi_upsizer_32_128` is a SoC-specific adapter, not a general AXI width
  converter. It assumes the CPU-side path is single-beat and single-outstanding,
  as produced by `axi_lite_to_axi_full`. It does not implement generic burst
  coalescing, multiple outstanding transactions, or a full W-before-AW write
  skid path.
- Combinational SRAM Port-B read (`COMB_B=1`): `axi_dma`, `img_expand`,
  `sram_copy`, and the skip-read path all assume Port B returns `dob` the SAME
  cycle the address is driven. This is a sim-model convention shared design-wide,
  NOT specific to the new engines. The large Act/Out SRAMs (256KB/128KB) cannot be
  combinational-read on FPGA (they map to registered-output BRAM, data valid +1
  cycle; LUTRAM is only for tiny RAMs). A real FPGA port must flip the wrappers to
  `COMB_B=0` AND add one read-latency pipeline stage to EVERY Port-B consumer
  together (DMA + both engines + skip-read). Treat it as a whole-path change.
- A parity/validation build for `NPU_GEMM_PARITY=1` if changing GEMM/FC logic.
  Note: parity reads Pool3 from DDR, which SRAM residency no longer populates, so
  a meaningful parity run needs a non-residency (DDR-drain) build.
- Directed hardware-padding and weight-reuse tests if touching either FSM path.
- `deepnet_run.c` Test11 is stale for the current weight layout: it loads only
  9 Wgt SRAM words, while the current 16-OC conv weight layout needs 16*9 words
  for that test. Its failure should not be used to judge the 128-bit AXI update
  until the test is refreshed.

## Main Gaps / Backlog

Priority order from the current architecture and RTL counters:

1. **Update report/docs to match the code.** `CLAUDE.md` is now refreshed, but
   `docs/superpowers` checkboxes still read like unexecuted plans. Reconcile or
   annotate them before using them as project status.
2. **One-time setup/weight preload.** `TRAP - cyc_total ~= 0.12M` now -- just the
   one-shot conv/FC weight preload DMA (8768 words DDR -> Wgt SRAM). Much smaller
   share than before, but if contest scoring counts it, a pre-initialized Wgt-SRAM
   image or faster boot DMA would shave it.
3. **CPU data formatting / scheduling.** Conv1 input formatting is HW
   (`img_expand`), the Pool3->FC1 reorder is folded into the FC1 weight packing,
   and FC2 is now an NPU INT32 pass -- so the CPU runs NO conv/FC arithmetic. The
   only remaining CPU work is per-layer NPU/DMA register programming and the final
   argmax. The dominant remaining cost is the MMIO register writes per layer
   (bias/scale/shift for up to 64 OCs + dims); batching or descriptor-driven
   programming is the next lever.
4. **Array-utilization METRIC is now misleading, not the bottleneck.** After the
   compute-core port `npu_busy` is only 225,360 of 818,640 cyc_total (~27%), and
   `arr_active` (27,170) collapsed because `row_par`/`oc_single` do ~15x more work
   per array-active cycle -- so `array_util=arr/total=3%` UNDER-reads. Report
   `npu_busy` reduction (1.23M->225K, -82%) or a MAC/active-cycle figure instead.
   The remaining cyc_total is now CPU register programming + DMA/copy overhead, so
   the next real lever is cutting per-layer CPU MMIO writes, not the array.
5. **Low-power architecture.** Add real clock-enable/operand-isolation/SRAM-bank
   gating and counters or trace evidence. Current enables are useful, but the
   design still lacks a strong low-power proof.
6. **Verification package.** Keep expanding directed tests. Current tests cover
   pooling, post-process pooling, AXI read backpressure, and CPU 32->128 upsizer
   lane/strobe behavior; still add DMA round-trip, GEMM parity, hardware-padding
   bit-identical output, weight-reuse fallback, and coverage. The contest text
   mentions a 95% path coverage target.
7. **FC2/logit handling. (DONE)** FC2 now runs on the NPU via `int32_out`
   (CTRL[13], decision Q): scaled un-clamped INT32 logits, so no INT8 saturation
   of the final logits. The CPU only reads the 10 logits and runs argmax. The
   whole CNN+MLP is now on the NPU.
8. **On-chip residency. (DONE)** Inter-layer conv activations now stay in Act
    SRAM via the `sram_copy` engine (R0/R1 ping-pong); Conv1 input is built in SRAM
    by `img_expand`. DDR read/write beats dropped 26,040/18,240 -> 530/80 in the
    latest run. Remaining DDR traffic is raw image input, FC1 output staging
    (write then read for FC2), and the final FC2 INT32 logits. The next residency
    step is keeping FC1->FC2 fully on-chip or replacing per-layer MMIO scheduling
    with descriptors.
9. **Simulation ergonomics.** `modelsim/run_modelsim.bat batch` via `run.do`
   logs all signals and can be extremely slow. Use `bash run_all.sh sim` for
   normal regression, or change `run.do` to avoid `log -r /*` in batch mode.

## Key Files

```text
Build/sim:
  run_all.sh
  modelsim/{compile.do,run.do,wave.do,run_modelsim.bat}

Firmware:
  firmware/{start7.S,irq.c,print.c,deepnet_deploy.c,firmware.h,
            deepnet_weights.h,mnist_test_images.h}

NPU RTL:
  rtl/{npu_top,npu_axi_wrapper,param_regfile,axi_dma,top_controller_fsm,
       im2col_line_buffer,wgt_reader,systolic_16x16,post_process_top,
       max_pooling_2x2,vector_alu,sram_models}.v
  rtl/{img_expand,sram_copy}.v   # on-chip SRAM data-movement engines

SoC/TB:
  rtl/{axi_sys,axi_sys_tb,picorv32,axi_lite_ram,axi_lite_to_axi_full,
       axi_arbiter_2to1,axi_full_slave_v1_0_S00_AXI,axi_upsizer_32_128}.v

Unit tests:
  tests/{tb_max_pooling_2x2,tb_post_process_pool,
         tb_axi_read_backpressure,tb_axi_upsizer}.v

Design docs:
  docs/superpowers/plans/*.md
  docs/superpowers/specs/*.md

Logs:
  sim/modelsim/{compile.log,direct_batch.log,sim.log,*_batch.wlf}
```

## Address Map

```text
0x0000_0000..0x00FF_FFFF  private RAM (firmware)
0x1000_0000               UART MMIO
0x2000_0000               TEST MMIO
0x3000_0000..0x3000_0FFF  NPU registers (param_regfile)
0x4000_0000+              shared memory / DDR model
```

Important NPU register/control bits:

```text
CTRL[0] start        CTRL[9]  row_par      (16-row spatial parallelism, task E)
CTRL[1] ping_pong    CTRL[10] gemm_reduce  (GEMM 16-row IC-reduction, decision M)
CTRL[2] pool_en      CTRL[11] row_block    (row-block packing, narrow layers, #4)
CTRL[3] eltwise_en   CTRL[12] oc_single    (all OC tiles in one start, decision O)
CTRL[4] clear_done   CTRL[13] int32_out    (raw INT32 output, decision Q, FC2)
CTRL[5] relu_en      CTRL[14] pw_en        (1x1 pointwise conv, im2col bypass)
CTRL[6] out_ping     CTRL[15] dw_en        (depthwise conv, channel-parallel MAC)
CTRL[7] gemm_en      CTRL[16] pool_avg     (2x2 average pooling, vs max)
CTRL[8] hw_pad       CTRL[17] gpool_en     (global average pooling)

Operator-generality block (2026-06-16, opt-in, default OFF = bit-identical; each
has a directed/integration TB; stride>1 conv deferred):
  pw_en[14]   1x1 pointwise: reuses conv sweep + weight reuse, reads each pixel
              directly (GEMM-style array feed), bypassing im2col. IC<=16.
  dw_en[15]   depthwise: rtl/depthwise_engine.v, 16 channel-parallel INT32 MACs
              bypassing the array, fed by the im2col window; KO=9 per-tap weight
              words prefetched from Wgt Port A (borrowed during LOAD_ROW).
  pool_avg[16]/gpool_en[17] avg + global-avg pooling (rtl/global_avg.v + NPU_GAVG_CFG).
  NPU_CLIP_MAX configurable post-process upper clamp (ReLU6). NPU_SKIP_BASE
  configurable residual skip base (light INT8 residual). tests/tb_npu_integ.v is a
  reusable npu_top integration harness (drive config + preload SRAM + golden check).

NPU_CLIP_MAX       0x118  post-process upper clamp [7:0] (default 127; ReLU6 = q(6.0))
NPU_SKIP_BASE      0x11C  residual skip-source Out-SRAM base (0 = same-addr legacy)
NPU_PAD            0x150  {pad_h[15:8], pad_w[7:0]}
NPU_DMA_COPY_TRIG  0x154  write: start sram_copy (Out->Act residency)
NPU_DMA_EXPAND_TRIG 0x158 write: start img_expand (raw->tile-major Conv1 input)
NPU_GAVG_CFG       0x15C  global avgpool reciprocal: [25:0]mul [31:26]shift (mean=(sum*mul)>>shift)
NPU_BIAS/SCALE/SHIFT  ch0-15 @ 0x40/0x80/0xC0; ch16-63 @ 0x160/0x220/0x2E0
                      (64-OC resident regfile for oc_single, decision O)
NPU_PERF_CLR       0x3A0  (relocated above the 0x160..0x39C resident-param block)
NPU_PERF_*         0x3A4..0x3BC
NPU_DMA_STATUS     0x140  [0]rd_done [1]wr_done [2]copy_done [3]expand_done
```

Data path:

```text
print:    print_str/hex -> store 0x1000_0000 -> axi_sys UART decode -> $write
NPU reg:  store 0x3000_xxxx -> axi_sys MMIO -> npu_axi_wrapper -> param_regfile
CPU data: store/load 0x4000_xxxx -> axi_lite_to_axi_full (32-bit single beat)
              -> axi_upsizer_32_128 -> 128-bit AXI arbiter
              -> 128-bit shared DDR model
NPU data: DDR <-> 128-bit AXI arbiter <-> npu_axi_wrapper
              <-> 128-bit axi_dma <-> Act/Wgt/Out SRAM
              <-> im2col / wgt_reader / systolic / post-process / vector ALU
```

AXI verification notes:

```text
tests/tb_axi_read_backpressure.v:
  Runs the shared-memory read channel with RREADY gaps and the "last beat held
  while a new AR arrives" conflict. Use -gDW=128 for the current shared-memory
  width.

tests/tb_axi_upsizer.v:
  Checks CPU 32-bit writes into all four lanes of one 128-bit word, partial
  WSTRB behavior, and readback lane muxing through axi_upsizer_32_128.
```

## Working Rules

- Keep changes small and focused; preserve the 10/10 baseline.
- Verify with smoke first, then batch (`bash run_all.sh sim`).
- For risky RTL timing changes in FSM, padding, GEMM, weight reuse, or pooling,
  add a directed testbench before spending time on full MNIST simulation.
- Use `bash run_all.sh sim` for normal full regression.
- Do not revert the ModelSim flow to VCS; do not restore old VCS/Verdi artifacts.

## Conventions

- Comments in RTL and firmware are bilingual (Chinese/English); keep matching surrounding style.
- Register-map changes require touching both sides: `rtl/param_regfile.v` and `firmware/firmware.h`.
- New RTL files must be added to `axi_sys.f`.
- Firmware C code must be warning-clean under strict CFLAGS.
- Multiple test programs exist (`deepnet_run.c`, `deepnet_deploy.c`), each defining `usercode7()`. Only one may be linked at a time; switch by editing `FW_C_SRCS` in `run_all.sh`.
- `makehex.py` pads firmware binary to 524288 words for `$readmemh`.
- DeepConvNet model assets: `deepnet.h` (dimensions), `deepnet_weights.h` (weights/biases), `mnist_test_images.h` (test images, generated by `extract_images.py`).
