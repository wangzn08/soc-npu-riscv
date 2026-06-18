# YOLOv8n Shared SoC RTL Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring up YOLOv8n INT8 in RTL simulation on the existing CPU+NPU SoC without creating a second model-specific hardware path.

**Architecture:** The PicoRV32 CPU remains the scheduler: it programs NPU MMIO registers, launches DMA transfers, and checks/decodes software-visible results. The NPU remains one shared hardware instance: systolic array, SRAMs, DMA, post-process, and Port-B data-movement engines are extended with optional modes. YOLO is executed by spatial strip/tile blocks because full 640x640 feature maps do not fit in the existing Act/Out SRAMs.

**Tech Stack:** Verilog/SystemVerilog RTL, PicoRV32 C firmware, ModelSim/Questa through `bash run_all.sh`, existing YOLOv8n INT8 C/metadata golden files.

## Global Constraints

- Use one shared SoC/NPU hardware path for MNIST and YOLO; do not add a parallel YOLO-only accelerator.
- Default reset and default MNIST firmware behavior must remain unchanged.
- YOLO execution is block/strip based; do not try to place whole YOLO feature maps on chip.
- Every new hardware capability needs a directed RTL test and a CPU C smoke test.
- After each RTL milestone, run `bash run_all.sh sim` and preserve MNIST `10/10`, `DEPLOY SUCCESS.`, and `ALL TESTS PASSED.`
- Use ModelSim/Questa through `run_all.sh`; do not switch to VCS.

---

## Files And Responsibilities

- `tools/yolo_deploy_sizing.py`: graph-aware YOLOv8n layer shape, SRAM, and traffic estimator.
- `docs/notes/yolov8n-deploy-sizing.md`: generated sizing report and strip budget evidence.
- `rtl/post_process_top.v`: shared activation/quant/pool output path, including optional SiLU.
- `rtl/param_regfile.v`: shared NPU MMIO control/status register map.
- `rtl/npu_top.v`: shared NPU integration, SRAM muxing, and data-movement engine wiring.
- `rtl/upsample2x.v`: shared Act-SRAM nearest-neighbor 2x upsample engine.
- `firmware/firmware.h`: CPU-visible register macros for all shared NPU controls.
- `firmware/upsample2x_smoke.c`: pattern for CPU/NPU cooperative smoke tests.
- Future `firmware/yolo_*.c`: CPU scheduler and C golden comparison entry points for YOLO subgraphs.

## Milestone 0: Graph Sizing And Strip Policy

**Status:** Complete.

**Deliverable:** A reproducible report proving YOLO must run in strips/tiles and identifying which layers are NPU conv work vs CPU decode-tail work.

- [x] Parse `yolov8n_int8/yolov8n_layers.h`.
- [x] Follow graph topology from `yolov8n_int8/yolov8n_infer.c`.
- [x] Mark conv0..conv62 as NPU conv candidates and conv63/DFL as CPU decode-tail.
- [x] Generate `docs/notes/yolov8n-deploy-sizing.md`.
- [x] Run `python tests/test_yolo_deploy_sizing.py`.
- [x] Run `bash run_all.sh sim` to preserve MNIST.

## Milestone 1: Shared SiLU Activation

**Status:** Complete.

**Deliverable:** Optional SiLU activation in the existing post-process path. MNIST keeps ReLU/clip defaults.

- [x] Add `tests/tb_silu_lut.v`.
- [x] Add `rtl/silu_lut_q4_4.hex`.
- [x] Add `i_silu_en` to `rtl/post_process_top.v`.
- [x] Add `CTRL[18]` wiring through `rtl/param_regfile.v`, `rtl/npu_top.v`, and `firmware/firmware.h`.
- [x] Run `tb_silu_lut`, existing post-process tests, and `bash run_all.sh sim`.

## Milestone 2: Shared Act-SRAM Upsample2x

**Status:** Complete.

**Deliverable:** CPU-triggered 2x nearest-neighbor upsample in the shared Act SRAM path.

- [x] Add standalone `tests/tb_upsample2x.v`.
- [x] Add npu_top integration `tests/tb_npu_upsample2x.v`.
- [x] Add `rtl/upsample2x.v`.
- [x] Add `NPU_UPSAMPLE_CFG0`, `NPU_UPSAMPLE_CFG1`, and `NPU_DMA_UPSAMPLE_TRIG`.
- [x] Add `NPU_DMA_STATUS[4]` as `upsample_done`.
- [x] Add CPU C smoke `firmware/upsample2x_smoke.c`.
- [x] Verify `tb_upsample2x`, `tb_npu_upsample2x`, `bash run_all.sh sim upsample2x_smoke.c`, and `bash run_all.sh sim`.

## Milestone 3: YOLO Tensor Layout And Concat

**Status:** In progress. The CPU/DMA block-concat primitive is complete; full per-layer block-plan generation is still pending.

**Deliverable:** A block-level tensor layout contract and concat path for FPN/PAN without whole-feature-map SRAM residency.

- [x] Define a single DDR/Act SRAM layout for YOLO feature blocks: tile-major words with `addr = base + ic_group * H * W + y * W + x`.
- [ ] Extend the sizing tool to emit per-layer block plans: source DDR base, destination DDR base, input shape, output shape, and strip rows.
- [x] Add a CPU/NPU smoke for block concat: two source tensors with different channel-group ranges produce one concatenated tile-major tensor.
- [x] Add firmware helpers in `firmware/yolo_ops.c` that DMA-load two DDR blocks into one Act-SRAM concat layout, DMA-drain the result, and compare in software from `firmware/yolo_concat_smoke.c`.
- [x] Run MNIST regression after the concat path is added.

## Milestone 4: YOLO Conv Block Scheduler

**Status:** In progress. Synthetic and tiny real-weight CPU-scheduled 1x1 pointwise conv blocks are complete. Tiny real YOLO 3x3 stride-1, stride-2, and multi-tile OC32/IC32 padded blocks also run with C-reference qparam/SiLU/requant semantics after adding configurable hardware pad fill. A concat-channel pointwise block (real conv5, IC48/OC32) now runs through the same shared NPU path with `oc_single`. Full per-layer strip/block-plan integration is still pending.

**Deliverable:** CPU firmware can run one YOLO conv layer by strips using the existing systolic array and post-process path.

- [x] Choose one low-risk YOLO-style layer for the first conv block smoke: a synthetic 1x1 pointwise conv whose 2x3 block fits in SRAM.
- [x] Generate packed weights and expected block output from the existing INT8 C/ONNX-derived assets for a tiny real YOLOv8n conv2 pointwise block.
- [x] Add firmware helpers for YOLO pointwise conv register programming: dimensions, Act/Wgt/Out base addresses, scale/bias, and control flags.
- [x] Run one synthetic block through CPU -> DMA -> NPU pointwise conv -> DMA -> CPU compare.
- [x] Run one real-weight YOLOv8n conv2 block through CPU -> DMA -> shared NPU pointwise conv -> DMA -> CPU compare.
- [x] Fix shared `pw_en` for IC>16: tile-major Act SRAM addressing and a fresh Act SRAM read per IC group.
- [x] Add configurable `NPU_PAD_VALUE` so hardware padding can inject the YOLO input zero-point instead of hard-coded INT8 zero.
- [x] Run one real YOLOv8n `conv3` 3x3 stride-1 padded block through CPU -> DMA -> shared NPU im2col conv -> DMA -> C-reference tolerance compare.
- [x] Run one real YOLOv8n `conv1` 3x3 stride-2 padded OC tile through CPU -> DMA -> shared NPU im2col conv -> DMA -> C-reference tolerance compare.
- [x] Run one real YOLOv8n `conv8` OC32/IC32 3x3 padded block through CPU -> DMA -> shared NPU `oc_single` conv -> DMA -> C-reference tolerance compare.
- [x] Run one real YOLOv8n `conv5` concat-channel IC48/OC32 1x1 block through CPU -> DMA -> shared NPU `oc_single` pointwise conv -> DMA -> C-reference tolerance compare.
- [x] Run MNIST regression.

## Milestone 5: YOLO Subgraph RTL Smoke

**Status:** In progress. Tiny synthetic and real-weight YOLO-neck-style subgraphs now run through CPU scheduling and shared NPU hardware. Real conv2 zero-point/per-channel-qparam/SiLU and SiLU-output requant smokes pass, the tiny real conv2 block matches the YOLO C/float reference within +/-1 INT8 LSB, and a tiny real multi-op `conv -> upsample -> concat -> conv` subgraph matches the C reference within a 3-LSB final-output tolerance. Full per-layer strip/block scheduling is still pending.

**Deliverable:** A real YOLO subgraph runs through CPU scheduling and shared NPU hardware.

- [x] Start with a small synthetic subgraph shaped like `conv -> upsample -> concat -> conv`.
- [x] Add reusable CPU helper `yolo_run_upsample2x`.
- [x] Add `firmware/yolo_subgraph_smoke.c` to run pointwise conv, upsample2x, concat, and a second pointwise conv through MMIO/DMA/NPU.
- [x] Add `firmware/yolo_subgraph_real_smoke.c` and `tools/gen_yolo_subgraph_real_smoke.py` for a real `conv2_w.bin` tiny subgraph.
- [x] Use block-level golden data from the C reference for a tiny real conv2 pointwise block, including input zero-point correction, per-channel weight scales, float bias, SiLU, and output requantization.
- [x] Extend that C-reference golden path from one block to a tiny multi-op real subgraph.
- [x] Keep each intermediate tensor either in Act SRAM for the block or in DDR with explicit layout metadata in the smoke firmware.
- [x] Compare final synthetic block output against exact INT8 CPU golden values.
- [x] Compare final real-weight block output against exact RTL-integer golden values.
- [x] Add real conv2 pointwise smoke with input zero-point correction, per-channel qparams, and `NPU_CTRL_SILU_EN`.
- [x] Add final output-scale/zero-point requantization after SiLU (`CTRL[19]`, `NPU_SILU_REQUANT_CFG`) and verify it with a real conv2 CPU smoke.
- [x] Use the qparam/requant controls in a real C-reference pointwise-block golden.
- [x] Use the qparam/requant controls in a tiny real C-reference multi-op subgraph golden.
- [x] Run MNIST regression.

## Milestone 6: Detect Head And Decode Tail

**Status:** Planned.

**Deliverable:** NPU produces YOLO detect-head logits; CPU handles DFL/decode/NMS first.

- [ ] Run detect-head conv outputs through shared NPU block scheduling.
- [ ] DMA detect logits to DDR.
- [ ] Reuse or port the existing C golden DFL/decode logic in firmware or host-side checker.
- [ ] Compare boxes/classes/scores against the existing INT8 golden.
- [ ] Defer DFL hardening until the software-tail path is correct.
- [ ] Run MNIST regression.

## Milestone 7: Full YOLO RTL Simulation

**Status:** Planned.

**Deliverable:** Full YOLOv8n INT8 inference runs in RTL simulation under CPU control.

- [ ] Build a firmware entry point such as `firmware/yolo_smoke.c`.
- [ ] Use CPU scheduling for every layer and block.
- [ ] Use shared NPU hardware for conv, SiLU, upsample, pooling/copy, and any completed concat helper.
- [ ] Use CPU for DFL/decode/NMS unless Milestone 6 proves a better split.
- [ ] Compare final detections against `yolov8n_int8/dets_int8.json` or a generated deterministic golden.
- [ ] Run final MNIST regression to prove shared-hardware compatibility.

## Verification Ladder

Each milestone must pass this ladder before the next begins:

- Directed RTL test for the new primitive.
- `npu_top` integration test when the primitive touches shared SRAM or register wiring.
- CPU C smoke through PicoRV32, MMIO, DMA, and DDR-visible comparison.
- Default MNIST deploy: `bash run_all.sh sim`.
- YOLO golden comparison at the smallest available block/subgraph level.
