# YOLOv8n RTL Bring-Up Milestone 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the first reproducible gate for running YOLOv8n INT8 in RTL simulation: graph-aware layer sizing, strip/SRAM budgeting, DDR traffic estimates, and an MNIST regression gate.

**Architecture:** This milestone does not change RTL. It parses `yolov8n_int8/yolov8n_layers.h`, follows the topology encoded in the C golden model `yolov8n_int8/yolov8n_infer.c`, marks conv0..conv62 as NPU conv work, marks conv63/DFL as CPU decode-tail work, and generates `docs/notes/yolov8n-deploy-sizing.md`. Later milestones must add directed RTL tests before any datapath change.

**Tech Stack:** Python 3 standard library, existing YOLO INT8 metadata, Bash/ModelSim via `bash run_all.sh sim`.

## Global Constraints

- Work from `E:\code\6-10\soc`.
- Preserve MNIST: `=== Result: 10/10 correct ===`, `DEPLOY SUCCESS.`, and `ALL TESTS PASSED.`
- Do not switch ModelSim/Questa flow to VCS.
- Do not modify RTL in Milestone 0.
- Treat `yolov8n_int8/yolov8n_infer.c` as the graph/topology golden reference.
- Keep YOLO input at 640x640 and use spatial strip tiling, not MB-scale SRAM.

---

## Files

- Create `tests/test_yolo_deploy_sizing.py`: executable Python assertions for parser, topology shapes, strip budget, and summary.
- Create `tools/yolo_deploy_sizing.py`: parser, topology estimator, strip budget helper, report renderer.
- Create `docs/notes/yolov8n-deploy-sizing.md`: generated sizing report.

## Task 1: RED Parser And Topology Test

**Files:**
- Create: `tests/test_yolo_deploy_sizing.py`

**Interfaces Expected By Test:**
- `parse_layers(path: Path) -> list[Layer]`
- `build_yolov8n_graph(layers: list[Layer], input_h: int, input_w: int) -> YoloGraph`
- `choose_strip_rows(shape: LayerShape, act_sram_bytes: int = 256 * 1024) -> int`
- `summarize(shapes: list[LayerShape]) -> dict[str, int]`

- [x] **Step 1: Write failing test**

The test checks:
- 64 layer rows parse from `yolov8n_layers.h`.
- conv0 output is 320x320 and conv1 output is 160x160.
- graph-aware concat inputs are correct: conv27 sees 40x40x384, conv31 sees 80x80x192, conv58 sees 20x20x256.
- conv63 is marked `cpu_dfl`.
- conv0 has exactly 8 output rows under a 256KB Act/Out strip budget.
- total NPU MACs are between 4G and 5G.

- [x] **Step 2: Verify RED**

Run:

```bash
python tests/test_yolo_deploy_sizing.py
```

Observed failure:

```text
ModuleNotFoundError: No module named 'tools'
```

## Task 2: GREEN Graph-Aware Sizing Tool

**Files:**
- Create: `tools/yolo_deploy_sizing.py`

**Produced Interfaces:**
- `Layer`, `TensorShape`, `LayerShape`, `YoloGraph`
- `parse_layers`, `build_yolov8n_graph`, `strip_act_bytes`, `choose_strip_rows`, `summarize`, `render_report`, `main`

- [x] **Step 1: Implement parser and graph builder**

Implementation follows the same sequence as `yolo_infer()`:
- Stem: conv0, conv1.
- Backbone C2f blocks: conv2..5, conv7..12, conv14..19, conv21..24.
- SPPF: conv25, conv26.
- FPN/PAN concat paths: conv27..35, conv40, conv43..46, conv51, conv54..56.
- Detection branches: conv36..42, conv47..53, conv57..62.
- CPU DFL tail: conv63.

- [x] **Step 2: Verify GREEN**

Run:

```bash
python tests/test_yolo_deploy_sizing.py
```

Expected:

```text
PASS: yolo_deploy_sizing
```

## Task 3: Generate Sizing Report

**Files:**
- Create: `docs/notes/yolov8n-deploy-sizing.md`

- [x] **Step 1: Render report**

Run:

```bash
python tools/yolo_deploy_sizing.py
```

Expected:

```text
wrote E:\code\6-10\soc\docs\notes\yolov8n-deploy-sizing.md
```

- [x] **Step 2: Check key numbers**

Expected report summary:
- NPU conv layers: 63.
- CPU decode/DFL tail layers: 1.
- NPU MACs: about 4.37G.
- conv0 strip rows at 256KB: 8.

## Task 4: MNIST Regression Gate

**Files:**
- No source edits.

- [ ] **Step 1: Run full regression**

Run:

```bash
bash run_all.sh sim
```

Expected output includes:

```text
=== Result: 10/10 correct ===
DEPLOY SUCCESS.
ALL TESTS PASSED.
```

- [ ] **Step 2: Inspect log if console is truncated**

Run:

```bash
tail -80 sim/modelsim/direct_batch.log
```

Expected: same success markers.

## Milestone 0 Exit Criteria

- `python tests/test_yolo_deploy_sizing.py` passes.
- `docs/notes/yolov8n-deploy-sizing.md` exists and is graph-aware.
- No RTL files changed by this milestone.
- `bash run_all.sh sim` preserves MNIST 10/10.

## Next Milestone

Milestone 1 should add the first shared-NPU optional activation path behind default-off controls: SiLU LUT post-process. MNIST and YOLO use the same hardware instance; the model-specific behavior is selected only by CTRL bits and layer parameters. It must begin with a directed RTL test comparing representative INT8 inputs against the C/ONNX-derived LUT before production RTL changes.
