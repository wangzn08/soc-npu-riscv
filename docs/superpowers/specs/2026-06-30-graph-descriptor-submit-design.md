# Graph Descriptor Submit Design

## Goal

Move the deploy baseline from stage-level descriptor submissions to graph-level
jobs:

- MNIST: one descriptor submission covers one full image inference.
- YOLOv8n @320: one descriptor submission covers the backbone, neck, and head
  tensor-producing graph. CPU decode, thresholding, DFL geometry, and NMS remain
  after the descriptor job completes.

The CPU should no longer write `NPU_DESC_BASE`, `NPU_DESC_COUNT`, and
`NPU_DESC_CTRL.START` once per layer, strip, maxpool, or residual-add program in
the deploy paths. It should prepare the descriptor stream, start it once, wait for
one final descriptor done/IRQ, then continue with result handling.

## Baseline

Base commit: `checkpoint-yolo320-desc-precompile` (`8ca52ae`).

Relevant existing behavior:

- YOLO @320 full-net passes with descriptor queue and precompiled descriptor
  image: `YOLO FULL NET PASS 4/4`, `TRAP 22,245,379 cyc`.
- MNIST deploy passes: `10/10 DEPLOY SUCCESS`.
- YOLO descriptor runtime already supports record/replay and hardware execution
  of conv, DMA movement, 5x5 maxpool, residual add, activation config, LUT load,
  and qparam-table reads.
- YOLO record/replay currently stores many cataloged descriptor programs and
  submits them by call order through `desc_submit_cataloged()`.
- MNIST already has an optional `NPU_HW_DESC` path that builds one descriptor
  list for one image and submits it once, but it is not the default deploy path.

## Non-Goals

- Do not port YOLO decode/NMS to hardware in this change.
- Do not change YOLO numerical output or the C oracle criteria.
- Do not revive the 640x640 branch work.
- Do not remove the stage-level descriptor helpers; keep them for directed tests
  and debug.
- Do not redesign the descriptor RTL ABI unless a concrete limit blocks the
  graph stream.

## Architecture

Add a graph-builder mode to the YOLO descriptor runtime.

The existing stage helpers continue to construct descriptor records. In legacy
mode, they submit each constructed program immediately, preserving existing smoke
tests. In graph mode, they append records to one active descriptor stream and
return after local validation only. The top-level YOLO firmware starts graph mode,
calls the same stage helpers in the same order, appends one final `STOP_IRQ`, and
submits the whole main graph once.

For replay builds, the precompiled descriptor image should contain one catalog
entry for the main graph, not hundreds of per-stage entries. The firmware should
read that single catalog entry and submit it once.

MNIST should make the full-inference descriptor path first-class. The existing
single-submit descriptor builder is the preferred implementation path; the change
should ensure the normal descriptor deploy path uses one submit per image and
prints or otherwise exposes evidence of that submit count.

## Interfaces

YOLO runtime additions in `firmware/yolo_desc.h`:

```c
void yolo_desc_graph_begin(void);
int yolo_desc_graph_end_and_submit(void);
int yolo_desc_graph_active(void);
uint32_t yolo_desc_submit_count(void);
```

Expected behavior:

- `yolo_desc_graph_begin()` resets the current graph stream and enters append
  mode.
- Existing `yolo_run_*_desc()` helpers append records instead of submitting when
  graph mode is active.
- `yolo_desc_graph_end_and_submit()` appends `STOP_IRQ`, exits append mode, and
  performs exactly one hardware descriptor submission.
- `yolo_desc_submit_count()` returns the number of hardware descriptor jobs
  submitted since the last `yolo_desc_reset()`.

The stage-level helper API remains source-compatible.

## Descriptor Image Layout

Keep the existing descriptor regions:

- `DESC_IMAGE_BASE`: descriptor records.
- `DESC_QPARAM_BASE`: qparam records.
- `DESC_CATALOG_BASE`: catalog entries.

Record mode should write one catalog entry for the YOLO main graph. Replay mode
should use that one catalog entry. If debug builds still need stage-level replay,
guard it behind a separate compatibility path rather than mixing it with deploy
graph replay.

The graph record count may be larger than the largest current per-stage program.
Before implementation, audit `DESC_IMAGE_BASE..DESC_QPARAM_BASE` capacity against
the recorded `total_records * NPU_HW_DESC_WORDS * 4`.

## CPU Responsibilities After This Change

MNIST:

- Pick image index.
- Submit one full-inference descriptor job.
- Wait for one done/IRQ.
- Read final logits and run argmax/checking.

YOLO:

- Preload the input image, weights, LUTs, descriptor image, and qparam image as
  before.
- Submit one backbone/neck/head descriptor job.
- Wait for one done/IRQ.
- Run existing CPU decode/NMS and compare final boxes to the 320 golden.

## Error Handling

Graph submit should surface descriptor-engine failures with enough context:

- descriptor status
- descriptor PC
- descriptor error code
- graph record count
- submit count

If graph build overflows the descriptor region or qparam region, firmware should
print a clear failure and return before starting the engine.

## Verification

Directed tests:

- Add or update a descriptor-runtime smoke that builds multiple logical programs
  in graph mode and asserts only one hardware submit occurred.
- Keep an existing stage-level descriptor smoke passing to prove debug mode still
  works.

MNIST:

- Build the descriptor deploy path.
- Run simulation and require `10/10`, `DEPLOY SUCCESS`, and submit count equal to
  one per image inference.

YOLO @320:

- Record mode: regenerate the graph descriptor image.
- Replay mode: run the precompiled graph descriptor image.
- Require `YOLO FULL NET PASS (4 boxes match C oracle)`.
- Require descriptor submit count equal to one for the main graph before CPU
  decode/NMS starts.

Regression:

- Ensure the normal non-YOLO MNIST baseline still runs.
- Ensure stage-level YOLO descriptor smokes remain available for debugging.

## Reporting

The final report should distinguish three CPU/NPU scheduling levels:

1. Legacy layer-level MMIO scheduling.
2. Current checkpoint stage-level precompiled descriptor replay.
3. New graph-level descriptor job: CPU starts one main inference graph and waits
   for final descriptor done.

The honest YOLO wording is: graph-level submit covers backbone, neck, and head
tensor generation; CPU still performs final detection decode/NMS.
