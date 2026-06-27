# Pre-compiled Descriptor Image (A2 + qparam 方案1) — Design

## Problem / Motivation

Measured on the full YOLOv8n RTL run (probe in `yolo_desc.c`):

```
[DESCPROF] build=9,661,458  run(submit+dma+wait)=13,174,876  wgt_preload=1,475,840
[DESCSIZE] total_records=15,173  submits=148  max_prog_records=434
           total_words=242,768   bytes=971,072  (~0.95 MB)
TRAP total = 31,840,227
```

The CPU spends **9.66M cyc = 30.6% of the whole net** building descriptor records
in DDR at runtime, because each 32-bit record word is written through the
single-beat `axi_upsizer_32_128` path at ~26 cyc/word. The descriptor programs are
fully **static** for a fixed network + memory map (op/addr/dims/qparam values all
known ahead of time), and total only ~0.95 MB.

**Goal:** pre-generate every descriptor program + its qparam table once, load them
into DDR at boot (like the weight blob), and have the deploy firmware *replay*
them (set base/count, submit) instead of rebuilding. Eliminates ~9.66M build +
~0.8M qparam writes per inference. Target: **31.84M → ~22M cyc (-30%)**, 4/4 boxes
preserved, and the same dynamic-power/DDR-write saving each frame.

## Non-Goals

- No change to any compute/synthesizable RTL (descriptor engine, DMA, array,
  post-process all untouched). Only one **testbench** `$readmemh` is added to
  preload the descriptor image (record-mode adds a `$writememh` dump task).
- No graph-import / generic memory planner. The image is tied to the current fixed
  YOLO memory map; regenerate it if the map changes.
- MNIST path untouched (it does not use `yolo_desc.c`).

## Constraints

- YOLO must stay `YOLO FULL NET PASS` 4/4 vs the C oracle.
- New firmware C must be warning-clean under strict CFLAGS.
- The descriptor image + per-layer qparam regions must avoid the image
  (`0x4040_0000..0x405A_0000`) and weight blob (`0x4080_0000..0x40B0_1000`).
- The Phase-0 probe (`g_desc_build/run/...`) stays so the saving is re-measurable;
  in replay mode `build` must read ~0.

## Architecture

Two firmware modes selected by a compile flag, sharing the same call sequence:

- **Record mode** (`-DDESC_RECORD`): a generator build. `yolo_run_*_desc` bump-
  allocates the next slot in a resident **descriptor image region** and a
  per-layer **qparam region**, writes the records + qparam values there, records
  `{offset,count}` into an in-DDR catalog, and submits (producing a correct 4/4
  inference). At end-of-run the testbench `$writememh`-dumps the descriptor image,
  qparam region, and catalog to hex files.
- **Replay mode** (default deploy): the testbench `$readmemh`-loads the three hex
  files into DDR at boot. `yolo_run_*_desc` becomes: take the next catalog entry
  (by call order), set `NPU_DESC_BASE`/`COUNT`, submit. No build, no qparam write.

Because record and replay run the identical deterministic call sequence, the
call-order index alone keys the catalog — no per-call ID plumbing needed.

### qparam 方案1 (per-layer resident qparam)

Today all layers reuse one `YOLO_QPARAM_DDR` region (overwritten per layer). For
replay to skip qparam writes, record mode gives **each layer its own qparam slot**
(bump-allocated in the qparam region), and the CONV2D record points at that slot.
The qparam values are dumped in the image, so replay needs zero qparam writes.
Extra DDR: ≤ sum(out_c)·4 words ≈ ~150 KB (fits).

## DDR Memory Map (new regions, in free space)

```
DESC_IMAGE_BASE   0x40C0_0000   descriptor records blob   (~1 MB,  cap 0x100000)
DESC_QPARAM_BASE  0x40D0_0000   per-layer qparam blob      (~256 KB, cap 0x40000)
DESC_CATALOG_BASE 0x40D4_0000   catalog: N×{offset,count}  (~2 KB)
```

(0x40C0_0000.. is the documented free region above the weight blob. Final bases
chosen in the plan to not collide with head/neck scratch which also live there —
the plan’s first task audits and, if needed, relocates these or the scratch.)

## Components / Data Flow

### 1. `yolo_desc.c` record/replay split

- Add a mode flag (compile-time `DESC_RECORD`, plus a runtime `g_desc_mode` so a
  single binary could do record-then-replay if wanted; v1 uses the compile flag).
- Bump allocators: `g_img_top` (descriptor image), `g_qp_top` (qparam), reset at
  net start via a new `yolo_desc_reset(void)`.
- A catalog array in DDR at `DESC_CATALOG_BASE`: `struct { u32 off; u32 count; }`,
  appended per submit; `g_prog_idx` indexes it.
- **Record path** (`yolo_run_conv2d_tiled_desc` + the maxpool/eltwise builders):
  build into `DESC_IMAGE_BASE + g_img_top` instead of the reused `YOLO_DESC_DDR`
  scratch; write qparam into `DESC_QPARAM_BASE + g_qp_top` and point the record
  there; append `{g_img_top, di}` to the catalog; submit; advance the bump tops.
- **Replay path**: `c = catalog[g_prog_idx++]; submit(DESC_IMAGE_BASE+c.off, c.count)`.
  Skip all record/qparam writes.
- The non-conv programs (`yolo_run_maxpool5x5_desc`, `yolo_run_eltwise_add_desc`)
  go through the same record/replay split (they are also in the catalog, in call
  order), so the entire descriptor stream is pre-compiled.

### 2. Testbench dump / load (`rtl/axi_sys_tb.v`, record-only `$writememh`)

- Record build: after `ALL TESTS PASSED`, a dump task `$writememh`s the three DDR
  ranges to `firmware/desc_image.hex`, `desc_qparam.hex`, `desc_catalog.hex`
  (128-bit/line, same format as the weight blob).
- Replay build: `$readmemh` those three into the shared memory model at boot,
  guarded by a `+define+DESC_REPLAY` (set by `run_all.sh`), mirroring how
  `yolo_weights_ddr.hex` is loaded.

### 3. `run_all.sh`

- Add a `desc-record` target: builds firmware with `-DDESC_RECORD` and runs sim to
  emit the three hex files.
- Default YOLO deploy: build replay firmware and pass `+define+DESC_REPLAY` so the
  testbench preloads the images.

## Verification

- **Record run**: must still print `YOLO FULL NET PASS` 4/4 (it is a normal build,
  just writing to resident slots) and emit the three non-empty hex files.
- **Replay run**: `YOLO FULL NET PASS` 4/4, `[DESCPROF] build≈0`, and
  `TRAP ≈ 22M cyc` (down from 31.84M). This is the acceptance gate.
- **MNIST**: `bash run_all.sh sim` still 10/10 / 941,155 (yolo files unlinked).
- If replay diverges from 4/4, the catalog/offset bump logic mismatches between
  record and replay — debug by dumping the first differing program’s base/count.

## Risks / Mitigations

- **Determinism**: record and replay must take the identical call path. They share
  the same `usercode7` code; the only difference is inside `yolo_run_*_desc`. The
  catalog is keyed purely by call order — safe as long as no mode-dependent
  branching changes the call sequence (it does not).
- **DDR collision**: the three new regions sit in the free zone above the weight
  blob, which the head/neck scratch also use. Plan task 1 audits the live ranges
  during the head/neck stages and places the descriptor regions so they never
  overlap a buffer that is live when a conv runs. (The descriptor image is read by
  the engine *while* convs run, so it must stay clear of every conv’s in/out/wgt.)
- **Hex format**: `$writememh` of the 128-bit DDR model yields the same line format
  `$readmemh` expects (verified against the existing weight-blob load).

## Self-Review

- Placeholders: none; bases are concrete with a plan-task audit for collisions.
- Consistency: record writes the same records replay submits; qparam 方案1 removes
  the only per-layer runtime write the records depend on.
- Scope: single, focused change (firmware record/replay + tb load/dump + run_all);
  no RTL compute change; one acceptance metric (4/4 @ ~22M, build≈0).
- Ambiguity: qparam 方案1 (per-layer resident) chosen explicitly over 方案2.
