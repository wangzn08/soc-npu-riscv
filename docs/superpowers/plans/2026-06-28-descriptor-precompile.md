# Pre-compiled Descriptor Image (A2③ self-dump) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the ~9.66M-cyc/inference CPU descriptor-build by recording all YOLO descriptor programs + per-layer qparams into resident DDR once, dumping them to hex from the testbench, and replaying them (submit base/count) in the deploy build.

**Architecture:** Two firmware modes share one call sequence. `-DDESC_RECORD` builds programs into resident DDR slots (bump allocator) + per-layer qparam slots, appends a `{offset,count}` catalog, submits (correct 4/4 run); the testbench `$writememh`-dumps the three DDR regions. The default deploy build `$readmemh`-loads them (`+define+DESC_REPLAY`) and replays via the catalog — no build, no qparam write. No compute-RTL change.

**Tech Stack:** Bare-metal C (`firmware/yolo_desc.c`), ModelSim/Questa, `rtl/axi_full_slave_v1_0_S00_AXI.v` (sim memory model), `run_all.sh`.

---

## Constants (fixed DDR layout, collision-audited)

```
DESC_IMAGE_BASE   0x40000000   (word idx 0)        descriptor records blob, cap 65536 words (1MB)
DESC_QPARAM_BASE  0x40200000   (word idx 131072)   per-layer qparam blob,  cap 16384 words (256KB)
DESC_CATALOG_BASE 0x40280000   (word idx 163840)   catalog: 256 x {off,count}, cap small
```

These sit in the low region below the YOLO image (`0x40400000`), clear of: the
unconditional MNIST preloads (word 65536 / 81920), the YOLO image (262144) and
weight blob (524288), and every YOLO conv in/out/scratch buffer (all >= 0x405C0000
or 0x40C00000). The descriptor image is only *read* by the engine while convs run;
no conv writes 0x40000000..0x40400000. Word index = (cpu_addr - 0x40000000) / 16.

---

### Task 1: Record/replay mode infrastructure in yolo_desc.c

**Files:**
- Modify: `firmware/yolo_desc.c`
- Modify: `firmware/yolo_desc.h`

- [ ] **Step 1: Add layout constants + mode globals**

In `firmware/yolo_desc.h`, add above `#endif`:

```c
// Pre-compiled descriptor image layout (see plan). Word idx = (addr-0x40000000)/16.
#define DESC_IMAGE_BASE    0x40000000u
#define DESC_QPARAM_BASE   0x40200000u
#define DESC_CATALOG_BASE  0x40280000u
#define DESC_CATALOG_MAX   256u

// Reset the record/replay program cursor + bump allocators at net start.
void yolo_desc_reset(void);
```

In `firmware/yolo_desc.c`, after the existing probe globals, add:

```c
// Catalog entry: program location in the descriptor image (record) / to submit (replay).
typedef struct { uint32_t off_words; uint32_t count; } desc_cat_t;
static volatile desc_cat_t *const g_catalog = (volatile desc_cat_t *)DESC_CATALOG_BASE;
static uint32_t g_prog_idx = 0u;   // call-order cursor (record appends, replay reads)
static uint32_t g_img_top  = 0u;   // descriptor-image bump top (words), record only
static uint32_t g_qp_top   = 0u;   // qparam bump top (words), record only

void yolo_desc_reset(void)
{ g_prog_idx = 0u; g_img_top = 0u; g_qp_top = 0u; }
```

- [ ] **Step 2: Point the descriptor builders at the bump base (record) / nothing (replay)**

The record builders must write records into `DESC_IMAGE_BASE + g_img_top*64` instead
of the reused `YOLO_DESC_DDR` scratch. Change `dword()` to use a per-program base.
Replace the existing `dword`:

```c
static uint32_t g_prog_base = YOLO_DESC_DDR;   // current program's DDR base (bytes)
static volatile uint32_t *dword(uint32_t idx)
{ return (volatile uint32_t *)(g_prog_base + idx * NPU_HW_DESC_WORDS * 4u); }
```

- [ ] **Step 3: Compile-check**

Run:
```bash
bash run_all.sh fw yolo_full_stem.c yolo_c2f.c yolo_ops.c npu_desc.c yolo_desc.c
```
Expected: `[OK] 固件编译完成`

- [ ] **Step 4: Commit**

```bash
git add firmware/yolo_desc.c firmware/yolo_desc.h
git commit -m "Add descriptor record/replay layout + bump cursors"
```

---

### Task 2: Record path in the conv descriptor runner

**Files:**
- Modify: `firmware/yolo_desc.c` (`yolo_run_conv2d_tiled_desc`)

- [ ] **Step 1: Wrap each submit in record/replay**

In `yolo_run_conv2d_tiled_desc`, the per-strip block currently builds records into
`di` then `d_submit(di)`. Restructure so a helper does program emit + submit. Add
near the top of the function body (after `uint32_t icg = in_c/16u;`):

```c
#ifdef DESC_RECORD
    /* point qparam writes (Task 3) + records at the next resident slots */
    g_prog_base = DESC_IMAGE_BASE + g_img_top * NPU_HW_DESC_WORDS * 4u;
#else
    g_prog_base = YOLO_DESC_DDR;   /* replay rebuilds nothing; base unused */
#endif
```

Replace the per-strip submit block:

```c
        g_desc_recs += di;
        g_desc_calls += 1u;
        if (di > g_desc_maxrec) g_desc_maxrec = di;
        {
            uint32_t _tr = d_cyc();
            int ok = d_submit(di);
            g_desc_run += d_cyc() - _tr;
            if (!ok)
                return 0;
        }
```

with:

```c
        g_desc_recs += di; g_desc_calls += 1u;
        if (di > g_desc_maxrec) g_desc_maxrec = di;
        {
            uint32_t _tr = d_cyc();
            int ok;
#ifdef DESC_RECORD
            g_catalog[g_prog_idx].off_words = (g_prog_base - DESC_IMAGE_BASE) / 4u;
            g_catalog[g_prog_idx].count = di;
            g_prog_idx++;
            g_img_top += di * NPU_HW_DESC_WORDS;
            ok = d_submit(di);              /* g_prog_base already set this program */
            g_prog_base += di * NPU_HW_DESC_WORDS * 4u;  /* next program slot */
#else
            (void)di;
            { desc_cat_t c = { g_catalog[g_prog_idx].off_words, g_catalog[g_prog_idx].count };
              g_prog_idx++;
              d_npu_wr(NPU_DESC_BASE_LO, DESC_IMAGE_BASE + c.off_words * 4u);
              d_npu_wr(NPU_DESC_BASE_HI, 0u);
              d_npu_wr(NPU_DESC_COUNT, c.count);
              d_npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_CLEAR_DONE);
              d_npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_START);
              { int t = 8000000; ok = 0;
                while (t-- > 0) { uint32_t st = d_npu_rd(NPU_DESC_STATUS);
                  if (st & NPU_DESC_STATUS_ERR) { ok = 0; break; }
                  if (st & NPU_DESC_STATUS_DONE) { ok = 1; break; } }
                d_npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_CLEAR_DONE); } }
#endif
            g_desc_run += d_cyc() - _tr;
            if (!ok)
                return 0;
        }
```

Note: in replay (`#else`), the loop still computes `di` (cheap, no DDR writes via
`dword` because `d_submit` is never reached for building) — the record-building
calls (`d_act_cfg`, `d_dma_in`, `d_conv`, `d_drain`) DO still write to
`g_prog_base = YOLO_DESC_DDR` scratch but are immediately ignored; the engine runs
the pre-loaded image instead. (Replay correctness depends only on the catalog +
loaded image, not on these throwaway writes.)

- [ ] **Step 2: Compile both modes**

```bash
bash run_all.sh fw yolo_full_stem.c yolo_c2f.c yolo_ops.c npu_desc.c yolo_desc.c
```
Expected: `[OK] 固件编译完成` (replay/default build).

- [ ] **Step 3: Commit**

```bash
git add firmware/yolo_desc.c
git commit -m "Record catalog + replay submit in conv desc runner"
```

---

### Task 3: Per-layer resident qparam (方案1)

**Files:**
- Modify: `firmware/yolo_desc.c` (qparam table block + `d_conv` qbase)

- [ ] **Step 1: Allocate qparam per layer in record mode**

Replace the qparam table block:

```c
    {
        uint32_t _t = d_cyc();
        volatile uint32_t *q = (volatile uint32_t *)YOLO_QPARAM_DDR;
        uint32_t oc;
        for (oc = 0u; oc < out_c; oc++) {
            q[oc*4+0] = (uint32_t)(bias ? bias[oc] : 0);
            q[oc*4+1] = scale_mul[oc];
            q[oc*4+2] = 0u;
            q[oc*4+3] = scale_shift[oc];
        }
        g_desc_build += d_cyc() - _t;
    }
```

with (introduce `qbase_ddr` used by `d_conv`):

```c
    uint32_t qbase_ddr = YOLO_QPARAM_DDR;
#ifdef DESC_RECORD
    qbase_ddr = DESC_QPARAM_BASE + g_qp_top * 4u;
    g_qp_top += out_c * 4u;
    {
        uint32_t _t = d_cyc();
        volatile uint32_t *q = (volatile uint32_t *)qbase_ddr;
        uint32_t oc;
        for (oc = 0u; oc < out_c; oc++) {
            q[oc*4+0] = (uint32_t)(bias ? bias[oc] : 0);
            q[oc*4+1] = scale_mul[oc];
            q[oc*4+2] = 0u;
            q[oc*4+3] = scale_shift[oc];
        }
        g_desc_build += d_cyc() - _t;
    }
#else
    (void)bias; (void)scale_mul; (void)scale_shift;   /* qparams pre-loaded in DDR */
#endif
```

- [ ] **Step 2: Point d_conv at the per-layer qbase**

Find the `d_conv(&di, ...)` call inside the OC-chunk loop. It currently passes
`YOLO_QPARAM_DDR + done * 16u`. Change to `qbase_ddr + done * 16u`.

- [ ] **Step 3: Compile**

```bash
bash run_all.sh fw yolo_full_stem.c yolo_c2f.c yolo_ops.c npu_desc.c yolo_desc.c
```
Expected: `[OK] 固件编译完成`

- [ ] **Step 4: Commit**

```bash
git add firmware/yolo_desc.c
git commit -m "Per-layer resident qparam for descriptor replay"
```

---

### Task 4: Reset cursors at net start

**Files:**
- Modify: `firmware/yolo_full_stem.c`

- [ ] **Step 1: Call yolo_desc_reset before the first layer**

After `prof_reset();` near the top of `usercode7()`, add:

```c
    yolo_desc_reset();
```

- [ ] **Step 2: Compile**

```bash
bash run_all.sh fw yolo_full_stem.c yolo_c2f.c yolo_ops.c npu_desc.c yolo_desc.c
```
Expected: `[OK] 固件编译完成`

- [ ] **Step 3: Commit**

```bash
git add firmware/yolo_full_stem.c
git commit -m "Reset descriptor cursors at YOLO net start"
```

---

### Task 5: Testbench dump (record) + load (replay)

**Files:**
- Modify: `rtl/axi_full_slave_v1_0_S00_AXI.v`

- [ ] **Step 1: Load the three images in replay mode**

In the `initial begin` preload block, after the `YOLO_DDR` weight load (`for (ywi ...)`),
add inside the `` `ifdef YOLO_DDR `` region:

```verilog
`ifdef DESC_REPLAY
            $readmemh("firmware/desc_image.hex",   dimg_pre);
            for (dii = 0; dii < 60692; dii = dii + 1)
                byte_ram[0 + dii] = dimg_pre[dii][mem_byte_index*8 +: 8];
            $readmemh("firmware/desc_qparam.hex",  dqp_pre);
            for (dqi = 0; dqi < 16384; dqi = dqi + 1)
                byte_ram[131072 + dqi] = dqp_pre[dqi][mem_byte_index*8 +: 8];
            $readmemh("firmware/desc_catalog.hex", dcat_pre);
            for (dci = 0; dci < 512; dci = dci + 1)
                byte_ram[163840 + dci] = dcat_pre[dci][mem_byte_index*8 +: 8];
`endif
```

Declare the arrays + integers next to the existing `yimg_pre` declarations:

```verilog
`ifdef DESC_REPLAY
        reg [127:0] dimg_pre [0:60691];
        reg [127:0] dqp_pre  [0:16383];
        reg [127:0] dcat_pre [0:511];
        integer dii, dqi, dci;
`endif
```

- [ ] **Step 2: Dump the three regions in record mode**

The shared memory is `BRAM_GEN[0].BYTE_BRAM_GEN[k].byte_ram` (k=0..15). Add a dump
task in `rtl/axi_sys_tb.v` (the testbench has full hierarchical access). After the
`ALL TESTS PASSED` detection / before `$finish`, guarded by `` `ifdef DESC_RECORD ``:

```verilog
`ifdef DESC_RECORD
    task dump_desc;
        integer w, k; reg [127:0] word;
        reg [127:0] img [0:60691];
        reg [127:0] qp  [0:16383];
        reg [127:0] cat [0:511];
        begin
            for (w = 0; w < 60692; w = w + 1) begin
                for (k = 0; k < 16; k = k + 1)
                    word[k*8 +: 8] = `DESC_MEM_PATH[k].byte_ram[0 + w];
                img[w] = word;
            end
            $writememh("firmware/desc_image.hex", img);
            for (w = 0; w < 16384; w = w + 1) begin
                for (k = 0; k < 16; k = k + 1)
                    word[k*8 +: 8] = `DESC_MEM_PATH[k].byte_ram[131072 + w];
                qp[w] = word;
            end
            $writememh("firmware/desc_qparam.hex", qp);
            for (w = 0; w < 512; w = w + 1) begin
                for (k = 0; k < 16; k = k + 1)
                    word[k*8 +: 8] = `DESC_MEM_PATH[k].byte_ram[163840 + w];
                cat[w] = word;
            end
            $writememh("firmware/desc_catalog.hex", cat);
            $display("DESC DUMP DONE");
        end
    endtask
`endif
```

Resolve `` `DESC_MEM_PATH `` to the actual hierarchical instance path of the shared
memory's `BRAM_GEN[0].BYTE_BRAM_GEN` from `axi_sys_tb` (find it by grepping the
instance names in `rtl/axi_sys.v` for the `axi_full_slave_v1_0_S00_AXI` instance,
then `<tb>.<sys_inst>.<mem_inst>.BRAM_GEN[0].BYTE_BRAM_GEN`). Call `dump_desc;`
right where the testbench currently prints `ALL TESTS PASSED.` (record build only).

- [ ] **Step 3: Verify it elaborates (record compile)**

```bash
bash run_all.sh clean
touch .yolo_ddr
bash run_all.sh compile
```
Expected: compile log has no errors (DESC_RECORD/DESC_REPLAY not yet defined, so the
guarded blocks are inert; this just checks the new Verilog parses).

- [ ] **Step 4: Commit**

```bash
git add rtl/axi_full_slave_v1_0_S00_AXI.v rtl/axi_sys_tb.v
git commit -m "Testbench dump (record) + load (replay) for descriptor image"
```

---

### Task 6: run_all.sh — desc-record target + DESC_REPLAY for deploy

**Files:**
- Modify: `run_all.sh`

- [ ] **Step 1: Add a vlog define hook + desc-record target**

Add `DESC_RECORD` / `DESC_REPLAY` to the vlog `+define+` list driven by env vars
`DESC_RECORD=1` / `DESC_REPLAY=1` (mirror how `YOLO_DDR` is added via `.yolo_ddr`).
Add a `desc-record` subcommand that: sets `DESC_RECORD`, builds the record firmware,
runs sim, and confirms `DESC DUMP DONE` + the three `firmware/desc_*.hex` exist.

- [ ] **Step 2: Generate the descriptor image**

```bash
DESC_RECORD=1 bash run_all.sh sim yolo_full_stem.c yolo_c2f.c yolo_ops.c npu_desc.c yolo_desc.c
```
Expected: `YOLO FULL NET PASS` 4/4, `DESC DUMP DONE`, and non-empty
`firmware/desc_image.hex`, `desc_qparam.hex`, `desc_catalog.hex`.

- [ ] **Step 3: Commit**

```bash
git add run_all.sh firmware/desc_image.hex firmware/desc_qparam.hex firmware/desc_catalog.hex
git commit -m "Generate + commit pre-compiled descriptor image"
```

---

### Task 7: Verify replay (acceptance gate)

**Files:** none (verification only)

- [ ] **Step 1: Run the deploy (replay) build**

```bash
DESC_REPLAY=1 bash run_all.sh sim yolo_full_stem.c yolo_c2f.c yolo_ops.c npu_desc.c yolo_desc.c
```
Expected:
```
YOLO FULL NET PASS (4 boxes match C oracle)
[DESCPROF] build=<small, ~qparam-free ~0>  run(...)=~13M  ...
TRAP after ~22,0xx,xxx clock cycles
```
Acceptance: 4/4 boxes AND `TRAP` ~22M (down from 31.84M) AND `build` near 0.

- [ ] **Step 2: Confirm MNIST untouched**

```bash
rm -f .yolo_ddr
bash run_all.sh sim
```
Expected: `=== Result: 10/10 correct ===`, `DEPLOY SUCCESS.`, `TRAP after 941155`.

- [ ] **Step 3: If replay != 4/4, debug**

The catalog/offset bump in record must match the submit order in replay. Add a
temporary print of `g_catalog[0..3].off_words/count` in both modes and compare;
the first mismatch localizes the bug. Do NOT add a second fix without re-checking
the record/replay symmetry (one cause: a non-conv submit — maxpool/eltwise — also
consumes a catalog slot and must be recorded too; see Task 8).

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "Fix descriptor replay parity"
```

---

### Task 8: Route maxpool/eltwise descriptor programs through the catalog

**Files:**
- Modify: `firmware/yolo_desc.c` (`yolo_run_maxpool5x5_desc`, `yolo_run_eltwise_add_desc`)

The conv runner is not the only `d_submit` caller — the SPPF maxpool and C2f
eltwise also submit programs and so must take catalog slots in call order, or the
conv replay indices will desync.

- [ ] **Step 1: Factor the record/replay submit into a helper**

Add in `firmware/yolo_desc.c`:

```c
// Record: append {current g_prog_base offset, di} to the catalog, submit, bump.
// Replay: ignore di, submit the pre-loaded program at catalog[g_prog_idx].
static int desc_submit_cataloged(uint32_t di)
{
#ifdef DESC_RECORD
    g_catalog[g_prog_idx].off_words = (g_prog_base - DESC_IMAGE_BASE) / 4u;
    g_catalog[g_prog_idx].count = di;
    g_prog_idx++; g_img_top += di * NPU_HW_DESC_WORDS;
    { int ok = d_submit(di); g_prog_base += di * NPU_HW_DESC_WORDS * 4u; return ok; }
#else
    desc_cat_t c = { g_catalog[g_prog_idx].off_words, g_catalog[g_prog_idx].count };
    (void)di; g_prog_idx++;
    d_npu_wr(NPU_DESC_BASE_LO, DESC_IMAGE_BASE + c.off_words * 4u);
    d_npu_wr(NPU_DESC_BASE_HI, 0u);
    d_npu_wr(NPU_DESC_COUNT, c.count);
    d_npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_CLEAR_DONE);
    d_npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_START);
    { int t = 8000000; while (t-- > 0) { uint32_t st = d_npu_rd(NPU_DESC_STATUS);
        if (st & NPU_DESC_STATUS_ERR) return 0;
        if (st & NPU_DESC_STATUS_DONE) break; }
      if (t <= 0) return 0; }
    d_npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_CLEAR_DONE);
    return 1;
#endif
}
```

Replace the inline submit in `yolo_run_conv2d_tiled_desc` (Task 2/Step 1) and in
`yolo_run_maxpool5x5_desc` / `yolo_run_eltwise_add_desc` with
`g_prog_base = DESC_RECORD ? (DESC_IMAGE_BASE + g_img_top*64) : YOLO_DESC_DDR;`
followed by building records then `return desc_submit_cataloged(di);` (maxpool/
eltwise build into the same `g_prog_base`).

- [ ] **Step 2: Regenerate the image and re-verify**

```bash
DESC_RECORD=1 bash run_all.sh sim yolo_full_stem.c yolo_c2f.c yolo_ops.c npu_desc.c yolo_desc.c
DESC_REPLAY=1 bash run_all.sh sim yolo_full_stem.c yolo_c2f.c yolo_ops.c npu_desc.c yolo_desc.c
```
Expected: both `YOLO FULL NET PASS` 4/4; replay `TRAP` ~22M, `build`~0.

- [ ] **Step 3: Commit**

```bash
git add firmware/yolo_desc.c firmware/desc_image.hex firmware/desc_qparam.hex firmware/desc_catalog.hex
git commit -m "Route maxpool/eltwise programs through the descriptor catalog"
```

---

## Final Verification

- [ ] **Replay YOLO:** `DESC_REPLAY=1 bash run_all.sh sim yolo_full_stem.c yolo_c2f.c yolo_ops.c npu_desc.c yolo_desc.c` → `YOLO FULL NET PASS` 4/4, `TRAP` ~22M, `[DESCPROF] build`~0.
- [ ] **MNIST:** `rm -f .yolo_ddr; bash run_all.sh sim` → 10/10, `TRAP after 941155`.
- [ ] **git status:** only intended files changed.

---

## Self-Review

- **Spec coverage:** record/replay split (Task 1-2,8), per-layer qparam 方案1 (Task 3),
  reset (Task 4), tb dump+load no-compute-RTL (Task 5), run_all targets (Task 6),
  acceptance 4/4@~22M + MNIST (Task 7), non-conv programs in catalog (Task 8). All
  spec sections mapped.
- **Placeholder scan:** the one deferred concrete is `` `DESC_MEM_PATH `` (the
  hierarchical instance path) — Task 5/Step 2 gives the exact grep to resolve it
  before writing; not a silent TODO.
- **Type consistency:** `desc_cat_t {off_words,count}`, `g_catalog`, `g_prog_idx`,
  `g_img_top`, `g_qp_top`, `desc_submit_cataloged`, `yolo_desc_reset` consistent
  across tasks. Task 8 generalizes the Task-2 inline submit into the shared helper
  (the inline version is replaced, not duplicated).
- **Scope:** single focused change; no compute-RTL; one acceptance metric.
