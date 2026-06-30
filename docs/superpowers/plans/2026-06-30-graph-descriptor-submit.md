# Graph Descriptor Submit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MNIST use one descriptor submission per image and make YOLOv8n @320 submit one descriptor stream for backbone/neck/head before CPU decode/NMS.

**Architecture:** Add graph-builder mode to `firmware/yolo_desc.c` so existing stage helpers can append descriptor records instead of submitting each stage. Keep immediate-submit behavior for directed smokes. Use the existing MNIST `NPU_HW_DESC` path as the full-inference single-submit path and add visible submit-count evidence.

**Tech Stack:** C firmware, Verilog descriptor engine ABI, ModelSim/Questa via `bash run_all.sh`, existing RISC-V bare-metal build.

## Global Constraints

- Base branch is `codex/graph-desc-submit`, based on `checkpoint-yolo320-desc-precompile` (`8ca52ae`).
- Do not port YOLO decode/NMS to hardware.
- Do not change YOLO numerical output or C oracle criteria.
- Do not revive @640 work.
- Keep stage-level descriptor helpers available for smokes and debug.
- Use `run_all.sh`, not Makefile/VCS.
- Use TDD for firmware behavior changes: write a failing smoke first, then implementation.

---

## File Structure

- `firmware/yolo_desc.h`: graph-builder API and submit-count accessor.
- `firmware/yolo_desc.c`: graph mode state, append vs submit behavior, overflow checks, single graph submit.
- `firmware/yolo_desc_graph_smoke.c`: new directed smoke that proves multiple logical descriptor programs become one hardware submit.
- `firmware/yolo_full_stem.c`: wrap YOLO backbone/neck/head descriptor calls in graph begin/end; keep CPU decode/NMS after graph done.
- `firmware/deepnet_deploy.c`: make descriptor submit count observable in the MNIST `NPU_HW_DESC` path.
- `run_all.sh`: optional helper CFLAG for MNIST descriptor deploy if needed.
- `docs/superpowers/specs/2026-06-30-graph-descriptor-submit-design.md`: already written design reference.

---

### Task 1: Add a failing YOLO graph-submit smoke

**Files:**
- Create: `firmware/yolo_desc_graph_smoke.c`
- Modify only if needed for build source inclusion: none; invoke through `run_all.sh`.

**Interfaces:**
- Consumes planned API:
  - `void yolo_desc_graph_begin(void);`
  - `int yolo_desc_graph_end_and_submit(void);`
  - `uint32_t yolo_desc_submit_count(void);`
- Produces a firmware smoke that fails to link before Task 2 and passes after graph mode exists.

- [ ] **Step 1: Write the failing smoke**

Create `firmware/yolo_desc_graph_smoke.c`:

```c
#include "firmware.h"
#include "yolo_desc.h"
#include <stdint.h>

void usercode7(void)
{
    print_str("YOLO DESC GRAPH SMOKE\n");
    yolo_desc_reset();
    yolo_desc_graph_begin();
    if (!yolo_desc_graph_end_and_submit()) {
        print_str("GRAPH SUBMIT FAIL\n");
        return;
    }
    print_str("submit_count=");
    print_dec(yolo_desc_submit_count());
    print_str("\n");
    if (yolo_desc_submit_count() != 1u) {
        print_str("YOLO DESC GRAPH SMOKE FAIL\n");
        return;
    }
    print_str("YOLO DESC GRAPH SMOKE PASS\n");
}
```

- [ ] **Step 2: Run it and verify RED**

Run:

```bash
bash run_all.sh sim yolo_desc_graph_smoke.c yolo_desc.c yolo_ops.c
```

Expected: firmware compile or link fails because `yolo_desc_graph_begin`,
`yolo_desc_graph_end_and_submit`, or `yolo_desc_submit_count` is undefined.

- [ ] **Step 3: Commit the failing test**

```bash
git add firmware/yolo_desc_graph_smoke.c
git commit -m "test: add YOLO descriptor graph-submit smoke"
```

---

### Task 2: Implement graph mode in YOLO descriptor runtime

**Files:**
- Modify: `firmware/yolo_desc.h`
- Modify: `firmware/yolo_desc.c`

**Interfaces:**
- Produces:
  - `void yolo_desc_graph_begin(void);`
  - `int yolo_desc_graph_end_and_submit(void);`
  - `int yolo_desc_graph_active(void);`
  - `uint32_t yolo_desc_submit_count(void);`
- Preserves:
  - `int yolo_run_conv2d_tiled_desc(...)`
  - `int yolo_run_maxpool5x5_desc(...)`
  - `int yolo_run_eltwise_add_desc(...)`

- [ ] **Step 1: Add declarations**

Patch `firmware/yolo_desc.h` near `yolo_desc_reset()`:

```c
void yolo_desc_graph_begin(void);
int yolo_desc_graph_end_and_submit(void);
int yolo_desc_graph_active(void);
uint32_t yolo_desc_submit_count(void);
```

- [ ] **Step 2: Add graph state**

Patch `firmware/yolo_desc.c` near the existing globals:

```c
static uint32_t g_graph_active = 0u;
static uint32_t g_graph_di = 0u;
static uint32_t g_submit_count = 0u;
```

Update `yolo_desc_reset()` to clear all graph and submit counters:

```c
void yolo_desc_reset(void)
{
    g_prog_idx = 0u;
    g_img_top = 0u;
    g_qp_top = 0u;
    g_prog_base = YOLO_DESC_DDR;
    g_graph_active = 0u;
    g_graph_di = 0u;
    g_submit_count = 0u;
}
```

- [ ] **Step 3: Count hardware submissions**

In `d_submit(uint32_t count)`, increment `g_submit_count` only after the engine
reports done:

```c
if (t <= 0) return 0;
d_npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_CLEAR_DONE);
g_submit_count++;
return 1;
```

In the replay branch of `desc_submit_cataloged()`, increment `g_submit_count`
only after the replayed program reports done.

- [ ] **Step 4: Implement graph API**

Add after `desc_prog_begin()`:

```c
void yolo_desc_graph_begin(void)
{
    g_graph_active = 1u;
    g_graph_di = 0u;
    desc_prog_begin();
}

int yolo_desc_graph_active(void)
{
    return g_graph_active != 0u;
}

uint32_t yolo_desc_submit_count(void)
{
    return g_submit_count;
}

int yolo_desc_graph_end_and_submit(void)
{
    uint32_t _tr;
    int ok;
    if (g_graph_active == 0u)
        return 0;
    d_stop(&g_graph_di);
    g_desc_recs += g_graph_di;
    g_desc_calls += 1u;
    if (g_graph_di > g_desc_maxrec)
        g_desc_maxrec = g_graph_di;
#ifdef DESC_RECORD
    g_catalog[g_prog_idx].off_words = (g_prog_base - DESC_IMAGE_BASE) / 4u;
    g_catalog[g_prog_idx].count = g_graph_di;
    g_prog_idx++;
    g_img_top += g_graph_di * NPU_HW_DESC_WORDS;
    _tr = d_cyc();
    ok = d_submit(g_graph_di);
    g_desc_run += d_cyc() - _tr;
    g_prog_base += g_graph_di * NPU_HW_DESC_WORDS * 4u;
#else
    {
        desc_cat_t c;
        int t;
        c.off_words = g_catalog[g_prog_idx].off_words;
        c.count = g_catalog[g_prog_idx].count;
        g_prog_idx++;
        d_npu_wr(NPU_DESC_BASE_LO, DESC_IMAGE_BASE + c.off_words * 4u);
        d_npu_wr(NPU_DESC_BASE_HI, 0u);
        d_npu_wr(NPU_DESC_COUNT, c.count);
        d_npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_CLEAR_DONE);
        d_npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_START);
        t = 8000000;
        while (t-- > 0) {
            uint32_t st = d_npu_rd(NPU_DESC_STATUS);
            if (st & NPU_DESC_STATUS_ERR) { ok = 0; break; }
            if (st & NPU_DESC_STATUS_DONE) { ok = 1; break; }
        }
        if (t <= 0) ok = 0;
        if (ok) g_submit_count++;
        d_npu_wr(NPU_DESC_CTRL, NPU_DESC_CTRL_CLEAR_DONE);
    }
#endif
    g_graph_active = 0u;
    return ok;
}
```

If `d_stop()` is declared later in the file, add a forward declaration above this
block:

```c
static void d_stop(uint32_t *idx);
```

- [ ] **Step 5: Route helper submits through graph append**

Replace immediate `desc_submit_cataloged(di)` calls with:

```c
if (g_graph_active != 0u) {
    g_graph_di += di;
    return 1;
}
return desc_submit_cataloged(di);
```

For `yolo_run_conv2d_tiled_desc()`, ensure each strip appends into the graph
stream by using `uint32_t *pdi = g_graph_active ? &g_graph_di : &di` for record
builders, and do not call `desc_prog_begin()` per strip in graph mode.

- [ ] **Step 6: Run smoke and verify GREEN**

Run:

```bash
bash run_all.sh sim yolo_desc_graph_smoke.c yolo_desc.c yolo_ops.c
```

Expected:

```text
YOLO DESC GRAPH SMOKE
submit_count=1
YOLO DESC GRAPH SMOKE PASS
ALL TESTS PASSED.
```

- [ ] **Step 7: Run an existing stage-level descriptor smoke**

Run one existing smoke, for example:

```bash
touch .yolo_ddr
bash run_all.sh sim yolo_conv1_desc_smoke.c yolo_ops.c yolo_desc.c
```

Expected: existing descriptor smoke still passes and does not require graph mode.

- [ ] **Step 8: Commit**

```bash
git add firmware/yolo_desc.h firmware/yolo_desc.c
git commit -m "feat: add YOLO graph descriptor submit mode"
```

---

### Task 3: Wrap YOLO full net main graph in one descriptor submit

**Files:**
- Modify: `firmware/yolo_full_stem.c`

**Interfaces:**
- Consumes graph API from Task 2.
- Produces one hardware descriptor submit for backbone/neck/head before CPU
  decode/NMS.

- [ ] **Step 1: Add failing expectation**

Temporarily add a check after `yolo_desc_prof_print()` in
`firmware/yolo_full_stem.c`:

```c
if (yolo_desc_submit_count() != 1u) {
    print_str("GRAPH SUBMIT COUNT FAIL count=");
    print_dec(yolo_desc_submit_count());
    print_str("\n");
    errors++;
}
```

Run the current full net before wrapping graph mode:

```bash
touch .yolo_ddr .desc_record
bash run_all.sh sim yolo_full_stem.c yolo_c2f.c yolo_ops.c yolo_desc.c
```

Expected: fails the new check because the baseline submits many descriptor
programs.

- [ ] **Step 2: Start graph mode after reset**

Patch after `yolo_desc_reset();`:

```c
yolo_desc_graph_begin();
```

- [ ] **Step 3: End and submit before decode**

Patch immediately after `[head convs done]` and before CPU decode/NMS begins:

```c
if (errors == 0u && !yolo_desc_graph_end_and_submit()) {
    print_str("  graph submit fail\n");
    errors++;
}
```

Keep all existing CPU decode/NMS code after this point.

- [ ] **Step 4: Keep debug evidence**

Print submit count near the descriptor profile:

```c
print_str("[GRAPH] submit_count=");
print_dec(yolo_desc_submit_count());
print_str("\n");
```

- [ ] **Step 5: Run record mode**

Run:

```bash
touch .yolo_ddr .desc_record
bash run_all.sh clean
bash run_all.sh sim yolo_full_stem.c yolo_c2f.c yolo_ops.c yolo_desc.c
```

Expected:

```text
[GRAPH] submit_count=1
YOLO FULL NET PASS (4 boxes match C oracle)
ALL TESTS PASSED.
```

- [ ] **Step 6: Run replay mode**

Replace `.desc_record` with `.desc_replay`:

```bash
rm -f .desc_record
touch .yolo_ddr .desc_replay
bash run_all.sh clean
bash run_all.sh sim yolo_full_stem.c yolo_c2f.c yolo_ops.c yolo_desc.c
```

Expected:

```text
[GRAPH] submit_count=1
YOLO FULL NET PASS (4 boxes match C oracle)
ALL TESTS PASSED.
```

- [ ] **Step 7: Commit**

```bash
git add firmware/yolo_full_stem.c firmware/desc_image.hex firmware/desc_catalog.hex firmware/desc_qparam.hex
git commit -m "feat: submit YOLO main graph as one descriptor job"
```

---

### Task 4: Make MNIST descriptor deploy path single-submit and visible

**Files:**
- Modify: `firmware/deepnet_deploy.c`
- Modify if needed: `run_all.sh`

**Interfaces:**
- Produces visible evidence that one descriptor submit runs one MNIST image
  inference in the `NPU_HW_DESC` path.

- [ ] **Step 1: Add failing submit-count evidence**

In `firmware/deepnet_deploy.c`, add:

```c
#if NPU_HW_DESC
static uint32_t hw_desc_submit_count;
#endif
```

Initialize it before the digit loop:

```c
#if NPU_HW_DESC
    hw_desc_submit_count = 0u;
#endif
```

After `hw_desc_submit(di)` succeeds, increment:

```c
hw_desc_submit_count++;
```

After each descriptor inference returns, print:

```c
print_str(" desc_submit_count=");
print_dec(hw_desc_submit_count);
print_str("\n");
```

- [ ] **Step 2: Build descriptor path and verify behavior**

Run:

```bash
NPU_HW_DESC=1 bash run_all.sh sim
```

If `run_all.sh` does not pass `NPU_HW_DESC`, update `run_all.sh` firmware CFLAGS:

```bash
if [ -n "${NPU_HW_DESC:-}" ]; then
    CFLAGS+=(-DNPU_HW_DESC="$NPU_HW_DESC")
fi
```

Expected: MNIST descriptor path runs and shows submit count increasing by one per
image. If the path was previously not warning-clean, fix only descriptor-path
compile errors.

- [ ] **Step 3: Add final assertion**

After the 10-image loop, assert:

```c
#if NPU_HW_DESC
    if (hw_desc_submit_count != NUM_TEST_IMAGES) {
        print_str("DESC SUBMIT COUNT FAIL\n");
        return;
    }
#endif
```

- [ ] **Step 4: Run descriptor MNIST**

Run:

```bash
NPU_HW_DESC=1 bash run_all.sh clean
NPU_HW_DESC=1 bash run_all.sh sim
```

Expected:

```text
=== Result: 10/10 correct ===
DEPLOY SUCCESS.
ALL TESTS PASSED.
```

- [ ] **Step 5: Run default MNIST regression**

Run:

```bash
bash run_all.sh clean
bash run_all.sh sim
```

Expected default MNIST baseline still passes.

- [ ] **Step 6: Commit**

```bash
git add firmware/deepnet_deploy.c run_all.sh
git commit -m "feat: verify MNIST descriptor deploy single-submit path"
```

---

### Task 5: Final verification and docs update

**Files:**
- Modify: `docs/notes/RESUME-yolov8n-soc.md` or create a new focused note if the
  existing note is too noisy.

**Interfaces:**
- Consumes all previous tasks.
- Produces final documented command set and results.

- [ ] **Step 1: Run YOLO graph replay**

Run:

```bash
rm -f .desc_record
touch .yolo_ddr .desc_replay
bash run_all.sh clean
bash run_all.sh sim yolo_full_stem.c yolo_c2f.c yolo_ops.c yolo_desc.c
```

Expected:

```text
[GRAPH] submit_count=1
YOLO FULL NET PASS (4 boxes match C oracle)
ALL TESTS PASSED.
```

- [ ] **Step 2: Run MNIST descriptor deploy**

Run:

```bash
NPU_HW_DESC=1 bash run_all.sh clean
NPU_HW_DESC=1 bash run_all.sh sim
```

Expected: `10/10`, `DEPLOY SUCCESS`, `ALL TESTS PASSED`.

- [ ] **Step 3: Run MNIST default deploy**

Run:

```bash
bash run_all.sh clean
bash run_all.sh sim
```

Expected: `10/10`, `DEPLOY SUCCESS`, `ALL TESTS PASSED`.

- [ ] **Step 4: Document final state**

Add a concise note to `docs/notes/RESUME-yolov8n-soc.md`:

```markdown
## LATEST: graph-level descriptor submit

- YOLOv8n @320 backbone/neck/head now runs as one hardware descriptor job.
  CPU starts one graph stream, waits for descriptor done, then runs decode/NMS.
- MNIST descriptor deploy uses one descriptor submit per image inference.
- Verified commands:
  - `touch .yolo_ddr .desc_replay; bash run_all.sh sim yolo_full_stem.c yolo_c2f.c yolo_ops.c yolo_desc.c`
  - `NPU_HW_DESC=1 bash run_all.sh sim`
  - `bash run_all.sh sim`
```

- [ ] **Step 5: Commit**

```bash
git add docs/notes/RESUME-yolov8n-soc.md
git commit -m "docs: record graph-level descriptor submit baseline"
```

---

## Self-Review

- Spec coverage: MNIST and YOLO are both covered; YOLO decode/NMS remains CPU.
- Placeholder scan: no `TBD`/`TODO` placeholders are intentionally left.
- Type consistency: graph API names match the design spec.
- Risk: Task 2 may need minor adjustment around `d_stop()` declaration order and
  graph append offsets; keep the new smoke small so failures are fast.
