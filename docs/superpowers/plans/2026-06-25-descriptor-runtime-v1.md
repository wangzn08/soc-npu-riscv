# Descriptor Runtime V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small generic descriptor-based NPU runtime so MNIST and YOLO can be described as layer/data-movement records instead of only hand-written per-model scheduling calls.

**Architecture:** Keep the current RTL and existing C helpers as the execution backend. Add a firmware-only descriptor layer that maps typed descriptors to existing DMA/NPU helper calls, then migrate a small, verified subset of MNIST/YOLO calls to descriptors while preserving the old direct-call path as fallback. This improves SoC system-level generality without changing datasets or making the hardware YOLO-specific.

**Tech Stack:** Bare-metal C firmware, existing NPU MMIO register map in `firmware/firmware.h`, current YOLO helpers in `firmware/yolo_ops.c`, ModelSim/Questa flow via `bash run_all.sh sim`.

## Global Constraints

- Do not change the MNIST dataset or expected MNIST labels.
- MNIST must remain `10/10 correct` and print `DEPLOY SUCCESS.`
- YOLO default image must remain the current `bus320.ppm` deployment unless a test explicitly regenerates `firmware/yolo_img_ddr.hex` and restores it afterward.
- Descriptor runtime v1 must be generic: no YOLO layer-number-specific logic in the descriptor executor.
- Keep existing direct helper functions as fallback; do not delete the current working YOLO/MNIST paths in v1.
- New firmware C must be warning-clean under the existing strict CFLAGS.
- Normal full regression command is `bash run_all.sh sim`.

---

## File Structure

- Create `firmware/npu_desc.h`: descriptor types, op enum, flags, and public runner prototypes.
- Create `firmware/npu_desc.c`: descriptor validation and dispatch to existing helper functions.
- Modify `run_all.sh`: include `npu_desc.c` only for firmware targets that need descriptor runtime, or include it globally if there is no duplicate symbol risk.
- Modify `firmware/yolo_full_stem.c`: migrate a small group of YOLO calls to descriptors after the runner is verified.
- Modify `firmware/deepnet_deploy.c`: optionally migrate one MNIST DMA/NPU sequence only after YOLO smoke succeeds; v1 can stop after adding a no-op descriptor smoke if MNIST code risk is too high.
- Add `tests/tb_desc_compile.c` only if the build flow supports firmware-only compile checks; otherwise use `run_all.sh fw ...` as the compile test.

---

### Task 1: Add Descriptor Data Model and Validator

**Files:**
- Create: `firmware/npu_desc.h`
- Create: `firmware/npu_desc.c`
- Modify: `run_all.sh`

**Interfaces:**
- Produces:
  - `typedef enum npu_desc_op_t`
  - `typedef struct npu_desc_t`
  - `int npu_desc_validate(const npu_desc_t *d)`
  - `int npu_desc_run(const npu_desc_t *d)`
  - `int npu_desc_run_many(const npu_desc_t *list, uint32_t count)`
- Consumes: existing `firmware/firmware.h`; no YOLO helper dependency yet.

- [ ] **Step 1: Write the descriptor header**

Create `firmware/npu_desc.h` with:

```c
#ifndef NPU_DESC_H
#define NPU_DESC_H

#include <stdint.h>

typedef enum {
    NPU_DESC_OP_NOP = 0,
    NPU_DESC_OP_DMA_DDR_TO_ACT = 1,
    NPU_DESC_OP_DMA_ACT_TO_DDR = 2,
    NPU_DESC_OP_UPSAMPLE2X_DDR = 3,
    NPU_DESC_OP_COPY_DDR_TO_DDR = 4,
    NPU_DESC_OP_CONV2D_TILED = 5
} npu_desc_op_t;

typedef struct {
    uint32_t op;
    uint32_t src0;
    uint32_t src1;
    uint32_t dst;
    uint32_t wgt;
    uint32_t scratch0;
    uint32_t scratch1;
    uint16_t in_w;
    uint16_t in_h;
    uint16_t in_c;
    uint16_t out_c;
    uint8_t  kh;
    uint8_t  kw;
    uint8_t  stride;
    uint8_t  pad;
    uint32_t words;
    uint32_t flags;
    const int32_t  *bias;
    const uint32_t *scale_mul;
    const uint32_t *scale_shift;
    const uint8_t  *lut;
    int32_t pad_value;
    uint32_t wgt_words_per_oc;
    uint32_t strip_out_rows;
} npu_desc_t;

int npu_desc_validate(const npu_desc_t *d);
int npu_desc_run(const npu_desc_t *d);
int npu_desc_run_many(const npu_desc_t *list, uint32_t count);

#endif
```

- [ ] **Step 2: Write a minimal implementation**

Create `firmware/npu_desc.c` with only NOP and validation:

```c
#include "npu_desc.h"

int npu_desc_validate(const npu_desc_t *d)
{
    if (d == (const npu_desc_t *)0)
        return 0;
    switch ((npu_desc_op_t)d->op) {
    case NPU_DESC_OP_NOP:
        return 1;
    case NPU_DESC_OP_DMA_DDR_TO_ACT:
    case NPU_DESC_OP_DMA_ACT_TO_DDR:
    case NPU_DESC_OP_UPSAMPLE2X_DDR:
    case NPU_DESC_OP_COPY_DDR_TO_DDR:
    case NPU_DESC_OP_CONV2D_TILED:
        return 1;
    default:
        return 0;
    }
}

int npu_desc_run(const npu_desc_t *d)
{
    if (!npu_desc_validate(d))
        return 0;
    if (d->op == NPU_DESC_OP_NOP)
        return 1;
    return 0;
}

int npu_desc_run_many(const npu_desc_t *list, uint32_t count)
{
    uint32_t i;
    if (list == (const npu_desc_t *)0 && count != 0u)
        return 0;
    for (i = 0u; i < count; i++) {
        if (!npu_desc_run(&list[i]))
            return 0;
    }
    return 1;
}
```

- [ ] **Step 3: Add `npu_desc.c` to firmware build**

Modify `run_all.sh` so `firmware/npu_desc.c` is compiled with common firmware sources or is accepted as an additional source. If common source list exists, add:

```bash
firmware/npu_desc.c
```

- [ ] **Step 4: Compile firmware to verify descriptor scaffolding**

Run:

```bash
bash run_all.sh fw yolo_full_stem.c yolo_c2f.c yolo_ops.c
```

Expected:

```text
[OK] 固件编译完成
```

- [ ] **Step 5: Commit**

```bash
git add firmware/npu_desc.h firmware/npu_desc.c run_all.sh
git commit -m "Add generic NPU descriptor scaffolding"
```

---

### Task 2: Dispatch Generic Data-Movement Descriptors

**Files:**
- Modify: `firmware/npu_desc.c`
- Modify: `firmware/npu_desc.h`

**Interfaces:**
- Consumes:
  - `yolo_dma_ddr_to_act(uint32_t ddr_addr, uint32_t act_base, uint32_t words)`
  - `yolo_dma_act_to_ddr(uint32_t ddr_addr, uint32_t act_base, uint32_t words)`
  - `yolo_run_upsample2x_ddr(uint32_t src_ddr, uint32_t dst_ddr, uint32_t scratch_act_base, uint32_t in_w, uint32_t in_h, uint32_t ic_groups)`
  - `yolo_copy_ddr_to_ddr_via_act(uint32_t src_ddr, uint32_t dst_ddr, uint32_t scratch_act_base, uint32_t words)`
- Produces: working descriptor dispatch for data movement and upsample.

- [ ] **Step 1: Include existing helper declarations**

In `firmware/npu_desc.c`, add:

```c
#include "yolo_ops.h"
```

- [ ] **Step 2: Strengthen validation for data movement**

Replace `npu_desc_validate` body with:

```c
int npu_desc_validate(const npu_desc_t *d)
{
    if (d == (const npu_desc_t *)0)
        return 0;
    switch ((npu_desc_op_t)d->op) {
    case NPU_DESC_OP_NOP:
        return 1;
    case NPU_DESC_OP_DMA_DDR_TO_ACT:
    case NPU_DESC_OP_DMA_ACT_TO_DDR:
    case NPU_DESC_OP_COPY_DDR_TO_DDR:
        return d->words != 0u;
    case NPU_DESC_OP_UPSAMPLE2X_DDR:
        return d->in_w != 0u && d->in_h != 0u && d->in_c != 0u &&
               (d->in_c & 15u) == 0u;
    case NPU_DESC_OP_CONV2D_TILED:
        return d->in_w != 0u && d->in_h != 0u && d->in_c != 0u &&
               d->out_c != 0u && d->kh != 0u && d->kw != 0u &&
               d->stride != 0u && d->bias != (const int32_t *)0 &&
               d->scale_mul != (const uint32_t *)0 &&
               d->scale_shift != (const uint32_t *)0;
    default:
        return 0;
    }
}
```

- [ ] **Step 3: Implement data-movement dispatch**

Replace `npu_desc_run` with:

```c
int npu_desc_run(const npu_desc_t *d)
{
    if (!npu_desc_validate(d))
        return 0;

    switch ((npu_desc_op_t)d->op) {
    case NPU_DESC_OP_NOP:
        return 1;
    case NPU_DESC_OP_DMA_DDR_TO_ACT:
        return yolo_dma_ddr_to_act(d->src0, d->dst, d->words);
    case NPU_DESC_OP_DMA_ACT_TO_DDR:
        return yolo_dma_act_to_ddr(d->dst, d->src0, d->words);
    case NPU_DESC_OP_UPSAMPLE2X_DDR:
        return yolo_run_upsample2x_ddr(d->src0, d->dst, d->scratch0,
                                       d->in_w, d->in_h, d->in_c / 16u);
    case NPU_DESC_OP_COPY_DDR_TO_DDR:
        return yolo_copy_ddr_to_ddr_via_act(d->src0, d->dst, d->scratch0, d->words);
    case NPU_DESC_OP_CONV2D_TILED:
        return 0;
    default:
        return 0;
    }
}
```

- [ ] **Step 4: Compile YOLO firmware**

Run:

```bash
bash run_all.sh fw yolo_full_stem.c yolo_c2f.c yolo_ops.c npu_desc.c
```

Expected:

```text
[OK] 固件编译完成
```

- [ ] **Step 5: Commit**

```bash
git add firmware/npu_desc.c firmware/npu_desc.h
git commit -m "Dispatch generic NPU data descriptors"
```

---

### Task 3: Migrate YOLO Neck Data Movement to Descriptors

**Files:**
- Modify: `firmware/yolo_full_stem.c`
- Modify: `run_all.sh` if needed so `npu_desc.c` is linked for YOLO.

**Interfaces:**
- Consumes: `npu_desc_run_many(const npu_desc_t *list, uint32_t count)`
- Produces: YOLO neck upsample/cat movement driven by descriptors, replacing direct helper calls for four movement groups.

- [ ] **Step 1: Include descriptor header**

Add near other includes in `firmware/yolo_full_stem.c`:

```c
#include "npu_desc.h"
```

- [ ] **Step 2: Replace `neck_up1_cat1_DMA` direct calls with descriptors**

Replace the current direct calls:

```c
if (errors == 0u && !yolo_run_upsample2x_ddr(SPPF_OUT, NK_CAT1, 0u, 10u, 10u, 256u/16u)) {
    print_str("  neck up1 fail\n"); errors++;
}
if (errors == 0u && !yolo_copy_ddr_to_ddr_via_act(C2F6_OUT, NK_CAT1 + (256u/16u)*(20u*20u)*16u,
                                                   0u, (128u/16u)*(20u*20u))) {
    print_str("  neck cat1 tap fail\n"); errors++;
}
```

with:

```c
{
    const npu_desc_t descs[] = {
        {
            .op = NPU_DESC_OP_UPSAMPLE2X_DDR,
            .src0 = SPPF_OUT,
            .dst = NK_CAT1,
            .scratch0 = 0u,
            .in_w = 10u,
            .in_h = 10u,
            .in_c = 256u
        },
        {
            .op = NPU_DESC_OP_COPY_DDR_TO_DDR,
            .src0 = C2F6_OUT,
            .dst = NK_CAT1 + (256u/16u)*(20u*20u)*16u,
            .scratch0 = 0u,
            .words = (128u/16u)*(20u*20u)
        }
    };
    if (errors == 0u && !npu_desc_run_many(descs, 2u)) {
        print_str("  neck cat1 desc fail\n"); errors++;
    }
}
```

- [ ] **Step 3: Replace `neck_up2_cat2_DMA` direct calls with descriptors**

Use:

```c
{
    const npu_desc_t descs[] = {
        {
            .op = NPU_DESC_OP_UPSAMPLE2X_DDR,
            .src0 = NK_FMID,
            .dst = NK_CAT2,
            .scratch0 = 0u,
            .in_w = 20u,
            .in_h = 20u,
            .in_c = 128u
        },
        {
            .op = NPU_DESC_OP_COPY_DDR_TO_DDR,
            .src0 = C2F4_OUT,
            .dst = NK_CAT2 + (128u/16u)*(40u*40u)*16u,
            .scratch0 = 0u,
            .words = (64u/16u)*(40u*40u)
        }
    };
    if (errors == 0u && !npu_desc_run_many(descs, 2u)) {
        print_str("  neck cat2 desc fail\n"); errors++;
    }
}
```

- [ ] **Step 4: Replace `neck_cat3_DMA` direct calls with descriptors**

Use:

```c
{
    const npu_desc_t descs[] = {
        {
            .op = NPU_DESC_OP_COPY_DDR_TO_DDR,
            .src0 = NK_C35,
            .dst = NK_CAT3,
            .scratch0 = 0u,
            .words = (64u/16u)*(20u*20u)
        },
        {
            .op = NPU_DESC_OP_COPY_DDR_TO_DDR,
            .src0 = NK_FMID,
            .dst = NK_CAT3 + (64u/16u)*(20u*20u)*16u,
            .scratch0 = 0u,
            .words = (128u/16u)*(20u*20u)
        }
    };
    if (errors == 0u && !npu_desc_run_many(descs, 2u)) {
        print_str("  neck cat3 desc fail\n"); errors++;
    }
}
```

- [ ] **Step 5: Replace `neck_cat4_DMA` direct calls with descriptors**

Use:

```c
{
    const npu_desc_t descs[] = {
        {
            .op = NPU_DESC_OP_COPY_DDR_TO_DDR,
            .src0 = NK_C46,
            .dst = NK_CAT4,
            .scratch0 = 0u,
            .words = (128u/16u)*(10u*10u)
        },
        {
            .op = NPU_DESC_OP_COPY_DDR_TO_DDR,
            .src0 = C2F8_OUT,
            .dst = NK_CAT4 + (128u/16u)*(10u*10u)*16u,
            .scratch0 = 0u,
            .words = (256u/16u)*(10u*10u)
        }
    };
    if (errors == 0u && !npu_desc_run_many(descs, 2u)) {
        print_str("  neck cat4 desc fail\n"); errors++;
    }
}
```

- [ ] **Step 6: Run YOLO full simulation**

Run:

```bash
touch .yolo_ddr
bash run_all.sh sim yolo_full_stem.c yolo_c2f.c yolo_ops.c npu_desc.c
```

Expected:

```text
YOLO FULL NET PASS
ALL TESTS PASSED.
```

Cycle count should remain close to the current ICG=16 baseline, about `42.8M` cycles. Small compile/log variation is acceptable; correctness is the gate.

- [ ] **Step 7: Commit**

```bash
git add firmware/yolo_full_stem.c run_all.sh
git commit -m "Run YOLO neck movement from descriptors"
```

---

### Task 4: Add Conv2D Tiled Descriptor Dispatch

**Files:**
- Modify: `firmware/npu_desc.c`

**Interfaces:**
- Consumes:
  - `yolo_run_conv2d_tiled(...)`
  - descriptor fields `.src0`, `.wgt`, `.scratch0`, `.dst`, `.scratch1`, shape, qparams, flags
- Produces: `NPU_DESC_OP_CONV2D_TILED` dispatch.

- [ ] **Step 1: Implement conv descriptor dispatch**

In `npu_desc_run`, replace the `NPU_DESC_OP_CONV2D_TILED` case with:

```c
    case NPU_DESC_OP_CONV2D_TILED:
        return yolo_run_conv2d_tiled(d->src0, d->wgt, d->scratch0, d->dst, d->scratch1,
                                     d->in_w, d->in_h, d->in_c, d->out_c,
                                     d->kh, d->kw, d->stride, d->pad,
                                     d->bias, d->scale_mul, d->scale_shift,
                                     d->flags, d->wgt_words_per_oc,
                                     d->strip_out_rows, d->pad_value);
```

- [ ] **Step 2: Compile YOLO firmware**

Run:

```bash
bash run_all.sh fw yolo_full_stem.c yolo_c2f.c yolo_ops.c npu_desc.c
```

Expected:

```text
[OK] 固件编译完成
```

- [ ] **Step 3: Commit**

```bash
git add firmware/npu_desc.c
git commit -m "Dispatch tiled conv descriptors"
```

---

### Task 5: Migrate Two Safe YOLO Conv Calls to Descriptors

**Files:**
- Modify: `firmware/yolo_full_stem.c`

**Interfaces:**
- Consumes: `NPU_DESC_OP_CONV2D_TILED`
- Produces: at least two normal YOLO convs launched through descriptors.

- [ ] **Step 1: Migrate conv25 to descriptor**

Replace the direct `yolo_run_conv2d_tiled` call for `s9_sppf_conv25` with:

```c
{
    const npu_desc_t d = {
        .op = NPU_DESC_OP_CONV2D_TILED,
        .src0 = C2F8_OUT,
        .wgt = WGT_OF(25),
        .scratch0 = WGT_BASE,
        .dst = S_CV1,
        .scratch1 = PAD_ROW,
        .in_w = SPPFE_IN_W,
        .in_h = SPPFE_IN_H,
        .in_c = SPPFE_C25_IC,
        .out_c = SPPFE_C25_OC,
        .kh = 1u,
        .kw = 1u,
        .stride = 1u,
        .pad = 0u,
        .bias = yolo_sppf_e_c25_bias,
        .scale_mul = yolo_sppf_e_c25_mul,
        .scale_shift = yolo_sppf_e_c25_shift,
        .flags = NPU_CTRL_SILU_EXACT_EN,
        .wgt_words_per_oc = SPPFE_C25_IC/16u,
        .strip_out_rows = 16u,
        .pad_value = 0
    };
    if (errors == 0u && !npu_desc_run(&d)) {
        print_str("  conv25 desc fail\n"); errors++;
    }
}
```

- [ ] **Step 2: Migrate conv26 to descriptor**

Replace the direct `yolo_run_conv2d_tiled` call for `s9_sppf_conv26` with:

```c
{
    const npu_desc_t d = {
        .op = NPU_DESC_OP_CONV2D_TILED,
        .src0 = S_CAT,
        .wgt = WGT_OF(26),
        .scratch0 = WGT_BASE,
        .dst = SPPF_OUT,
        .scratch1 = PAD_ROW,
        .in_w = SPPFE_IN_W,
        .in_h = SPPFE_IN_H,
        .in_c = SPPFE_C26_IC,
        .out_c = SPPFE_C26_OC,
        .kh = 1u,
        .kw = 1u,
        .stride = 1u,
        .pad = 0u,
        .bias = yolo_sppf_e_c26_bias,
        .scale_mul = yolo_sppf_e_c26_mul,
        .scale_shift = yolo_sppf_e_c26_shift,
        .flags = NPU_CTRL_SILU_EXACT_EN,
        .wgt_words_per_oc = SPPFE_C26_IC/16u,
        .strip_out_rows = 16u,
        .pad_value = 0
    };
    if (errors == 0u && !npu_desc_run(&d)) {
        print_str("  conv26 desc fail\n"); errors++;
    }
}
```

- [ ] **Step 3: Run YOLO full simulation**

Run:

```bash
touch .yolo_ddr
bash run_all.sh sim yolo_full_stem.c yolo_c2f.c yolo_ops.c npu_desc.c
```

Expected:

```text
YOLO FULL NET PASS
ALL TESTS PASSED.
```

- [ ] **Step 4: Commit**

```bash
git add firmware/yolo_full_stem.c
git commit -m "Launch SPPF convs through descriptors"
```

---

### Task 6: Add a MNIST Descriptor Smoke Without Changing Baseline

**Files:**
- Modify: `firmware/deepnet_deploy.c`
- Modify: `run_all.sh` if `npu_desc.c` is not globally linked.

**Interfaces:**
- Consumes: `npu_desc_run_many`
- Produces: MNIST build includes descriptor runtime and proves it does not perturb deployment.

- [ ] **Step 1: Include descriptor header in MNIST firmware**

Add:

```c
#include "npu_desc.h"
```

- [ ] **Step 2: Add a no-op descriptor smoke before MNIST inference**

Near the start of `usercode7()` in `deepnet_deploy.c`, after the first banner print, add:

```c
{
    const npu_desc_t smoke[] = {
        { .op = NPU_DESC_OP_NOP },
        { .op = NPU_DESC_OP_NOP }
    };
    if (!npu_desc_run_many(smoke, 2u)) {
        print_str("DESC SMOKE FAIL\n");
        return;
    }
}
```

- [ ] **Step 3: Run MNIST regression**

Run:

```bash
rm -f .yolo_ddr
bash run_all.sh sim
```

Expected:

```text
=== Result: 10/10 correct ===
DEPLOY SUCCESS.
TRAP after 941155 clock cycles
ALL TESTS PASSED.
```

The exact `TRAP` may increase by a tiny number due to two NOP descriptors. If the project requires the exact old cycle count, move this smoke under `#ifdef DESC_SMOKE`.

- [ ] **Step 4: Commit**

```bash
git add firmware/deepnet_deploy.c run_all.sh
git commit -m "Add MNIST descriptor runtime smoke"
```

---

### Task 7: Document Current Capability and Limits

**Files:**
- Modify: `docs/notes/RESUME-yolov8n-soc.md` or create `docs/notes/descriptor-runtime-v1.md`

**Interfaces:**
- Consumes: final verified command outputs.
- Produces: project-facing explanation that hardware is generic, v1 runtime is descriptor-backed for selected paths, and 640 support still needs memory planning.

- [ ] **Step 1: Create docs note**

Create `docs/notes/descriptor-runtime-v1.md`:

```markdown
# Descriptor Runtime V1

Descriptor runtime v1 adds a firmware-level layer descriptor interface above the
existing NPU/DMA helper functions. It does not change the RTL programming model:
the descriptor executor still writes the same NPU MMIO registers and calls the
same generic data movement and convolution helpers.

Verified scope:

- YOLO neck upsample/copy movement can be launched from descriptors.
- YOLO SPPF conv25/conv26 can be launched from descriptors.
- MNIST includes a descriptor runtime smoke and still reaches 10/10 correctness.

Non-goals for v1:

- Full graph import.
- Automatic DDR memory planning.
- Full 640x640 YOLO migration.
- Removing the existing hand-written fallback paths.

Why this matters:

The SoC hardware already exposes generic Conv/GEMM/Pool/Upsample/DFL/Elementwise
and DMA capabilities. The descriptor runtime starts moving the deployment model
from per-network handwritten scheduling toward a reusable CPU+NPU runtime.
```

- [ ] **Step 2: Commit docs**

```bash
git add docs/notes/descriptor-runtime-v1.md
git commit -m "Document descriptor runtime v1 scope"
```

---

## Final Verification

- [ ] **Run YOLO default regression**

```bash
touch .yolo_ddr
bash run_all.sh sim yolo_full_stem.c yolo_c2f.c yolo_ops.c npu_desc.c
```

Expected:

```text
YOLO FULL NET PASS
ALL TESTS PASSED.
```

- [ ] **Run MNIST default regression**

```bash
rm -f .yolo_ddr
bash run_all.sh sim
```

Expected:

```text
=== Result: 10/10 correct ===
DEPLOY SUCCESS.
ALL TESTS PASSED.
```

- [ ] **Check git status**

```bash
git status --short
```

Expected: only pre-existing unrelated dirty files remain.

---

## Self-Review

- Spec coverage: descriptor data model, data movement, conv dispatch, YOLO migration, MNIST safety, docs, and final verification are each covered by a task.
- Placeholder scan: no TODO/TBD placeholders remain.
- Type consistency: `npu_desc_t`, `npu_desc_run`, and `npu_desc_run_many` signatures are consistent across tasks.
- Scope check: v1 intentionally does not implement full 640x640 dynamic shape support; it creates the descriptor foundation needed for that later.
