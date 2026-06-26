# Hardware Descriptor Queue V1 Design

## Goal

Add a hardware descriptor queue so the CPU can submit an NPU command stream with
one start instead of programming every layer through many MMIO writes.

The first implementation target is one MNIST image inference per queue start.
The architecture and descriptor format must remain generic enough for later YOLO
subgraphs, especially SPPF, neck movement, tiled conv, C2F blocks, and head ops.

## Non-Goals For V1

- No YOLO full-network migration in v1.
- No dynamic graph importer.
- No branches, loops, or conditional execution in the hardware queue.
- No automatic DDR memory planner.
- No removal of the existing direct MMIO firmware path.
- No RTL hardcoding of MNIST layer numbers or MNIST-specific addresses.

## High-Level Architecture

```text
CPU
  writes NPU_DESC_BASE
  writes NPU_DESC_COUNT
  writes NPU_DESC_CTRL.start
  waits for descriptor_done IRQ

descriptor_engine.v
  reads fixed-size descriptors from DDR through the existing 128-bit AXI path
  decodes op/flags/shape/address fields
  programs internal NPU/DMA/copy/expand configuration path
  loads qparams when requested
  starts the selected engine
  waits for done
  advances to the next descriptor

npu_top
  arbitrates CPU MMIO direct mode vs descriptor-engine mode
  preserves all existing direct-call behavior when descriptor engine is idle
```

The queue is a linear command stream. One descriptor completes before the next
one starts. Later versions may prefetch or overlap setup, but v1 keeps ordering
simple for correctness.

## New MMIO Registers

Add registers in the NPU register map above the existing performance counter
region, or in another clearly unused range after checking `firmware/firmware.h`
and `rtl/param_regfile.v`.

```text
NPU_DESC_BASE_LO   descriptor list byte address in DDR
NPU_DESC_BASE_HI   reserved for future 64-bit address support, write 0 in v1
NPU_DESC_COUNT     number of descriptors
NPU_DESC_CTRL      bit0=start, bit1=abort, bit2=irq_en
NPU_DESC_STATUS    bit0=busy, bit1=done, bit2=err, bit3=aborted
NPU_DESC_PC        current descriptor index
NPU_DESC_ERR       error code
```

`NPU_DESC_STATUS.done` raises the existing NPU IRQ bit when `irq_en` is set. The
firmware acknowledges completion by writing a clear bit in `NPU_DESC_CTRL` or by
writing the existing NPU clear-done path if that is cleaner in the current IRQ
scheme. The final implementation plan must pick one clear mechanism and keep it
consistent with `irq.c`.

## Descriptor Format

Use a fixed 16-word descriptor. The first four 32-bit words are fetched as one
128-bit AXI beat. The full descriptor is four 128-bit beats.

All unused fields must be zero. The descriptor version is explicit so YOLO phase
2 can extend behavior without guessing old layouts.

```text
word  0: [7:0] op, [15:8] version, [31:16] flags_lo
word  1: flags_hi / ctrl_flags
word  2: src0 byte address or Act SRAM base, op-defined
word  3: src1 byte address or Wgt SRAM base, op-defined
word  4: dst byte address or Out/Act SRAM base, op-defined
word  5: scratch0, op-defined
word  6: scratch1, op-defined
word  7: words / element count / descriptor-specific count
word  8: in_w[15:0], in_h[31:16]
word  9: in_c[15:0], out_c[31:16]
word 10: kh[7:0], kw[15:8], stride[23:16], pad[31:24]
word 11: qparam_base byte address in DDR, or 0 if no qparam load
word 12: qparam_count channels
word 13: wgt_words_per_oc
word 14: strip_out_rows[15:0], pad_value[31:16] signed low 16 bits
word 15: next/reserved, must be 0 in v1
```

Address convention:

- DDR addresses are byte addresses matching existing firmware constants.
- SRAM base fields are 128-bit word indexes matching existing NPU registers.
- `qparam_base` points to packed qparam records in DDR.

## Opcodes

V1 must implement only the opcodes needed by MNIST and the generic substrate.
YOLO opcodes are reserved in the enum now but may return an unsupported-op error
until implemented.

```text
0x00 NOP
0x01 DMA_DDR_TO_ACT
0x02 DMA_ACT_TO_DDR
0x03 DMA_OUT_TO_DDR
0x04 IMG_EXPAND
0x05 SRAM_COPY_OUT_TO_ACT
0x06 CONV2D
0x07 GEMM
0x08 STOP_IRQ

Reserved for YOLO phase 2+:
0x20 UPSAMPLE2X
0x21 MAXPOOL5X5
0x22 ELTWISE_ADD
0x23 DFL
0x24 LUT_LOAD
0x25 SIGMOID_OR_SILU_CFG
```

Unsupported reserved opcodes set `NPU_DESC_STATUS.err`, write
`NPU_DESC_ERR=ERR_UNSUPPORTED_OP`, stop the queue, and optionally raise the done
IRQ as an error completion.

## QParam Loading

Conv/GEMM descriptors may set `qparam_base != 0`. Before starting compute, the
descriptor engine loads `qparam_count` records into the existing resident
parameter regfile.

Packed qparam record:

```text
word0: bias int32
word1: scale_mul uint32
word2: scale_shift uint32
word3: reserved, 0
```

Each record is 16 bytes, so the descriptor engine can read one channel qparam
record per 128-bit beat and write:

```text
NPU_BIAS(ch)
NPU_SCALE(ch)
NPU_SHIFT(ch)
```

For MNIST, qparam tables are emitted by firmware into DDR or included in the
firmware image and copied to the shared-memory model. For YOLO later, the same
format handles per-channel quantization for conv chunks.

V1 only needs to load up to 64 channels per descriptor, matching the current
resident param regfile. If a future YOLO layer has more than 64 output channels,
firmware should emit multiple conv descriptors for OC chunks, exactly like the
current helper path does.

## Descriptor Engine FSM

Recommended top-level states:

```text
IDLE
FETCH_DESC_AR
FETCH_DESC_R
DECODE
QPARAM_AR
QPARAM_R
PROGRAM
START_OP
WAIT_OP
ADVANCE
DONE
ERROR
ABORT
```

Behavior:

- `IDLE`: waits for `NPU_DESC_CTRL.start` when NPU is not already busy.
- `FETCH_DESC_*`: reads four 128-bit beats from `base + pc*64`.
- `DECODE`: validates version, op, count, shape, and alignment.
- `QPARAM_*`: optional qparam read loop.
- `PROGRAM`: drives the same config values currently written by CPU MMIO.
- `START_OP`: pulses the selected trigger/start.
- `WAIT_OP`: waits for the selected done condition.
- `ADVANCE`: increments PC; if PC reaches count, enters `DONE`.
- `DONE`: raises done status/IRQ.
- `ERROR`: records error and stops the queue.
- `ABORT`: stops cleanly after an explicit abort request.

V1 should not overlap qparam loading with compute. Once this version passes, a
future version can add descriptor prefetch or shadow config banks to hide setup
behind compute.

## Integration With Existing Blocks

### CPU Direct MMIO Compatibility

The existing direct MMIO path remains the default when `desc_busy=0`.
Descriptor-engine writes must not corrupt CPU-visible registers during direct
mode.

The simplest implementation is an internal config mux in `npu_top`:

```text
if desc_busy:
  config source = descriptor_engine
else:
  config source = param_regfile CPU MMIO registers
```

For v1, descriptor mode should reject start if the regular NPU core, DMA, copy,
or expand engine is already busy.

### DMA / Copy / Expand

Descriptor op mapping:

- `DMA_DDR_TO_ACT`: program `NPU_DMA_*` fields and wait for rd_done.
- `DMA_ACT_TO_DDR`: program write path from Act and wait for wr_done.
- `DMA_OUT_TO_DDR`: program write path from Out and wait for wr_done.
- `IMG_EXPAND`: program Act source/destination/count and wait for expand_done.
- `SRAM_COPY_OUT_TO_ACT`: program Out->Act source/destination/count and wait for
  copy_done.

These ops should reuse existing engines rather than duplicating data movers.

### CONV2D

Descriptor fields map to existing NPU compute config:

```text
in_w/in_h/in_c/out_c
kh/kw/stride/pad
src0      Act SRAM base
src1      Wgt SRAM base
dst       Out SRAM base
flags     NPU_CTRL_* bits such as RELU, POOL, HW_PAD, ROW_PAR, ROW_BLOCK, OC_SINGLE
qparams   optional bias/scale/shift load
```

The engine starts `NPU_CTRL_START | flags`, waits for NPU done, then advances.

### GEMM

GEMM uses the same shape/address fields with `flags` including `NPU_CTRL_GEMM`
and optionally `NPU_CTRL_GEMM_REDUCE` or `NPU_CTRL_INT32_OUT`.

For FC2, descriptor done must wait until INT32 output serialization is complete,
matching the fixed direct MMIO semantics.

## MNIST V1 Command Stream

Firmware emits one descriptor list per image:

```text
DMA_DDR_TO_ACT        raw 49 words -> Act scratch
IMG_EXPAND            raw scratch -> Conv1 Act region
CONV2D                Conv1
SRAM_COPY_OUT_TO_ACT  Conv1 out -> next Act region
CONV2D                Conv2 + Pool1
SRAM_COPY_OUT_TO_ACT
CONV2D                Conv3
SRAM_COPY_OUT_TO_ACT
CONV2D                Conv4 + Pool2
SRAM_COPY_OUT_TO_ACT
CONV2D                Conv5
SRAM_COPY_OUT_TO_ACT
CONV2D                Conv6 + Pool3 -> FC1 Act region
GEMM                  FC1 -> FC1_OUT staging
DMA_OUT_TO_DDR or SRAM_COPY equivalent for FC1 staging, depending current path
DMA_DDR_TO_ACT        FC1 staging -> FC2 input, unless v1 keeps FC1->FC2 on-chip
GEMM                  FC2 INT32_OUT
DMA_OUT_TO_DDR        FC2 logits -> DDR
STOP_IRQ
```

The exact FC1->FC2 handoff should mirror the current verified firmware path in
v1. A later optimization can keep FC1->FC2 entirely on-chip once the queue
engine is correct.

CPU loop for 10 images:

```text
build/update descriptor list for image d
write NPU_DESC_BASE
write NPU_DESC_COUNT
write NPU_DESC_CTRL.start | irq_en
wait descriptor_done IRQ
read FC2 logits
argmax/check label
```

This still uses CPU for argmax and checking, but removes per-layer CPU register
scheduling.

## YOLO Compatibility Requirements

Even though YOLO migration is not part of v1 implementation, the v1 descriptor
format must preserve fields needed by YOLO:

- `flags` carries existing `NPU_CTRL_*` feature bits such as pointwise,
  depthwise, average/global pooling, SiLU/exact activation, row parallelism, and
  OC single-start.
- `scratch0/scratch1` are generic and can name Act bases, pad-row DDR, psum DDR,
  LUT base, or temporary buffers in later opcodes.
- `wgt_words_per_oc` and `strip_out_rows` are present for tiled conv.
- `pad_value` is present for YOLO quantized pad handling.
- reserved opcodes cover upsample, maxpool5x5, eltwise add, DFL, and LUT loads.

YOLO phase 2 should start with the already firmware-descriptorized SPPF and neck
movement path, not the full graph.

## Error Handling

Minimum v1 errors:

```text
ERR_NONE
ERR_BAD_VERSION
ERR_BAD_OPCODE
ERR_UNSUPPORTED_OP
ERR_BAD_COUNT
ERR_BAD_ALIGNMENT
ERR_BAD_SHAPE
ERR_BUSY_AT_START
ERR_AXI_DESC_READ
ERR_AXI_QPARAM_READ
ERR_ENGINE_TIMEOUT
```

On error:

- stop the queue
- set `NPU_DESC_STATUS.err`
- write `NPU_DESC_ERR`
- leave `NPU_DESC_PC` at the failing descriptor
- raise IRQ if enabled

Firmware prints the failing PC and error code.

## Verification Plan

Directed RTL tests first:

- Descriptor fetch reads four 128-bit beats and assembles 16 words correctly.
- NOP + STOP_IRQ queue completes.
- Bad version/op/count set the expected error.
- Qparam loader writes bias/scale/shift for 16 and 64 channels.
- DMA_DDR_TO_ACT descriptor performs a known small copy.
- SRAM_COPY_OUT_TO_ACT descriptor performs a known small copy.
- IMG_EXPAND descriptor matches the existing raw->tile-major output.
- Tiny CONV2D descriptor matches the existing direct `npu_top` integration
  harness.
- Tiny GEMM descriptor, including INT32_OUT completion, matches direct mode.

Integration tests:

- MNIST direct path still reaches `10/10 correct`.
- MNIST descriptor path reaches `10/10 correct`.
- Descriptor path prints `DEPLOY SUCCESS.` and `ALL TESTS PASSED.`
- Compare cycle counters:
  - CPU-visible total trap cycles
  - descriptor busy cycles
  - NPU busy cycles
  - setup/qparam cycles

## Firmware Plan

Add a new build-time switch:

```text
NPU_HW_DESC=1
```

When off, `deepnet_deploy.c` keeps the existing direct helper path.

When on:

- generate qparam tables in descriptor-compatible packed form
- generate one descriptor list for each image
- submit queue through descriptor MMIO registers
- wait for descriptor IRQ
- read FC2 logits and run argmax

The old direct helper path remains the fallback until descriptor mode has stable
regression evidence.

## Performance Story

The expected improvement is not from faster MAC execution. It is from removing
or hiding CPU per-layer scheduling overhead:

```text
before:
  CPU writes many NPU/DMA/qparam registers per layer
  NPU waits between layers

after:
  CPU submits descriptor queue once
  hardware programs the next op locally
  CPU waits only for queue completion
```

V1 may still spend cycles loading qparams serially, but those cycles are now
measurable inside the hardware queue and no longer burn CPU instruction time.
Later versions can add qparam prefetch and shadow config banks to hide setup
behind compute.

## Acceptance Criteria

- No RTL MNIST layer-number hardcoding.
- Existing direct MNIST path still passes.
- Descriptor MNIST path passes `10/10 correct`, `DEPLOY SUCCESS.`, and
  `ALL TESTS PASSED.`
- Descriptor engine reports useful PC/error status on malformed descriptors.
- Descriptor format contains reserved YOLO fields/opcodes without requiring YOLO
  phase 2 implementation.
- New RTL files are added to `axi_sys.f`.
- Register map updates are reflected in both `rtl/param_regfile.v` and
  `firmware/firmware.h`.
