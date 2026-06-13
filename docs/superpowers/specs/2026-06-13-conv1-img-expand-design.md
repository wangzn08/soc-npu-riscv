# Conv1 Image Formatting via HW `img_expand` Engine — Design Spec

> **Status:** design approved. Next: writing-plans → implementation (inline,
> incremental, bit-identical gate 10/10).

## Context & motivation

Conv1's input formatting (`prof_pad`, ~101K cycles/image, the largest per-image
inference item after SRAM residency) is done by the CPU in two DDR loops:
```c
for (i=0; i<3136; i++) abuf[i] = 0;            // zero-fill 784 words (~75% of pad)
for (i=0; i<784;  i++) ((int8*)abuf)[i*16] = input[i];  // scatter pixels (~25%)
```
Both write the DDR shared buffer (`ACT_BUF_A`), ~3920 uncached stores. The NPU needs
the image in **tile-major Act SRAM**: 784 words, each a 16-channel 128-bit word with
the pixel in ch0 and ch1..15 = 0. This is per-image and CPU-bound — the target.

**Two observations drive the design:**
1. **The zero-fill is unnecessary.** Conv1 has IC=1 (only ch0 is a real input channel);
   its packed weights for channels 1..15 are zero, so those activation lanes are
   multiplied by 0 and never affect the result. Act SRAM ch1..15 may be garbage.
   ⇒ delete the zero-fill (free, ~75K/image).
2. **The scatter can move to hardware.** The remaining work — place each image byte into
   ch0 of a zero word — is a fixed "1 byte → 1 zero-extended 16-ch word" expansion, a
   natural HW primitive.

The image (`mnist_images[d]`) lives in **private RAM** (firmware .rodata at 0x0), which
the NPU DMA cannot read, so the raw bytes must reach the DDR shared mem first. The CPU
does one **contiguous** copy of the 784 bytes to DDR (~196 word-writes, ~5K — the fast
kind, not the strided scatter), then HW does the expansion.

**Saving:** pad ~101K → ~6K/image (zero-fill deleted + scatter moved to HW, residual =
the ~5K contiguous copy + ~1K expand). Per-image inference ~289K → ~195K (**~−33%**),
bit-identical 10/10. Generality: the engine is a generic "1 byte → 1 zero-extended
word" primitive (any length via RD_LEN); only Conv1 triggers it.

## Confirmed datapath facts

- Conv1 input is read by the FSM from Act SRAM at `NPU_ACT_ADDR_A = 0` (region R0 in the
  residency design). The `img_expand` output therefore targets Act word base 0.
- Act SRAM **Port B read is combinational** (`COMB_B=1`): `act_sram_dob = mem[addrb]`
  same cycle. Port B is a single port (read OR write per cycle).
- `npu_top` already muxes Act Port B (DMA write, DMA read, and `copy_busy` from the
  residency `sram_copy`). The expander adds another priority arm (`expand_busy`).
- The residency `sram_copy` engine + the `0x154` COPY_TRIG / `STATUS[2]` pattern are the
  template for the new engine's trigger/poll plumbing.
- Act regions in use (PING bank): R0=word 0, R1=word 1024 (residency ping-pong). The
  expander's packed-image **scratch** uses a free region (word 2048, 49 words).

## Architecture

### Data flow (Conv1 input)

1. **CPU (firmware):** copy the 784 image bytes from private RAM to a DDR buffer
   `IMG_BUF` as packed contiguous words (196 `int32` writes). No zero-fill, no scatter.
2. **`dma_ddr_to_act` (existing, unchanged):** DMA `IMG_BUF` → Act SRAM **scratch**
   region (word 2048), 49 beats (784 bytes = 49×16). Packed: 16 bytes per word.
3. **`img_expand` engine (new):** read the 49 packed words from Act scratch, write 784
   zero-extended words to Act R0 (word 0): output word `i` = `{120'b0, scratch[i/16]
   byte (i%16)}`.
4. **Conv1 (unchanged):** runs reading Act R0 = 0.

### `img_expand` engine (`rtl/img_expand.v`)

A small FSM, isolated from `axi_dma` and from `sram_copy`. Because Act Port B is a
single port, it reads one packed word then writes its 16 expanded words:

- **READ (1 cycle):** drive Act Port B `addrb = src_base + w`, `enb=1`, `web=0`; latch
  `act_sram_dob` (128 bits = 16 bytes).
- **WRITE (16 cycles):** for `b = 0..15`, drive `addrb = dst_base + w*16 + b`, `enb=1`,
  `web=1`, `dib = {120'b0, latched[b*8 +: 8]}`.
- Repeat for `w = 0 .. ceil(n_out/16)-1`; `n_out` = output word count (=784). Assert
  `expand_busy` throughout; raise `expand_done` at the end. ~17 cycles/word ⇒ ~833
  cycles for Conv1.
- Reads and writes the same Act SRAM (scratch + R0, both PING) on different cycles — no
  same-cycle Port-B conflict; scratch (2048..2096) and R0 (0..783) don't overlap.

### Registers (reuse + one trigger)

- **src** (Act scratch word base): reuse `NPU_DMA_RD_SRAM_BASE`.
- **dst** (Act output word base): reuse `NPU_DMA_WR_SRAM_BASE`.
- **n_out** (output word count): reuse `NPU_DMA_RD_LEN`.
- **New:** `NPU_EXPAND_TRIG` (0x158) → regfile `o_expand_trig` pulse; completion polled
  via `NPU_DMA_STATUS[3]` (`expand_done`) — no IRQ (avoids start7.S mask plumbing).

### npu_top Port-B mux

Extend the Act Port B mux with the expander as a priority arm (mutually exclusive with
copy/DMA in time): `expand_busy ? expand_* : copy_busy ? copy_* : <dma mux>`. The
expander drives `act_sram_addrb/enb/web/dib` (writes) and reads `act_sram_dob`.

### Firmware

- Replace the Conv1 formatting block (zero-fill + scatter) with:
  1. contiguous copy `input[784]` → `IMG_BUF` (DDR), 196 words.
  2. `dma_ddr_to_act(IMG_BUF, SCRATCH=2048, 49)`.
  3. `img_expand(dst=0, src=2048, n_out=784)` helper: set src/dst/n, pulse
     `NPU_EXPAND_TRIG`, poll `STATUS[3]`.
- The old `dma_ddr_to_act(ACT_BUF_A, 0, 784)` after the block is removed (the expander
  produces Conv1's input directly at R0).

## Scope

- Conv1 image formatting only. Conv2..Conv6 inputs are already resident (decision J) —
  untouched. The expander is generic but only Conv1 uses it (IC=1 input stage).
- Not in scope: pre-loading images into DDR offline (would remove even the ~5K CPU
  copy, but is sim-specific; the contiguous copy models real sensor→DDR input).

## Generality

`img_expand` is a register-driven "1 byte → 1 zero-extended 128-bit word" primitive
parameterized by src/dst/n_out — works for any single-channel input of any size. It is
a hardware width primitive (16 bytes/word = the SRAM width), not a model dimension. When
not triggered, the datapath is byte-identical to today.

## Verification

- **Bit-identical gate:** Conv1's output (hence Pool1, predictions) must stay byte-for-
  byte identical → 10/10, ALL TESTS PASSED, Pool1 nz 711/379/784/838 byte-match golden.
  (The expander produces the same Act SRAM contents the CPU formatting did, modulo the
  don't-care ch1..15.)
- **Incremental bring-up:** (a) first delete only the zero-fill, verify 10/10 (confirms
  ch1..15 are don't-cares); (b) then add the engine + staging, verify 10/10.
- **Profile:** `-DNPU_PROFILE=1` — confirm `prof_pad` drops from ~1.01M to ~60K (×10),
  report per-image inference vs the ~289K baseline.
- Build/sim via the PowerShell tool (`bash run_all.sh sim`; Bash tool runs from the
  wrong dir). RTL change → `bash run_all.sh clean` first.

## Risks

- **ch1..15 don't-care assumption:** if Conv1's packed weights for channels 1..15 are
  NOT zero, deleting the zero-fill corrupts the result. Bring-up step (a) verifies this
  in isolation before building the engine.
- **Port-B contention:** the expander shares Act Port B with `sram_copy` and `axi_dma`.
  They're sequential (firmware never overlaps), so a priority mux suffices. The expander
  reads and writes Act on different cycles (single port).
- **Scratch/region aliasing:** scratch (2048) must not overlap R0 (0..783), R1
  (1024..1807), or anything live. 2048 is free in the PING bank.
- Lower-risk than residency (no compute-path change; the engine only fills Act SRAM,
  which the FSM reads as before; `axi_dma` and `sram_copy` untouched).

## Files

- New: `rtl/img_expand.v` (the engine); add to `axi_sys.f`.
- Modify: `rtl/npu_top.v` (instantiate, Act Port-B mux, status), `rtl/param_regfile.v` +
  `firmware/firmware.h` (EXPAND_TRIG 0x158 + STATUS[3]), `firmware/deepnet_deploy.c`
  (replace Conv1 formatting: contiguous copy + stage DMA + img_expand; remove zero-fill
  + scatter + the old 784-word load).
- Unchanged: im2col, systolic/post/FSM, max_pooling, wgt_reader, axi_dma, sram_copy,
  SRAM wrappers.
