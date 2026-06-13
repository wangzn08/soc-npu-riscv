# On-Chip SRAM Residency for conv→conv Boundaries — Design Spec

> **Status:** design approved (sections 1–4). Next: writing-plans → implementation
> (subagent-driven). Incremental bring-up behind a per-boundary firmware switch,
> bit-identical gate (10/10).

## Context & motivation

Today every conv layer round-trips its result through DDR: layer N drains Out SRAM
→ DDR (`dma_out_to_ddr`), then layer N+1 loads DDR → Act SRAM (`dma_ddr_to_act`)
before computing. The CPU does **nothing** between conv layers anymore (hardware
padding, decision H, moved padding into the FSM), so this DDR bounce is now **pure
transport overhead**.

**Confirmed layout match:** the Out SRAM output (post-process writes tile-major:
OC-tile-major spatial, 16-channel/128-bit words) is **byte-identical** to what the
next layer expects in Act SRAM (IC-tile-major spatial, same word format). The data
that round-trips through DDR is exactly what the next layer needs — so an on-chip
**Out SRAM → Act SRAM copy** replaces the round-trip with no reformatting.

**Saving (measured-data estimate, `NPU_OC_OVERLAP=1`):** the input load
(`prof_load`) is fully serial and not hidden by the intra-layer OC-pass overlap;
the conv→conv portion is ~33K cycles/image. The output drain is mostly hidden by
overlap already. Replacing the DDR round-trip with an on-chip copy (~1 cyc/word vs
~19 cyc/word for DDR) saves the load and the unhidden drain tail, minus a small copy
cost. **Net ≈ 30–40K cycles/image (~8–11% of the 374K/image inference)**, composes
with task E, doesn't touch preload/cold-start. Exact figure verified post-implementation.

## Confirmed datapath facts (verified this session)

- `post_process_top` → Out SRAM Port A write, tile-major (`out_wr_addr` non-pool;
  contiguous `pool_out_addr_cnt` pooled). The pooled output is contiguous = tile-major
  for the single OC tile, and per-pass for multi-OC layers — matches Act read layout.
- FSM/im2col read input from **Act SRAM Port A** via `tilemaj_addr` (hw_pad reads the
  unpadded tile-major stream and injects border zeros in the FSM). **Unchanged by this
  feature** — the copy only changes how Act SRAM gets *filled*, not how it's read.
- `axi_dma` bridges **DDR↔SRAM only**; there is no Out→Act path today. The copy is the
  new on-chip path.
- SRAM Port B is already muxed in `npu_top` for DMA: Act Port B (`act_sram_addrb/enb/web/dib`)
  is DMA-write; Out Port B (`out_sram_addrb/enb`, data `out_sram_dob`) is DMA-read.
  The copy engine time-shares these Port-B paths with `axi_dma`.
- Bank decoupling (Issue C, reg `0x14C`): `dma_out_ping_sel`[2] selects which Out bank
  Port B reads; `dma_act_ping_sel`[0] selects which Act bank Port B writes — exactly
  the two selects the copy needs.

## Architecture

### 1. Hardware copy engine (`sram_copy`, in `npu_top`)

A small dedicated FSM, **isolated from the proven `axi_dma` DDR path** (risk
containment):

- On trigger, for `i = 0 .. len-1`: read **Out SRAM Port B** `[src_base + i]`, and one
  cycle later (SRAM read latency) write **Act SRAM Port B** `[dst_base + i]` with the
  read data. Assert `copy_busy` during; pulse/raise `copy_done` at the end.
- **Port-B arbitration:** when `copy_busy`, the copy engine drives Out Port B
  (addr/en) and Act Port B (addr/en/we/data); otherwise `axi_dma` drives them (current
  behavior). Copy and DMA are mutually exclusive in time (firmware never overlaps
  them), so a simple `copy_busy ? copy : dma` mux on the Port-B signals suffices.
- Banks: reads Out bank = `dma_out_ping_sel`, writes Act bank = `dma_act_ping_sel`
  (reuse Issue-C selects).

### 2. Register interface (reuse + one trigger)

No new register block — reuse the existing DMA SRAM-base/len/bank registers:
- **src** (Out SRAM word base): reuse `NPU_DMA_RD_SRAM_BASE`.
- **dst** (Act SRAM word base): reuse `NPU_DMA_WR_SRAM_BASE`.
- **len** (words): reuse `NPU_DMA_RD_LEN`.
- **banks:** reuse `dma_out_ping_sel` (src) / `dma_act_ping_sel` (dst) in reg `0x14C`.
- **New:** a copy trigger (a new DMA-block offset, e.g. `NPU_DMA_COPY_TRIG`) and a
  **polled** `copy_done`/`copy_busy` STATUS bit. Polling (not IRQ) is chosen to avoid
  the `start7.S` IRQ-mask plumbing (a new IRQ source would need its mask bit cleared);
  the copy is short (~hundreds–2K cycles) so a busy-poll is cheap and simple.

### 3. Bank handling — serial flow needs NO Act ping-pong

Conv reads Act bank = global `cfg_ping_pong_sel` (= PING for all convs; PONG is GEMM).
**The global select couples Act and Wgt banks** (conv weights are resident in Wgt PING),
so we must NOT flip it to alternate Act banks. We don't need to: in the serial inter-
layer flow, layer N fully completes before the copy, and the copy completes before
layer N+1 starts — nothing reads Act PING while the copy writes it. So:

```
layer N      : read Act PING → compute → write Out SRAM (bank = CTRL[6] per OC-pass)
copy (per pass): read Out bank → write Act PING  (no reader active → safe overwrite)
layer N+1    : read Act PING → compute → ...
```

The copy always targets **Act PING** (the conv read bank); each layer's input fully
replaces the previous, so overwriting Act PING every boundary is correct. The Wgt bank
is untouched (stays PING, resident conv weights). Multi-OC-pass layers do **one copy
per pass** (mirroring today's one `dma_out_to_ddr` per pass), reading that pass's Out
bank to `Act PING + pass*out_words`.

### 4. Firmware

- New helper `dma_out_to_act(dst_word, src_word, len_words, out_bank)`: set
  `dma_out_ping_sel=out_bank`, `dma_act_ping_sel=PING`, src/dst/len, pulse the copy
  trigger, poll `copy_done`.
- `npu_conv_pass` gains a destination mode: **copy-to-Act** (resident) vs **DMA-to-DDR**
  (current). Per OC-pass, the resident path calls `dma_out_to_act(Act_PING + pass*out_words,
  Out_bank, out_words)` instead of `dma_out_to_ddr(...)`.
- The **consumer** (next conv) **skips its `dma_ddr_to_act`** — input already in Act PING.
- Caller wires the 5 conv→conv boundaries to resident; the producer for the FC path
  (Conv6) keeps DDR (the CPU reorder needs DDR).

## Scope

- **Resident: 5 conv→conv boundaries** — Conv1→2, Conv2→3, Conv3→4, Conv4→5, Conv5→6.
- **Stays on DDR:** Conv1 input (the test image, CPU-formatted), and **Conv6 → reorder
  → FC** (CPU channels-first reorder needs DDR; FC2/argmax on CPU).
- Pooled boundaries (Conv2→3, Conv4→5) are included — pooled Out SRAM layout is
  contiguous = tile-major, matching the next layer's Act read.

## Generality (per [[soc-npu-general-purpose]])

The copy is a **layout-agnostic word copy** parameterized by src/dst/len/bank — works
for any layer, any dims/IC/OC/tiles, pooled or not. It is a hardware transport
primitive (like the existing DMA), not model-specific. When a boundary uses the DDR
path instead (the mode switch off), behavior is **byte-identical** to today. The "16-
channel word" granularity is the existing SRAM width, not a model dimension.

## Verification

- **Bit-identical gate:** the resident data equals what DDR carried, so the full run
  must stay **10/10, ALL TESTS PASSED**, predictions unchanged, and per-layer spot
  checks (Conv outputs, Pool nz) byte-match the pre-residency golden.
- **Incremental bring-up:** enable residency **one boundary at a time** (Conv1→2 first),
  behind the firmware mode switch, validating bit-identical after each — mirrors the E
  and PAD bring-ups.
- **Profile:** with `-DNPU_PROFILE=1`, confirm `prof_load` drops for the resident
  boundaries and report the per-image inference reduction vs the ~374K baseline.
- RTL change → `bash run_all.sh clean` first. Build/sim via the PowerShell tool
  (`bash run_all.sh sim`; the Bash tool runs from the wrong dir).

## Risks

- **Port-B contention:** the copy and `axi_dma` share Out/Act Port B. Firmware must not
  trigger a copy while a DMA is in flight (it won't — they're sequential). The mux must
  cleanly hand Port B to the copy engine only when `copy_busy`.
- **Bank aliasing:** the copy must write the Act bank the next conv reads (PING) and read
  the Out bank the producing pass wrote. Wrong bank → stale/garbage input. Verified per
  boundary in bring-up.
- **Multi-OC-pass placement:** each pass's output must land at `Act PING + pass*out_words`;
  off-by-one in the per-pass offset corrupts channel tiles. Covered by the bit-identical
  gate on the multi-tile layers (Conv3/4/5/6).
- **Interaction with `NPU_OC_OVERLAP`:** v1 does the inter-layer copy **serially**
  (correctness-first). Overlapping the copy with compute (copy on Port B while compute
  uses Port A, like the current DMA drain overlap) is a **follow-on** optimization, not
  in v1 scope.
- Lower-risk than E (no change to im2col/systolic/post/FSM compute path; the copy only
  fills Act SRAM, which the FSM reads exactly as before).

## Files

- New: a copy engine — either a small module `rtl/sram_copy.v` instantiated in
  `npu_top.v`, or an inline FSM in `npu_top.v` (decide in planning; prefer a separate
  module for testability). Add to `axi_sys.f` if a new file.
- Modify: `rtl/npu_top.v` (instantiate copy engine, Port-B mux, status), `rtl/param_regfile.v`
  + `firmware/firmware.h` (copy trigger + status bit), `firmware/deepnet_deploy.c`
  (`dma_out_to_act`, `npu_conv_pass` destination mode, wire the 5 boundaries, skip the
  consumer loads).
- Unchanged: im2col, systolic/gp/pe, wgt_reader, post_process, max_pooling, top_controller_fsm,
  axi_dma (DDR path untouched), SRAM wrappers.
