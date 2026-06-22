# Conv large-IC streaming via INT32 psum accumulation (design)

Date: 2026-06-22. Branch: `codex/yolov8n-rtl-m0`.
Status: design (approved approach: performance-first, INT32 psum accumulate in Out SRAM).

## Problem

A 3x3 (im2col) conv with `ic_groups > ICG_MAX (=4)` produces wrong output: the
`im2col_line_buffer` window holds only `ICG_MAX=4` IC tiles, so tiles 4+ read
garbage. First seen on YOLOv8n c2f_8 bottleneck (3x3, half_c=128 => icg=8): the
stage-checksum smoke shows CV1 OK but ADD0 = all 0x80 (saturated). Proven to be
the activation window, not the weight buffer: forcing the 3x3 onto the PW
weight-streaming path changed cycle count (27.2M->23.4M) but produced
byte-identical wrong output.

Related already-fixed sibling: 1x1 PW large-IC (cv2 IC=128, icg=8) was fixed by
per-IC-group **weight** streaming (drop `|| i_pw_en` from `reuse_mode`; firmware
tiles OC<=16 without oc_single). PW has no im2col, so the window limit does not
apply there. This spec covers the im2col (3x3 / general kernel) case.

Constraints discovered:
- The systolic array holds the psum for only the current output position(s)
  (1, or 16 under row_par); it cannot hold a whole strip's psums across IC chunks.
- `int32_out` (CTRL[13]) is FC-single-position only: its serializer captures one
  output write pulse and ignores further pulses while serializing 4 words; its
  `i32_base = fsm_out_wr_addr` is not ×4-spaced for multiple positions. So the
  existing INT32 path cannot drain a spatial conv as-is.
- Firmware IC-tiling with INT32 readback is therefore not possible today.

## Goal

Make any conv with `ic_groups > ICG_MAX` produce bit-identical output to a
single-pass conv, by streaming IC in chunks of `<= ICG_MAX` tiles, accumulating
INT32 partial sums in an Out-SRAM psum region across chunks, and applying
bias + requant + SiLU only on the last chunk. OFF (icg<=ICG_MAX) ⇒ byte-identical
to today. General over any model (capacity-knob driven, per
[[soc-npu-general-purpose]]).

## Architecture

### Dataflow (FSM, when `ic_groups > ICG_MAX`)

IC chunk becomes the OUTER loop; OC is tiled to <=16 so the INT32 psum fits Out
SRAM; oc_single is OFF (it forces full-IC weight residency). Per OC tile:

```
for ic_chunk c in 0 .. ceil(icg/ICG_MAX)-1:
    LOAD_ROW: load chunk c's <=ICG_MAX IC tiles' rows into the line buffer
              (full row-strip; spatial line-buffer reuse preserved WITHIN a chunk)
    sweep all spatial positions (cur_ox/cur_oy, row_par allowed):
        CALC: ko-loop x chunk's IC tiles -> array psum (this chunk's contribution)
        post_process accumulate mode:
            c == 0        : ACC_FIRST -> write INT32 array psum to OutSRAM psum region
            0 < c < last  : ACC_ADD   -> read OutSRAM psum + array psum -> write INT32
            c == last     : ACC_FINAL -> read OutSRAM psum + array psum + bias,
                                         requant (scale>>shift) + SiLU LUT -> INT8 out
```

The array psum is drained per position per chunk (existing K_END/DRAIN). The
cross-chunk accumulator lives in Out SRAM, not the array.

### Out-SRAM psum region

INT32 layout mirrors `int32_out`: per OC-group(16) per output position = 16×INT32
= 4×128-bit words. Psum region size = `out_spatial * 4` words per OC tile. With
OC tiled to 16 and a bounded strip, this fits the 128KB Out SRAM. The FINAL pass
overwrites with the compact INT8 output (1 word per OC-group per position) in the
normal Out-SRAM output region, which the firmware then DMAs to DDR as usual.

Open sub-decision (resolve in plan): reuse Out SRAM for psum (4× footprint, must
bound strip/OC so it fits) vs add a dedicated Psum SRAM. Lean: reuse Out SRAM +
tile OC to 16 + cap strip rows so `strip*out_w*4 <= OUT_SRAM_DEPTH/2`.

### post_process accumulate modes

New FSM-driven control (e.g. `acc_mode[1:0]`: NONE/FIRST/ADD/FINAL), OFF ⇒ legacy
path byte-identical. Reuses the existing s1 (psum+bias) / s2 ((s1*scale)>>shift) /
SiLU-LUT stages; ACC_FINAL is exactly the legacy post-process but with
`psum_in = array_psum + outsram_psum_readback`. ACC_FIRST/ACC_ADD bypass bias and
s2/SiLU and emit raw INT32 (= array_psum [+ readback]) to the psum region.

### INT32 spatial write + readback

- WRITE: generalize the INT32 serializer so consecutive output positions write to
  ×4-spaced Out-SRAM addresses and back-to-back position pulses are not dropped
  (positions in a conv sweep are tens of cycles apart, so the 4-cycle serialize
  fits, but the address spacing and the `!i32_active` gate must be made
  position-aware).
- READBACK: ACC_ADD/ACC_FINAL read the prior INT32 psum (4 words / OC-group /
  position) from Out SRAM (Port A `doa`, or Port B), pipelined ahead of the
  post-process add. Out SRAM is `sdp_bram` with a Port A read output (`doa`);
  confirm read/write timing (registered doa, COMB_B convention) in the plan.

### Firmware

`yolo_run_conv2d_tiled`: re-add the streamed conv arm (icg>ICG_BUF): tile OC<=16,
drop oc_single, mask row_par if needed, and drive the IC-chunk outer loop +
acc-mode config (or expose the chunking to the FSM via a new "ic_stream" CTRL bit
and let HW loop). Keep `YOLO_ICG_BUF`/`ICG_MAX` mirrors in sync.

## Capacity / counter widths

- `i_ic_groups` is `[3:0]` today (caps ~16). Widen the IC-group counters
  (`i_ic_groups`, `pf_icg`, ic-tile selects) as needed for the target max icg.
- YOLOv8n actual max: 3x3 convs <= IC 256 (icg=16); 1x1 PW up to IC 512 (icg=32,
  handled by weight streaming, no im2col). Size knobs accordingly.

## Testing

- Directed TB on `npu_top` (extend `tests/tb_npu_integ.v`): a 3x3 conv with
  icg=8 and icg=16, golden-checked, ACC modes OFF vs ON bit-identical where icg<=4.
- Integration: `yolo_c2f8_320_exact_smoke.c` (stage-checksum: CV1/ADD0/CONCAT/OUT
  must all be OK; ~27s sim). Then c2f6 (regression), c2f4 (regression), MNIST 10/10.
- Gate: ACC OFF path must be byte-identical to current RTL (MNIST 941,155 cyc).

## Out of scope

- Stride>1 large-IC interactions beyond what conv already supports.
- FPGA BRAM mapping of the psum region (sim-first; note the COMB_B / registered
  read caveat from CLAUDE.md when porting).

## References

- `rtl/im2col_line_buffer.v` (ICG_MAX), `rtl/top_controller_fsm.v` (reuse_mode,
  S_CALC_KERNEL IC-tile loop, S_POST), `rtl/post_process_top.v` (s1/s2/SiLU,
  int32_out / s2_quant), `rtl/npu_top.v` (int32 write sequencer ~line 1160).
- Sibling fix + history: `docs/notes/RESUME-yolov8n-soc.md`,
  memory `[[soc-npu-weight-reuse]]`, `[[project_yolov8n_soc_resume]]`.
