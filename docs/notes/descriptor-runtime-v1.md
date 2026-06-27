# Descriptor Runtime V1

Descriptor runtime v1 adds a firmware-level layer descriptor interface above the
existing NPU/DMA helper functions. It does not change the RTL programming model:
the descriptor executor still writes the same NPU MMIO registers and calls the
same generic data movement and convolution helpers.

Verified scope:

- YOLO neck upsample/copy movement can be launched from descriptors.
- YOLO SPPF conv25/conv26 can be launched from descriptors (npu_desc struct
  wrapper, CPU-helper backend).
- MNIST includes a descriptor runtime smoke and still reaches 10/10 correctness.

Hardware-descriptor-queue conv migration (yolo_run_conv2d_tiled_desc) — full-net
yolo_full_stem.c, each batch re-confirmed YOLO FULL NET PASS 4/4 on RTL:

- Backbone standalone convs conv0/1/6/13/20 (commit "Migrate full-net backbone
  convs"), 41.2M cyc.
- Neck downsample convs conv35 (3x3 s2 icg4) / conv46 (3x3 s2 icg8), 40.85M cyc.
- Head non-ic_stream path in run_head_conv: small 3x3 stems/mids (icg<=4) and
  1x1 linear-out convs (icg4/5), 40.54M cyc.

All migrated convs are directly analogous to a standalone desc smoke (icg<=8 3x3,
icg<=5 1x1 PW). Still on non-desc-queue paths (need runtime capability validation
before migrating, NOT just a rename):

- conv25/conv26: large-IC 1x1 PW (IC=256/512); desc PW only proven at icg2.
- run_head_conv ic_stream branch: large-IC 3x3 head stems 47/48 (icg8), 57/58
  (icg16 — at the ICG_MAX=16 boundary, unproven on desc), cls mids 39/50/60 (icg5).
- c2f-internal convs (yolo_run_c2f_block) — needs a desc path added to the runner.

Non-goals for v1:

- Full graph import.
- Automatic DDR memory planning.
- Full 640x640 YOLO migration.
- Removing the existing hand-written fallback paths.

Why this matters:

The SoC hardware already exposes generic Conv/GEMM/Pool/Upsample/DFL/Elementwise
and DMA capabilities. The descriptor runtime starts moving the deployment model
from per-network handwritten scheduling toward a reusable CPU+NPU runtime.
