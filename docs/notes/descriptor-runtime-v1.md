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

RESOLVED 2026-06-27 -- EVERY conv in the full net now runs on the descriptor
queue, YOLO FULL NET PASS 4/4 at 31.9M cyc:

- Large-IC 1x1 PW fix (conv25/26 + c2f cv1/cv2): the first conv25/26 desc attempt
  regressed to 3/4 + 141.9M cyc. Root cause (RTL-traced): yolo_desc.c set
  NPU_CTRL_OC_SINGLE unconditionally; wgt_reader pf_all = prefetch_all||oc_single
  holds all IC groups resident but pf_icg_store is [3:0] (16) so icg>16 aliases.
  Fix: for 1x1 PW with icg>YOLO_ICG_BUF (pw_stream), drop OC_SINGLE + cap OC
  chunks at 16 (mirrors yolo_run_conv2d_tiled). The 141.9M cyc was a symptom:
  garbage SPPF -> decode false positives -> NMS explosion.
- C2f convs: yolo_run_c2f_block_desc routes cv1/mcv/cv2 through the desc queue
  via a c2f_impl(cfg,use_desc) split (CPU yolo_run_c2f_block + the 10 standalone
  smokes byte-identical). residual-add + concat glue stay on CPU.

Still on CPU (not conv; descriptor engine has ops for some but kept as glue per
v1 scope): C2f residual eltwise-add + concat-requant, decode (DFL/sigmoid/NMS).
Neck upsample/copy run via the npu_desc movement ops.
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
