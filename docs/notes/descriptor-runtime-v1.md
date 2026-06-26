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
