#ifndef DEEPNET_H
#define DEEPNET_H

#include <stdint.h>

// DeepConvNet inference on PicoRV32
// Architecture: Conv-Relu-Conv-Relu-Pool-Conv-Relu-Conv-Relu-Pool-Conv-Relu-Conv-Relu-Pool-Affine-Relu-Affine

// Feature map sizes
#define IMG_C 1
#define IMG_H 28
#define IMG_W 28

// Conv1 output: 16x28x28
#define Conv1_OUT_C 16
#define Conv1_OUT_H 28
#define Conv1_OUT_W 28

// Conv2 output: 16x28x28
#define Conv2_OUT_C 16
#define Conv2_OUT_H 28
#define Conv2_OUT_W 28

// Pool1 output: 16x14x14
#define Pool1_OUT_C 16
#define Pool1_OUT_H 14
#define Pool1_OUT_W 14

// Conv3 output: 32x14x14
#define Conv3_OUT_C 32
#define Conv3_OUT_H 14
#define Conv3_OUT_W 14

// Conv4 output: 32x16x16 (pad=2)
#define Conv4_OUT_C 32
#define Conv4_OUT_H 16
#define Conv4_OUT_W 16

// Pool2 output: 32x8x8
#define Pool2_OUT_C 32
#define Pool2_OUT_H 8
#define Pool2_OUT_W 8

// Conv5 output: 64x8x8
#define Conv5_OUT_C 64
#define Conv5_OUT_H 8
#define Conv5_OUT_W 8

// Conv6 output: 64x8x8
#define Conv6_OUT_C 64
#define Conv6_OUT_H 8
#define Conv6_OUT_W 8

// Pool3 output: 64x4x4
#define Pool3_OUT_C 64
#define Pool3_OUT_H 4
#define Pool3_OUT_W 4

// Affine1: 1024 -> 50
#define AFFINE1_IN  (Pool3_OUT_C * Pool3_OUT_H * Pool3_OUT_W)  // 1024
#define AFFINE1_OUT 50

// Affine2: 50 -> 10
#define AFFINE2_IN  AFFINE1_OUT  // 50
#define AFFINE2_OUT 10

// Buffer sizes (int8)
#define MAX_FEATUREMAP (16*28*28)  // 12544 bytes
#define MAX_IM2COL (28*28*16*3*3)  // 112896 bytes (for Conv2)

// Main inference function
// Input: 784 int8 values (28x28 image)
// Output: 10 int32 values (raw scores)
void deepnet_inference(const int8_t *input, int32_t *output, int img_idx);

#endif
