#!/usr/bin/env python3
"""Extract MNIST test images and generate C arrays for firmware testing."""
import gzip
import struct
import os
import numpy as np

DATASET_DIR = '/home/ICer/projects/wzn_prj/飞腾杯/2026-05-03/training/dataset'

def read_idx_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)

def read_idx_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

images = read_idx_images(os.path.join(DATASET_DIR, 't10k-images-idx3-ubyte.gz'))
labels = read_idx_labels(os.path.join(DATASET_DIR, 't10k-labels-idx1-ubyte.gz'))

print(f"Total test images: {len(labels)}")

# Extract one image of each digit 0-9
found = {}
for i in range(len(labels)):
    digit = labels[i]
    if digit not in found:
        img = images[i]
        img_int8 = (img.astype(np.float32) * 127.0 / 255.0).astype(np.int8)
        found[int(digit)] = (i, img_int8)
        if len(found) == 10:
            break

print(f"Found {len(found)} digits: {sorted(found.keys())}")

# Generate C code
print("\n// ============================================================")
print("// MNIST test images (extracted from t10k test set)")
print("// ============================================================")

for digit in sorted(found.keys()):
    idx, img = found[digit]
    print(f"// Digit {digit}, MNIST index {idx}")
    print(f"static const int8_t mnist_img_{digit}[784] = {{")
    flat = img.flatten()
    for j in range(0, 784, 16):
        line = flat[j:j+16]
        print("    " + ", ".join([f"{int(x):4d}" for x in line]) + ",")
    print("};")
    print()

# Generate array of pointers and labels
print("static const int8_t *mnist_images[10] = {")
for digit in range(10):
    print(f"    mnist_img_{digit},")
print("};")
print()
print("static const int8_t mnist_labels[10] = {")
print("    " + ", ".join([str(d) for d in range(10)]) + "")
print("};")
