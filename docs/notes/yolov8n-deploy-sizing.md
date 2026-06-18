# YOLOv8n RTL Deploy Sizing

Generated from `yolov8n_int8/yolov8n_layers.h` and the C golden topology in
`yolov8n_int8/yolov8n_infer.c`.

## Summary

- NPU conv layers: 63
- CPU decode/DFL tail layers: 1
- NPU MACs: 4,371,456,000
- CPU DFL MACs: 537,600
- Raw NPU weight bytes: 3,146,160
- 16-channel-aligned NPU weight bytes: 3,148,032
- Aligned activation reads if every conv streams from memory: 25,305,600 bytes
- Aligned activation writes if every conv streams to memory: 15,097,600 bytes

The activation numbers are a conservative per-conv streaming upper bound for
Milestone 0. Later strip/layer fusion work should reduce external DDR traffic.

## Layer Table

| idx | role | source | input | output | k/s/p | MACs | strip_rows@256KB | strip_bytes |
|---:|:---|:---|---:|---:|:---:|---:|---:|---:|
| 0 | npu_conv | stem.0 | 640x640x3 | 320x320x16 | 3x3/2/1 | 44,236,800 | 8 | 215,040 |
| 1 | npu_conv | stem.1 | 320x320x16 | 160x160x32 | 3x3/2/1 | 117,964,800 | 16 | 250,880 |
| 2 | npu_conv | c2f_2.cv1 | 160x160x32 | 160x160x32 | 1x1/1/0 | 26,214,400 | 16 | 163,840 |
| 3 | npu_conv | c2f_2.m0.cv1 | 160x160x16 | 160x160x16 | 3x3/1/1 | 58,982,400 | 48 | 250,880 |
| 4 | npu_conv | c2f_2.m0.cv2 | 160x160x16 | 160x160x16 | 3x3/1/1 | 58,982,400 | 48 | 250,880 |
| 5 | npu_conv | c2f_2.cv2 | 160x160x48 | 160x160x32 | 1x1/1/0 | 39,321,600 | 16 | 204,800 |
| 6 | npu_conv | backbone.down3 | 160x160x32 | 80x80x64 | 3x3/2/1 | 117,964,800 | 16 | 250,880 |
| 7 | npu_conv | c2f_4.cv1 | 80x80x64 | 80x80x64 | 1x1/1/0 | 26,214,400 | 16 | 163,840 |
| 8 | npu_conv | c2f_4.m0.cv1 | 80x80x32 | 80x80x32 | 3x3/1/1 | 58,982,400 | 48 | 250,880 |
| 9 | npu_conv | c2f_4.m0.cv2 | 80x80x32 | 80x80x32 | 3x3/1/1 | 58,982,400 | 48 | 250,880 |
| 10 | npu_conv | c2f_4.m1.cv1 | 80x80x32 | 80x80x32 | 3x3/1/1 | 58,982,400 | 48 | 250,880 |
| 11 | npu_conv | c2f_4.m1.cv2 | 80x80x32 | 80x80x32 | 3x3/1/1 | 58,982,400 | 48 | 250,880 |
| 12 | npu_conv | c2f_4.cv2 | 80x80x128 | 80x80x64 | 1x1/1/0 | 52,428,800 | 16 | 245,760 |
| 13 | npu_conv | backbone.down5 | 80x80x64 | 40x40x128 | 3x3/2/1 | 117,964,800 | 16 | 250,880 |
| 14 | npu_conv | c2f_6.cv1 | 40x40x128 | 40x40x128 | 1x1/1/0 | 26,214,400 | 16 | 163,840 |
| 15 | npu_conv | c2f_6.m0.cv1 | 40x40x64 | 40x40x64 | 3x3/1/1 | 58,982,400 | 32 | 168,960 |
| 16 | npu_conv | c2f_6.m0.cv2 | 40x40x64 | 40x40x64 | 3x3/1/1 | 58,982,400 | 32 | 168,960 |
| 17 | npu_conv | c2f_6.m1.cv1 | 40x40x64 | 40x40x64 | 3x3/1/1 | 58,982,400 | 32 | 168,960 |
| 18 | npu_conv | c2f_6.m1.cv2 | 40x40x64 | 40x40x64 | 3x3/1/1 | 58,982,400 | 32 | 168,960 |
| 19 | npu_conv | c2f_6.cv2 | 40x40x256 | 40x40x128 | 1x1/1/0 | 52,428,800 | 16 | 245,760 |
| 20 | npu_conv | backbone.down7 | 40x40x128 | 20x20x256 | 3x3/2/1 | 117,964,800 | 16 | 250,880 |
| 21 | npu_conv | c2f_8.cv1 | 20x20x256 | 20x20x256 | 1x1/1/0 | 26,214,400 | 16 | 163,840 |
| 22 | npu_conv | c2f_8.m0.cv1 | 20x20x128 | 20x20x128 | 3x3/1/1 | 58,982,400 | 16 | 87,040 |
| 23 | npu_conv | c2f_8.m0.cv2 | 20x20x128 | 20x20x128 | 3x3/1/1 | 58,982,400 | 16 | 87,040 |
| 24 | npu_conv | c2f_8.cv2 | 20x20x384 | 20x20x256 | 1x1/1/0 | 39,321,600 | 16 | 204,800 |
| 25 | npu_conv | sppf.cv1 | 20x20x256 | 20x20x128 | 1x1/1/0 | 13,107,200 | 16 | 122,880 |
| 26 | npu_conv | sppf.cv2 | 20x20x512 | 20x20x256 | 1x1/1/0 | 52,428,800 | 16 | 245,760 |
| 27 | npu_conv | c2f_12.cv1 | 40x40x384 | 40x40x128 | 1x1/1/0 | 78,643,200 | 8 | 163,840 |
| 28 | npu_conv | c2f_12.m0.cv1 | 40x40x64 | 40x40x64 | 3x3/1/1 | 58,982,400 | 32 | 168,960 |
| 29 | npu_conv | c2f_12.m0.cv2 | 40x40x64 | 40x40x64 | 3x3/1/1 | 58,982,400 | 32 | 168,960 |
| 30 | npu_conv | c2f_12.cv2 | 40x40x192 | 40x40x128 | 1x1/1/0 | 39,321,600 | 16 | 204,800 |
| 31 | npu_conv | c2f_15.cv1 | 80x80x192 | 80x80x64 | 1x1/1/0 | 78,643,200 | 8 | 163,840 |
| 32 | npu_conv | c2f_15.m0.cv1 | 80x80x32 | 80x80x32 | 3x3/1/1 | 58,982,400 | 48 | 250,880 |
| 33 | npu_conv | c2f_15.m0.cv2 | 80x80x32 | 80x80x32 | 3x3/1/1 | 58,982,400 | 48 | 250,880 |
| 34 | npu_conv | c2f_15.cv2 | 80x80x96 | 80x80x64 | 1x1/1/0 | 39,321,600 | 16 | 204,800 |
| 35 | npu_conv | pan.down_p3 | 80x80x64 | 40x40x64 | 3x3/2/1 | 58,982,400 | 16 | 209,920 |
| 36 | npu_conv | detect.p3.bbox0 | 80x80x64 | 80x80x64 | 3x3/1/1 | 235,929,600 | 16 | 174,080 |
| 37 | npu_conv | detect.p3.cls0 | 80x80x64 | 80x80x80 | 3x3/1/1 | 294,912,000 | 16 | 194,560 |
| 38 | npu_conv | detect.p3.bbox1 | 80x80x64 | 80x80x64 | 3x3/1/1 | 235,929,600 | 16 | 174,080 |
| 39 | npu_conv | detect.p3.cls1 | 80x80x80 | 80x80x80 | 3x3/1/1 | 368,640,000 | 16 | 217,600 |
| 40 | npu_conv | c2f_18.cv1 | 40x40x192 | 40x40x128 | 1x1/1/0 | 39,321,600 | 16 | 204,800 |
| 41 | npu_conv | detect.p3.bbox2 | 80x80x64 | 80x80x64 | 1x1/1/0 | 26,214,400 | 16 | 163,840 |
| 42 | npu_conv | detect.p3.cls2 | 80x80x80 | 80x80x80 | 1x1/1/0 | 40,960,000 | 16 | 204,800 |
| 43 | npu_conv | c2f_18.m0.cv1 | 40x40x64 | 40x40x64 | 3x3/1/1 | 58,982,400 | 32 | 168,960 |
| 44 | npu_conv | c2f_18.m0.cv2 | 40x40x64 | 40x40x64 | 3x3/1/1 | 58,982,400 | 32 | 168,960 |
| 45 | npu_conv | c2f_18.cv2 | 40x40x192 | 40x40x128 | 1x1/1/0 | 39,321,600 | 16 | 204,800 |
| 46 | npu_conv | pan.down_p4 | 40x40x128 | 20x20x128 | 3x3/2/1 | 58,982,400 | 16 | 209,920 |
| 47 | npu_conv | detect.p4.bbox0 | 40x40x128 | 40x40x64 | 3x3/1/1 | 117,964,800 | 32 | 256,000 |
| 48 | npu_conv | detect.p4.cls0 | 40x40x128 | 40x40x80 | 3x3/1/1 | 147,456,000 | 16 | 143,360 |
| 49 | npu_conv | detect.p4.bbox1 | 40x40x64 | 40x40x64 | 3x3/1/1 | 58,982,400 | 32 | 168,960 |
| 50 | npu_conv | detect.p4.cls1 | 40x40x80 | 40x40x80 | 3x3/1/1 | 92,160,000 | 32 | 211,200 |
| 51 | npu_conv | c2f_21.cv1 | 20x20x384 | 20x20x256 | 1x1/1/0 | 39,321,600 | 16 | 204,800 |
| 52 | npu_conv | detect.p4.bbox2 | 40x40x64 | 40x40x64 | 1x1/1/0 | 6,553,600 | 32 | 163,840 |
| 53 | npu_conv | detect.p4.cls2 | 40x40x80 | 40x40x80 | 1x1/1/0 | 10,240,000 | 32 | 204,800 |
| 54 | npu_conv | c2f_21.m0.cv1 | 20x20x128 | 20x20x128 | 3x3/1/1 | 58,982,400 | 16 | 87,040 |
| 55 | npu_conv | c2f_21.m0.cv2 | 20x20x128 | 20x20x128 | 3x3/1/1 | 58,982,400 | 16 | 87,040 |
| 56 | npu_conv | c2f_21.cv2 | 20x20x384 | 20x20x256 | 1x1/1/0 | 39,321,600 | 16 | 204,800 |
| 57 | npu_conv | detect.p5.bbox0 | 20x20x256 | 20x20x64 | 3x3/1/1 | 58,982,400 | 16 | 112,640 |
| 58 | npu_conv | detect.p5.cls0 | 20x20x256 | 20x20x80 | 3x3/1/1 | 73,728,000 | 16 | 117,760 |
| 59 | npu_conv | detect.p5.bbox1 | 20x20x64 | 20x20x64 | 3x3/1/1 | 14,745,600 | 16 | 43,520 |
| 60 | npu_conv | detect.p5.cls1 | 20x20x80 | 20x20x80 | 3x3/1/1 | 23,040,000 | 16 | 54,400 |
| 61 | npu_conv | detect.p5.bbox2 | 20x20x64 | 20x20x64 | 1x1/1/0 | 1,638,400 | 16 | 40,960 |
| 62 | npu_conv | detect.p5.cls2 | 20x20x80 | 20x20x80 | 1x1/1/0 | 2,560,000 | 16 | 51,200 |
| 63 | cpu_dfl | cpu.decode.dfl | 1x33600x16 | 1x33600x1 | 1x1/1/0 | 537,600 | 0 | 0 |

## Gate For RTL Work

- The next RTL change must start with a directed test for the single-conv strip path.
- `conv0` is the first stress case: 640-wide input, stride 2, and only 8 output rows fit in a 256KB Act/Out strip budget.
- After every RTL milestone, run `bash run_all.sh sim` and preserve MNIST 10/10.
