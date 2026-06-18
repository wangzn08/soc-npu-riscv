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

## Block Plan Preview

This is the first deterministic CPU scheduler skeleton. It assigns DDR
tensor slots, weight slots, SRAM bases, strip rows, and control flags for
each NPU conv layer. It is intentionally conservative: tensors are kept
in separate DDR ranges first, then later firmware work can add reuse or
fusion once correctness is stable.

| idx | input tensor | output tensor | input DDR | output DDR | weight DDR | strips | flags |
|---:|:---|:---|---:|---:|---:|---:|:---|
| 0 | input | conv0 | 0x40000000 | 0x41000000 | 0x48000000 | 40x8 | HW_PAD+SILU+SILU_REQUANT |
| 1 | conv0 | conv1 | 0x41000000 | 0x41190000 | 0x48000900 | 10x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 2 | conv1 | conv2 | 0x41190000 | 0x41258000 | 0x48001B00 | 10x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 3 | c2f_2.split1 | conv3 | 0x41320000 | 0x41384000 | 0x48001F00 | 4x48 | HW_PAD+SILU+SILU_REQUANT |
| 4 | conv3 | conv4 | 0x41384000 | 0x413E8000 | 0x48002800 | 4x48 | HW_PAD+SILU+SILU_REQUANT |
| 5 | c2f_2.concat | conv5 | 0x4144C000 | 0x41578000 | 0x48003100 | 10x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 6 | conv5 | conv6 | 0x41578000 | 0x41640000 | 0x48003700 | 5x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 7 | conv6 | conv7 | 0x41640000 | 0x416A4000 | 0x48007F00 | 5x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 8 | c2f_4.split1 | conv8 | 0x41708000 | 0x4173A000 | 0x48008F00 | 2x48 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 9 | conv8 | conv9 | 0x4173A000 | 0x4176C000 | 0x4800B300 | 2x48 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 10 | c2f_4.m0.add | conv10 | 0x4179E000 | 0x417D0000 | 0x4800D700 | 2x48 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 11 | conv10 | conv11 | 0x417D0000 | 0x41802000 | 0x4800FB00 | 2x48 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 12 | c2f_4.concat | conv12 | 0x41834000 | 0x418FC000 | 0x48011F00 | 5x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 13 | conv12 | conv13 | 0x418FC000 | 0x41960000 | 0x48013F00 | 3x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 14 | conv13 | conv14 | 0x41960000 | 0x41992000 | 0x48025F00 | 3x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 15 | c2f_6.split1 | conv15 | 0x419C4000 | 0x419DD000 | 0x48029F00 | 2x32 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 16 | conv15 | conv16 | 0x419DD000 | 0x419F6000 | 0x48032F00 | 2x32 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 17 | c2f_6.m0.add | conv17 | 0x41A0F000 | 0x41A28000 | 0x4803BF00 | 2x32 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 18 | conv17 | conv18 | 0x41A28000 | 0x41A41000 | 0x48044F00 | 2x32 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 19 | c2f_6.concat | conv19 | 0x41A5A000 | 0x41ABE000 | 0x4804DF00 | 3x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 20 | conv19 | conv20 | 0x41ABE000 | 0x41AF0000 | 0x48055F00 | 2x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 21 | conv20 | conv21 | 0x41AF0000 | 0x41B09000 | 0x4809DF00 | 2x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 22 | c2f_8.split1 | conv22 | 0x41B22000 | 0x41B2E800 | 0x480ADF00 | 2x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 23 | conv22 | conv23 | 0x41B2E800 | 0x41B3B000 | 0x480D1F00 | 2x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 24 | c2f_8.concat | conv24 | 0x41B47800 | 0x41B6D000 | 0x480F5F00 | 2x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 25 | conv24 | conv25 | 0x41B6D000 | 0x41B86000 | 0x4810DF00 | 2x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 26 | sppf.concat | conv26 | 0x41B92800 | 0x41BC4800 | 0x48115F00 | 2x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 27 | fpn.cat1 | conv27 | 0x41BDD800 | 0x41C73800 | 0x48135F00 | 5x8 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 28 | c2f_12.split1 | conv28 | 0x41CA5800 | 0x41CBE800 | 0x48141F00 | 2x32 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 29 | conv28 | conv29 | 0x41CBE800 | 0x41CD7800 | 0x4814AF00 | 2x32 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 30 | c2f_12.concat | conv30 | 0x41CF0800 | 0x41D3B800 | 0x48153F00 | 3x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 31 | fpn.cat2 | conv31 | 0x41D6D800 | 0x41E99800 | 0x48159F00 | 10x8 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 32 | c2f_15.split1 | conv32 | 0x41EFD800 | 0x41F2F800 | 0x4815CF00 | 2x48 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 33 | conv32 | conv33 | 0x41F2F800 | 0x41F61800 | 0x4815F300 | 2x48 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 34 | c2f_15.concat | conv34 | 0x41F93800 | 0x42029800 | 0x48161700 | 5x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 35 | conv34 | conv35 | 0x42029800 | 0x4208D800 | 0x48162F00 | 3x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 36 | conv34 | conv36 | 0x42029800 | 0x420A6800 | 0x4816BF00 | 5x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 37 | conv34 | conv37 | 0x42029800 | 0x4210A800 | 0x48174F00 | 5x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 38 | conv36 | conv38 | 0x420A6800 | 0x42187800 | 0x48180300 | 5x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 39 | conv37 | conv39 | 0x4210A800 | 0x421EB800 | 0x48189300 | 5x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 40 | pan.cat3 | conv40 | 0x42268800 | 0x422B3800 | 0x48197400 | 3x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 41 | conv38 | conv41 | 0x42187800 | 0x422E5800 | 0x4819D400 | 5x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 42 | conv39 | conv42 | 0x421EB800 | 0x42349800 | 0x4819E400 | 5x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 43 | c2f_18.split1 | conv43 | 0x423C6800 | 0x423DF800 | 0x4819FD00 | 2x32 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 44 | conv43 | conv44 | 0x423DF800 | 0x423F8800 | 0x481A8D00 | 2x32 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 45 | c2f_18.concat | conv45 | 0x42411800 | 0x4245C800 | 0x481B1D00 | 3x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 46 | conv45 | conv46 | 0x4245C800 | 0x4248E800 | 0x481B7D00 | 2x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 47 | conv45 | conv47 | 0x4245C800 | 0x4249B000 | 0x481DBD00 | 2x32 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 48 | conv45 | conv48 | 0x4245C800 | 0x424B4000 | 0x481EDD00 | 3x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 49 | conv47 | conv49 | 0x4249B000 | 0x424D3400 | 0x48204500 | 2x32 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 50 | conv48 | conv50 | 0x424B4000 | 0x424EC400 | 0x4820D500 | 2x32 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 51 | pan.cat4 | conv51 | 0x4250B800 | 0x42531000 | 0x4821B600 | 2x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 52 | conv49 | conv52 | 0x424D3400 | 0x4254A000 | 0x48233600 | 2x32 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 53 | conv50 | conv53 | 0x424EC400 | 0x42563000 | 0x48234600 | 2x32 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 54 | c2f_21.split1 | conv54 | 0x42582400 | 0x4258EC00 | 0x48235F00 | 2x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 55 | conv54 | conv55 | 0x4258EC00 | 0x4259B400 | 0x48259F00 | 2x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 56 | c2f_21.concat | conv56 | 0x425A7C00 | 0x425CD400 | 0x4827DF00 | 2x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 57 | conv56 | conv57 | 0x425CD400 | 0x425E6400 | 0x48295F00 | 2x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 58 | conv56 | conv58 | 0x425CD400 | 0x425EC800 | 0x482B9F00 | 2x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 59 | conv57 | conv59 | 0x425E6400 | 0x425F4500 | 0x482E6F00 | 2x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 60 | conv58 | conv60 | 0x425EC800 | 0x425FA900 | 0x482EFF00 | 2x16 | HW_PAD+OC_SINGLE+SILU+SILU_REQUANT |
| 61 | conv59 | conv61 | 0x425F4500 | 0x42602600 | 0x482FE000 | 2x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |
| 62 | conv60 | conv62 | 0x425FA900 | 0x42608A00 | 0x482FF000 | 2x16 | PW_EN+OC_SINGLE+SILU+SILU_REQUANT |

## Gate For RTL Work

- The next RTL change must start with a directed test for the single-conv strip path.
- `conv0` is the first stress case: 640-wide input, stride 2, and only 8 output rows fit in a 256KB Act/Out strip budget.
- After every RTL milestone, run `bash run_all.sh sim` and preserve MNIST 10/10.
