/* Auto-generated YOLOv8n INT8 layer configuration */
#ifndef YOLOV8N_LAYERS_H
#define YOLOV8N_LAYERS_H

#define YOLO_NUM_CONV 64
#define YOLO_INPUT_H 640
#define YOLO_INPUT_W 640
#define YOLO_INPUT_C 3
#define YOLO_NUM_CLASSES 80

typedef struct {
    int oc, ic, kh, kw, stride, pad;
    const char *w_file, *b_file, *s_file;
} YoloConvCfg;

static const YoloConvCfg yolo_conv[64] = {
    {16,3,3,3,2,1,"weights/conv0_w.bin","weights/conv0_b.bin","weights/conv0_s.bin"},
    {32,16,3,3,2,1,"weights/conv1_w.bin","weights/conv1_b.bin","weights/conv1_s.bin"},
    {32,32,1,1,1,0,"weights/conv2_w.bin","weights/conv2_b.bin","weights/conv2_s.bin"},
    {16,16,3,3,1,1,"weights/conv3_w.bin","weights/conv3_b.bin","weights/conv3_s.bin"},
    {16,16,3,3,1,1,"weights/conv4_w.bin","weights/conv4_b.bin","weights/conv4_s.bin"},
    {32,48,1,1,1,0,"weights/conv5_w.bin","weights/conv5_b.bin","weights/conv5_s.bin"},
    {64,32,3,3,2,1,"weights/conv6_w.bin","weights/conv6_b.bin","weights/conv6_s.bin"},
    {64,64,1,1,1,0,"weights/conv7_w.bin","weights/conv7_b.bin","weights/conv7_s.bin"},
    {32,32,3,3,1,1,"weights/conv8_w.bin","weights/conv8_b.bin","weights/conv8_s.bin"},
    {32,32,3,3,1,1,"weights/conv9_w.bin","weights/conv9_b.bin","weights/conv9_s.bin"},
    {32,32,3,3,1,1,"weights/conv10_w.bin","weights/conv10_b.bin","weights/conv10_s.bin"},
    {32,32,3,3,1,1,"weights/conv11_w.bin","weights/conv11_b.bin","weights/conv11_s.bin"},
    {64,128,1,1,1,0,"weights/conv12_w.bin","weights/conv12_b.bin","weights/conv12_s.bin"},
    {128,64,3,3,2,1,"weights/conv13_w.bin","weights/conv13_b.bin","weights/conv13_s.bin"},
    {128,128,1,1,1,0,"weights/conv14_w.bin","weights/conv14_b.bin","weights/conv14_s.bin"},
    {64,64,3,3,1,1,"weights/conv15_w.bin","weights/conv15_b.bin","weights/conv15_s.bin"},
    {64,64,3,3,1,1,"weights/conv16_w.bin","weights/conv16_b.bin","weights/conv16_s.bin"},
    {64,64,3,3,1,1,"weights/conv17_w.bin","weights/conv17_b.bin","weights/conv17_s.bin"},
    {64,64,3,3,1,1,"weights/conv18_w.bin","weights/conv18_b.bin","weights/conv18_s.bin"},
    {128,256,1,1,1,0,"weights/conv19_w.bin","weights/conv19_b.bin","weights/conv19_s.bin"},
    {256,128,3,3,2,1,"weights/conv20_w.bin","weights/conv20_b.bin","weights/conv20_s.bin"},
    {256,256,1,1,1,0,"weights/conv21_w.bin","weights/conv21_b.bin","weights/conv21_s.bin"},
    {128,128,3,3,1,1,"weights/conv22_w.bin","weights/conv22_b.bin","weights/conv22_s.bin"},
    {128,128,3,3,1,1,"weights/conv23_w.bin","weights/conv23_b.bin","weights/conv23_s.bin"},
    {256,384,1,1,1,0,"weights/conv24_w.bin","weights/conv24_b.bin","weights/conv24_s.bin"},
    {128,256,1,1,1,0,"weights/conv25_w.bin","weights/conv25_b.bin","weights/conv25_s.bin"},
    {256,512,1,1,1,0,"weights/conv26_w.bin","weights/conv26_b.bin","weights/conv26_s.bin"},
    {128,384,1,1,1,0,"weights/conv27_w.bin","weights/conv27_b.bin","weights/conv27_s.bin"},
    {64,64,3,3,1,1,"weights/conv28_w.bin","weights/conv28_b.bin","weights/conv28_s.bin"},
    {64,64,3,3,1,1,"weights/conv29_w.bin","weights/conv29_b.bin","weights/conv29_s.bin"},
    {128,192,1,1,1,0,"weights/conv30_w.bin","weights/conv30_b.bin","weights/conv30_s.bin"},
    {64,192,1,1,1,0,"weights/conv31_w.bin","weights/conv31_b.bin","weights/conv31_s.bin"},
    {32,32,3,3,1,1,"weights/conv32_w.bin","weights/conv32_b.bin","weights/conv32_s.bin"},
    {32,32,3,3,1,1,"weights/conv33_w.bin","weights/conv33_b.bin","weights/conv33_s.bin"},
    {64,96,1,1,1,0,"weights/conv34_w.bin","weights/conv34_b.bin","weights/conv34_s.bin"},
    {64,64,3,3,2,1,"weights/conv35_w.bin","weights/conv35_b.bin","weights/conv35_s.bin"},
    {64,64,3,3,1,1,"weights/conv36_w.bin","weights/conv36_b.bin","weights/conv36_s.bin"},
    {80,64,3,3,1,1,"weights/conv37_w.bin","weights/conv37_b.bin","weights/conv37_s.bin"},
    {64,64,3,3,1,1,"weights/conv38_w.bin","weights/conv38_b.bin","weights/conv38_s.bin"},
    {80,80,3,3,1,1,"weights/conv39_w.bin","weights/conv39_b.bin","weights/conv39_s.bin"},
    {128,192,1,1,1,0,"weights/conv40_w.bin","weights/conv40_b.bin","weights/conv40_s.bin"},
    {64,64,1,1,1,0,"weights/conv41_w.bin","weights/conv41_b.bin","weights/conv41_s.bin"},
    {80,80,1,1,1,0,"weights/conv42_w.bin","weights/conv42_b.bin","weights/conv42_s.bin"},
    {64,64,3,3,1,1,"weights/conv43_w.bin","weights/conv43_b.bin","weights/conv43_s.bin"},
    {64,64,3,3,1,1,"weights/conv44_w.bin","weights/conv44_b.bin","weights/conv44_s.bin"},
    {128,192,1,1,1,0,"weights/conv45_w.bin","weights/conv45_b.bin","weights/conv45_s.bin"},
    {128,128,3,3,2,1,"weights/conv46_w.bin","weights/conv46_b.bin","weights/conv46_s.bin"},
    {64,128,3,3,1,1,"weights/conv47_w.bin","weights/conv47_b.bin","weights/conv47_s.bin"},
    {80,128,3,3,1,1,"weights/conv48_w.bin","weights/conv48_b.bin","weights/conv48_s.bin"},
    {64,64,3,3,1,1,"weights/conv49_w.bin","weights/conv49_b.bin","weights/conv49_s.bin"},
    {80,80,3,3,1,1,"weights/conv50_w.bin","weights/conv50_b.bin","weights/conv50_s.bin"},
    {256,384,1,1,1,0,"weights/conv51_w.bin","weights/conv51_b.bin","weights/conv51_s.bin"},
    {64,64,1,1,1,0,"weights/conv52_w.bin","weights/conv52_b.bin","weights/conv52_s.bin"},
    {80,80,1,1,1,0,"weights/conv53_w.bin","weights/conv53_b.bin","weights/conv53_s.bin"},
    {128,128,3,3,1,1,"weights/conv54_w.bin","weights/conv54_b.bin","weights/conv54_s.bin"},
    {128,128,3,3,1,1,"weights/conv55_w.bin","weights/conv55_b.bin","weights/conv55_s.bin"},
    {256,384,1,1,1,0,"weights/conv56_w.bin","weights/conv56_b.bin","weights/conv56_s.bin"},
    {64,256,3,3,1,1,"weights/conv57_w.bin","weights/conv57_b.bin","weights/conv57_s.bin"},
    {80,256,3,3,1,1,"weights/conv58_w.bin","weights/conv58_b.bin","weights/conv58_s.bin"},
    {64,64,3,3,1,1,"weights/conv59_w.bin","weights/conv59_b.bin","weights/conv59_s.bin"},
    {80,80,3,3,1,1,"weights/conv60_w.bin","weights/conv60_b.bin","weights/conv60_s.bin"},
    {64,64,1,1,1,0,"weights/conv61_w.bin","weights/conv61_b.bin","weights/conv61_s.bin"},
    {80,80,1,1,1,0,"weights/conv62_w.bin","weights/conv62_b.bin","weights/conv62_s.bin"},
    {1,16,1,1,1,0,"weights/conv63_w.bin","weights/conv63_b.bin","weights/conv63_s.bin"},
};

/* Layer dimensions (OC x IC x KH x KW, stride, pad):
 * conv 0:   16 x    3 x 3x3 s=2 p=1
 * conv 1:   32 x   16 x 3x3 s=2 p=1
 * conv 2:   32 x   32 x 1x1 s=1 p=0
 * conv 3:   16 x   16 x 3x3 s=1 p=1
 * conv 4:   16 x   16 x 3x3 s=1 p=1
 * conv 5:   32 x   48 x 1x1 s=1 p=0
 * conv 6:   64 x   32 x 3x3 s=2 p=1
 * conv 7:   64 x   64 x 1x1 s=1 p=0
 * conv 8:   32 x   32 x 3x3 s=1 p=1
 * conv 9:   32 x   32 x 3x3 s=1 p=1
 * conv10:   32 x   32 x 3x3 s=1 p=1
 * conv11:   32 x   32 x 3x3 s=1 p=1
 * conv12:   64 x  128 x 1x1 s=1 p=0
 * conv13:  128 x   64 x 3x3 s=2 p=1
 * conv14:  128 x  128 x 1x1 s=1 p=0
 * conv15:   64 x   64 x 3x3 s=1 p=1
 * conv16:   64 x   64 x 3x3 s=1 p=1
 * conv17:   64 x   64 x 3x3 s=1 p=1
 * conv18:   64 x   64 x 3x3 s=1 p=1
 * conv19:  128 x  256 x 1x1 s=1 p=0
 * conv20:  256 x  128 x 3x3 s=2 p=1
 * conv21:  256 x  256 x 1x1 s=1 p=0
 * conv22:  128 x  128 x 3x3 s=1 p=1
 * conv23:  128 x  128 x 3x3 s=1 p=1
 * conv24:  256 x  384 x 1x1 s=1 p=0
 * conv25:  128 x  256 x 1x1 s=1 p=0
 * conv26:  256 x  512 x 1x1 s=1 p=0
 * conv27:  128 x  384 x 1x1 s=1 p=0
 * conv28:   64 x   64 x 3x3 s=1 p=1
 * conv29:   64 x   64 x 3x3 s=1 p=1
 * conv30:  128 x  192 x 1x1 s=1 p=0
 * conv31:   64 x  192 x 1x1 s=1 p=0
 * conv32:   32 x   32 x 3x3 s=1 p=1
 * conv33:   32 x   32 x 3x3 s=1 p=1
 * conv34:   64 x   96 x 1x1 s=1 p=0
 * conv35:   64 x   64 x 3x3 s=2 p=1
 * conv36:   64 x   64 x 3x3 s=1 p=1
 * conv37:   80 x   64 x 3x3 s=1 p=1
 * conv38:   64 x   64 x 3x3 s=1 p=1
 * conv39:   80 x   80 x 3x3 s=1 p=1
 * conv40:  128 x  192 x 1x1 s=1 p=0
 * conv41:   64 x   64 x 1x1 s=1 p=0
 * conv42:   80 x   80 x 1x1 s=1 p=0
 * conv43:   64 x   64 x 3x3 s=1 p=1
 * conv44:   64 x   64 x 3x3 s=1 p=1
 * conv45:  128 x  192 x 1x1 s=1 p=0
 * conv46:  128 x  128 x 3x3 s=2 p=1
 * conv47:   64 x  128 x 3x3 s=1 p=1
 * conv48:   80 x  128 x 3x3 s=1 p=1
 * conv49:   64 x   64 x 3x3 s=1 p=1
 * conv50:   80 x   80 x 3x3 s=1 p=1
 * conv51:  256 x  384 x 1x1 s=1 p=0
 * conv52:   64 x   64 x 1x1 s=1 p=0
 * conv53:   80 x   80 x 1x1 s=1 p=0
 * conv54:  128 x  128 x 3x3 s=1 p=1
 * conv55:  128 x  128 x 3x3 s=1 p=1
 * conv56:  256 x  384 x 1x1 s=1 p=0
 * conv57:   64 x  256 x 3x3 s=1 p=1
 * conv58:   80 x  256 x 3x3 s=1 p=1
 * conv59:   64 x   64 x 3x3 s=1 p=1
 * conv60:   80 x   80 x 3x3 s=1 p=1
 * conv61:   64 x   64 x 1x1 s=1 p=0
 * conv62:   80 x   80 x 1x1 s=1 p=0
 * conv63:    1 x   16 x 1x1 s=1 p=0
 */


typedef struct {
    float in_scale, out_scale;
    int   in_zp, out_zp;
    int   has_silu;
} YoloActQuant;

static const YoloActQuant yolo_act_quant[64] = {
    {0.0039215689f, 0.2352971584f, -128, -127, 1},
    {0.2352971584f, 0.6557820439f, -127, -128, 1},
    {0.6557820439f, 0.1601515412f, -128, -126, 1},
    {0.1601515412f, 0.1423473954f, -126, -126, 1},
    {0.1423473954f, 0.1425503194f, -126, -126, 1},
    {0.1612435579f, 0.0763198882f, -125, -124, 1},
    {0.0763198882f, 0.0334292874f, -124, -120, 1},
    {0.0334292874f, 0.0397584029f, -120, -121, 1},
    {0.0397584029f, 0.0204943288f, -121, -114, 1},
    {0.0204943288f, 0.0271009300f, -114, -118, 1},
    {0.0365898795f, 0.0219360031f, -113, -115, 1},
    {0.0219360031f, 0.0483939648f, -115, -122, 1},
    {0.0560306832f, 0.0337357186f, -113, -120, 1},
    {0.0337357186f, 0.0319866650f, -120, -119, 1},
    {0.0319866650f, 0.0365007035f, -119, -120, 1},
    {0.0365007035f, 0.0307744816f, -120, -119, 1},
    {0.0307744816f, 0.0272509996f, -119, -118, 1},
    {0.0315765068f, 0.0313216858f, -110, -119, 1},
    {0.0313216858f, 0.0618021861f, -119, -123, 1},
    {0.0707620457f, 0.0331330635f, -116, -120, 1},
    {0.0331330635f, 0.0388779789f, -120, -121, 1},
    {0.0388779789f, 0.0546551012f, -121, -123, 1},
    {0.0546551012f, 0.0524734072f, -123, -123, 1},
    {0.0524734072f, 0.0760652423f, -123, -124, 1},
    {0.0779053494f, 0.0433438867f, -121, -122, 1},
    {0.0433438867f, 0.0334274098f, -122, -120, 1},
    {0.0334274098f, 0.0468565077f, -120, -122, 1},
    {0.0468565077f, 0.0443715937f, -122, -122, 1},
    {0.0443715937f, 0.0349353999f, -122, -120, 1},
    {0.0349353999f, 0.0368834250f, -120, -120, 1},
    {0.0443715937f, 0.0303480383f, -122, -119, 1},
    {0.0337357186f, 0.0244056843f, -120, -117, 1},
    {0.0244056843f, 0.0180846192f, -117, -113, 1},
    {0.0180846192f, 0.0324782133f, -113, -119, 1},
    {0.0324782133f, 0.0308109522f, -119, -119, 1},
    {0.0308109522f, 0.0312041230f, -119, -119, 1},
    {0.0308109522f, 0.0312871225f, -119, -119, 1},
    {0.0308109522f, 0.0366929248f, -119, -120, 1},
    {0.0312871225f, 0.1906583309f, -119, -127, 1},
    {0.0366929248f, 0.4357864559f, -120, -127, 1},
    {0.0312041230f, 0.0423709862f, -119, -121, 1},
    {0.1906583309f, 0.1402078271f, -127, -60, 0},
    {0.4357864559f, 0.1624340564f, -127, 109, 0},
    {0.0423709862f, 0.0233757310f, -121, -116, 1},
    {0.0233757310f, 0.0458762310f, -116, -122, 1},
    {0.0458762310f, 0.0320586488f, -122, -119, 1},
    {0.0320586488f, 0.0383166559f, -119, -121, 1},
    {0.0320586488f, 0.0502213985f, -119, -122, 1},
    {0.0320586488f, 0.0406545065f, -119, -121, 1},
    {0.0502213985f, 0.2595561445f, -122, -127, 1},
    {0.0406545065f, 0.5205523968f, -121, -127, 1},
    {0.0468565077f, 0.0409575403f, -122, -121, 1},
    {0.2595561445f, 0.1008327380f, -127, -45, 0},
    {0.5205523968f, 0.2252567559f, -127, 115, 0},
    {0.0409575403f, 0.0303888749f, -121, -119, 1},
    {0.0303888749f, 0.0516663790f, -119, -123, 1},
    {0.0516663790f, 0.0412088670f, -123, -121, 1},
    {0.0412088670f, 0.0554671213f, -121, -123, 1},
    {0.0412088670f, 0.0441324860f, -121, -122, 1},
    {0.0554671213f, 0.2104742825f, -123, -127, 1},
    {0.0441324860f, 0.3582480252f, -122, -127, 1},
    {0.2104742825f, 0.0867320970f, -127, -32, 0},
    {0.3582480252f, 0.2458649278f, -127, 101, 0},
    {0.0037991831f, 0.0576510578f, -128, -128, 0},
};

#define YOLO_NUM_GLUE_OPS 24
typedef struct {
    const char *name;
    float out_scale;
    int   out_zp;
} YoloGlueQuant;

static const YoloGlueQuant yolo_glue_quant[24] = {
    {"/model.2/m.0/Add", 0.1549137533f, -124},
    {"/model.2/Concat", 0.1612435579f, -125},
    {"/model.4/m.0/Add", 0.0365898795f, -113},
    {"/model.4/m.1/Add", 0.0560306832f, -113},
    {"/model.4/Concat", 0.0560306832f, -113},
    {"/model.6/m.0/Add", 0.0315765068f, -110},
    {"/model.6/m.1/Add", 0.0707620457f, -116},
    {"/model.6/Concat", 0.0707620457f, -116},
    {"/model.8/m.0/Add", 0.0779053494f, -121},
    {"/model.8/Concat", 0.0779053494f, -121},
    {"/model.9/m/MaxPool", 0.0334274098f, -120},
    {"/model.9/m_1/MaxPool", 0.0334274098f, -120},
    {"/model.9/m_2/MaxPool", 0.0334274098f, -120},
    {"/model.9/Concat", 0.0334274098f, -120},
    {"/model.10/Resize", 0.0468565077f, -122},
    {"/model.11/Concat", 0.0468565077f, -122},
    {"/model.12/Concat", 0.0443715937f, -122},
    {"/model.13/Resize", 0.0303480383f, -119},
    {"/model.14/Concat", 0.0337357186f, -120},
    {"/model.15/Concat", 0.0324782133f, -119},
    {"/model.17/Concat", 0.0312041230f, -119},
    {"/model.18/Concat", 0.0458762310f, -122},
    {"/model.20/Concat", 0.0468565077f, -122},
    {"/model.21/Concat", 0.0516663790f, -123},
};


#endif
