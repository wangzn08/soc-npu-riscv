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
    {0.0039215689f, 0.2229382843f, -128, -127, 1},
    {0.2229382843f, 0.5254408717f, -127, -127, 1},
    {0.5254408717f, 0.1119460166f, -127, -126, 1},
    {0.1119460166f, 0.0997420177f, -126, -125, 1},
    {0.0997420177f, 0.1117400080f, -125, -126, 1},
    {0.1257445961f, 0.0623452179f, -124, -124, 1},
    {0.0623452179f, 0.0294791609f, -124, -119, 1},
    {0.0294791609f, 0.0288450532f, -119, -118, 1},
    {0.0288450532f, 0.0178034566f, -118, -112, 1},
    {0.0178034566f, 0.0203037187f, -112, -114, 1},
    {0.0272450428f, 0.0148501666f, -108, -109, 1},
    {0.0148501666f, 0.0394611359f, -109, -121, 1},
    {0.0445279926f, 0.0263596177f, -109, -117, 1},
    {0.0263596177f, 0.0214071870f, -117, -115, 1},
    {0.0214071870f, 0.0268922318f, -115, -118, 1},
    {0.0268922318f, 0.0241889823f, -118, -116, 1},
    {0.0241889823f, 0.0332387984f, -116, -120, 1},
    {0.0332784094f, 0.0206421297f, -111, -115, 1},
    {0.0206421297f, 0.0330366381f, -115, -120, 1},
    {0.0376032330f, 0.0266261417f, -106, -118, 1},
    {0.0266261417f, 0.0238745771f, -118, -116, 1},
    {0.0238745771f, 0.0371915363f, -116, -121, 1},
    {0.0371915363f, 0.0230140053f, -121, -116, 1},
    {0.0230140053f, 0.0320745781f, -116, -119, 1},
    {0.0382835455f, 0.0251241773f, -113, -117, 1},
    {0.0251241773f, 0.0273747370f, -117, -118, 1},
    {0.0273747370f, 0.0172612965f, -118, -112, 1},
    {0.0266261417f, 0.0230489317f, -118, -116, 1},
    {0.0230489317f, 0.0157706924f, -116, -110, 1},
    {0.0157706924f, 0.0269427709f, -110, -118, 1},
    {0.0269427709f, 0.0272754543f, -118, -118, 1},
    {0.0272754543f, 0.0134569565f, -118, -107, 1},
    {0.0134569565f, 0.0142951915f, -107, -109, 1},
    {0.0142951915f, 0.0200350154f, -109, -114, 1},
    {0.0200350154f, 0.0175402369f, -114, -112, 1},
    {0.0175402369f, 0.0250063110f, -112, -117, 1},
    {0.0175402369f, 0.0232884549f, -112, -116, 1},
    {0.0175402369f, 0.0216259696f, -112, -115, 1},
    {0.0232884549f, 0.1562104821f, -116, -126, 1},
    {0.0216259696f, 0.1673392802f, -115, -126, 1},
    {0.0272754543f, 0.0250679478f, -118, -117, 1},
    {0.1562104821f, 0.0954652056f, -126, -56, 0},
    {0.1673392802f, 0.0881871134f, -126, 127, 0},
    {0.0250679478f, 0.0133610778f, -117, -107, 1},
    {0.0133610778f, 0.0272369962f, -107, -118, 1},
    {0.0272369962f, 0.0221300740f, -118, -115, 1},
    {0.0221300740f, 0.0258114282f, -115, -117, 1},
    {0.0221300740f, 0.0331719555f, -115, -120, 1},
    {0.0221300740f, 0.0305516310f, -115, -119, 1},
    {0.0331719555f, 0.1100535318f, -120, -125, 1},
    {0.0305516310f, 0.1615440398f, -119, -126, 1},
    {0.0258114282f, 0.0246042069f, -117, -117, 1},
    {0.1100535318f, 0.0759554431f, -125, -71, 0},
    {0.1615440398f, 0.1057757810f, -126, 119, 0},
    {0.0246042069f, 0.0241170228f, -117, -116, 1},
    {0.0241170228f, 0.0329095162f, -116, -120, 1},
    {0.0329095162f, 0.0281161219f, -120, -118, 1},
    {0.0281161219f, 0.0485097989f, -118, -122, 1},
    {0.0281161219f, 0.0337256826f, -118, -120, 1},
    {0.0485097989f, 0.1670174748f, -122, -126, 1},
    {0.0337256826f, 0.2197803706f, -120, -127, 1},
    {0.1670174748f, 0.0714144185f, -126, -33, 0},
    {0.2197803706f, 0.1535685211f, -127, 113, 0},
    {0.0039215689f, 0.0542175286f, -128, -128, 0},
};

#define YOLO_NUM_GLUE_OPS 30
typedef struct {
    const char *name;
    float out_scale;
    int   out_zp;
} YoloGlueQuant;

static const YoloGlueQuant yolo_glue_quant[30] = {
    {"/model.2/m.0/Add", 0.1257445961f, -124},
    {"/model.2/Concat", 0.1257445961f, -124},
    {"/model.4/m.0/Add", 0.0272450428f, -108},
    {"/model.4/m.1/Add", 0.0445279926f, -109},
    {"/model.4/Concat", 0.0445279926f, -109},
    {"/model.6/m.0/Add", 0.0332784094f, -111},
    {"/model.6/m.1/Add", 0.0376032330f, -106},
    {"/model.6/Concat", 0.0376032330f, -106},
    {"/model.8/m.0/Add", 0.0328725837f, -111},
    {"/model.8/Concat", 0.0382835455f, -113},
    {"/model.9/m/MaxPool", 0.0273747370f, -118},
    {"/model.9/m_1/MaxPool", 0.0273747370f, -118},
    {"/model.9/m_2/MaxPool", 0.0273747370f, -118},
    {"/model.9/Concat", 0.0273747370f, -118},
    {"/model.10/Resize", 0.0172612965f, -112},
    {"/model.11/Concat", 0.0266261417f, -118},
    {"/model.12/Concat", 0.0269427709f, -118},
    {"/model.13/Resize", 0.0272754543f, -118},
    {"/model.14/Concat", 0.0272754543f, -118},
    {"/model.15/Concat", 0.0200350154f, -114},
    {"/model.17/Concat", 0.0272754543f, -118},
    {"/model.18/Concat", 0.0272369962f, -118},
    {"/model.20/Concat", 0.0258114282f, -117},
    {"/model.21/Concat", 0.0329095162f, -120},
    {"/model.22/Concat", 0.0954652056f, -56},
    {"/model.22/Concat_1", 0.1535685211f, 113},
    {"/model.22/Add_1", 0.3187699914f, -128},
    {"/model.22/Add_2", 0.6248660684f, -128},
    {"/model.22/Concat_2", 0.3124330342f, -128},
    {"/model.22/Concat_3", 2.5000445843f, -128},
};

#endif
