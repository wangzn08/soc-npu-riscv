/* YOLOv8n INT8 pure-C inference engine
 *
 * 64 conv layers, 3.15M params, ~8.7 GFLOPs
 * Input: 640x640 RGB uint8
 * Output: detections [x,y,w,h,conf,class_id]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "yolov8n_infer.h"
#include "yolov8n_layers.h"

/* env-gated debug dump: print dequantized first-8 values of a conv output,
 * keyed by conv index, so it lines up with ONNX checkpoint tensors. */
static void dbg_dump_conv(int ci, const void *tp);

/* ================================================================ */
/* Tensor: int8 CHW layout, with its own per-tensor scale/zero_point */
/* ================================================================ */
typedef struct {
    int8_t *data;
    int c, h, w;
    float scale;
    int zero_point;
} Tensor;

static inline int t_size(const Tensor *t) { return t->c * t->h * t->w; }
static inline int8_t *t_at(const Tensor *t, int ch, int row, int col) {
    return t->data + ((long long)ch * t->h * t->w + row * t->w + col);
}

/* ================================================================ */
/* Simple arena allocator                                          */
/* ================================================================ */
static char arena[128 * 1024 * 1024];  /* 128 MB */
static size_t arena_off = 0;

/* Input resolution (square, multiple of 32). Default 640; set to 320 for the
 * SoC deploy golden. Same weights/quant scales are reused at any resolution. */
int g_yolo_input = 640;

static void *arena_alloc(size_t bytes) {
    bytes = (bytes + 15) & ~15;  /* 16-byte align */
    void *p = arena + arena_off;
    arena_off += bytes;
    if (arena_off > sizeof(arena)) {
        fprintf(stderr, "arena overflow\n");
        exit(1);
    }
    return p;
}

static Tensor tensor_make(int c, int h, int w, float scale, int zero_point) {
    Tensor t;
    t.data = (int8_t *)arena_alloc(c * h * w * sizeof(int8_t));
    t.c = c; t.h = h; t.w = w;
    t.scale = scale; t.zero_point = zero_point;
    memset(t.data, 0, c * h * w * sizeof(int8_t));
    return t;
}

static inline int8_t clamp_round_int8(float v) {
    int r = (int)lroundf(v);
    if (r < -128) r = -128;
    if (r > 127) r = 127;
    return (int8_t)r;
}

/* ================================================================ */
/* Weight storage                                                  */
/* ================================================================ */
typedef struct {
    const int8_t *w;       /* [OC, IC, KH, KW] */
    float *b;               /* [OC] bias, float */
    float *wscale;           /* [OC] per-channel weight scale */
    int32_t *wsum;            /* [OC] sum of int8 weights, for zero-point correction */
    int oc, ic, kh, kw, stride, pad;
    float in_scale, out_scale;
    int in_zp, out_zp, has_silu;
} ConvW;

static ConvW conv_w[YOLO_NUM_CONV];

void load_weights(const char *dir) {
    (void)dir;  /* w_file/b_file/s_file already include "weights/" prefix */
    for (int i = 0; i < YOLO_NUM_CONV; i++) {
        const YoloConvCfg *c = &yolo_conv[i];
        const YoloActQuant *q = &yolo_act_quant[i];
        int ws = c->oc * c->ic * c->kh * c->kw;

        conv_w[i].w = (const int8_t *)malloc(ws);
        if (!conv_w[i].w) { fprintf(stderr, "malloc failed for %s\n", c->w_file); exit(1); }
        FILE *f = fopen(c->w_file, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", c->w_file); exit(1); }
        if (fread((void *)conv_w[i].w, 1, ws, f) != (size_t)ws) {
            fprintf(stderr, "short read on %s\n", c->w_file); exit(1);
        }
        fclose(f);

        conv_w[i].b = (float *)malloc(c->oc * sizeof(float));
        if (!conv_w[i].b) { fprintf(stderr, "malloc failed for %s\n", c->b_file); exit(1); }
        f = fopen(c->b_file, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", c->b_file); exit(1); }
        if (fread(conv_w[i].b, sizeof(float), c->oc, f) != (size_t)c->oc) {
            fprintf(stderr, "short read on %s\n", c->b_file); exit(1);
        }
        fclose(f);

        conv_w[i].wscale = (float *)malloc(c->oc * sizeof(float));
        if (!conv_w[i].wscale) { fprintf(stderr, "malloc failed for %s\n", c->s_file); exit(1); }
        f = fopen(c->s_file, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", c->s_file); exit(1); }
        if (fread(conv_w[i].wscale, sizeof(float), c->oc, f) != (size_t)c->oc) {
            fprintf(stderr, "short read on %s\n", c->s_file); exit(1);
        }
        fclose(f);

        conv_w[i].oc = c->oc;
        conv_w[i].ic = c->ic;
        conv_w[i].kh = c->kh;
        conv_w[i].kw = c->kw;
        conv_w[i].stride = c->stride;
        conv_w[i].pad = c->pad;
        conv_w[i].in_scale = q->in_scale;
        conv_w[i].out_scale = q->out_scale;
        conv_w[i].in_zp = q->in_zp;
        conv_w[i].out_zp = q->out_zp;
        conv_w[i].has_silu = q->has_silu;

        /* precompute per-OC weight sum for input zero-point correction
         * (weight zero_point is verified 0 / symmetric for all 64 convs,
         * so no weight-zp term is needed here) */
        conv_w[i].wsum = (int32_t *)malloc(c->oc * sizeof(int32_t));
        if (!conv_w[i].wsum) { fprintf(stderr, "malloc failed for wsum (conv %d)\n", i); exit(1); }
        int per_oc = c->ic * c->kh * c->kw;
        for (int oc = 0; oc < c->oc; oc++) {
            int32_t s = 0;
            const int8_t *woc = conv_w[i].w + (long long)oc * per_oc;
            for (int k = 0; k < per_oc; k++) s += (int32_t)woc[k];
            conv_w[i].wsum[oc] = s;
        }
    }
}

/* ================================================================ */
/* Conv with full INT8 datapath:                                    */
/*   acc_int32 = sum(int8_act * int8_w)  (real integer MAC)         */
/*   acc_corrected = acc_int32 - in_zp * wsum[oc]                    */
/*   preact = acc_corrected * in_scale * wscale[oc] + bias            */
/*   y = has_silu ? SiLU(preact) : preact                            */
/*   out_int8 = clamp_round(y / out_scale + out_zp)                  */
/* ================================================================ */
static Tensor conv_int8(int ci, const Tensor *in) {
    const ConvW *cw = &conv_w[ci];
    int OC = cw->oc, IC = cw->ic, KH = cw->kh, KW = cw->kw;
    int S = cw->stride, P = cw->pad;
    int OH = (in->h + 2*P - KH) / S + 1;
    int OW = (in->w + 2*P - KW) / S + 1;

    Tensor out = tensor_make(OC, OH, OW, cw->out_scale, cw->out_zp);

    for (int oc = 0; oc < OC; oc++) {
        float bias = cw->b[oc];
        float wscale = cw->wscale[oc];
        int32_t wsum = cw->wsum[oc];
        const int8_t *woc = cw->w + (long long)oc * IC * KH * KW;
        int8_t *out_ch = t_at(&out, oc, 0, 0);

        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int32_t acc = 0;
                for (int ic = 0; ic < IC; ic++) {
                    const int8_t *in_ch = t_at(in, ic, 0, 0);
                    const int8_t *wic = woc + (long long)ic * KH * KW;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * S - P + kh;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * S - P + kw;
                            /* ONNX QLinearConv pads with the input zero-point, so
                             * padded taps contribute in_zp * weight (NOT skipped).
                             * Skipping them desyncs from the in_zp*wsum correction
                             * below and corrupts every conv edge. */
                            int8_t ival = (ih < 0 || ih >= in->h || iw < 0 || iw >= in->w)
                                              ? (int8_t)in->zero_point
                                              : in_ch[ih * in->w + iw];
                            acc += (int32_t)ival * (int32_t)wic[kh * KW + kw];
                        }
                    }
                }
                int32_t acc_corrected = acc - in->zero_point * wsum;
                float preact = (float)acc_corrected * in->scale * wscale + bias;
                float y = cw->has_silu ? (preact / (1.0f + expf(-preact))) : preact;
                out_ch[oh * OW + ow] = clamp_round_int8(y / cw->out_scale + cw->out_zp);
            }
        }
    }
    if (getenv("YOLO_DEBUG_DUMP")) dbg_dump_conv(ci, &out);
    return out;
}

/* Dump dequantized first-8 values for the conv indices that map to the
 * ONNX checkpoint tensors (conv0, C2f.2 cv1, SPPF cv2, scale2 bbox head). */
static void dbg_dump_conv(int ci, const void *tp) {
    const Tensor *t = (const Tensor *)tp;
    /* Full per-layer golden dump for the SoC assembly oracle:
     * <YOLO_DUMP_DIR>/conv<ci>.bin = [int32 c,h,w, float scale, int32 zp] + int8 CHW */
    const char *dd = getenv("YOLO_DUMP_DIR");
    if (dd) {
        char path[512];
        snprintf(path, sizeof(path), "%s/conv%d.bin", dd, ci);
        FILE *fp = fopen(path, "wb");
        if (fp) {
            int32_t hdr[3] = { t->c, t->h, t->w };
            float sc = t->scale; int32_t zp = t->zero_point;
            fwrite(hdr, sizeof(int32_t), 3, fp);
            fwrite(&sc, sizeof(float), 1, fp);
            fwrite(&zp, sizeof(int32_t), 1, fp);
            fwrite(t->data, sizeof(int8_t), (size_t)t->c * t->h * t->w, fp);
            fclose(fp);
        }
    }
    if (ci != 0 && ci != 2 && ci != 26 && ci != 61) return;
    const char *tag = "?";
    if (ci == 0)  tag = "conv0   /model.0/act/Mul_output_0";
    if (ci == 2)  tag = "conv2   /model.2/cv1/act/Mul_output_0";
    if (ci == 26) tag = "conv26  /model.9/cv2/act/Mul_output_0";
    if (ci == 61) tag = "conv61  /model.22/cv2.2/cv2.2.2/Conv_output_0";
    fprintf(stderr, "DEBUG %s: shape=(%d,%d,%d) scale=%.8f zp=%d first8= ",
            tag, t->c, t->h, t->w, t->scale, t->zero_point);
    for (int i = 0; i < 8; i++)
        fprintf(stderr, "%.6f ", (t->data[i] - t->zero_point) * t->scale);
    fprintf(stderr, "\n");
    if (ci == 0 && getenv("YOLO_DEBUG_INTERIOR")) {
        int W = t->w, H = t->h;
        int idxs[6][2] = {{160,160},{160,161},{100,100},{200,50},{0,0},{1,1}};
        fprintf(stderr, "  conv0 ch0 interior:");
        for (int k = 0; k < 6; k++) {
            int r = idxs[k][0], c = idxs[k][1];
            (void)H;
            int8_t v = t->data[r*W + c];
            fprintf(stderr, " [%d,%d]=%.4f", r, c, (v - t->zero_point) * t->scale);
        }
        fprintf(stderr, "\n");
    }
}

/* ================================================================ */
/* Requantize a tensor from its current (scale, zero_point) to a   */
/* target (out_scale, out_zp). No-op when already matching.        */
/* ================================================================ */
static void requant_inplace(Tensor *t, float out_scale, int out_zp) {
    /* Exact float == is deliberate: scales come from the same calibration
     * constant table and are bit-identical, so this is a real no-op fast-path.
     * Do NOT "fix" this into an epsilon compare. */
    if (t->scale == out_scale && t->zero_point == out_zp) return;  /* no-op */
    int n = t_size(t);
    for (int i = 0; i < n; i++) {
        float real_val = (t->data[i] - t->zero_point) * t->scale;
        t->data[i] = clamp_round_int8(real_val / out_scale + out_zp);
    }
    t->scale = out_scale;
    t->zero_point = out_zp;
}

/* ================================================================ */
/* C2f block:                                                       */
/*   cv1(1x1) -> Split(2) -> N x bottleneck -> Concat -> cv2(1x1)  */
/* ================================================================ */
typedef struct {
    int cv1;
    int m_cv1[4];
    int m_cv2[4];
    int cv2;
    int n_bottleneck;
    int shortcut;
    int add_glue_idx[4];   /* index into yolo_glue_quant, only used when shortcut */
} C2fCfg;

static Tensor c2f_block(const Tensor *in, const C2fCfg *cfg) {
    Tensor cv1_out = conv_int8(cfg->cv1, in);

    int half_c = cv1_out.c / 2;
    Tensor s0 = {cv1_out.data, half_c, cv1_out.h, cv1_out.w, cv1_out.scale, cv1_out.zero_point};
    Tensor s1 = {cv1_out.data + half_c * cv1_out.h * cv1_out.w,
                 half_c, cv1_out.h, cv1_out.w, cv1_out.scale, cv1_out.zero_point};

    Tensor prev = s1;
    Tensor adds[4];
    adds[0] = s1;

    for (int i = 0; i < cfg->n_bottleneck; i++) {
        Tensor t = conv_int8(cfg->m_cv1[i], &prev);
        t = conv_int8(cfg->m_cv2[i], &t);

        if (cfg->shortcut) {
            const YoloGlueQuant *gq = &yolo_glue_quant[cfg->add_glue_idx[i]];
            Tensor add_out = tensor_make(t.c, t.h, t.w, gq->out_scale, gq->out_zp);
            for (int j = 0; j < t_size(&t); j++) {
                float real_prev = (prev.data[j] - prev.zero_point) * prev.scale;
                float real_t    = (t.data[j]    - t.zero_point)    * t.scale;
                add_out.data[j] = clamp_round_int8((real_prev + real_t) / gq->out_scale + gq->out_zp);
            }
            adds[i + 1] = add_out;
            prev = add_out;
        } else {
            adds[i + 1] = t;
            prev = t;
        }
    }

    int total_c = half_c;
    for (int i = 0; i <= cfg->n_bottleneck; i++) total_c += adds[i].c;

    /* Concat: requant each piece to the next conv's calibrated input scale.
     * requant_inplace is safe here because s0/adds[] never escape this
     * function. */
    Tensor pieces[5];
    int n_pieces = 0;
    pieces[n_pieces++] = s0;
    for (int i = 0; i <= cfg->n_bottleneck; i++) pieces[n_pieces++] = adds[i];

    float target_scale = conv_w[cfg->cv2].in_scale;
    int target_zp = conv_w[cfg->cv2].in_zp;

    Tensor concat_out = tensor_make(total_c, cv1_out.h, cv1_out.w, target_scale, target_zp);
    int8_t *dst = concat_out.data;
    for (int i = 0; i < n_pieces; i++) {
        requant_inplace(&pieces[i], target_scale, target_zp);
        long long sz = (long long)pieces[i].c * pieces[i].h * pieces[i].w;
        memcpy(dst, pieces[i].data, sz);
        dst += sz;
    }

    return conv_int8(cfg->cv2, &concat_out);
}

/* ================================================================ */
/* SPPF: cv1(1x1) -> 3x MaxPool5x5 -> Concat(4) -> cv2(1x1)       */
/* MaxPool is int8-domain (same scale/zp), element-wise max.       */
/* ================================================================ */
static void maxpool_k_int8(const int8_t *in, int8_t *out, int C, int H, int W, int k) {
    int OH = H, OW = W;
    int P = k / 2;
    for (int c = 0; c < C; c++) {
        const int8_t *chin = in + (long long)c * H * W;
        int8_t *chout = out + (long long)c * OH * OW;
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int8_t mx = -128;
                for (int kh = 0; kh < k; kh++) {
                    for (int kw = 0; kw < k; kw++) {
                        int ih = oh - P + kh;
                        int iw = ow - P + kw;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            int8_t v = chin[ih * W + iw];
                            if (v > mx) mx = v;
                        }
                    }
                }
                chout[oh * OW + ow] = mx;
            }
        }
    }
}

static Tensor sppf(const Tensor *in, int cv1_idx, int cv2_idx) {
    Tensor cv1_out = conv_int8(cv1_idx, in);
    int C = cv1_out.c, H = cv1_out.h, W = cv1_out.w;

    int8_t *mp_buf = (int8_t *)arena_alloc(C * H * W);
    int8_t *mp_buf2 = (int8_t *)arena_alloc(C * H * W);
    int8_t *mp_buf3 = (int8_t *)arena_alloc(C * H * W);

    maxpool_k_int8(cv1_out.data, mp_buf, C, H, W, 5);
    maxpool_k_int8(mp_buf, mp_buf2, C, H, W, 5);
    maxpool_k_int8(mp_buf2, mp_buf3, C, H, W, 5);

    float target_scale = conv_w[cv2_idx].in_scale;
    int target_zp = conv_w[cv2_idx].in_zp;

    Tensor cat = tensor_make(4*C, H, W, target_scale, target_zp);
    Tensor parts[4] = {
        {cv1_out.data, C, H, W, cv1_out.scale, cv1_out.zero_point},
        {mp_buf, C, H, W, cv1_out.scale, cv1_out.zero_point},
        {mp_buf2, C, H, W, cv1_out.scale, cv1_out.zero_point},
        {mp_buf3, C, H, W, cv1_out.scale, cv1_out.zero_point},
    };
    int8_t *dst = cat.data;
    for (int i = 0; i < 4; i++) {
        requant_inplace(&parts[i], target_scale, target_zp);
        memcpy(dst, parts[i].data, (long long)C * H * W);
        dst += C * H * W;
    }

    return conv_int8(cv2_idx, &cat);
}

/* ================================================================ */
/* Upsample nearest-neighbor 2x (no value change, keeps scale/zp)  */
/* ================================================================ */
static Tensor upsample2x(const Tensor *in) {
    Tensor out = tensor_make(in->c, in->h * 2, in->w * 2, in->scale, in->zero_point);
    for (int c = 0; c < in->c; c++)
        for (int oh = 0; oh < out.h; oh++)
            for (int ow = 0; ow < out.w; ow++)
                *t_at(&out, c, oh, ow) = *t_at(in, c, oh/2, ow/2);
    return out;
}

/* ================================================================ */
/* Concat two tensors, requant to next conv's input scale.         */
/* Uses fresh-buffer output (no aliasing of source tensors).       */
/* ================================================================ */
static Tensor concat2_to_conv(const Tensor *a, const Tensor *b, int next_conv_idx) {
    float target_scale = conv_w[next_conv_idx].in_scale;
    int target_zp = conv_w[next_conv_idx].in_zp;
    Tensor out = tensor_make(a->c + b->c, a->h, a->w, target_scale, target_zp);

    long long sz_a = (long long)a->c * a->h * a->w;
    long long sz_b = (long long)b->c * b->h * b->w;

    if (a->scale == target_scale && a->zero_point == target_zp) {
        memcpy(out.data, a->data, sz_a);
    } else {
        for (long long i = 0; i < sz_a; i++) {
            float real_val = (a->data[i] - a->zero_point) * a->scale;
            out.data[i] = clamp_round_int8(real_val / target_scale + target_zp);
        }
    }
    if (b->scale == target_scale && b->zero_point == target_zp) {
        memcpy(out.data + sz_a, b->data, sz_b);
    } else {
        for (long long i = 0; i < sz_b; i++) {
            float real_val = (b->data[i] - b->zero_point) * b->scale;
            out.data[sz_a + i] = clamp_round_int8(real_val / target_scale + target_zp);
        }
    }
    return out;
}

/* ================================================================ */
/* NMS                                                             */
/* ================================================================ */
static float iou(const YoloDet *a, const YoloDet *b) {
    float x1 = fmaxf(a->x - a->w/2, b->x - b->w/2);
    float y1 = fmaxf(a->y - a->h/2, b->y - b->h/2);
    float x2 = fminf(a->x + a->w/2, b->x + b->w/2);
    float y2 = fminf(a->y + a->h/2, b->y + b->h/2);
    float inter = fmaxf(0, x2-x1) * fmaxf(0, y2-y1);
    return inter / (a->w*a->h + b->w*b->h - inter + 1e-7f);
}

static int nms(YoloDet *d, int n, float thr) {
    /* Sort by confidence desc */
    for (int i = 0; i < n-1; i++)
        for (int j = i+1; j < n; j++)
            if (d[j].conf > d[i].conf) { YoloDet t=d[i]; d[i]=d[j]; d[j]=t; }
    int keep = 0;
    for (int i = 0; i < n; i++) {
        if (d[i].conf < 0) continue;  /* suppressed */
        d[keep++] = d[i];
        for (int j = i+1; j < n; j++) {
            if (d[j].conf < 0) continue;
            if (d[i].class_id == d[j].class_id && iou(&d[i], &d[j]) > thr)
                d[j].conf = -1;
        }
    }
    return keep;
}

/* ================================================================ */
/* Main inference                                                   */
/* ================================================================ */

/* YOLOv8n conv index mapping (from ONNX graph analysis):
 *
 * Backbone:
 *   conv0: model.0   (3->16, s=2)        -> 320x320
 *   conv1: model.1   (16->32, s=2)       -> 160x160
 *   C2f(2):  cv1=2, m0={3,4}, cv2=5      -> 160x160x32
 *   conv6: model.3   (32->64, s=2)       -> 80x80
 *   C2f(4):  cv1=7, m0={8,9}, m1={10,11}, cv2=12 -> 80x80x64
 *   conv13: model.5  (64->128, s=2)      -> 40x40
 *   C2f(6):  cv1=14, m0={15,16}, m1={17,18}, cv2=19 -> 40x40x128
 *   conv20: model.7  (128->256, s=2)     -> 20x20
 *   C2f(8):  cv1=21, m0={22,23}, cv2=24  -> 20x20x256
 *
 * SPPF (model.9):
 *   conv25: 256->128 (1x1), maxpool x3, conv26: 512->256 (1x1) -> 20x20x256
 *
 * FPN Neck:
 *   upsample 20x20->40x40
 *   concat with backbone 40x40 (conv19 output)
 *   C2f(12): cv1=27, m0={28,29}, cv2=30 -> 40x40x128
 *   upsample 40x40->80x80
 *   concat with backbone 80x80 (conv12 output)
 *   C2f(15): cv1=31, m0={32,33}, cv2=34 -> 80x80x64
 *
 * PAN Neck:
 *   conv35: 64->64 s=2 -> 40x40x64
 *   concat with FPN 40x40
 *   C2f(18): cv1=40, m0={43,44}, cv2=45 -> 40x40x128
 *   conv46: 128->128 s=2 -> 20x20x128
 *   concat with SPPF 20x20
 *   C2f(21): cv1=51, m0={54,55}, cv2=56 -> 20x20x256
 *
 * Detection Head (model.22):
 *   Scale 80x80: stem=36+37, cls0=38+39+41+42, bbox=40+43+44+45
 *   Scale 40x40: stem=47+48, cls1=49+50+52+53, bbox=51+54+55+56
 *   Scale 20x20: stem=57+58, cls2=59+60+61+62
 *   DFL: conv63 (16->1)
 */

int yolo_infer(const uint8_t *image_rgb, YoloDet *dets, int max_dets,
               float conf_thr, float nms_thr) {
    arena_off = 0;

    /* Convert input to int8 CHW (quantize per conv0's in_scale/in_zp).
     * Resolution-parametric: g_yolo_input (default 640) lets the same weights/
     * scales run at e.g. 320x320. Must be a multiple of 32 (5 stride-2 stages). */
    int IN = g_yolo_input;
    Tensor input = tensor_make(3, IN, IN, conv_w[0].in_scale, conv_w[0].in_zp);
    for (int c = 0; c < 3; c++)
        for (int h = 0; h < IN; h++)
            for (int w = 0; w < IN; w++) {
                float real_val = image_rgb[(h*IN+w)*3+c] / 255.0f;
                *t_at(&input, c, h, w) = clamp_round_int8(real_val / input.scale + input.zero_point);
            }

    /* ---- BACKBONE ---- */
    Tensor t;
    t = conv_int8(0, &input);       /* 320x320x16 */
    t = conv_int8(1, &t);           /* 160x160x32 */

    C2fCfg c2f_2  = {2,  {3,0,0,0},   {4,0,0,0},   5,  1, 1, {0,0,0,0}};
    t = c2f_block(&t, &c2f_2);     /* 160x160x32 */

    t = conv_int8(6, &t);           /* 80x80x64 */

    Tensor p4 = t;                  /* save 40x40 level */
    C2fCfg c2f_4  = {7,  {8,10,0,0},  {9,11,0,0},  12, 2, 1, {2,3,0,0}};
    t = c2f_block(&t, &c2f_4);     /* 80x80x64 */
    p4 = t;  /* update: 80x80x64 is p4 for FPN concat */

    t = conv_int8(13, &t);          /* 40x40x128 */
    C2fCfg c2f_6  = {14, {15,17,0,0}, {16,18,0,0}, 19, 2, 1, {5,6,0,0}};
    t = c2f_block(&t, &c2f_6);     /* 40x40x128 */

    Tensor p5 = t;                  /* 20x20 level */
    t = conv_int8(20, &t);          /* 20x20x256 */
    C2fCfg c2f_8  = {21, {22,0,0,0},  {23,0,0,0},  24, 1, 1, {8,0,0,0}};
    Tensor sppf_in = c2f_block(&t, &c2f_8); /* 20x20x256 */

    /* ---- SPPF (model.9) ---- */
    t = sppf(&sppf_in, 25, 26);    /* 20x20x256 */

    /* ---- FPN Neck ---- */
    Tensor up1 = upsample2x(&t);    /* 40x40x256 */
    Tensor cat1 = concat2_to_conv(&up1, &p5, 27); /* 40x40x(256+128=384) */
    C2fCfg c2f_12 = {27, {28,0,0,0},  {29,0,0,0},  30, 1, 0, {0,0,0,0}};
    Tensor fpn_mid = c2f_block(&cat1, &c2f_12); /* 40x40x128 */

    Tensor up2 = upsample2x(&fpn_mid); /* 80x80x128 */
    Tensor cat2 = concat2_to_conv(&up2, &p4, 31);   /* 80x80x(128+64=192) */
    C2fCfg c2f_15 = {31, {32,0,0,0},  {33,0,0,0},  34, 1, 0, {0,0,0,0}};
    Tensor pan_p3 = c2f_block(&cat2, &c2f_15); /* 80x80x64 */

    /* ---- PAN Neck ---- */
    t = conv_int8(35, &pan_p3);     /* 40x40x64 (s=2 downsample) */
    Tensor cat3 = concat2_to_conv(&t, &fpn_mid, 40); /* 40x40x(64+128=192) */
    C2fCfg c2f_18 = {40, {43,0,0,0},  {44,0,0,0},  45, 1, 0, {0,0,0,0}};
    Tensor pan_p4 = c2f_block(&cat3, &c2f_18); /* 40x40x128 */

    t = conv_int8(46, &pan_p4);     /* 20x20x128 (s=2 downsample) */
    Tensor cat4 = concat2_to_conv(&t, &sppf_in, 51); /* 20x20x(128+256=384) */
    C2fCfg c2f_21 = {51, {54,0,0,0},  {55,0,0,0},  56, 1, 0, {0,0,0,0}};
    Tensor pan_p5 = c2f_block(&cat4, &c2f_21); /* 20x20x256 */

    /* ---- Detection Head (model.22) ----
     *
     * 3 scales: P3(80x80), P4(40x40), P5(20x20) from PAN neck.
     * Each scale -> parallel bbox(64ch) + cls(80ch) branches.
     * conv35/46 are PAN internal downsamples, NOT detection inputs.
     * ONNX node connectivity: conv36,conv37 both read pan_p3 output (fork).
     *
     * bbox branch: stem(3x3 SiLU) -> mid(3x3 SiLU) -> out(1x1 linear) [64ch]
     * cls branch:  stem(3x3 SiLU) -> mid(3x3 SiLU) -> out(1x1 linear) [80ch]
     *
     * Scale 0 (80x80): bbox=conv36->38->41, cls=conv37->39->42
     * Scale 1 (40x40): bbox=conv47->49->52, cls=conv48->50->53
     * Scale 2 (20x20): bbox=conv57->59->61, cls=conv58->60->62
     * DFL: conv63 (16->1 1x1)
     */

    /* Scale 0: 80x80 (pan_p3, 64ch) */
    Tensor d0_bbox = conv_int8(36, &pan_p3);
    d0_bbox = conv_int8(38, &d0_bbox);
    d0_bbox = conv_int8(41, &d0_bbox);

    Tensor d0_cls = conv_int8(37, &pan_p3);
    d0_cls = conv_int8(39, &d0_cls);
    d0_cls = conv_int8(42, &d0_cls);

    /* Scale 1: 40x40 (pan_p4, 128ch) */
    Tensor d1_bbox = conv_int8(47, &pan_p4);
    d1_bbox = conv_int8(49, &d1_bbox);
    d1_bbox = conv_int8(52, &d1_bbox);

    Tensor d1_cls = conv_int8(48, &pan_p4);
    d1_cls = conv_int8(50, &d1_cls);
    d1_cls = conv_int8(53, &d1_cls);

    /* Scale 2: 20x20 (pan_p5, 256ch) */
    Tensor d2_bbox = conv_int8(57, &pan_p5);
    d2_bbox = conv_int8(59, &d2_bbox);
    d2_bbox = conv_int8(61, &d2_bbox);

    Tensor d2_cls = conv_int8(58, &pan_p5);
    d2_cls = conv_int8(60, &d2_cls);
    d2_cls = conv_int8(62, &d2_cls);

    /* === Concatenate bbox/cls across 3 scales === */
    int a0 = d0_bbox.h * d0_bbox.w, a1 = d1_bbox.h * d1_bbox.w, a2 = d2_bbox.h * d2_bbox.w;
    int num_anchors = a0 + a1 + a2; /* 8400 */

    /* bbox/cls outputs are int8; dequant to float while assembling the
     * per-scale [C,H,W] tensors into the flat [C, 8400] buffers. */
    float *bbox_raw = (float *)arena_alloc(64LL * num_anchors * sizeof(float));
    float *cls_raw = (float *)arena_alloc(80LL * num_anchors * sizeof(float));

    Tensor *bbox_scales[3] = {&d0_bbox, &d1_bbox, &d2_bbox};
    Tensor *cls_scales[3] = {&d0_cls, &d1_cls, &d2_cls};
    int areas[3] = {a0, a1, a2};
    /* Assemble into GLOBAL channel-major [C, num_anchors] (matching the ONNX
     * output layout [84, 8400]): each channel spans all 8400 anchors, and the
     * 3 scales' anchors are concatenated (P3 6400, P4 1600, P5 400). The source
     * conv tensors are channel-major per scale: data[c*A + hw]. */
    int anchor_off = 0;
    for (int s = 0; s < 3; s++) {
        int A = areas[s];
        Tensor *bt = bbox_scales[s];
        for (int c = 0; c < 64; c++)
            for (int hw = 0; hw < A; hw++)
                bbox_raw[(long long)c * num_anchors + (anchor_off + hw)] =
                    (bt->data[(long long)c * A + hw] - bt->zero_point) * bt->scale;

        Tensor *ct = cls_scales[s];
        for (int c = 0; c < 80; c++)
            for (int hw = 0; hw < A; hw++)
                cls_raw[(long long)c * num_anchors + (anchor_off + hw)] =
                    (ct->data[(long long)c * A + hw] - ct->zero_point) * ct->scale;

        anchor_off += A;
    }

    /* === DFL: bbox [64, 8400] -> [4, 8400] ===
     * bbox_raw layout: [64, N] where 64 = 4 coords * 16 bins.
     * Channel order: coord0_bin0..15, coord1_bin0..15, ...
     * For each (coord, anchor): softmax over 16 bins, then weighted sum
     * with dequantized conv63 weights (IC=16, OC=1). */
    float *bbox_dfl = (float *)arena_alloc(4LL * num_anchors * sizeof(float));
    /* conv63 (the DFL conv) is bias-free, and its input zero-point is
     * intentionally not applied here: standard YOLOv8 DFL is a softmax-
     * expectation projection (softmax over 16 bins, then weighted sum). */
    const ConvW *dfl_cw = &conv_w[63];

    for (int a = 0; a < num_anchors; a++) {
        for (int c = 0; c < 4; c++) {
            float dist[16];
            float mx = -1e30f;
            for (int k = 0; k < 16; k++) {
                dist[k] = bbox_raw[(c * 16 + k) * num_anchors + a];
                if (dist[k] > mx) mx = dist[k];
            }
            float s = 0;
            for (int k = 0; k < 16; k++) { dist[k] = expf(dist[k] - mx); s += dist[k]; }
            for (int k = 0; k < 16; k++) dist[k] /= s;

            float val = 0;
            for (int k = 0; k < 16; k++)
                val += dist[k] * dfl_cw->wscale[0] * (float)dfl_cw->w[k];  /* w[k] is INT8, wscale[0] is scale */
            bbox_dfl[c * num_anchors + a] = val;
        }
    }

    /* cls scores -> sigmoid (in place) */
    for (int i = 0; i < 80 * num_anchors; i++)
        cls_raw[i] = 1.0f / (1.0f + expf(-cls_raw[i]));

    /* === Decode detections ===
     * bbox_dfl [4, 8400] = [lt_x, lt_y, rb_x, rb_y] per anchor (grid-cell units).
     * anchor centers: (ax+0.5, ay+0.5) in grid units.
     * x1 = ax - lt_x, y1 = ay - lt_y, x2 = ax + rb_x, y2 = ay + rb_y.
     * center = (x1+x2)/2, wh = x2-x1. Then * stride -> pixel coords. */
    int n_dets = 0;
    YoloDet raw_dets[8400];

    int scale_hw[3] = { d0_bbox.h, d1_bbox.h, d2_bbox.h };
    for (int scale = 0; scale < 3; scale++) {
        int hw = scale_hw[scale];
        int stride = g_yolo_input / hw;   /* 8/16/32 at any resolution */

        for (int ay = 0; ay < hw; ay++) {
            for (int ax = 0; ax < hw; ax++) {
                int a;
                if (scale == 0)      a = ay * hw + ax;
                else if (scale == 1) a = a0 + ay * hw + ax;
                else                 a = a0 + a1 + ay * hw + ax;

                float lt_x = bbox_dfl[0 * num_anchors + a];
                float lt_y = bbox_dfl[1 * num_anchors + a];
                float rb_x = bbox_dfl[2 * num_anchors + a];
                float rb_y = bbox_dfl[3 * num_anchors + a];

                float x1 = (ax + 0.5f) - lt_x;
                float y1 = (ay + 0.5f) - lt_y;
                float x2 = (ax + 0.5f) + rb_x;
                float y2 = (ay + 0.5f) + rb_y;
                float cx = (x1 + x2) * 0.5f * stride;
                float cy = (y1 + y2) * 0.5f * stride;
                float bw = (x2 - x1) * stride;
                float bh = (y2 - y1) * stride;

                int best_cls = 0;
                float best_score = cls_raw[0 * num_anchors + a];
                for (int c = 1; c < 80; c++) {
                    float sc = cls_raw[c * num_anchors + a];
                    if (sc > best_score) { best_score = sc; best_cls = c; }
                }

                if (best_score < conf_thr) continue;
                if (n_dets < 8400) {
                    raw_dets[n_dets].x = cx;
                    raw_dets[n_dets].y = cy;
                    raw_dets[n_dets].w = bw;
                    raw_dets[n_dets].h = bh;
                    raw_dets[n_dets].conf = best_score;
                    raw_dets[n_dets].class_id = best_cls;
                    n_dets++;
                }
            }
        }
    }

    /* === NMS === */
    int final = nms(raw_dets, n_dets, nms_thr);
    if (final > max_dets) final = max_dets;
    memcpy(dets, raw_dets, final * sizeof(YoloDet));
    return final;
}
