/* Simple test: load a PPM image, run YOLOv8n INT8 inference, print detections */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "yolov8n_infer.h"

/* Load a PPM P6 image (640x640 RGB) */
static uint8_t *load_ppm(const char *path, int *w, int *h) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    char magic[4];
    if (!fgets(magic, sizeof(magic), f) || magic[0] != 'P' || magic[1] != '6') {
        fprintf(stderr, "Not a PPM P6 file\n");
        fclose(f);
        return NULL;
    }
    /* skip comments */
    int c;
    while ((c = fgetc(f)) == '#') { while (fgetc(f) != '\n'); }
    ungetc(c, f);
    fscanf(f, "%d %d\n", w, h);
    int maxval;
    fscanf(f, "%d\n", &maxval);
    (void)maxval;
    uint8_t *img = (uint8_t *)malloc(*w * *h * 3);
    fread(img, 1, *w * *h * 3, f);
    fclose(f);
    return img;
}

/* Convert a tiny 4x4 test image (no file needed) */
static uint8_t *make_test_image(void) {
    uint8_t *img = (uint8_t *)malloc(640 * 640 * 3);
    memset(img, 128, 640 * 640 * 3);
    /* draw a bright rectangle in the center */
    for (int y = 200; y < 440; y++)
        for (int x = 200; x < 440; x++) {
            img[(y*640+x)*3+0] = 200;
            img[(y*640+x)*3+1] = 50;
            img[(y*640+x)*3+2] = 50;
        }
    return img;
}

int main(int argc, char **argv) {
    const char *weight_dir = "weights";
    const char *image_path = NULL;
    float conf_thr = 0.25f;
    float nms_thr = 0.45f;

    if (argc > 1) image_path = argv[1];
    if (argc > 2) weight_dir = argv[2];

    printf("Loading weights from: %s\n", weight_dir);
    load_weights(weight_dir);

    uint8_t *image;
    int iw, ih;
    if (image_path) {
        image = load_ppm(image_path, &iw, &ih);
        if (!image) return 1;
        printf("Loaded image: %dx%d\n", iw, ih);
    } else {
        printf("No image specified, using synthetic test image\n");
        image = make_test_image();
        iw = ih = 640;
    }

    if (iw != 640 || ih != 640) {
        fprintf(stderr, "Image must be 640x640, got %dx%d\n", iw, ih);
        free(image);
        return 1;
    }

    printf("Running YOLOv8n INT8 inference...\n");
    YoloDet dets[100];
    int n = yolo_infer(image, dets, 100, conf_thr, nms_thr);

    printf("\nDetected %d objects:\n", n);
    for (int i = 0; i < n; i++) {
        printf("  [%d] conf=%.3f  bbox=(%.1f, %.1f, %.1f, %.1f)  class=%d\n",
               i, dets[i].conf, dets[i].x, dets[i].y, dets[i].w, dets[i].h,
               dets[i].class_id);
    }

    free(image);
    return 0;
}
