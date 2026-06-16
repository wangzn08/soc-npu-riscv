/* YOLOv8n INT8 pure-C inference engine */
#ifndef YOLOV8N_INFER_H
#define YOLOV8N_INFER_H

#include <stdint.h>

#define YN_IMAGE_SIZE  640
#define YN_NUM_CLASSES 80
#define YN_NUM_ANCHORS 8400  /* 80*80 + 40*40 + 20*20 */

/* Detection result */
typedef struct {
    float x, y, w, h;   /* center x/y, width, height (pixel coords) */
    float conf;           /* class confidence = obj_conf * class_score */
    int   class_id;
} YoloDet;

/* Load INT8 weights + activation quant metadata from a directory of .bin
 * files. Must be called before yolo_infer(). */
void load_weights(const char *dir);

/* Run full YOLOv8n INT8 inference (genuine int8 activations between every
 * layer, not just int8 weights) on a 640x640 RGB image.
 * Returns number of detections written to `dets` (max `max_dets`).
 * `nms_thr` is the IoU threshold for NMS (0.45 typical). */
int yolo_infer(const uint8_t *image_rgb /* 640*640*3 */,
               YoloDet *dets, int max_dets,
               float conf_thr, float nms_thr);

#endif
