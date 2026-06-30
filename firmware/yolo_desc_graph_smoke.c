#include "firmware.h"
#include "yolo_desc.h"
#include <stdint.h>

void usercode7(void)
{
    print_str("YOLO DESC GRAPH SMOKE\n");
    yolo_desc_reset();
    yolo_desc_graph_begin();
    if (!yolo_desc_graph_end_and_submit()) {
        print_str("GRAPH SUBMIT FAIL\n");
        return;
    }
    print_str("submit_count=");
    print_dec(yolo_desc_submit_count());
    print_str("\n");
    if (yolo_desc_submit_count() != 1u) {
        print_str("YOLO DESC GRAPH SMOKE FAIL\n");
        return;
    }
    print_str("YOLO DESC GRAPH SMOKE PASS\n");
}
