from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.yolo_deploy_sizing import (  # noqa: E402
    build_yolov8n_graph,
    choose_strip_rows,
    parse_layers,
    summarize,
)


def test_parse_yolov8n_layer_table():
    layers = parse_layers(ROOT / "yolov8n_int8" / "yolov8n_layers.h")

    assert len(layers) == 64
    assert layers[0].idx == 0
    assert (layers[0].oc, layers[0].ic, layers[0].kh, layers[0].kw) == (16, 3, 3, 3)
    assert (layers[0].stride, layers[0].pad) == (2, 1)
    assert layers[63].idx == 63
    assert (layers[63].oc, layers[63].ic, layers[63].kh, layers[63].kw) == (1, 16, 1, 1)


def test_builds_graph_aware_feature_shapes():
    layers = parse_layers(ROOT / "yolov8n_int8" / "yolov8n_layers.h")
    graph = build_yolov8n_graph(layers, input_h=640, input_w=640)

    assert (graph.by_idx[0].out_h, graph.by_idx[0].out_w) == (320, 320)
    assert (graph.by_idx[1].out_h, graph.by_idx[1].out_w) == (160, 160)
    assert (graph.by_idx[27].in_c, graph.by_idx[27].in_h, graph.by_idx[27].in_w) == (384, 40, 40)
    assert (graph.by_idx[31].in_c, graph.by_idx[31].in_h, graph.by_idx[31].in_w) == (192, 80, 80)
    assert (graph.by_idx[58].in_c, graph.by_idx[58].in_h, graph.by_idx[58].in_w) == (256, 20, 20)
    assert graph.by_idx[63].role == "cpu_dfl"


def test_strip_budget_and_summary_are_plausible():
    layers = parse_layers(ROOT / "yolov8n_int8" / "yolov8n_layers.h")
    graph = build_yolov8n_graph(layers, input_h=640, input_w=640)

    rows = choose_strip_rows(graph.by_idx[0], act_sram_bytes=256 * 1024)
    assert rows == 8

    summary = summarize(graph.shapes)
    assert summary["npu_conv_layers"] == 63
    assert summary["cpu_tail_layers"] == 1
    assert 4_000_000_000 < summary["total_macs"] < 5_000_000_000
    assert summary["total_weight_bytes_aligned"] > 3_000_000


if __name__ == "__main__":
    test_parse_yolov8n_layer_table()
    test_builds_graph_aware_feature_shapes()
    test_strip_budget_and_summary_are_plausible()
    print("PASS: yolo_deploy_sizing")
