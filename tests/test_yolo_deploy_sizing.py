from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.yolo_deploy_sizing import (  # noqa: E402
    build_block_plan,
    build_strip_plan,
    build_yolov8n_graph,
    choose_strip_rows,
    encode_ctrl_flags,
    parse_act_quant,
    parse_layers,
    render_block_plan_header,
    summarize,
)


def test_parse_yolov8n_layer_table():
    layers = parse_layers(ROOT / "yolov8n_int8" / "yolov8n_layers.h")
    quant = parse_act_quant(ROOT / "yolov8n_int8" / "yolov8n_layers.h")

    assert len(layers) == 64
    assert len(quant) == 64
    assert layers[0].idx == 0
    assert (layers[0].oc, layers[0].ic, layers[0].kh, layers[0].kw) == (16, 3, 3, 3)
    assert (layers[0].stride, layers[0].pad) == (2, 1)
    assert quant[0].in_zp == -128
    assert quant[0].out_zp == -127
    assert quant[0].has_silu == 1
    assert quant[5].in_zp == -125
    assert quant[5].out_zp == -124
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


def test_block_plan_emits_scheduler_addresses_and_flags():
    layers = parse_layers(ROOT / "yolov8n_int8" / "yolov8n_layers.h")
    graph = build_yolov8n_graph(layers, input_h=640, input_w=640)
    plan = build_block_plan(graph)
    by_idx = {item.idx: item for item in plan}

    assert len(plan) == 63
    assert by_idx[0].input_ddr == 0x40000000
    assert by_idx[0].strip_rows == 8
    assert by_idx[0].strip_count == 40
    assert "HW_PAD" in by_idx[0].ctrl_flags

    assert by_idx[5].input_name == "c2f_2.concat"
    assert by_idx[5].output_name == "conv5"
    assert by_idx[5].input_words == (160 * 160 * 48) // 16
    assert by_idx[5].output_words == (160 * 160 * 32) // 16
    assert by_idx[5].weight_words == (32 * 48) // 16
    assert "PW_EN" in by_idx[5].ctrl_flags
    assert "OC_SINGLE" in by_idx[5].ctrl_flags

    output_ranges = [
        (item.output_ddr, item.output_ddr + item.output_words * 16)
        for item in plan
    ]
    assert output_ranges == sorted(output_ranges)
    for prev, curr in zip(output_ranges, output_ranges[1:]):
        assert prev[1] <= curr[0]


def test_block_plan_header_is_firmware_consumable():
    layers = parse_layers(ROOT / "yolov8n_int8" / "yolov8n_layers.h")
    graph = build_yolov8n_graph(layers, input_h=640, input_w=640)
    plan = build_block_plan(graph)
    by_idx = {item.idx: item for item in plan}
    header = render_block_plan_header(graph)

    assert "#define YOLO_BLOCK_PLAN_COUNT 63u" in header
    assert "typedef struct" in header
    assert "static const yolo_block_plan_entry_t yolo_block_plan" in header
    assert encode_ctrl_flags(by_idx[5].ctrl_flags) == 0x0000001D
    assert (
        "{5u, 160u, 160u, 48u, 160u, 160u, 32u, 1u, 1u, 1u, 0u,"
        in header
    )


def test_block_plan_header_exports_conv0_strip_table():
    layers = parse_layers(ROOT / "yolov8n_int8" / "yolov8n_layers.h")
    graph = build_yolov8n_graph(layers, input_h=640, input_w=640)
    strip_plan = build_strip_plan(graph)
    header = render_block_plan_header(graph)

    assert sum(len(items) for items in strip_plan.values()) == 241
    assert "#define YOLO_STRIP_PLAN_COUNT 241u" in header
    assert "#define YOLO_CONV0_STRIP_PLAN_COUNT 40u" in header
    assert "typedef struct" in header
    assert "uint16_t strip_offset;" in header
    assert "static const yolo_strip_plan_entry_t yolo_strip_plan[YOLO_STRIP_PLAN_COUNT]" in header
    assert "#define yolo_conv0_strip_plan (&yolo_strip_plan[0])" in header
    assert "{0u, 0u, 8u, 0u, 16u, 1u, 0u}," in header
    assert "{1u, 320u, 320u, 16u, 160u, 160u, 32u, 3u, 3u, 2u, 1u, 16u, 10u, 40u," in header
    assert "{0u, 0u, 16u, 0u, 32u, 1u, 0u}," in header
    assert "{1u, 8u, 8u, 15u, 17u, 0u, 0u}," in header
    assert "{39u, 312u, 8u, 623u, 17u, 0u, 0u}," in header


def test_block_plan_header_exports_quant_table():
    layers = parse_layers(ROOT / "yolov8n_int8" / "yolov8n_layers.h")
    graph = build_yolov8n_graph(layers, input_h=640, input_w=640)
    header = render_block_plan_header(graph)

    assert "#define YOLO_ACT_QUANT_COUNT 64u" in header
    assert "static const yolo_act_quant_entry_t yolo_act_quant_plan" in header
    assert "{0.0039215689f, 0.2352971584f, -128, -127, 1u}," in header
    assert "{0.1612435579f, 0.0763198882f, -125, -124, 1u}," in header


def test_strip_plan_tracks_conv0_halo_rows():
    layers = parse_layers(ROOT / "yolov8n_int8" / "yolov8n_layers.h")
    graph = build_yolov8n_graph(layers, input_h=640, input_w=640)
    strips = build_strip_plan(graph)[0]

    assert len(strips) == 40
    assert strips[0].out_y == 0
    assert strips[0].out_rows == 8
    assert strips[0].in_y == 0
    assert strips[0].in_rows == 16
    assert strips[0].top_pad_rows == 1
    assert strips[0].bottom_pad_rows == 0

    assert strips[1].out_y == 8
    assert strips[1].in_y == 15
    assert strips[1].in_rows == 17
    assert strips[1].top_pad_rows == 0

    assert strips[-1].out_y == 312
    assert strips[-1].out_rows == 8
    assert strips[-1].in_y == 623
    assert strips[-1].in_rows == 17
    assert strips[-1].bottom_pad_rows == 0


if __name__ == "__main__":
    test_parse_yolov8n_layer_table()
    test_builds_graph_aware_feature_shapes()
    test_strip_budget_and_summary_are_plausible()
    test_block_plan_emits_scheduler_addresses_and_flags()
    test_block_plan_header_is_firmware_consumable()
    test_block_plan_header_exports_conv0_strip_table()
    test_block_plan_header_exports_quant_table()
    test_strip_plan_tracks_conv0_halo_rows()
    print("PASS: yolo_deploy_sizing")
