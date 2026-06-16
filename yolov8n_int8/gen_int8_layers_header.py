"""Regenerate yolov8n_layers.h: keep the existing YoloConvCfg dims/files
table byte-for-byte, append a new YoloActQuant table built from
act_quant_meta.json."""
import json
import os
import re

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LAYERS_H_PATH = os.path.join(OUT_DIR, "yolov8n_layers.h")
META_PATH = os.path.join(OUT_DIR, "act_quant_meta.json")


def main():
    with open(LAYERS_H_PATH) as f:
        original = f.read()
    with open(META_PATH) as f:
        meta = json.load(f)

    convs = meta["convs"]
    assert len(convs) == 64

    lines = []
    lines.append("\ntypedef struct {")
    lines.append("    float in_scale, out_scale;")
    lines.append("    int   in_zp, out_zp;")
    lines.append("    int   has_silu;")
    lines.append("} YoloActQuant;\n")
    lines.append("static const YoloActQuant yolo_act_quant[64] = {")
    for m in convs:
        lines.append(
            "    {%.10ff, %.10ff, %d, %d, %d},"
            % (m["in_scale"], m["out_scale"], m["in_zp"], m["out_zp"],
               1 if m["has_silu"] else 0)
        )
    lines.append("};\n")

    glue = meta["glue_ops"]
    lines.append("#define YOLO_NUM_GLUE_OPS %d" % len(glue))
    lines.append("typedef struct {")
    lines.append("    const char *name;")
    lines.append("    float out_scale;")
    lines.append("    int   out_zp;")
    lines.append("} YoloGlueQuant;\n")
    lines.append("static const YoloGlueQuant yolo_glue_quant[%d] = {" % len(glue))
    for g in glue:
        safe_name = g["name"].replace('"', '\\"')
        lines.append('    {"%s", %.10ff, %d},' % (safe_name, g["out_scale"], g["out_zp"]))
    lines.append("};\n")

    # Idempotent: strip any previously-generated block so re-running does not
    # stack duplicate definitions. The generated block always starts with the
    # YoloActQuant typedef; that exact text never appears in the hand-authored
    # part of the header.
    marker = "\ntypedef struct {\n    float in_scale, out_scale;"
    cut = original.find(marker)
    if cut != -1:
        base = original[:cut].rstrip()
    else:
        base = original[:original.rindex("#endif")].rstrip()

    new_content = base + "\n\n" + "\n".join(lines) + "\n\n#endif\n"

    with open(LAYERS_H_PATH, "w") as f:
        f.write(new_content)
    print(f"Wrote YoloActQuant[64] and YoloGlueQuant[{len(glue)}] to {LAYERS_H_PATH} (idempotent)")


if __name__ == "__main__":
    main()
