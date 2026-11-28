"""Bind the two verified CC0 renders to the existing native SF3D material reader."""

import argparse
from io import BytesIO
import json
from pathlib import Path

from PIL import Image

import prepare as first


RECIPE = "sf3d-cc0-constant-material-two-object-v1"
RENDERS = (
    ("metal_tool_chest", "sf3d-cc0-constant-material-fixed-view-v1",
     "6e2eaaf6e2389cfa3797cf8011142424931e9f6b7688d74c75b0b56fc9d8707b",
     20600, {"albedo": [.375, .125, .0625], "roughness": .375, "metallic": .625}),
    ("painted_wooden_stool", "sf3d-cc0-painted-stool-nonmetal-fixed-view-v1",
     "5ed91db1e42851f539b7b89242e3f6eee072613dfda899cc09b21edf0ba42396",
     12457, {"albedo": [.1875, .375, .25], "roughness": .625, "metallic": 0.}),
)


def collect(root, group, folder, receipt_sha, visible, material):
    render = root / folder
    receipt = json.loads(first.bounded(render / "RESULT.json", receipt_sha, 8192))
    first.need(receipt["visible_pixels"] == visible, "EXACT_VISIBLE_RENDER")
    image = first.bounded(render / "sample.private/conditioning.png", receipt["image"]["SHA256"], 8192)
    target_pin = receipt.get("target", receipt.get("targets"))
    target = first.bounded(render / "sample.private/targets.npz", target_pin["SHA256"], 4096)
    first.need(first.pin(image) == receipt["image"] and first.pin(target) == target_pin,
               "EXACT_RENDER_MEMBER_EXTENTS")
    with Image.open(BytesIO(image), formats=("PNG",)) as decoded:
        png_alpha = decoded.getchannel("A").histogram()
        first.need(decoded.mode == "RGBA" and decoded.size == (256, 256)
                   and png_alpha[255] == visible and png_alpha[0] == 65536 - visible,
                   "EXACT_RGBA_MASK_AND_CONTEXT")
    row = dict(group=group, image=f"{group}.png", target=f"{group}.npz",
               image_pin=first.pin(image), target_pin=first.pin(target))
    return row, image, target, material


def render_reader(original, rows, identity):
    source = original.decode()
    substitutions = (
        ('RECIPE = "sf3d-public-surface-area-nearest-PBR4096-v1"', f"RECIPE = {RECIPE!r}"),
        ('GROUPS = ("WaterBottle", "Corset")', f"GROUPS = {tuple(row['group'] for row in rows)!r}"),
        ("expected the fixed WaterBottle/Corset training order", "expected the fixed two-object CC0 training order"),
        ("8567c9ce3dc0ee8382055c044aa2f837bc2f612000b77210acb8a8b32f902039", identity),
    )
    for old, new in substitutions:
        first.need(source.count(old) == 1, "EXACT_ORIGINAL_READER_ANCHOR")
        source = source.replace(old, new)
    start = source.index("    expected_pins = ")
    end = source.index("\n    for row in plan", start)
    pins = {row["group"]: {key: row[key] for key in ("image_pin", "target_pin")} for row in rows}
    return (source[:start] + "    expected_pins = " + repr(pins) + source[end:]).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = args.local_root.resolve(strict=True)
    folder = root / args.output
    first.need(folder.resolve().parent == root and not folder.exists(), "NEW_PRIVATE_OUTPUT_DIRECTORY")
    original_path = root / "training-sf3d-public-surface-native-route-v1-original/overlay/generation/src/gencuda/backends/surface_inputs.py"
    original = first.bounded(original_path, first.ORIGINAL, 8192)
    samples = [collect(root, *record) for record in RENDERS]
    rows = [sample[0] for sample in samples]
    identity = first.pin(first.encode(dict(recipe=RECIPE, ordered=rows,
                                          render_receipt_SHA256=[record[2] for record in RENDERS])))["SHA256"]
    manifest = first.encode(dict(recipe=RECIPE, train=rows, surface_train_identity_SHA256=identity))
    source = render_reader(original, rows, identity)
    folder.mkdir(mode=0o700)
    for row, image, target, _ in samples:
        first.write(folder / row["image"], image)
        first.write(folder / row["target"], target)
    first.write(folder / "train-manifest.json", manifest)
    reader = folder / "surface_inputs.py"
    first.write(reader, source)
    env = first.namespace(reader, source)
    config = dict(sf3d_surface=dict(manifest=str(folder / "train-manifest.json"), **first.pin(manifest)),
                  input_images=[str(folder / row["image"]) for row in rows])
    actual = env["load_surface"](config)
    first.need(len(actual["images"]) == len(actual["targets"]) == 2, "ORIGINAL_TWO_VIEW_MINIMUM")
    for observed, image_bytes, (row, _, _, material) in zip(actual["targets"], actual["images"], samples, strict=True):
        first.need(first.pin(image_bytes) == row["image_pin"] and observed["albedo"].tolist() == material["albedo"]
                   and all(observed[key] == material[key] for key in ("roughness", "metallic")), "EXACT_NATIVE_LABELS")
    try:
        env["load_surface"]({**config, "input_images": list(reversed(config["input_images"]))})
    except ValueError as error:
        first.need(str(error) == "configured native input order differs from the train-only manifest", "ORDER_REJECTION")
    else:
        raise ValueError("Native reader accepted reversed conditioning rows")
    result = dict(schema="SF3D_CC0_TWO_OBJECT_NATIVE_READER_CPU_V1", explicit_recipe=RECIPE,
                  groups=[row["group"] for row in rows], source_render_receipt_SHA256=[record[2] for record in RENDERS],
                  input_identity_SHA256=actual["input_sha256"], label_identity_SHA256=actual["labels_sha256"],
                  training_manifest=first.pin(manifest), generated_reader=first.pin(source), native_loader_rows=2,
                  reversed_input_order_rejected=True, model_loads=0, native_feature_rows=0, actual_GPU=0,
                  optimizer_updates=0, training_ready=False)
    first.write(folder / "RESULT.json", first.encode(result))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
