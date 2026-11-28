"""Bind the real CC0 one-view image-level labels to a new native SF3D input cohort."""

import argparse
import ast
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKENDS = HERE.with_name("training-sf3d-public-surface-native-route-v1-original") / "overlay/generation/src/gencuda/backends"
ORIGINAL = "7d1ff30f4323db2a8e734f0ad8c177c65e14196bd58d1a89f70b2e23cc6635ae"
BACKEND = "c81d32e0103dc7cee323bd434af099cb7267273501dc8334530c621f0dc5300b"
RENDER = HERE.with_name("sf3d-cc0-constant-material-fixed-view-v1")
RENDER_RESULT = "6e2eaaf6e2389cfa3797cf8011142424931e9f6b7688d74c75b0b56fc9d8707b"
RENDER_MANIFEST = "eeebc4e686e14122bdd0380e6506fceac249e2630e885004ce7af3075e76acdc"
GROUP = "metal_tool_chest"
RECIPE = "sf3d-cc0-image-level-constant-metal-tool-chest-v1"
MEDIA = {
    "conditioning.png": (3396, "fdbd86b57f2698187330439cf9f8abb460819dd9ec3c1ec1b2ec7e1e6a339a92"),
    "targets.npz": (683, "b55643c5abb2620a0daad721fa10b8902a9bd59349edd1676998291791c0e6dd"),
}
WEIGHT_LOCK = {
    "config.yaml": (2405, "5e880cfec3dec28d4e20afe07bfa1799c8a561f56f0532ebf1b4022ae4ddc8e4"),
    "model.safetensors": (4024289892, "a3416e1cf654e7d4f5e75f116cec2c3f0a14501a77d30c2f6068bbda178de388"),
}


def pin(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def bounded(path, expected, limit):
    need(path.is_file() and not path.is_symlink(), "EXACT_CANONICAL_MEMBER")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit and pin(raw)["SHA256"] == expected, "EXACT_MEMBER_SHA256_AND_EXTENT")
    return raw


def encode(doc):
    return (json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write(path, raw):
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())


def namespace(path, raw):
    module = {"__name__": "sf3d_isolated_input_boundary", "__file__": str(path)}
    exec(compile(raw, str(path), "exec"), module)
    return module


def render(original, identity, row):
    text = original.decode()
    substitutions = (
        ("Validate the opt-in two-object training input", "Validate the opt-in constant-material one-view input"),
        ('RECIPE = "sf3d-public-surface-area-nearest-PBR4096-v1"', f"RECIPE = {RECIPE!r}"),
        ('GROUPS = ("WaterBottle", "Corset")', f"GROUPS = ({GROUP!r},)"),
        ("expected the fixed WaterBottle/Corset training order", "expected the fixed constant-material tool-chest order"),
        ("8567c9ce3dc0ee8382055c044aa2f837bc2f612000b77210acb8a8b32f902039", identity),
    )
    for old, new in substitutions:
        need(text.count(old) == 1, "EXACT_ORIGINAL_LOADER_EDIT_ANCHOR")
        text = text.replace(old, new)
    start = text.index("    expected_pins = ")
    end = text.index("\n    for row in plan", start)
    value = {GROUP: {key: row[key] for key in ("image_pin", "target_pin")}}
    return (text[:start] + "    expected_pins = " + repr(value) + text[end:]).encode()


def material():
    sample = RENDER / "sample.private"
    prior = json.loads(bounded(RENDER / "RESULT.json", RENDER_RESULT, 8192))
    manifest = json.loads(bounded(sample / "MANIFEST.json", RENDER_MANIFEST, 8192))
    need(prior["specified_material"] == manifest["material"] and prior["verified"] is True
         and prior["visible_pixels"] == 20600, "EXACT_INDEPENDENT_CONSTANT_RENDER")
    payloads = {}
    for name, (extent, digest) in MEDIA.items():
        raw = bounded(sample / name, digest, extent)
        need(len(raw) == extent, "EXACT_SAMPLE_MEMBER_EXTENT")
        payloads[name] = raw
    with Image.open(BytesIO(payloads["conditioning.png"]), formats=("PNG",)) as original:
        original.load()
        need(original.mode == "RGBA" and original.size == (256, 256)
             and original.getchannel("A").histogram()[255] == 20600
             and original.getchannel("A").histogram()[0] == 65536 - 20600, "EXACT_ORIGINAL_RGBA_LAYOUT")
    doc = dict(group=GROUP, image="conditioning.png", target="targets.npz",
               image_pin=pin(payloads["conditioning.png"]), target_pin=pin(payloads["targets.npz"]))
    identity = pin(encode(dict(recipe=RECIPE, ordered=[doc], render_manifest_SHA256=RENDER_MANIFEST)))["SHA256"]
    return doc, identity, payloads, manifest["material"]


def config(folder):
    path = folder / "train-manifest.json"
    return dict(sf3d_surface=dict(manifest=str(path), **pin(path.read_bytes())),
                input_images=[str(folder / "conditioning.png")])


def exercise(new, old, folder, expected):
    plan = config(folder)
    output = new["load_surface"](plan)
    need(len(output["images"]) == len(output["targets"]) == 1
         and pin(output["images"][0])["SHA256"] == MEDIA["conditioning.png"][1], "EXACT_REAL_NATIVE_INPUT_ORDER")
    row, = output["targets"]
    need(row["albedo"].dtype == np.float32 and row["albedo"].shape == (3,)
         and row["albedo"].tolist() == expected["albedo_linear_RGB"]
         and row["roughness"] == expected["roughness"] and row["metallic"] == expected["metallic"],
         "EXACT_NATIVE_PARSED_CONSTANT_VALUES")
    # Confirm native loss-facing dimensions remain the original [1,3], [1], [1].
    need(new["DATA_KEYS"] == old["DATA_KEYS"] == {"albedo": (1, 3), "roughness": (1,), "metallic": (1,)},
         "ORIGINAL_NATIVE_MATERIAL_LOSS_SHAPE")
    malformed = (("legacy_reader", old, plan, "expected the fixed WaterBottle/Corset training order"),
                 ("wrong_order", new, {**plan, "input_images": []}, "configured native input order differs"),
                 ("wrong_manifest_sha", new, {**plan, "sf3d_surface": {
                      **plan["sf3d_surface"], "SHA256": "0" * 64}}, "train input digest differs"))
    rejects = []
    for name, reader, variant, prefix in malformed:
        try:
            reader["load_surface"](variant)
        except ValueError as error:
            need(str(error).startswith(prefix), "EXACT_NATIVE_REJECTION_REASON")
            rejects.append(name)
        else:
            raise ValueError("UNEXPECTED_NATIVE_ADMISSION_" + name)
    return output, rejects


def inspect_backend():
    raw = bounded(BACKENDS / "sf3d_surface_training.py", BACKEND, 65536)
    tree = ast.parse(raw)
    method = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "run_sf3d_training")
    source = ast.unparse(method)
    stages = ["load_surface(config)", "require_cuda()", "len(input_paths) < 2", "SF3D.from_pretrained",
              "_prepare_material_heads(model)", "_prepare_batch(model, image)", "model.get_scene_codes",
              "model.query_triplane(points, scene_codes[0])", "_image_features(torch, model, batch)"]
    order = [source.index(stage) for stage in stages]
    need(order == sorted(order), "ACTUAL_ORIGINAL_PROVIDER_CACHE_SEQUENCE")
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    need(folder.parent == HERE, "EXCLUSIVE_CHILD_PRIVATE_OUTPUT")
    old_path = BACKENDS / "surface_inputs.py"
    original = bounded(old_path, ORIGINAL, 8192)
    doc, identity, media, expected = material()
    generated = render(original, identity, doc)
    manifest = encode(dict(recipe=RECIPE, train=[doc], surface_train_identity_SHA256=identity))
    stages = inspect_backend()
    folder.mkdir(mode=0o700)
    for name, raw in media.items():
        write(folder / name, raw)
    write(folder / "train-manifest.json", manifest)
    fresh = folder / "surface_inputs.py"
    write(fresh, generated)
    observed, rejected = exercise(namespace(fresh, generated), namespace(old_path, original), folder, expected)
    result = dict(schema="SF3D_CC0_CONSTANT_ONE_VIEW_NATIVE_INPUT_BOUNDARY_V1",
                  actual_native_loader_inputs=1, loaded_shapes=namespace(fresh, generated)["DATA_KEYS"],
                  active_backend_source_SHA256=BACKEND, source_reader_SHA256=ORIGINAL,
                  active_backend_stages=stages, frozen_backend_minimum_views=2,
                  generated_reader=pin(generated), training_manifest=pin(manifest),
                  labels_SHA256=observed["labels_sha256"], input_SHA256=observed["input_sha256"],
                  local_negative_checks=rejected, original_backend_modified=False,
                  original_weight_identity={name: dict(bytes=size, SHA256=sha) for name, (size, sha) in WEIGHT_LOCK.items()},
                  target_checkpoint_verified=False, feature_cache_rows=0, model_loads=0, actual_GPU=0,
                  actual_training_rows=0, optimizer_updates=0, training_ready=False)
    raw = encode(result)
    write(folder / "RESULT.json", raw)
    print(json.dumps({"RESULT": pin(raw), **result}, sort_keys=True))


if __name__ == "__main__":
    main()
