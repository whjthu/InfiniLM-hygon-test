"""Produce a four-row train-only ImageFolder from the two original CC0 training objects."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local-only"
STOOL = LOCAL / "specialized-cc0-hard-surface-public-pilot-20261128"
TWO = {
    "metal_tool_chest": {
        "scope": LOCAL / "polyhaven-metal-tool-chest-1k-domain-v1",
        "admission": "91c6b769abcb038a5f3581bbb48675647eb57e02a66f139e60498ea2d17ab474",
        "original": LOCAL / "sdxl-metal-tool-chest-256-domain-source-v1/metal_tool_chest_0001.png",
        "original_sha": "df0806d6195e8b78ac45b5833356ae7ca656e6ce3bd2cce85434082d81cbf274",
        "caption": "A worn red metal tool chest with drawers and metal hardware, on a plain light gray background.",
        "caption_sha": "0a2f6b50e1631cb5a5dc5ac47c6c34e694275288f6176f2a01b0bdcea41a4506",
        "original_caption_suffix": "",
        "vertices": 11073,
        "faces": 13360,
    },
    "painted_wooden_stool": {
        "scope": STOOL / "painted_wooden_stool",
        "admission": "0667c0f1c722e226ec5ac5eeae5e3fcd13dfcb40fe1da320b6c6a8db18ddbcff",
        "original": STOOL / "painted-stool-fixed-view-sdxl-sf3d-v1/rgb.png",
        "original_sha": "ebc9641bdc1c3b1db45821abb81585fa92ee65da97bbcef04bf227f47b82cd73",
        "caption": "A single painted wooden stool with a round seat and braced legs on a plain background.",
        "caption_sha": "c27d943853f15401eafa9efc17bda3b8c32e19da316e45496afd4fb30d30435b",
        "original_caption_suffix": "\n",
        "vertices": 604,
        "faces": 676,
    },
}
EXCLUDED = ("cassette_player", "vintage_electric_kettle", "vintage_telephone_wall_clock")
CAMERA = (-2.0, 0.5, 1.0)
SIZE = 256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output.resolve()
    loader = REPO / "tools/sf3d-polyhaven-metal-tool-chest-fixed-view-v1/view.py"
    raw = loader.read_bytes()
    if sha256(raw).hexdigest() != "658a9127286ff023fba570e41bca9f75966b0500f6495c2ac433392d41101536":
        raise ValueError("Exact original admitted-view support is required")
    common = {"__name__": "frozen_cc0_view_support", "__file__": str(loader)}
    exec(compile(raw, str(loader), "exec"), common)
    need, read, pin = (common[name] for name in ("need", "read", "pin"))
    need(output.is_relative_to(LOCAL) and output != LOCAL and not output.exists(), "NEW_PRIVATE_OUTPUT_ONLY")
    refs = common["module"]("tools/render_sf3d_boombox_material_views.py", common["FIXED"])
    geometry = common["module"]("code/bw1000/training-trellis-public-mesh-quality-input-v1/mesh_quality.py", common["MESH"])
    need(refs["CAMERAS"][1] == CAMERA and not set(EXCLUDED).intersection(TWO), "PRESELECTED_SIDE_VIEW_AND_TRAIN_FAMILIES")
    camera = np.asarray(CAMERA, dtype=np.float64)
    assets, file_rows = {}, []
    for asset, row in TWO.items():
        root = row["scope"]
        admission = json.loads(read(root / "admission.json", 16 << 10, row["admission"]))
        material_names = {f"textures/{asset}_{suffix}_1k.jpg" for suffix in ("diff", "arm", "nor_gl")}
        need(admission["asset"] == asset and admission["license"] == "CC0-1.0"
             and set(admission["files"]) == {asset + ".gltf", asset + ".bin", *material_names},
             "EXACT_ORIGINAL_CC0_ADMISSION")
        source = {name: read(root / name, entry["bytes"], entry["SHA256"])
                  for name, entry in admission["files"].items()}
        gltf, raw_mesh = json.loads(source[asset + ".gltf"]), source[asset + ".bin"]
        need(gltf["asset"]["version"] == "2.0"
             and gltf["buffers"] == [{"uri": asset + ".bin", "byteLength": len(raw_mesh)}]
             and gltf["scenes"][gltf["scene"]]["nodes"] == list(range(len(gltf["nodes"])))
             and not any(gltf.get(key) for key in ("skins", "animations", "extensionsRequired")),
             "EXACT_STATIC_PUBLISHER_MESH")
        material, = gltf["materials"]
        texture = material["pbrMetallicRoughness"]["baseColorTexture"]
        uri = gltf["images"][gltf["textures"][texture["index"]]["source"]]["uri"]
        need(material.get("alphaMode", "OPAQUE") == "OPAQUE"
             and material["pbrMetallicRoughness"].get("baseColorFactor", [1, 1, 1, 1]) == [1, 1, 1, 1]
             and texture.get("texCoord", 0) == 0 and not texture.get("extensions")
             and uri == f"textures/{asset}_diff_1k.jpg", "EXACT_UNLIT_ALBEDO_SOURCE")
        vertices, uvs, faces, offset = [], [], [], 0
        for index, node in enumerate(gltf["nodes"]):
            need(node["mesh"] == index and set(node) <= {"mesh", "name", "translation"}, "ORIGINAL_NODE_TRANSFORM")
            primitive, = gltf["meshes"][index]["primitives"]
            need(primitive.get("mode", 4) == 4 and primitive["material"] == 0
                 and not primitive.get("extensions"), "ORIGINAL_OPAQUE_TRIANGLES")
            accessor = geometry["accessor"]
            xyz = accessor(gltf, raw_mesh, primitive["attributes"]["POSITION"]).astype(np.float64)
            uv = accessor(gltf, raw_mesh, primitive["attributes"]["TEXCOORD_0"])
            face = accessor(gltf, raw_mesh, primitive["indices"]).reshape(-1, 3)
            move = np.asarray(node.get("translation", (0, 0, 0)), dtype=np.float64)
            need(xyz.shape == (len(uv), 3) and uv.shape[1] == 2 and np.isfinite(xyz).all()
                 and np.isfinite(uv).all() and face.dtype.kind == "u" and face.max() < len(xyz)
                 and move.shape == (3,) and np.isfinite(move).all(), "EXACT_VALID_GEOMETRY")
            vertices.append(xyz + move)
            uvs.append(uv)
            faces.append(face.astype(np.int64) + offset)
            offset += len(xyz)
        world, uv, triangles = map(np.concatenate, (vertices, uvs, faces))
        need(len(world) == row["vertices"] and len(triangles) == row["faces"], "EXACT_FULL_GEOMETRY_COUNTS")
        if not material.get("doubleSided", False):
            a, b, c = world[triangles].transpose(1, 0, 2)
            triangles = triangles[np.cross(b - a, c - a) @ camera > 0]
        center = (world.max(0) + world.min(0)) / 2
        screen = refs["project"]((world - center) / np.ptp(world, axis=0).max(), camera, SIZE, refs["SPAN"])
        mask, coordinates = refs["visible_uv"](screen, triangles, uv, SIZE)
        need(0 < mask.sum() < SIZE**2 and not np.any(mask[[0, -1], :])
             and not np.any(mask[:, [0, -1]]), "VISIBLE_UNCLIPPED_OBJECT")
        rgb = np.full((SIZE, SIZE, 3), 232, dtype=np.uint8)
        rgb[mask] = refs["sample_texture"](common["decode"](source[uri]), coordinates[mask])
        generated = io.BytesIO()
        Image.fromarray(rgb).save(generated, format="PNG")
        original = read(row["original"], 2 << 20, row["original_sha"])
        side = generated.getvalue()
        need(pin(side) != pin(original), "SIDE_VIEW_DIFFERS_FROM_ORIGINAL")
        original_caption = (row["caption"] + row["original_caption_suffix"]).encode()
        need(pin(original_caption)["SHA256"] == row["caption_sha"], "ORIGINAL_CAPTION_IDENTITY")
        for name, payload in (("original", original), ("side", side)):
            with Image.open(io.BytesIO(payload), formats=("PNG",)) as image:
                need(image.mode == "RGB" and image.size == (SIZE, SIZE), "EXACT_RGB_256")
                image.load()
            file_rows.append((asset + "_" + name + ".png", row["caption"], payload))
        assets[asset] = dict(admission_SHA256=row["admission"], rgb_foreground_pixels=int(mask.sum()),
                             vertices=len(world), full_source_faces=row["faces"], raster_faces=len(triangles))
    output.mkdir(mode=0o700)
    images = output / "imagefolder"
    images.mkdir(mode=0o700)
    metadata = []
    file_pins = {}
    for filename, caption, payload in file_rows:
        with (images / filename).open("xb") as stream:
            stream.write(payload)
        metadata.append(dict(file_name=filename, text=caption))
        file_pins[filename] = pin(payload)
    with (images / "metadata.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in metadata:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    file_pins["metadata.jsonl"] = pin((images / "metadata.jsonl").read_bytes())
    receipt = dict(train_families=assets, holdout_excluded=EXCLUDED, fixed_side_camera=CAMERA,
                   background_rgb=[232, 232, 232], resolution=[SIZE, SIZE], rows=len(metadata),
                   views_per_train_family=2, native_imagefolder_rows=0, optimizer_updates=0, files=file_pins)
    with (output / "receipt.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
