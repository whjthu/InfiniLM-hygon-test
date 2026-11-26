"""Build SF3D target fields from a verified opaque Poly Haven PBR asset."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
from PIL import Image


PBR = Path(__file__).resolve().parents[1] / "code/bw1000/training-sf3d-public-pbr-atlas2-v1/pbr.py"
PBR_SHA = "3bc2279874fefbe2ebdb36ffa74824ee83661c63ea39359099dc266f7437d1e9"
FIELDS = ("albedo", "roughness", "metallic")
RECIPE = "sf3d-public-pbr-UV-atlas-v1"


def pin(raw):
    return dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source(path, maximum, expected=None):
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum and (expected is None or pin(raw) == expected), "identity mismatch")
    return raw


def image(raw):
    with Image.open(BytesIO(raw), formats=("JPEG", "PNG")) as data:
        require(data.mode == "RGB" and data.size == (1024, 1024), "exact 1K RGB atlas required")
        rgb = np.asarray(data).copy()
    require(rgb.dtype == np.uint8 and rgb.shape == (1024, 1024, 3), "native 8-bit atlas required")
    return rgb


def build(asset_dir, manifest_sha, output):
    admission_raw = source(asset_dir / "admission.json", 16_384)
    require(pin(admission_raw)["SHA256"] == manifest_sha, "admission digest mismatch")
    d = json.loads(admission_raw)
    name = d["asset"]
    require(re.fullmatch(r"[a-z0-9_]+", name) and d["publisher"] == "Poly Haven"
            and d["license"] == "CC0-1.0" and d["material_count"] == 1, "one CC0 material required")

    def asset_file(relative):
        require(relative == PurePosixPath(relative).as_posix() and re.fullmatch(r"(?:textures/)?[a-zA-Z0-9_.-]+", relative)
                and ".." not in PurePosixPath(relative).parts, "unsafe atlas member")
        expected = {k: d["files"][relative][k] for k in ("bytes", "SHA256")}
        require(0 < expected["bytes"] <= 2_000_000, "unbounded atlas member")
        return source(asset_dir / relative, expected["bytes"], expected)

    doc = json.loads(asset_file(name + ".gltf"))
    material, = doc["materials"]
    prims = [p for mesh in doc["meshes"] for p in mesh["primitives"]]
    require(prims and all(p.get("material") == 0 and "TEXCOORD_0" in p["attributes"] for p in prims)
            and material.get("alphaMode", "OPAQUE") == "OPAQUE", "opaque single material with UV0 required")

    def texture(slot):
        tag = material["pbrMetallicRoughness"][slot]
        require(tag.get("texCoord", 0) == 0 and not tag.get("extensions"), "unexpected texture transform or UV set")
        return doc["images"][doc["textures"][tag["index"]]["source"]]["uri"]

    base_uri = texture("baseColorTexture")
    orm_uri = texture("metallicRoughnessTexture")
    mapping, = d["material_texture_bindings"]
    require(mapping["albedo"] == base_uri and mapping["roughness_G_metallic_B"] == orm_uri,
            "publisher atlas mapping differs from verified material")
    base, orm = image(asset_file(base_uri)), image(asset_file(orm_uri))
    rgba = np.empty((1024, 1024, 4), dtype=np.uint8)
    rgba[..., :3] = base
    rgba[..., 3] = 255
    pbr = source(PBR, 8192)
    require(pin(pbr)["SHA256"] == PBR_SHA, "frozen SF3D PBR bridge changed")
    api = {"__name__": "frozen_pbr"}
    exec(compile(pbr, str(PBR), "exec"), api)
    uv, samples = api["uniform_uv_sample"](rgba, orm, material)
    native, ranges, atlas_count = api["atlas_targets"](rgba, orm, material)
    require(uv.shape == (256, 2) and len(np.unique(uv, axis=0)) == 256 and atlas_count == 1024 ** 2
            and all(np.isfinite(a).all() for a in (*samples.values(), *native.values())), "incomplete finite UV atlas")
    output.mkdir(mode=0o700)
    bundle = output / "material-targets-private.npz"
    with bundle.open("xb") as stream:
        np.savez_compressed(stream, UV=uv, **{k: samples[k].astype(np.float32) for k in FIELDS},
                            **{"native_batch_" + k: native[k] for k in FIELDS})
    result = dict(asset=name, recipe=RECIPE, admitted_member_SHA256=manifest_sha, PBR_source_SHA256=PBR_SHA,
                  material_count=1, alpha_mode="OPAQUE", primitive_count=len(prims), atlas_texels=atlas_count,
                  UV_samples=len(uv), bundle=pin(bundle.read_bytes()), source_atlas_ranges=ranges,
                  material_target_candidate_rows=1, native_model_training_rows=0, optimizer_updates=0,
                  native_image_features_created=False, visible_surface_reference_rendered=False,
                  split="unassigned_source_candidate")
    with (output / "atlas-receipt.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-dir", type=Path, required=True)
    p.add_argument("--admission-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build(args.asset_dir, args.admission_sha256, args.output), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
