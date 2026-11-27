"""Render one public painted-stool view for SDXL and SF3D input validation."""

import argparse
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "local-only/specialized-cc0-hard-surface-public-pilot-20261128"
ASSET = "painted_wooden_stool"
ADMISSION = "0667c0f1c722e226ec5ac5eeae5e3fcd13dfcb40fe1da320b6c6a8db18ddbcff"
ATLAS_RECEIPT = "bace4aff4a4809e8fdb44b45f9a623730b417c337becf77a20a7dbc9ef864afa"
CAPTION = "A single painted wooden stool with a round seat and braced legs on a plain background."
SIZE = 256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT or output.exists():
        raise ValueError("Use a new output directory inside the existing private data scope")
    legacy = REPO / "tools/sf3d-polyhaven-metal-tool-chest-fixed-view-v1/view.py"
    spec = importlib.util.spec_from_file_location("qualified_view", legacy)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    require, read, pin = common.need, common.read, common.pin
    refs = common.module("tools/render_sf3d_boombox_material_views.py", common.FIXED)
    geometry = common.module("code/bw1000/training-trellis-public-mesh-quality-input-v1/mesh_quality.py", common.MESH)
    pbr = common.module("code/bw1000/training-sf3d-public-pbr-atlas2-v1/pbr.py", common.PBR)
    admission = json.loads(read(ROOT / ASSET / "admission.json", 16384, ADMISSION))
    read(ROOT / (ASSET + "-atlas") / "atlas-receipt.json", 16384, ATLAS_RECEIPT)
    expected = {f"{ASSET}.gltf", f"{ASSET}.bin", *(f"textures/{ASSET}_{name}_1k.jpg" for name in ("diff", "arm", "nor_gl"))}
    require(admission["asset"] == ASSET and admission["license"] == "CC0-1.0"
            and set(admission["files"]) == expected, "Source admission differs")
    raw = {name: read(ROOT / ASSET / name, record["bytes"], record["SHA256"])
           for name, record in admission["files"].items()}
    gltf = json.loads(raw[f"{ASSET}.gltf"])
    binary = raw[f"{ASSET}.bin"]
    require(gltf["asset"]["version"] == "2.0" and gltf["buffers"] == [{"uri": f"{ASSET}.bin", "byteLength": len(binary)}]
            and gltf["nodes"] == [{"mesh": 0, "name": ASSET}] and gltf["scenes"][gltf["scene"]]["nodes"] == [0]
            and not any(gltf.get(key) for key in ("skins", "animations", "extensionsRequired")),
            "Expected original static source geometry")
    material, = gltf["materials"]
    mesh, = gltf["meshes"]
    primitive, = mesh["primitives"]
    require(material.get("alphaMode", "OPAQUE") == "OPAQUE" and material["doubleSided"] is True
            and primitive["material"] == 0 and primitive.get("mode", 4) == 4 and not primitive.get("extensions"),
            "Expected original opaque triangle material")
    attrs = primitive["attributes"]
    accessor = geometry["accessor"]
    points = accessor(gltf, binary, attrs["POSITION"]).astype(np.float64)
    uv = accessor(gltf, binary, attrs["TEXCOORD_0"])
    triangles = accessor(gltf, binary, primitive["indices"]).reshape(-1, 3).astype(np.int64)
    require(points.shape == (604, 3) and uv.shape == (604, 2) and triangles.shape == (676, 3)
            and 0 <= triangles.min() <= triangles.max() < len(points)
            and np.isfinite(points).all() and np.isfinite(uv).all(), "Expected full finite source mesh")
    surface = material["pbrMetallicRoughness"]
    names = []
    for field in ("baseColorTexture", "metallicRoughnessTexture"):
        entry = surface[field]
        require(entry.get("texCoord", 0) == 0 and not entry.get("extensions"), "Unexpected source texture mapping")
        names.append(gltf["images"][gltf["textures"][entry["index"]]["source"]]["uri"])
    require(names == [f"textures/{ASSET}_diff_1k.jpg", f"textures/{ASSET}_arm_1k.jpg"]
            and surface["metallicFactor"] == 0, "Expected genuinely nonmetal source material")
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    extent = float(np.ptp(points, axis=0).max())
    camera = np.asarray(refs["CAMERAS"][0], dtype=np.float64)
    projected = refs["project"]((points - center) / extent, camera, SIZE, refs["SPAN"])
    mask, locations = refs["visible_uv"](projected, triangles, uv, SIZE)
    require(0 < mask.sum() < SIZE**2 and not np.any(mask[[0, -1], :]) and not np.any(mask[:, [0, -1]]),
            "Expected visible and unclipped source")
    positions = locations[mask]
    rgb = refs["sample_texture"](common.decode(raw[names[0]]), positions)
    orm = refs["sample_texture"](common.decode(raw[names[1]]), positions)
    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    rgba[mask] = np.column_stack((rgb, np.full(len(rgb), 255, dtype=np.uint8)))
    targets = pbr["decode_pixels"](rgba[mask], orm, material)
    composite = np.full((SIZE, SIZE, 3), 232, dtype=np.uint8)
    composite[mask] = rgb
    output.mkdir(mode=0o700)
    for name, pixels in (("rgba.png", rgba), ("rgb.png", composite)):
        with (output / name).open("xb") as f:
            Image.fromarray(pixels).save(f, format="PNG")
    with (output / "surface.npz").open("xb") as f:
        np.savez_compressed(f, mask=mask, visible_uv=positions, **targets)
    with (output / "caption.txt").open("x", encoding="utf8", newline="\n") as f:
        f.write(CAPTION + "\n")
    with Image.open(io.BytesIO((output / "rgb.png").read_bytes()), formats=("PNG",)) as f:
        require(f.mode == "RGB" and np.array_equal(np.asarray(f), composite), "RGB read-back differs")
    with np.load(output / "surface.npz", allow_pickle=False) as f:
        require(np.array_equal(f["mask"], mask) and np.array_equal(f["visible_uv"], positions)
                and all(np.array_equal(f[name], arr) and np.isfinite(arr).all() for name, arr in targets.items())
                and np.all(f["metallic"] == 0), "PBR read-back differs")
    report = dict(asset=ASSET, source_group=ASSET, admission_SHA256=ADMISSION,
                  original_atlas_SHA256=ATLAS_RECEIPT, source_files=len(raw), width=SIZE, height=SIZE,
                  source_vertices=len(points), source_faces=len(triangles), foreground_pixels=int(mask.sum()),
                  camera=camera.tolist(), visible_nonmetal=True, unlit_texture_projection=True,
                  split="unassigned_source_candidate", native_conditioning_rows=0, optimizer_updates=0,
                  files={p.name: pin(p.read_bytes()) for p in output.iterdir()})
    with (output / "receipt.json").open("x", encoding="utf8") as f:
        json.dump(report, f, sort_keys=True, allow_nan=False)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
