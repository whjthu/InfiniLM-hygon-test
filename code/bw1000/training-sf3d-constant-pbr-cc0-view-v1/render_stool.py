"""Render a distinct CC0 stool with a single explicit nonmetal PBR material."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
DATA = HERE.with_name("specialized-cc0-hard-surface-public-pilot-20261128")
ASSET = "painted_wooden_stool"
ADMISSION = "0667c0f1c722e226ec5ac5eeae5e3fcd13dfcb40fe1da320b6c6a8db18ddbcff"
PREVIOUS_RGBA = "9ad9420ba123d17bc79e481175a3af4bfb900a613961c1bb4afbd9e0613e1199"
MATERIAL = dict(ALBEDO=(.1875, .375, .25), ROUGHNESS=.625, METALLIC=0.)


def load(path, wanted):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != wanted:
        raise ValueError("FROZEN_CPU_RENDER_SOURCE")
    namespace = dict(__file__=str(path), __name__="sf3d_second_constant_material")
    exec(compile(raw, str(path), "exec"), namespace)
    return namespace


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify"))
    mode = parser.parse_args().mode
    shader = load(HERE.with_name("sf3d-cc0-constant-material-fixed-view-v1") / "render.py",
                  "088b5020d9c8879658e0bab47ed0f2d27bd53732a7e2ab709e16c86274ef93ca")
    shader.update(MATERIAL)
    lineage = load(HERE.with_name("sf3d-metal-tool-chest-pixel-lineage-v1") / "lineage.py",
                   "dbd28c35f786ff68873db27bf4e29171ae52727c90ee47c45d14683c6c00d0fa")
    path = HERE.parents[1] / "tools/sf3d-polyhaven-metal-tool-chest-fixed-view-v1/view.py"
    view = load(path, "658a9127286ff023fba570e41bca9f75966b0500f6495c2ac433392d41101536")
    need, pin, write, encode = (shader[key] for key in ("need", "pin", "write", "json_bytes"))
    read = lineage["read"]
    receipt = json.loads(read(DATA / ASSET / "admission.json", 16384, ADMISSION))
    need(receipt["asset"] == ASSET and receipt["license"] == "CC0-1.0", "EXACT_CC0_STOOL")
    inputs = {}
    for ending in (".gltf", ".bin"):
        name = ASSET + ending
        fixed = receipt["files"][name]
        inputs[ending] = read(DATA / ASSET / name, fixed["bytes"], fixed["SHA256"])
        need(len(inputs[ending]) == fixed["bytes"], "EXACT_STOOL_MEMBER_SIZE")
    gltf = json.loads(inputs[".gltf"])
    need(gltf["nodes"] == [{"mesh": 0, "name": ASSET}] and gltf["scenes"][gltf["scene"]]["nodes"] == [0],
         "EXACT_STOOL_SOURCE_SCENE")
    primitive, = gltf["meshes"][0]["primitives"]
    mesh = view["module"]("code/bw1000/training-trellis-public-mesh-quality-input-v1/mesh_quality.py", view["MESH"])
    accessor = mesh["accessor"]
    vertices = accessor(gltf, inputs[".bin"], primitive["attributes"]["POSITION"]).astype(np.float64)
    uv = accessor(gltf, inputs[".bin"], primitive["attributes"]["TEXCOORD_0"])
    faces = accessor(gltf, inputs[".bin"], primitive["indices"]).reshape(-1, 3).astype(np.int64)
    need(vertices.shape == (604, 3) and faces.shape == (676, 3) and uv.shape == (604, 2), "EXACT_STOOL_GEOMETRY")
    refs = view["module"]("tools/render_sf3d_boombox_material_views.py", view["FIXED"])
    camera = np.asarray(refs["CAMERAS"][0], dtype=np.float64)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    extent = np.ptp(vertices, axis=0).max()
    screen = refs["project"]((vertices - center) / extent, camera, 256, refs["SPAN"])
    mask, _, ids, bary = lineage["capture_raster"](view)(screen, faces, uv, 256)
    old_path = DATA / "painted-stool-fixed-view-sdxl-sf3d-v1/rgba.png"
    with Image.open(io.BytesIO(read(old_path, 65536, PREVIOUS_RGBA)), formats=("PNG",)) as old:
        frozen = np.asarray(old).copy()
    need(mask.sum() == 12457 and np.array_equal(frozen[:, :, 3], mask * 255), "EXACT_PRIOR_FIXED_STOOL_ALPHA")
    triangle = faces[ids[mask]]
    centers = np.einsum("ni,nij->nj", bary[mask], screen[triangle])
    need(np.max(np.abs(centers[:, :2] - (np.argwhere(mask)[:, ::-1] + .5))) < 1e-10, "CAMERA_PIXEL_CENTERS")
    a, b, c = vertices[triangle].transpose((1, 0, 2))
    normal = np.cross(b - a, c - a)
    length = np.linalg.norm(normal, axis=1)
    need(np.all(length > 0), "FINITE_STOOL_NORMALS")
    normal /= length[:, None]
    normal[normal @ camera < 0] *= -1
    shade, rgb8 = shader["shade"], shader["rgb8"]
    albedo, roughness, metallic = MATERIAL.values()
    rgb = rgb8(shade(normal, camera, albedo, roughness, metallic))
    controls = {
        name: int(np.any(rgb != rgb8(shade(normal, camera, *value)), axis=1).sum())
        for name, value in dict(albedo=((.375, .375, .25), roughness, metallic),
                                roughness=(albedo, .375, metallic), metallic=(albedo, roughness, .5)).items()
    }
    need(all(count > 0 for count in controls.values()), "EACH_MATERIAL_CONTROL_AFFECTS_IMAGE")
    rgba = np.zeros_like(frozen)
    rgba[mask, :3], rgba[mask, 3] = rgb, 255
    need(not np.array_equal(rgba, frozen), "NEWLY_SHADED_RGBA")
    folder = HERE / "sample.private"
    image, fields = folder / "conditioning.png", folder / "targets.npz"
    labels = shader["native_targets"]()
    if mode == "build":
        folder.mkdir()
        with image.open("xb") as stream:
            Image.fromarray(rgba).save(stream, format="PNG", compress_level=9)
        with fields.open("xb") as stream:
            np.savez_compressed(stream, **labels)
        write(folder / "MANIFEST.json", encode(dict(asset=ASSET, admission_SHA256=ADMISSION,
              image=pin(image.read_bytes()), targets=pin(fields.read_bytes()))))
        return
    manifest = json.loads((folder / "MANIFEST.json").read_bytes())
    need(manifest["image"] == pin(image.read_bytes()) and manifest["targets"] == pin(fields.read_bytes()),
         "EXACT_AUTHORED_STOOL_OUTPUTS")
    with Image.open(image, formats=("PNG",)) as decoded:
        need(decoded.mode == "RGBA" and np.array_equal(np.asarray(decoded), rgba), "EXACT_INDEPENDENT_RGBA_READBACK")
    for j in np.linspace(0, len(normal) - 1, 97, dtype=np.int64):
        need(list(rgb[j]) == shader["scalar_rgb"](normal[j], camera), "INDEPENDENT_SCALAR_BRDF")
    with np.load(fields, allow_pickle=False) as source:
        need(set(source.files) == set(labels) and all(source[k].dtype == np.float32 and
             np.array_equal(source[k], v) for k, v in labels.items()), "EXACT_NATIVE_LABEL_VALUES")
    result = dict(**manifest, visible_pixels=int(mask.sum()), original_faces=676, scalar_checks=97,
                  field_controls_changed_pixels=controls, material=MATERIAL, actual_GPU=0,
                  native_feature_rows=0, optimizer_updates=0, actual_training_rows=0)
    raw = encode(result)
    write(HERE / "RESULT.json", raw)
    print(json.dumps(dict(result=pin(raw), **result), sort_keys=True))


if __name__ == "__main__":
    run()
