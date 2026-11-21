"""Freeze unlit fixed-view material references from the original BoomBox mesh."""

from hashlib import sha256
import io
import json
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
ASSETS = {
    "gltf": ("local-only/training-sf3d-public-pbr-atlas2-v1-original/BoomBox/glTF/BoomBox.gltf",
             2702, "b1d2e0e56864c71a4b65fc560bfc1572b1eb32cfb9c8e86d06e6f9ace2803a43"),
    "geometry": ("local-only/training-trellis-public-mesh-quality-input-v1-original/BoomBox.bin",
                 207816, "f6c61a95af3e0e2462a9ac01d1e681706149452e0191930e0591a81d50b5b944"),
    "base": ("local-only/training-sf3d-public-pbr-atlas2-v1-original/BoomBox/glTF/BoomBox_baseColor.png",
             3285844, "099816a7afc5f6690494313ac8039806fd6d5b84179126a808b2678aaab3563a"),
    "orm": ("local-only/training-sf3d-BoomBox-ORM-onefile-completion-v1-original/BoomBox_occlusionRoughnessMetallic.png",
            4471451, "496704a4836ff364dc4441f41651edb25178498926c859933e33df42d6361412"),
}
MESH = ("code/bw1000/training-trellis-public-mesh-quality-input-v1/mesh_quality.py",
        "8cd2e29e3708319e0ce161e350108b6e1217d43a0d48ceff8ed6d6f68e27456a")
PBR = "code/bw1000/training-sf3d-public-pbr-atlas2-v1"
PBR_MANIFEST_SHA = "9918612f5815bd99fad10321f20735a20a36cf9888adaa2979db04ca196a3b0f"
JOIN = ("code/bw1000/training-sf3d-public-atlas3-resume-target-join-v1/target_join3.py",
        "151af51e11692127e00ca6521901ed0c5503c48ecf8959d7ed5f654fb327f4f3")
CAMERAS = ((1.0, 0.5, 2.0), (-2.0, 0.5, 1.0), (1.0, 0.5, -2.0))
SPAN, SIZE = 1.80, 256
OUT = REPO / "local-only/training-sf3d-boombox-material-fixed-views-v1-original"


def digest(raw):
    return sha256(raw).hexdigest()


def read(path, maximum, expected_sha):
    with (REPO / path).open("rb") as stream:
        raw = stream.read(maximum + 1)
    if not 0 < len(raw) <= maximum or digest(raw) != expected_sha:
        raise ValueError("frozen file identity differs: " + Path(path).name)
    return raw


def checked_module(path, sha):
    scope = {"__file__": str(REPO / path)}
    exec(compile(read(path, 32_768, sha), path, "exec"), scope)
    return scope


def project(vertices, camera, size=SIZE, span=SPAN):
    forward = np.asarray(camera, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross((0.0, 1.0, 0.0), forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return vertices @ np.stack((right, -up, forward)).T * (size / span) + [size / 2, size / 2, 0.0]


def visible_uv(screen, faces, uv, size):
    depth = np.full((size, size), -np.inf)
    uv_map = np.zeros((size, size, 2), dtype=np.float64)
    for face in faces:
        triangle = screen[face]
        a, b, c = triangle[:, :2]
        edge_b, edge_c = b - a, c - a
        area = edge_b[0] * edge_c[1] - edge_b[1] * edge_c[0]
        if abs(area) < 1e-10:
            continue
        x0, y0 = np.maximum(0, np.floor(triangle[:, :2].min(axis=0)).astype(int))
        x1, y1 = np.minimum(size, np.ceil(triangle[:, :2].max(axis=0)).astype(int))
        if x1 <= x0 or y1 <= y0:
            continue
        x = np.arange(x0, x1)[None, :] + 0.5 - a[0]
        y = np.arange(y0, y1)[:, None] + 0.5 - a[1]
        wb = (x * edge_c[1] - y * edge_c[0]) / area
        wc = (edge_b[0] * y - edge_b[1] * x) / area
        wa = 1.0 - wb - wc
        z = wa * triangle[0, 2] + wb * triangle[1, 2] + wc * triangle[2, 2]
        clip = np.s_[y0:y1, x0:x1]
        replace = (wa >= -1e-12) & (wb >= -1e-12) & (wc >= -1e-12) & (z > depth[clip])
        depth[clip][replace] = z[replace]
        uv_map[clip][replace] = np.stack((wa[replace], wb[replace], wc[replace]), axis=1) @ uv[face]
    return np.isfinite(depth), uv_map


def sample_texture(texture, uv):
    h, w = texture.shape[:2]
    xy = np.floor(np.remainder(uv, 1.0) * [w, h]).astype(np.int64)
    return texture[xy[:, 1], xy[:, 0]]


def load_texture(raw, mode):
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "PNG" or image.size != (2048, 2048):
            raise ValueError("unexpected original texture geometry")
        return np.asarray(image.convert(mode)).copy()


def score(fields, prediction):
    return {name: float(np.abs(values - np.asarray(prediction[name]).reshape(-1)).mean())
            for name, values in fields.items()}


def render_views(document, geometry, base, orm, predictors, output):
    mesh = checked_module(*MESH)
    pbr_index = json.loads(read(PBR + "/SOURCE_FILES.json", 16_384, PBR_MANIFEST_SHA))
    pbr = checked_module(PBR + "/pbr.py", pbr_index["files"]["pbr.py"]["SHA256"])
    vertices, faces, frame = mesh["read_geometry"](document, geometry)
    primitive = document["meshes"][0]["primitives"][0]
    uv = mesh["accessor"](document, geometry, primitive["attributes"]["TEXCOORD_0"])
    if uv.shape != (len(vertices), 2) or len(faces) != 6036 or not np.isfinite(uv).all():
        raise ValueError("texture coordinate or triangle denominator differs")
    material = document["materials"][primitive["material"]]
    renders = []
    for index, camera in enumerate(CAMERAS):
        mask, uv_map = visible_uv(project(vertices, camera), faces, uv, SIZE)
        if not mask.any() or np.any(mask[[0, -1], :]) or np.any(mask[:, [0, -1]]):
            raise ValueError("invalid or clipped fixed-view silhouette")
        selected = uv_map[mask]
        rgba, rm = sample_texture(base, selected), sample_texture(orm, selected)
        unlit = np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)
        unlit[mask] = rgba[:, :3]
        path = output / f"view-{index}-basecolor.png"
        Image.fromarray(unlit).save(path)
        np.savez_compressed(output / f"view-{index}-reference.npz", mask=mask, uv=selected, orm=rm)
        fields = pbr["decode_pixels"](rgba, rm, material)
        metrics = {name: score(fields, prediction) for name, prediction in predictors.items()}
        renders.append(dict(view=index, camera_direction=list(camera), foreground_pixels=int(mask.sum()),
                            basecolor_png_SHA256=digest(path.read_bytes()),
                            reference_geometry_material_SHA256=digest((output / f"view-{index}-reference.npz").read_bytes()),
                            foreground_material_MAE=metrics))
    return renders, frame


def main():
    start = monotonic()
    all_assets = {name: read(*spec) for name, spec in ASSETS.items()}
    training_groups = ("WaterBottle", "Corset")
    join = checked_module(*JOIN)
    training = [join["targets_for_group"](group, purpose="train")[0] for group in training_groups]
    predictions = {
        "training_object_mean": {name: np.mean([target[name] for target in training], axis=0)
                                 for name in ("albedo", "roughness", "metallic")},
    }
    OUT.mkdir(mode=0o700)
    views, frame = render_views(json.loads(all_assets["gltf"]), all_assets["geometry"],
                                load_texture(all_assets["base"], "RGBA"), load_texture(all_assets["orm"], "RGB"),
                                predictions, OUT)
    report = dict(state="REAL_BOOMBOX_FIXED_VIEW_REFERENCES_AND_PIXEL_MAE_PASS", object="BoomBox", split="dev",
                  source_SHA256={name: spec[2] for name, spec in ASSETS.items()},
                  predictors={"training_object_mean": list(training_groups)}, training_material_join_SHA256=JOIN[1],
                  cameras_planned=3, cameras_rendered=len(views), resolution=[SIZE, SIZE], orthographic_span=SPAN,
                  frame=frame, nearest_repeat_uv=True, two_sided_visibility=True, views=views,
                  LPIPS=None, actual_training_or_model_outputs=0, training_READY_increment=0,
                  wall_seconds=monotonic()-start)
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with (OUT / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
