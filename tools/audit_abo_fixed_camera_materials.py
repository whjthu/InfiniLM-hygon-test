"""Read the one frozen ABO object and inspect actual base/alpha/ORM samples."""

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct

import numpy as np
from PIL import Image, __version__ as pillow_version


REPO = Path(__file__).resolve().parents[1]
OBJECT = "B07M6PHM8C"
GLB = (2073360, "352d13e2a41bc68da722f745ef5f86713786b4fb28c4ad7d8a9acdd877bb7ed8")
GEOMETRY = ("code/bw1000/training-trellis-public-mesh-quality-input-v1/mesh_quality.py", 5502,
            "8cd2e29e3708319e0ce161e350108b6e1217d43a0d48ceff8ed6d6f68e27456a")
PBR = ("code/bw1000/training-sf3d-public-pbr-atlas2-v1/pbr.py", 3808,
       "3bc2279874fefbe2ebdb36ffa74824ee83661c63ea39359099dc266f7437d1e9")
TRELLIS = ("tools/prepare_trellis_export_prediction.py", 8755,
           "8fe604ed390a226d21398f7f751c64a0151569f190a5066b6e34bd5be4c2c427")
ADMISSION = ("results/bw1000/2026-11-27/specialized-abo-first-complete-glb-and-alpha-admission.json", 3340,
             "8529a10995463d8a27b3bd65dd0d8209ea62ba518cf9a18b70414c8be242c365")
SIZE = 32


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, size, digest):
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    need(len(raw) == size and sha(raw) == digest, "EXACT_PIN_REQUIRED: " + path.name)
    return raw


def module(spec):
    relative, size, digest = spec
    path = REPO / relative
    namespace = {"__name__": path.stem, "__file__": str(path)}
    exec(compile(read(path, size, digest), str(path), "exec"), namespace)
    return namespace


def parse(raw):
    need(len(raw) >= 28 and struct.unpack_from("<4sII", raw) == (b"glTF", 2, len(raw)),
         "INVALID_GLB_HEADER")
    size, tag = struct.unpack_from("<I4s", raw, 12)
    stop = 20 + size
    need(tag == b"JSON" and size % 4 == 0 and stop + 8 <= len(raw), "INVALID_JSON_CHUNK")
    binary, tag = struct.unpack_from("<I4s", raw, stop)
    need(tag == b"BIN\0" and stop + 8 + binary == len(raw), "INVALID_BINARY_CHUNK")
    return json.loads(raw[20:stop]), raw[stop + 8:]


def embedded(doc, blob, ref, wanted_mime):
    need(ref.get("texCoord", 0) == 0 and "extensions" not in ref, "ORIGINAL_UV0_REQUIRED")
    texture = doc["textures"][ref["index"]]
    need(doc["samplers"][texture["sampler"]] == {}, "ORIGINAL_DEFAULT_REPEAT_REQUIRED")
    image = doc["images"][texture["source"]]
    need(image.get("mimeType") == wanted_mime and "uri" not in image, "PINNED_EMBEDDED_IMAGE_REQUIRED")
    view = doc["bufferViews"][image["bufferView"]]
    start = view["byteOffset"]
    stop = start + view["byteLength"]
    need(view["buffer"] == 0 and 0 <= start < stop <= len(blob), "IMAGE_VIEW_BOUNDS")
    return blob[start:stop]


def rejected(fn, want):
    try:
        fn()
    except ValueError as error:
        need(str(error) == want, "UNEXPECTED_NATIVE_REJECTION")
        return {"accepted": False, "error": str(error)}
    raise ValueError("EXPECTED_NATIVE_REJECTION_WAS_LOST")


def unit(v):
    return v / np.linalg.norm(v)


def front_hits(world, faces, uv):
    # Fixed orthographic camera in the pinned source-bounds coordinate frame.
    outward = unit(np.array([1., .65, 1.]))
    direction = -outward
    right = unit(np.cross(direction, [0., 1., 0.]))
    up = np.cross(right, direction)
    eye = 2. * outward
    width = 1.45
    grid = ((np.arange(SIZE, dtype=float) + .5) / SIZE - .5) * width
    triangle = world[faces].astype(np.float64)
    e1 = triangle[:, 1] - triangle[:, 0]
    e2 = triangle[:, 2] - triangle[:, 0]
    h = np.cross(direction, e2)
    det = np.sum(e1 * h, axis=1)
    front = det > 1e-12
    inv = np.divide(1., det, out=np.zeros_like(det), where=front)
    out = []
    for r, y in enumerate(grid[::-1]):
        for c, x in enumerate(grid):
            origin = eye + right * x + up * y
            d = origin - triangle[:, 0]
            u = np.sum(d * h, axis=1) * inv
            q = np.cross(d, e1)
            v = np.sum(q * direction, axis=1) * inv
            t = np.sum(q * e2, axis=1) * inv
            candidates = np.flatnonzero(front & (u >= 0.) & (v >= 0.) & (u + v <= 1.) & (t > 0.))
            if len(candidates):
                face = candidates[np.argmin(t[candidates])]
                bary = np.array([1. - u[face] - v[face], u[face], v[face]])
                out.append((r, c, int(face), *(bary @ uv[faces[face]])))
    return out, {"eye": eye.tolist(), "look_at": [0, 0, 0], "up_hint": [0, 1, 0],
                 "right": right.tolist(), "sensor_up": up.tolist(), "orthographic_width": width,
                 "grid": [SIZE, SIZE], "backface_culled": True, "nearest_geometry_hit_only": True}


def texels(uv, shape):
    height, width = shape[:2]
    return np.floor(np.mod(uv, 1.) * [width, height]).astype(np.int64)


def triplet(alpha):
    return dict(transparent=int(np.count_nonzero(alpha == 0)), partial=int(np.count_nonzero((alpha > 0) & (alpha < 255))),
                opaque=int(np.count_nonzero(alpha == 255)))


def inspect_asset(raw):
    geom, pbr, trellis = (module(spec) for spec in (GEOMETRY, PBR, TRELLIS))
    prior = json.loads(read(REPO / ADMISSION[0], *ADMISSION[1:]))["decoded_channels"]
    doc, blob = parse(raw)
    world, faces, frame = geom["read_geometry"](doc, blob)
    primitive, = doc["meshes"][0]["primitives"]
    uv = geom["accessor"](doc, blob, primitive["attributes"]["TEXCOORD_0"])
    normal = geom["accessor"](doc, blob, primitive["attributes"]["NORMAL"])
    need(world.shape == normal.shape == (16062, 3) and faces.shape == (23592, 3)
         and uv.shape == (16062, 2) and np.isfinite(uv).all() and np.isfinite(normal).all(),
         "PUBLISHER_MESH_SHAPE_REQUIRED")
    material = doc["materials"][primitive["material"]]
    need(material["alphaMode"] == "BLEND" and material["doubleSided"] is False, "ORIGINAL_BLEND_MATERIAL_REQUIRED")
    params = material["pbrMetallicRoughness"]
    factors = params["baseColorFactor"]
    need(factors == [1., 1., 1., 1.] and params["roughnessFactor"] == params["metallicFactor"] == 1.,
         "ORIGINAL_PBR_FACTORS_REQUIRED")
    color_raw = embedded(doc, blob, params["baseColorTexture"], "image/png")
    orm_raw = embedded(doc, blob, params["metallicRoughnessTexture"], "image/jpeg")
    need(sha(color_raw) == prior["albedo_source_SHA256"] and sha(orm_raw) == prior["orm_source_SHA256"],
         "ORIGINAL_COLOR_ORM_BYTES_REQUIRED")
    with Image.open(io.BytesIO(color_raw)) as image, Image.open(io.BytesIO(orm_raw)) as orm_image:
        need((image.format, image.mode, orm_image.format, orm_image.mode) == ("PNG", "RGBA", "JPEG", "RGB")
             and image.size == orm_image.size == (4096, 4096), "ORIGINAL_TEXTURE_FORMAT_REQUIRED")
        rgba, orm = np.asarray(image), np.asarray(orm_image)
        hist = image.getchannel("A").histogram()
        observed = triplet(rgba[..., 3])
        need(observed == dict(transparent=hist[0], partial=sum(hist[1:255]), opaque=hist[255])
             == dict(transparent=prior["transparent_texels"], partial=prior["translucent_texels"],
                     opaque=prior["fully_opaque_texels"]), "INDEPENDENT_ALPHA_COUNTS_DISAGREE")
        hits, camera = front_hits(world, faces, uv)
        need(0 < len(hits) < SIZE * SIZE, "FIXED_CAMERA_EMPTY_OR_NO_BACKGROUND")
        rows = np.asarray(hits, dtype=float)
        xy = texels(rows[:, -2:], rgba.shape)
        color_samples, orm_samples = rgba[xy[:, 1], xy[:, 0]], orm[xy[:, 1], xy[:, 0]]
        need(all(tuple(pixel) == image.getpixel(tuple(point)) and tuple(extra) == orm_image.getpixel(tuple(point))
                 for point, pixel, extra in zip(xy, color_samples, orm_samples)), "PIL_AND_ARRAY_CAMERA_TEXELS_DISAGREE")
        counts = triplet(color_samples[:, 3])
        # Anchor independently chosen actual full-texture examples of all 3 alpha classes.
        flat = rgba[..., 3].reshape(-1)
        indices = [int(np.flatnonzero(mask)[0]) for mask in (flat == 0, (flat > 0) & (flat < 255), flat == 255)]
        probe_y, probe_x = np.unravel_index(indices, rgba.shape[:2])
        probe_rgba = rgba[probe_y, probe_x]
        need(probe_rgba.tolist() == [list(image.getpixel((int(x), int(y)))) for y, x in zip(probe_y, probe_x)],
             "PIL_AND_ARRAY_ALPHA_PROBES_DISAGREE")
    original = trellis["original_export_functions"]()
    native_export = rejected(lambda: original["read_glb"](trellis["PinnedGLB"](raw),
        dict(max_bytes=GLB[0], max_faces=300000, texture_size=4096)), "unsupported GLB layout")
    native_export.update(actual_buffer_views=len(doc["bufferViews"]), expected_buffer_views=6,
                         framework_archive_SHA256=trellis["ARCHIVE_SHA"],
                         exporter_source_SHA256=trellis["EXPORTER_SHA"])
    opaque_atlas = rejected(lambda: pbr["decode_pixels"](color_samples, orm_samples, material),
                            "alpha visibility needs a separate recipe")
    linear = pbr["srgb_linear"](color_samples[:, :3].astype(float) / 255.)
    opacity = color_samples[:, 3].astype(float) / 255.
    material_values = np.column_stack([linear, opacity, orm_samples[:, 1:3].astype(float) / 255.])
    need(np.isfinite(material_values).all() and ((0 <= material_values) & (material_values <= 1)).all(),
         "UV_MATERIAL_VALUE_RANGE")
    return dict(schema="ABO_SINGLE_OBJECT_CAMERA_UV_PBR_DIAGNOSTIC_V1", object=OBJECT,
                glb=dict(bytes=len(raw), SHA256=sha(raw)), camera=camera, source_frame=frame,
                actual_geometry_loader=dict(passed=True, vertices=len(world), triangles=len(faces), uv0=len(uv)),
                native_trellis_export_reader=native_export, frozen_sf3d_opaque_pbr=opaque_atlas,
                texture_alpha=observed, camera_slots=dict(planned=SIZE * SIZE, hits=len(hits),
                    misses=SIZE * SIZE - len(hits), hit_alpha=counts, pillow_crosschecks=len(hits)),
                atlas_alpha_controls=[dict(x=int(x), y=int(y), alpha=int(a)) for x, y, a in
                    zip(probe_x, probe_y, probe_rgba[:, 3])],
                sample_channels=["linear_red", "linear_green", "linear_blue", "alpha", "roughness", "metallic"],
                sample_channels_SHA256=sha(material_values.astype("<f8").tobytes()),
                sample_association_SHA256=sha(np.column_stack([rows[:, :3], xy]).astype("<i8").tobytes()),
                sample_filter="base-level nearest diagnostic, repeat and glTF top-left UV; no lighting or alpha compositing",
                planned_calibrated_renders=1, actual_calibrated_renders=0,
                calibrated_render_status="BLOCKED_NO_QUALIFIED_LOCAL_RENDERER",
                actual_native_model_runs=0, actual_training_rows=0, actual_GPU=0,
                optional_renderer_providers={k: bool(importlib.util.find_spec(k)) for k in ("bpy", "pyrender", "OpenGL", "vtk")},
                blender_on_PATH=bool(shutil.which("blender")), numpy=np.__version__, pillow=pillow_version,
                source_pins={path: dict(bytes=size, SHA256=digest) for path, size, digest in (GEOMETRY, PBR, TRELLIS, ADMISSION)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glb", type=Path, default=REPO / "local-only/posttrain26-42-abo-official-bounded-v1"
                        / (OBJECT + "-complete-publisher-v1.glb"))
    args = parser.parse_args()
    report = inspect_asset(read(args.glb, *GLB))
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(bytes=len(raw), SHA256=sha(raw), camera_slots=report["camera_slots"],
                          calibrated_renders=report["actual_calibrated_renders"])))


if __name__ == "__main__":
    main()
