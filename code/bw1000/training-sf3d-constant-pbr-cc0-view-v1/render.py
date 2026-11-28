"""Shade the pinned CC0 tool-chest view with one opaque analytic GGX material."""

import argparse
from hashlib import sha256
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = HERE.with_name("sf3d-metal-tool-chest-pixel-lineage-v1")
LINEAGE = "dbd28c35f786ff68873db27bf4e29171ae52727c90ee47c45d14683c6c00d0fa"
POINTS = "438189fbe881377a00a5243eda9798db875a9c4c796544538d221c10fb8ea700"
ALBEDO = (.375, .125, .0625)
ROUGHNESS = .375
METALLIC = .625
LIGHTS = (((-.55, .78, .55), 1.1), ((.82, .42, .42), .65))


def need(ok, code):
    if not ok:
        raise ValueError(code)


def pin(raw):
    return dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())


def json_bytes(doc):
    return (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write(path, data):
    with path.open("xb") as out:
        out.write(data)


def load_geometry():
    path = PREVIOUS / "lineage.py"
    raw = path.read_bytes()
    need(pin(raw)["SHA256"] == LINEAGE, "EXACT_PRIOR_GEOMETRY_SOURCE")
    lineage = {"__name__": "sf3d_pinned_lineage", "__file__": str(path)}
    exec(compile(raw, str(path), "exec"), lineage)
    view, receipt, rgba, targets = lineage["original"]()
    _, _, screen, faces, uv, _, _, material, frame = view["native"]()
    need(all(frame[key] == receipt[key] for key in ("source", "center_m", "extent_m", "camera_direction", "span"))
         and material.get("alphaMode", "OPAQUE") == "OPAQUE", "EXACT_CC0_CAMERA_AND_OPAQUE_MESH")
    geom = lineage["geometry"](view, frame, faces, screen, uv)
    payload = lineage["read"](PREVIOUS / "RESULT/surface-pixel-lineage.private.npz", 4 << 20, POINTS)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        pixels = archive["pixel_yx"]
        ids = archive["source_global_face_id"]
        bary = archive["barycentric"]
    need(np.array_equal(pixels, np.argwhere(targets["mask"])) and pixels.shape == (20600, 2)
         and np.array_equal(rgba[:, :, 3], 255 * targets["mask"]) and ids.min() >= 0 and ids.max() < len(faces),
         "EXACT_FROZEN_20600_VISIBILITY")
    triangle = geom[1][faces[ids]]
    position = np.einsum("ni,nij->nj", bary, triangle)
    projected = ((position - frame["center_m"]) / frame["extent_m"]) @ geom[-2] + geom[-1]
    need(np.max(np.abs(projected[:, :2] - (pixels[:, ::-1] + .5))) < 1e-10, "EXACT_FIXED_CAMERA_PIXEL_CENTERS")
    view_direction = np.array(frame["camera_direction"])
    normals = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    length = np.linalg.norm(normals, axis=1)
    need(np.all(length > 0), "NONDEGENERATE_VISIBLE_FACES")
    normals /= length[:, None]
    normals[normals @ view_direction < 0] *= -1
    return pixels, normals, view_direction, rgba, frame


def shade(normals, camera, albedo, roughness, metallic):
    base = np.asarray(albedo, dtype=np.float64)
    f0 = .04 * (1 - metallic) + metallic * base
    alpha2 = roughness ** 4
    k = (roughness + 1) ** 2 / 8
    nv = np.maximum(1e-9, normals @ camera)
    rgb = np.broadcast_to(.08 * base * (1 - metallic), (len(normals), 3)).copy()
    for direction, power in LIGHTS:
        light = np.asarray(direction) / np.linalg.norm(direction)
        half = camera + light
        half /= np.linalg.norm(half)
        nl = np.maximum(0., normals @ light)
        nh = np.maximum(0., normals @ half)
        vh = min(1., float(camera @ half))
        fresnel = f0 + (1 - f0) * (1 - vh) ** 5
        distribution = alpha2 / (math.pi * (1 + nh * nh * (alpha2 - 1)) ** 2)
        attenuation = (nl / (nl * (1 - k) + k)) * (nv / (nv * (1 - k) + k))
        specular = fresnel * (distribution * attenuation / np.maximum(4 * nl * nv, 1e-9))[:, None]
        diffuse = (1 - fresnel) * (1 - metallic) * base / math.pi
        rgb += power * nl[:, None] * (diffuse + specular)
    return rgb


def rgb8(linear):
    need(np.isfinite(linear).all() and linear.min() >= 0, "FINITE_MATERIAL_LIGHT_TRANSPORT")
    # Exposure fixed at one, Reinhard tone mapping, then sRGB display transfer.
    tone = linear / (1. + linear)
    srgb = np.where(tone <= .0031308, tone * 12.92, 1.055 * tone ** (1 / 2.4) - .055)
    return np.rint(255 * srgb).astype(np.uint8)


def image_and_controls(normals, camera):
    rgb = rgb8(shade(normals, camera, ALBEDO, ROUGHNESS, METALLIC))
    controls = dict(albedo=((.25, .125, .0625), ROUGHNESS, METALLIC),
                    roughness=(ALBEDO, .625, METALLIC), metallic=(ALBEDO, ROUGHNESS, .25))
    effects = {name: int(np.any(rgb != rgb8(shade(normals, camera, *parameters)), axis=1).sum())
               for name, parameters in controls.items()}
    need(all(count > 0 for count in effects.values()) and len(np.unique(rgb, axis=0)) > 3,
         "ALL_THREE_FIELDS_CAUSALLY_AFFECT_ACTUAL_VISIBLE_RENDER")
    return rgb, effects


def native_targets():
    return dict(native_batch_albedo=np.array([ALBEDO], dtype=np.float32),
                native_batch_roughness=np.array([ROUGHNESS], dtype=np.float32),
                native_batch_metallic=np.array([METALLIC], dtype=np.float32))


def scalar_rgb(normal, camera):
    """Independent scalar evaluation of selected emitted pixels."""
    dot = lambda a, b: sum(float(x) * float(y) for x, y in zip(a, b))
    unit = lambda a: tuple(float(x) / math.sqrt(dot(a, a)) for x in a)
    nv = max(1e-9, dot(normal, camera))
    k = (ROUGHNESS + 1) ** 2 / 8
    result = []
    for color in ALBEDO:
        lightness = .08 * color * (1 - METALLIC)
        f0 = .04 + METALLIC * (color - .04)
        for direction, power in LIGHTS:
            light = unit(direction)
            half = unit(tuple(x + y for x, y in zip(light, camera)))
            nl = max(dot(normal, light), 0.)
            nh = max(dot(normal, half), 0.)
            fresnel = f0 + (1 - f0) * (1 - min(dot(half, camera), 1.)) ** 5
            ggx = ROUGHNESS ** 4 / (math.pi * (nh * nh * (ROUGHNESS ** 4 - 1) + 1) ** 2)
            visibility = (nv * nl) / ((nv + (1 - nv) * k) * (nl + (1 - nl) * k))
            spec = ggx * visibility * fresnel / max(1e-9, 4 * nl * nv)
            diffuse = ((1 - METALLIC) * (1 - fresnel) * color) / math.pi
            lightness += nl * power * (diffuse + spec)
        tone = lightness / (1 + lightness)
        value = 12.92 * tone if tone <= .0031308 else 1.055 * tone ** (1 / 2.4) - .055
        result.append(round(value * 255))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("build", "verify"))
    mode = cli.parse_args().mode
    pixels, normals, camera, old, frame = load_geometry()
    rgb, effects = image_and_controls(normals, camera)
    y, x = pixels.T
    rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    rgba[y, x, :3], rgba[y, x, 3] = rgb, 255
    need(np.array_equal(rgba[:, :, 3], old[:, :, 3]) and np.any(rgba != old), "NEW_SHADER_SAME_CAMERA_NO_RELABEL")
    folder = HERE / "sample.private"
    query = native_targets()
    image, targets = folder / "conditioning.png", folder / "targets.npz"
    if mode == "build":
        folder.mkdir()
        with image.open("xb") as out:
            Image.fromarray(rgba).save(out, format="PNG", compress_level=9)
        with targets.open("xb") as out:
            np.savez_compressed(out, **query)
        record = dict(asset="metal_tool_chest", source_license="CC0-1.0", source_faces=13360, kept_alpha_pixels=len(pixels),
                      source_view_lineage_SHA256=POINTS, source_object_payloads=frame["source"],
                      material=dict(albedo_linear_RGB=ALBEDO, roughness=ROUGHNESS, metallic=METALLIC, alpha_mode="OPAQUE"),
                      camera_direction=frame["camera_direction"], span=frame["span"], shader="flat-normal GGX direct lights",
                      actual_control_changed_pixels=effects, image=pin(image.read_bytes()), targets=pin(targets.read_bytes()))
        write(folder / "MANIFEST.json", json_bytes(record))
        return
    manifest = json.loads((folder / "MANIFEST.json").read_bytes())
    need(pin(image.read_bytes()) == manifest["image"] and pin(targets.read_bytes()) == manifest["targets"],
         "EXACT_SAVED_OUTPUT_BYTES")
    with Image.open(image, formats=("PNG",)) as decoded:
        need(decoded.mode == "RGBA" and decoded.size == (256, 256), "EXACT_RGBA_DECODE")
        actual = np.asarray(decoded)
        need(np.array_equal(actual, rgba), "FULL_RENDER_READBACK")
        scalar = np.linspace(0, len(pixels) - 1, 97, dtype=np.int64)
        for j in scalar:
            need(list(actual[y[j], x[j], :3]) == scalar_rgb(normals[j], camera), "INDEPENDENT_SCALAR_BRDF")
    with np.load(targets, allow_pickle=False) as opened:
        need(set(opened.files) == set(query) and all(
            opened[key].dtype == np.float32 and np.array_equal(opened[key], value)
            for key, value in query.items()), "EXACT_MATERIAL_AND_NATIVE_TARGET_SHAPES")
    result = dict(schema="SF3D_CC0_CONSTANT_PBR_FIXED_VIEW_CPU_V1", verified=True, specified_material=manifest["material"],
                  exact_independent_scalar_pixels=len(scalar), visible_pixels=len(pixels), image=manifest["image"],
                  field_shapes={k: list(v.shape) for k, v in query.items()}, target=manifest["targets"],
                  changed_pixels_in_field_controls=effects, verified_native_shape_target=1,
                  actual_native_cohort_admission=False, actual_native_feature_rows=0,
                  actual_native_training_rows=0, actual_optimizer_updates=0, actual_GPU=0)
    raw = json_bytes(result)
    write(HERE / "RESULT.json", raw)
    print(json.dumps({"result": pin(raw), **result}, sort_keys=True))


if __name__ == "__main__":
    main()
