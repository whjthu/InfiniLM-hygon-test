"""Render fixed, opaque geometry conditioning images from the four frozen references."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import time

import numpy as np
import PIL
from PIL import Image

from score import NAMES, RESULT, RESULT_SHA, pinned, require


SIDE = 256
ORTHOGRAPHIC_WIDTH = 1.9


def unit(vector):
    return vector / np.linalg.norm(vector)


TOWARD_CAMERA = unit(np.asarray([3., 2., 3.], dtype=np.float64))
RIGHT = unit(np.cross([0., 1., 0.], TOWARD_CAMERA))
UP = np.cross(TOWARD_CAMERA, RIGHT)
LIGHT = unit(TOWARD_CAMERA - 0.4 * RIGHT + 0.7 * UP)


def rasterize(projected, faces, face_color, side):
    """Depth is larger toward the camera; earlier triangles win exact depth ties."""
    source = projected[faces]
    low = np.maximum(0, np.ceil(source[:, :, :2].min(axis=1) - 0.5)).astype(np.int64)
    high = np.minimum(side - 1, np.floor(source[:, :, :2].max(axis=1) - 0.5)).astype(np.int64)
    depth = np.full((side, side), -np.inf)
    image = np.zeros((side, side, 4), dtype=np.uint8)
    for index in np.flatnonzero(np.all(low <= high, axis=1)):
        a, b, c = source[index]
        ab = b - a
        ac = c - a
        determinant = ab[0] * ac[1] - ab[1] * ac[0]
        if abs(determinant) < 1e-12:
            continue
        x0, y0 = low[index]
        x1, y1 = high[index] + 1
        x = np.arange(x0, x1)[None, :] + (0.5 - a[0])
        y = np.arange(y0, y1)[:, None] + (0.5 - a[1])
        w_b = (x * ac[1] - y * ac[0]) / determinant
        w_c = (y * ab[0] - x * ab[1]) / determinant
        z = a[2] + w_b * ab[2] + w_c * ac[2]
        region = depth[y0:y1, x0:x1]
        keep = (w_b >= -1e-10) & (w_c >= -1e-10) & (w_b + w_c <= 1 + 1e-10) & (z > region)
        region[keep] = z[keep]
        image[y0:y1, x0:x1][keep] = face_color[index]
    return image


def render(vertices, faces):
    require(vertices.dtype == np.dtype("<f4") and vertices.ndim == 2 and vertices.shape[1] == 3
            and 0 < len(vertices) <= 200000 and np.isfinite(vertices).all()
            and np.max(np.abs(vertices)) <= 0.500001, "Unexpected normalized vertices")
    require(faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype.kind in "iu" and 0 < len(faces) <= 250000
            and int(faces.min()) >= 0 and int(faces.max()) < len(vertices), "Unexpected triangle indices")
    vertices = vertices.astype(np.float64)
    points = vertices @ np.stack((RIGHT, UP, TOWARD_CAMERA), axis=-1)
    projected = points.copy()
    projected[:, :2] *= SIDE / ORTHOGRAPHIC_WIDTH
    projected[:, 1] *= -1
    projected[:, :2] += SIDE / 2
    require(np.all((projected[:, :2] > 0) & (projected[:, :2] < SIDE)), "Camera framing clips a vertex")
    triangle = vertices[faces]
    normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    lengths = np.linalg.norm(normal, axis=1)
    real = lengths > 0
    normal = normal[real] / lengths[real, None]
    require(real.any(), "No positive-area triangles")
    normal[normal @ TOWARD_CAMERA < 0] *= -1
    colors = np.full((len(normal), 4), 255, dtype=np.uint8)
    colors[:, :3] = np.rint(255 * (0.23 + 0.70 * np.clip(normal @ LIGHT, 0, 1))).astype(np.uint8)[:, None]
    image = rasterize(projected, faces[real], colors, SIDE)
    return image, len(faces) - int(real.sum())


def pin(raw):
    return {"SHA256": sha256(raw).hexdigest(), "bytes": len(raw)}


def build(reference_root, output):
    evidence = json.loads(pinned(RESULT, 32768, RESULT_SHA))
    output.mkdir(mode=0o700)
    camera = {"view": "fixed-orthographic-shaded-geometry-v1", "frame": evidence["frame"],
              "side_px": SIDE, "width_in_reference_frame": ORTHOGRAPHIC_WIDTH,
              "target": [0, 0, 0], "toward_camera": TOWARD_CAMERA.tolist(), "right": RIGHT.tolist(),
              "up": UP.tolist(), "light": LIGHT.tolist(), "coverage": "pixel-center",
              "z_ties": "first triangle", "textured_materials_used": False}
    report = {"schema": "TRELLIS_PUBLIC_HELDOUT4_CONDITIONS_V1", "reference_manifest_SHA256": RESULT_SHA,
              "camera": camera, "planned_objects": len(NAMES), "objects": {}, "actual_model_predictions": 0,
              "formal_T10_accepted": False, "runtime": {"numpy": np.__version__, "Pillow": PIL.__version__}}
    for name in NAMES:
        start = time.monotonic()
        row = {"passed": False}
        report["objects"][name] = row
        try:
            source = evidence["objects"][name]["artifact"]
            r = pinned(reference_root / f"{name}-reference.npz", source["bytes"], source["SHA256"])
            with np.load(io.BytesIO(r), allow_pickle=False) as data:
                image, zero = render(data["vertices"], data["faces"])
            y, x = np.nonzero(image[:, :, 3])
            require(len(x) > 64 and len(x) < SIDE**2 and x.min() > 0 and x.max() < SIDE - 1
                    and y.min() > 0 and y.max() < SIDE - 1, "Empty or clipped rendered silhouette")
            target = output / f"{name}-condition.png"
            with target.open("xb") as buffer:
                Image.fromarray(image).save(buffer, format="PNG", optimize=False, compress_level=6)
            with Image.open(target) as decoded:
                require(decoded.mode == "RGBA" and np.array_equal(np.asarray(decoded), image), "PNG round trip differs")
            row.update(passed=True, reference_SHA256=source["SHA256"], condition=pin(target.read_bytes()),
                       rgba_SHA256=sha256(image.tobytes()).hexdigest(), opaque_pixels=len(x),
                       bbox_inclusive_px=[int(x.min()), int(y.min()), int(x.max()), int(y.max())],
                       source_degenerate_triangles=zero)
        except (OSError, ValueError, KeyError, TypeError) as error:
            row["failure"] = str(error)[:200]
        row["wall_seconds"] = time.monotonic() - start
    report["completed_objects"] = sum(row["passed"] for row in report["objects"].values())
    with (output / "CONDITIONS.json").open("x", encoding="utf8") as file:
        json.dump(report, file, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.reference_root, args.output)
    print(json.dumps({"completed": result["completed_objects"], "objects": result["objects"]}, sort_keys=True))
    raise SystemExit(int(result["completed_objects"] != len(NAMES)))
