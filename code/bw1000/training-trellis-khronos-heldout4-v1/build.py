"""Build four held-out public geometry references without touching model artifacts."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.with_name("training-trellis-public-mesh-quality-input-v1")
ADAPTER_SHA256 = "8cd2e29e3708319e0ce161e350108b6e1217d43a0d48ceff8ed6d6f68e27456a"
NAMES = ("Avocado", "Lantern", "FlightHelmet", "ToyCar")
FRAME = "reference-normalized-glTF-Yup-v1"


def need(valid, label):
    if not valid:
        raise ValueError(label)


def pin(data):
    return {"SHA256": sha256(data).hexdigest(), "bytes": len(data)}


def read_exact(root, name, expected):
    path = root / name
    need(not path.is_symlink() and path.resolve(strict=True).parent == root, "source path differs")
    with path.open("rb") as file:
        data = file.read(expected["bytes"] + 1)
    need(pin(data) == expected, f"source pin differs: {name}")
    return data


def local_transform(node):
    need("matrix" not in node and "skin" not in node and "weights" not in node, "unexpected non-TRS node")
    q = np.asarray(node.get("rotation", [0, 0, 0, 1]), dtype=np.float64)
    scale = np.asarray(node.get("scale", [1, 1, 1]), dtype=np.float64)
    translation = np.asarray(node.get("translation", [0, 0, 0]), dtype=np.float64)
    need(q.shape == (4,) and scale.shape == translation.shape == (3,)
         and np.isfinite(np.r_[q, scale, translation]).all() and abs(q @ q - 1) < 1e-6
         and (scale != 0).all(), "invalid static node transform")
    x, y, z, w = q
    rotation = np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                         [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                         [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation * scale
    result[:3, 3] = translation
    return result


def geometry(document, blob, accessor):
    need(document["asset"]["version"] == "2.0" and len(document["buffers"]) == 1
         and document["buffers"][0]["byteLength"] == len(blob) and not document.get("extensionsRequired")
         and not document.get("skins") and not document.get("animations"), "unbound glTF scene features")
    nodes = document["nodes"]
    roots = document["scenes"][document.get("scene", 0)]["nodes"]
    parts, faces, ranges = [], [], []
    seen = set()
    offset = 0

    def visit(index, parent):
        nonlocal offset
        need(type(index) is int and 0 <= index < len(nodes) and index not in seen, "glTF node repeated or out of range")
        seen.add(index)
        node = nodes[index]
        world = parent @ local_transform(node)
        if "mesh" in node:
            mesh = document["meshes"][node["mesh"]]
            for primitive_index, primitive in enumerate(mesh["primitives"]):
                need(primitive.get("mode", 4) == 4 and not primitive.get("extensions")
                     and not primitive.get("targets"), "unbound non-static mesh primitive")
                position_id = primitive["attributes"]["POSITION"]
                xyz = accessor(document, blob, position_id)
                indices = accessor(document, blob, primitive["indices"])
                need(xyz.dtype == np.float32 and xyz.ndim == 2 and xyz.shape[1] == 3 and np.isfinite(xyz).all()
                     and indices.ndim == 2 and indices.shape[1] == 1 and indices.dtype.kind == "u"
                     and len(indices) % 3 == 0 and int(indices.max()) < len(xyz), "vertex or index schema differs")
                position = document["accessors"][position_id]
                # glTF specifies POSITION bounds at the precision of the accessor.
                need(np.array_equal(xyz.min(0), np.asarray(position["min"], dtype=xyz.dtype))
                     and np.array_equal(xyz.max(0), np.asarray(position["max"], dtype=xyz.dtype)),
                     "decoded POSITION bounds differ from glTF")
                xyz = xyz.astype(np.float64) @ world[:3, :3].T + world[:3, 3]
                parts.append(xyz)
                faces.append(indices.reshape(-1, 3).astype(np.int64) + offset)
                ranges.append({"node": index, "primitive": primitive_index,
                               "vertex_offset": offset, "vertices": len(xyz), "faces": len(indices) // 3})
                offset += len(xyz)
        for child in node.get("children", ()):
            visit(child, world)

    for index in roots:
        visit(index, np.eye(4))
    need(len(seen) == len(nodes) and parts and len(ranges) <= 16 and offset <= 200000,
         "active scene node or vertex denominator differs")
    xyz, tri = np.concatenate(parts), np.concatenate(faces)
    need(len(tri) <= 250000 and np.isfinite(xyz).all(), "geometry size or finite values differ")
    low, high = xyz.min(0), xyz.max(0)
    center, extent = (low + high) / 2, float((high - low).max())
    need(extent > 0, "reference has zero extent")
    normalized = ((xyz - center) / extent).astype(np.float32)
    return normalized, tri, center, extent, ranges


def build(metadata, buffers, output, plan):
    need(tuple(plan["objects"]) == NAMES and plan["frame"] == FRAME
         and plan["reference_points_per_object"] == 2048, "held-out roster or metric frame differs")
    adapter = PREVIOUS / "mesh_quality.py"
    need(sha256(adapter.read_bytes()).hexdigest() == ADAPTER_SHA256, "previous adapter source differs")
    adopted = runpy.run_path(str(adapter))
    old = json.loads((PREVIOUS / "REFERENCE_INPUTS.json").read_bytes())
    previous_sha = {asset["binary"]["SHA256"] for asset in old["objects"].values()}
    heldout = set()
    output.mkdir()
    result = {"case": "T10.shape-flow-lora", "split": "additional_public_holdout", "frame": FRAME,
              "planned_objects": len(NAMES), "objects": {}, "actual_model_predictions": 0,
              "formal_T10_accepted": False}
    for name in NAMES:
        spec = plan["objects"][name]
        row = {"completed": False}
        result["objects"][name] = row
        try:
            license_raw = read_exact(metadata, f"{name}-LICENSE.md", spec["license"])
            need(b'CC0-1.0' in license_raw and b'CC-BY-4.0' in license_raw, "license scope differs")
            document = json.loads(read_exact(metadata, f"{name}-{name}.gltf", spec["gltf"]))
            blob = read_exact(buffers, f"{name}.bin", spec["buffer"])
            digest = spec["buffer"]["SHA256"]
            need(name not in old["objects"] and digest not in previous_sha | heldout, "duplicate object or buffer")
            need(document["buffers"] == [{"uri": f"{name}.bin", "byteLength": len(blob)}], "declared buffer differs")
            vertices, faces, center, extent, ranges = geometry(document, blob, adopted["accessor"])
            tri = vertices[faces].astype(np.float64)
            nonzero = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) > 0
            need(bool(nonzero.any()), "no positive-area faces")
            points = adopted["sample_surface"](vertices, faces[nonzero], spec["seed"], 2048)
            need(points.shape == (2048, 3) and np.isfinite(points).all(), "reference point denominator differs")
            artifact = output / f"{name}-reference.npz"
            np.savez_compressed(artifact, vertices=vertices, faces=faces, reference_points=points,
                                center_m=center, longest_extent_m=np.asarray(extent))
            row.update(completed=True, vertices=len(vertices), native_faces=len(faces), ranges=ranges,
                       zero_area_faces_retained=int((~nonzero).sum()), sampled_points=len(points),
                       artifact=pin(artifact.read_bytes()))
            heldout.add(digest)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            row["failure"] = str(error)[:240]
    result["completed_objects"] = sum(row["completed"] for row in result["objects"].values())
    result["actual_reference_points"] = sum(row.get("sampled_points", 0) for row in result["objects"].values())
    with (output / "result.json").open("x", encoding="utf8") as file:
        json.dump(result, file, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--buffers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads((HERE / "SOURCE_INPUTS.json").read_bytes())
    result = build(args.metadata.resolve(strict=True), args.buffers.resolve(strict=True), args.output, plan)
    print(json.dumps({"completed": result["completed_objects"], "points": result["actual_reference_points"],
                      "failures": {k: v["failure"] for k, v in result["objects"].items() if "failure" in v}}))
    return int(result["completed_objects"] != 4)


if __name__ == "__main__":
    raise SystemExit(main())
