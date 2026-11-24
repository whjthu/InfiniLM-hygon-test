"""Verify publisher WoodenMesh and normalization assets for a local HY-Motion pilot."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from audit_hymotion_accad_skeletons import WOODEN_ASSETS, inspect_wooden


PUBLISHER_REVISION = "76957e3fedc64e80c7288d3630b3d32c7c17e6de"
ADDITIONAL_ASSETS = {
    "v_template.bin": (291072, "b9fbe1f34bfe8a07442d11166e169318022a18da8bc62ce0a9930dfdb3171050"),
    "faces.bin": (290376, "777b0806d2843c797ed18644eecc11466ff822b33b02d263c22f8ad3730e9bb5"),
    "skinWeights.bin": (388096, "343eac2902627ee6b45b547eb7d9f1526562eca7ba178d1dc71f9e466f307a77"),
    "skinIndice.bin": (194048, "846794fb90ea01e069435ad242caefb3d5c2f913fef3255c247f48836ddc1bda"),
    "Mean.npy": (1736, "a4bbe5f68881fe793d509f262ad3dc698a91f79158a9bde9d6f24d46ef9652bd"),
    "Std.npy": (1736, "6cc86d1ca4b47fbd0e84b677c4397e51b6615abcb34e38fdd9daabbe6a5685cc"),
}


def load_asset_bytes(path, name, pin):
    size, digest = pin
    with (path / name).open("rb") as stream:
        content = stream.read(size + 1)
    if len(content) != size or sha256(content).hexdigest() != digest:
        raise ValueError(f"publisher asset bytes differ: {name}")
    return content


def load_geometry(path):
    inspect_wooden(path)
    blob = {name: load_asset_bytes(path, name, pin) for name, pin in ADDITIONAL_ASSETS.items()}
    nodes = np.frombuffer(load_asset_bytes(path, "j_template.bin", WOODEN_ASSETS["j_template.bin"]),
                          dtype="<f4").reshape(-1, 3)
    parents = np.frombuffer(load_asset_bytes(path, "kintree.bin", WOODEN_ASSETS["kintree.bin"]), dtype="<i4")
    vertices = np.frombuffer(blob["v_template.bin"], dtype="<f4").reshape(-1, 3)
    faces = np.frombuffer(blob["faces.bin"], dtype="<u2").reshape(-1, 3)
    weights = np.frombuffer(blob["skinWeights.bin"], dtype="<f4").reshape(-1, 4)
    indices = np.frombuffer(blob["skinIndice.bin"], dtype="<u2").reshape(-1, 4)
    mean, std = (np.load(BytesIO(blob[name]), allow_pickle=False) for name in ("Mean.npy", "Std.npy"))
    if (len(vertices) != len(weights) or weights.shape != indices.shape or len(nodes) != len(parents)
            or faces.max() >= len(vertices) or indices.max() >= len(nodes)
            or not all(np.isfinite(a).all() for a in (vertices, nodes, weights, mean, std))
            or np.any(weights < 0) or np.any(std < 0)
            or not np.allclose(weights.sum(1), 1, atol=2e-7, rtol=0)
            or mean.shape != (201,) or std.shape != (201,)):
        raise ValueError("inconsistent WoodenMesh or 201-dimensional statistics")
    return vertices, faces, weights, indices, nodes, parents, mean.astype("f4"), std.astype("f4")


def audit(path):
    vertices, faces, weights, indices, nodes, parents, mean, std = load_geometry(path)
    replacement = np.flatnonzero(std < 1e-3)
    observed_neutral_pose_root = nodes[0]
    if not np.array_equal(replacement, [135, 136, 137]) or np.any(mean[replacement] != 0):
        raise ValueError("unexpected publisher standard-deviation fallback dimensions")
    return {
        "publisher_revision": PUBLISHER_REVISION,
        "asset_bytes_and_sha256": {
            name: {"bytes": size, "SHA256": pin}
            for name, (size, pin) in {**WOODEN_ASSETS, **ADDITIONAL_ASSETS}.items()
        },
        "vertices": len(vertices), "faces": len(faces), "joints": len(parents),
        "skin_weight_sums_range": [float(weights.sum(1).min()), float(weights.sum(1).max())],
        "skin_joint_max_index": int(indices.max()), "feature_dimensions": len(mean),
        "frozen_producer_std_fallback_indices": replacement.tolist(),
        "publisher_rest_root_not_zero": observed_neutral_pose_root.tolist(),
        "mounted_training_asset_equality_verified": False,
        "motion_201_rows_generated": 0, "text_condition_rows_generated": 0, "training_READY": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("publisher_assets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = (json.dumps(audit(args.publisher_assets), sort_keys=True, indent=2) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(sha256(data).hexdigest())


if __name__ == "__main__":
    main()
