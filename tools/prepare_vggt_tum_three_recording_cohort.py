"""Bind real inputs for the next VGGT domain cohort to the canonical RGB-D adapter."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "code/bw1000/training-vggt-domain-eight-pairs25-v1/data_contract.py"
ADAPTER = REPO / "code/bw1000/training-domain-pilot-v1/tum_vggt_pair.py"
CONTRACT_SHA = "aef0765500140c237fb8c57adfa53d0ba024064c67a1716e9610ca3448f4d898"
ADAPTER_SHA = "c453a9c561c6411b8c2f25652f5c004c928db423c03ca0a45eba7d47ed9e3962"
ORIGINAL_BUNDLE_SHA = "fdc6b68196db7403a00dada66bda274d08d737e83336cb4c3758362f7438b14c"
NEW_PAIR_SHA = "9a39e38c2acf5da7aacfc08729c60307cc048ea2293be81c83b40a69f6695cfa"
NEW_SEQUENCE = "freiburg3_sitting_xyz"
LAYOUTS = {
    "images": (2, 224, 224, 3),
    "depths": (2, 224, 224),
    "intrinsics": (2, 3, 3),
    "extrinsics": (2, 3, 4),
    "point_masks": (2, 224, 224),
}
RESERVED_RECORDINGS = {
    "freiburg3_cabinet", "freiburg3_teddy", "freiburg1_plant", "freiburg2_dishes",
    "freiburg3_long_office_household",
}


def digest(raw):
    return sha256(raw).hexdigest()


def read_pinned(path, expected, limit):
    if path.is_symlink():
        raise ValueError(f"Aliased input {path.name}")
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if not value or len(value) > limit or digest(value) != expected:
        raise ValueError(f"Missing or changed pin for {path.name}")
    return value


def verified_definitions(path, expected):
    raw = read_pinned(path, expected, 25000)
    scope = {"__name__": "vggt_cohort_dependency", "__file__": str(path)}
    exec(compile(raw, str(path), "exec"), scope)
    return scope


def fingerprint(value):
    contiguous = np.ascontiguousarray(value)
    return dict(shape=list(contiguous.shape), dtype=contiguous.dtype.str,
                SHA256=digest(contiguous.tobytes()))


def checked_input(raw, sequence):
    arrays = {key: np.stack(raw[key]) for key in LAYOUTS}
    if raw["seq_name"] != sequence:
        raise ValueError("Adapter returned a different recording")
    for key, shape in LAYOUTS.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Adapter layout or finite values differ for {key}")
    if arrays["images"].dtype != np.uint8 or arrays["point_masks"].dtype != np.bool_:
        raise ValueError("Adapter image or mask dtype differs")
    masks = arrays["point_masks"]
    if not np.array_equal(masks, arrays["depths"] > 0) or np.any(masks.sum(axis=(1, 2)) <= 100):
        raise ValueError("Canonical useful-depth requirement failed")
    return {name: fingerprint(array) for name, array in arrays.items()}


def load_cohort(base_root, new_root):
    """Return raw adapter batches and a data receipt without loading a model."""
    contract = verified_definitions(CONTRACT, CONTRACT_SHA)
    geometry = verified_definitions(ADAPTER, ADAPTER_SHA)
    verified = contract["verify_bundle"](base_root, ORIGINAL_BUNDLE_SHA)
    descriptors = [
        (base_root / item["root"], base_root / item["manifest"], item["manifest_SHA256"], item["sequence"])
        for item in verified["pairs"]
    ]
    descriptors.append((new_root, new_root / "pair-manifest.json", NEW_PAIR_SHA, NEW_SEQUENCE))
    rows = []
    batches = []
    seen = set()
    for directory, path, sha, sequence in descriptors:
        pair = json.loads(read_pinned(path, sha, 4096))
        if pair["sequence"] != sequence or sequence in RESERVED_RECORDINGS:
            raise ValueError("Pair collides with a development or blind recording")
        media = {value for name, value in pair["files_SHA256"].items() if name.endswith(".png")}
        if len(media) != 4 or seen.intersection(media):
            raise ValueError("Repeated RGB-D media in the training cohort")
        seen.update(media)
        sample, evidence = geometry["prepare_pair"](directory, pair)
        layout = checked_input(sample, sequence)
        batches.append(sample)
        rows.append(dict(sequence=sequence, source_manifest_SHA256=sha,
                         valid_pixels=evidence["valid_depth_pixels"], layouts=layout))
    identity = dict(schema="vggt-tum-three-recording-camera-inputs-v1", inputs=rows)
    report = dict(state="NINE_REAL_COMMON_ADAPTER_INPUTS_VERIFIED", planned_pairs=len(descriptors),
                  verified_pairs=len(rows), physical_recordings=dict(Counter(r["sequence"] for r in rows)),
                  unique_raw_pngs=len(seen),
                  input_order_SHA256=digest(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()),
                  original_bundle_SHA256=ORIGINAL_BUNDLE_SHA, new_source_manifest_SHA256=NEW_PAIR_SHA,
                  canonical_geometry_adapter_SHA256=ADAPTER_SHA, rows=rows, CO3D=False,
                  GPU_steps=0, training_updates=0, READY=False)
    return batches, report


def native_batches(torch, base_root, new_root, tensor_batch, valid_depth):
    """Call the existing VGGT collector's tensor conversion and validity check."""
    raw, evidence = load_cohort(base_root, new_root)
    batches = [tensor_batch(torch, sample) for sample in raw]
    expected = {name: (1,) + shape for name, shape in LAYOUTS.items()}
    expected["images"] = (1, 2, 3, 224, 224)
    for batch in batches:
        if (any(tuple(batch[name].shape) != shape for name, shape in expected.items())
                or not valid_depth(batch["point_masks"])):
            raise ValueError("Canonical tensor conversion changed the expected contract")
    return batches, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-bundle", type=Path, required=True)
    parser.add_argument("--new-pair", type=Path, required=True)
    parser.add_argument("--private-receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.private_receipt.exists():
        raise FileExistsError(args.private_receipt)
    _, evidence = load_cohort(args.original_bundle, args.new_pair)
    with args.private_receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in evidence.items() if k != "rows"}, allow_nan=False))


if __name__ == "__main__":
    main()
