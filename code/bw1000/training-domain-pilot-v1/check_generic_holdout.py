"""Check the frozen COCO64 training images against two pinned holdout inventories in place."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from prepare_qwen_inputs import pinned_bytes


TARGET = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-vlm-original-assets-v1")
DATASET_SHA256 = "1c9aaddad5b681b50a6ddfa79c08e30e4820ee6ffcdc81bd473d0c2e84b35a0f"
BINDINGS_SHA256 = "5d9e6132e937da172dd714313ba1dfdc7775e86b27fb623131fe9bbe2c1d1822"
COCO_FILE = re.compile(r"([0-9]{12})\.jpg")
SOURCE_ID = re.compile(r"[a-z][a-z0-9_.-]*:[A-Za-z0-9_.-]+")
SHA256 = re.compile(r"[0-9a-f]{64}")


def file_sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def train_images(root, expected_sha, bindings):
    raw = pinned_bytes(root / "vlm_report_coco64.hip.jsonl", expected_sha)
    rows = [json.loads(line) for line in raw.splitlines()]
    if not rows or len(rows) != len(bindings):
        raise ValueError("Unexpected frozen training row count")
    output = {}
    base = root.resolve(strict=True)
    for index, (row, original) in enumerate(zip(rows, bindings), start=1):
        images = row["images"]
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise ValueError(f"Unexpected image list in training row {index}")
        path = Path(images[0])
        resolved = path.resolve(strict=True)
        match = COCO_FILE.fullmatch(path.name)
        role = Path("coco-val2017", "images", path.name)
        if (path != root / "datasets" / role or not resolved.is_relative_to(base) or not match
                or original["ordered_index"] != index - 1 or original["relative_role"] != role.as_posix()
                or resolved.stat().st_size != original["bytes"] or file_sha(resolved) != original["sha256"]):
            raise ValueError(f"Unexpected image provenance or bytes in training row {index}")
        identity = f"coco:{int(match[1])}"
        if identity in output:
            raise ValueError("Repeated training source image ID")
        output[identity] = original["sha256"]
    if len(set(output.values())) != len(bindings):
        raise ValueError("Repeated exact image content within training")
    return output


def inventory(path, expected_sha, expected_images, metric):
    value = json.loads(pinned_bytes(path, expected_sha))
    if (not isinstance(value, dict) or set(value) != {"metric", "images"} or value["metric"] != metric
            or not isinstance(value["images"], list) or expected_images < 1 or len(value["images"]) != expected_images):
        raise ValueError(f"Unexpected frozen {metric} inventory schema or denominator")
    records = []
    seen = set()
    for row in value["images"]:
        if (not isinstance(row, dict) or set(row) != {"source_image_ids", "image_SHA256"}
                or not isinstance(row["source_image_ids"], list) or not row["source_image_ids"]
                or not all(isinstance(i, str) and SOURCE_ID.fullmatch(i) for i in row["source_image_ids"])
                or not isinstance(row["image_SHA256"], str) or not SHA256.fullmatch(row["image_SHA256"])):
            raise ValueError(f"Invalid image identity in {metric} inventory")
        identifiers = set(row["source_image_ids"])
        if (len(identifiers) != len(row["source_image_ids"]) or identifiers & seen
                or (metric == "coco_cider" and sum(i.startswith("coco:") for i in identifiers) != 1)):
            raise ValueError(f"Repeated ID or missing canonical COCO ID in {metric} inventory")
        for identity in identifiers:
            if identity.startswith("coco:"):
                numeric = identity.removeprefix("coco:")
                if not numeric.isascii() or not numeric.isdecimal() or str(int(numeric)) != numeric:
                    raise ValueError("COCO image ID must be canonical decimal")
        seen.update(identifiers)
        records.append((identifiers, row["image_SHA256"]))
    return records


def reconcile(train, inputs):
    hashes = set(train.values())
    reports = {}
    for metric, (path, sha, count) in inputs.items():
        records = inventory(path, sha, count, metric)
        by_id = by_sha = union = 0
        for ids, image_hash in records:
            id_match = bool(ids.intersection(train))
            hash_match = image_hash in hashes
            by_id += id_match
            by_sha += hash_match
            union += id_match or hash_match
        reports[metric] = {"inventory_SHA256": sha, "images": len(records), "matches_by_source_id": by_id,
                           "matches_by_exact_content_SHA256": by_sha, "matches_by_either": union}
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-bindings", type=Path, required=True)
    for arm in ("coco-cider", "textvqa"):
        parser.add_argument(f"--{arm}-inventory", type=Path, required=True)
        parser.add_argument(f"--{arm}-sha256", required=True)
        parser.add_argument(f"--{arm}-images", type=int, required=True)
    args = parser.parse_args()
    inputs = {metric: (getattr(args, f"{metric}_inventory"), getattr(args, f"{metric}_sha256"),
                       getattr(args, f"{metric}_images")) for metric in ("coco_cider", "textvqa")}
    bindings = json.loads(pinned_bytes(args.original_bindings, BINDINGS_SHA256))["common"]["image_identity"]
    if len(bindings) != 64:
        raise ValueError("Unexpected original training image inventory count")
    train = train_images(TARGET, DATASET_SHA256, bindings)
    reports = reconcile(train, inputs)
    overlap = any(report["matches_by_either"] for report in reports.values())
    print(json.dumps({"training_JSONL_SHA256": DATASET_SHA256, "original_bindings_SHA256": BINDINGS_SHA256,
                      "training_images": len(train),
                      "verified_inventory_reports": reports, "no_exact_overlap": not overlap}, sort_keys=True))
    return 1 if overlap else 0


if __name__ == "__main__":
    raise SystemExit(main())
