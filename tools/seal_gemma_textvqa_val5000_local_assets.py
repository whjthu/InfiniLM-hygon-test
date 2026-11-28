"""Seal the complete official TextVQA media against the original range receipts."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct

from PIL import Image

import read_gemma_cp25_textvqa_retention_gate as gate


IDENTITY_SHA = "c3e42949ca5d35b655995e4991d8f73a12f12f8da7d1428881f1258d3e0b1886"
IDENTITY_ROOT = "training-qwen-generic-public-holdout-v1"
PILOT = "d6710c7f9b0f2c2cb76e31dadfffaa5290bfe52192b75cabe50386a8f24cf73a"
MPO = "61aa6351a47f704cd67f01fb5b8f0ba418dd561ca7d3cc6f32686d4e95e75f84"
BATCHES = (
    (24, "a83826cf24c7378d1ffb6ccaec84be8381740da539becaf692cca62e9771b980"),
    (96, "515c6f5fb4149b27a00b33ed080d60b8542898d4c772cdd9e355da18ac326aa6"),
    (256, "d05b1ff44736d58f255a6805cf6c40d92c5243a2e1219ca5405d1a8ead1e906a"),
    (512, "0e75965e40c15df8d4812f964e71e56bb6a20102075d8615a9153506c3f311ea"),
    (640, "33e3787d04dbd6f27206459f12d59250bd3f15e962ce6589839a0a7ec1c04a0d"),
    (448, "a20d5ab106223b32824ab5ecaccf3b50d018f84bf704e1bb6c4179aa6c4fb8f1"),
    (384, "38b2a54bd1a2e9408097aa5383feab9e0ce74decdcf1e3aa0674ff8776769726"),
    (320, "3383fd674764f4b1e073ec767d3e9065d0d0b0f90b29bf618efc0e398db51548"),
    (216, "e44a3262bdaf1d12668031972ee96ab955f13ba8fe5a137f7bf6ed7c1030319e"),
    (140, "8945a1e65f8a482ce465d193b6889ea34f18dd6f5b41ffb796a2e0c18106ed8a"),
    (127, "d64827e402f041c8efcfe02f37b9aacba6c91a51b83395ab1a9c0fa6cd6affd2"),
)


def frozen_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(raw):
    return sha256(raw).hexdigest()


def write(path, value):
    data = frozen_json(value)
    with path.open("xb") as stream:
        stream.write(data)
    return sha(data)


def metadata(path, pin):
    return json.loads(gate.metadata(path, 1 << 20, pin), object_pairs_hook=gate.unique_object)


def receipts(root):
    pilot = root / IDENTITY_ROOT / "TEXTVQA_VAL_051_PUBLISHER_MEDIA_PILOT3"
    yield pilot, metadata(pilot / "batch.json", PILOT)["rows"], PILOT, 3, set()
    for count, pin in BATCHES:
        scope = root / f"gemma-textvqa-val-publisher-range{count}-v1"
        batch = metadata(scope / "RESULT.json", pin)
        if count in (24, 96):
            gate.need(batch["failure"] is None, "Earlier original image batch was incomplete")
            yield scope / "images", batch["rows"], pin, count, set()
        else:
            shards = batch["worker_outcomes"]
            failures = {shard["failure"]["image_id"] for shard in shards if shard["failure"] is not None}
            gate.need(len(shards) == 4 and len(failures) == (2 if count == 127 else 0),
                      "Original worker completion pattern changed")
            rows = [row for shard in shards for row in shard["rows"]]
            yield scope / "images", rows, pin, count - len(failures), failures
    recovery = root / "gemma-textvqa-val-publisher-format2-mpo-v1"
    yield recovery, metadata(recovery / "private-receipt.json", MPO)["members"], MPO, 2, set()


def scan_one(root, item):
    name, path, expected_bytes, expected_sha = item
    gate.need(path.resolve(strict=True) == path and path.is_relative_to(root), "Media path changed or escaped root")
    raw = gate.metadata(path, 6 << 20, expected_sha, expected_bytes)
    with Image.open(BytesIO(raw), formats=("JPEG", "MPO")) as image:
        actual_format = image.format
        image.seek(0)
        dims = image.size
        gate.need(actual_format in ("JPEG", "MPO") and 0 < dims[0] * dims[1] <= 32 << 20,
                  "Official media format or pixel extent differs")
        image.load()
        image_sha = sha(struct.pack("<II", *dims) + image.convert("RGB").tobytes())
    return {"image_id": name, "relative_local_path": path.relative_to(root).as_posix(), "bytes": expected_bytes,
            "SHA256": expected_sha, "decoded_first_frame_RGB_SHA256": image_sha, "format": actual_format}


def seal(root, output):
    identity = metadata(root / IDENTITY_ROOT / "TEXTVQA_VAL_051_IDENTITIES.json", IDENTITY_SHA)
    qids, expected_images = gate.projection(identity)
    expected_train = gate.training_hashes(gate.fixed("training_input_manifest"), gate.fixed("original_training_binding"))
    planned, pinned_sources = {}, []
    partial = set()
    for directory, rows, digest, count, failures in receipts(root):
        gate.need(len(rows) == count, "Original media receipts have a different denominator")
        partial |= failures
        if digest == MPO:
            gate.need({row["image_id"] for row in rows} == partial, "Recovered images differ from original failed IDs")
        for row in rows:
            image = row["image_id"]
            gate.need(image in expected_images and image not in planned and row["SHA256"] not in expected_train,
                      "Media receipt repeats validation or exact training media")
            planned[image] = (image, directory / f"{image}.jpg", row["bytes"], row["SHA256"])
        pinned_sources.append(digest)
    gate.need(planned.keys() == expected_images and len({x[3] for x in planned.values()}) == len(planned),
              "Validation media closure or encoded uniqueness differs")
    output.mkdir()
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda item: scan_one(root, item), (planned[name] for name in sorted(planned))))
    encoded = write(output / "asset-manifest.private.json", rows)
    witness = write(output / "exact-original-readback.private.json", pinned_sources)
    projection = {"schema": "TEXTVQA_V051_VAL_MEDIA_READBACK_V1", "annotation_SHA256": gate.ANN,
                  "identities_SHA256": IDENTITY_SHA, "all_local_asset_bytes_verified": True,
                  "original_readback_SHA256": witness, "asset_manifest_SHA256": encoded,
                  "rows": [{k: row[k] for k in ("image_id", "bytes", "SHA256")} | {"readable": True} for row in rows]}
    checked = gate.media_evidence(projection, IDENTITY_SHA, expected_images, expected_train)
    projection_pin = write(output / "retention-media.private.json", projection)
    pixel_duplicates = len(rows) - len({r["decoded_first_frame_RGB_SHA256"] for r in rows})
    result = {"schema": "GEMMA_TEXTVQA_ALL_OFFICIAL_LOCAL_ASSETS_REREAD_V1", "questions": len(qids),
              "images": len(rows), "file_bytes_reread": sum(row["bytes"] for row in rows),
              "formats": dict(Counter(row["format"] for row in rows)), "validation_exact_pixel_duplicates": pixel_duplicates,
              "asset_manifest_SHA256": encoded, "retention_compatible_private_receipt_SHA256": projection_pin,
              "ordered_original_readbacks_SHA256": witness, "encoded_integrity_check": checked,
              "cross_split_actual_pixel_check": None, "near_duplicate_check": None,
              "target_GPU_inference": 0, "answers_opened": False}
    write(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(seal(args.local_root.resolve(strict=True), args.out), sort_keys=True))
