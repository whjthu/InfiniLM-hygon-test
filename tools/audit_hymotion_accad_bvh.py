"""Inventory publisher BVH files for a distinct real 128-frame motion cohort."""

import argparse
from collections import Counter
from decimal import Decimal
from hashlib import sha256
from io import BytesIO
import json
from math import isfinite
from pathlib import Path
import re
from stat import S_ISLNK
from zipfile import ZipFile


ARCHIVES = {
    "Female1_bvh.zip": (81, 5171377, "84543297a24fe9900449675837b5b7758c5208d0f56b699d10a3eb362a46530a", "training"),
    "Male1_bvh.zip": (69, 3283636, "1cd26aeac2ee7f20e3c0fe5082e633201bb136a714ae8f684a4f34c4421f4627", "training"),
    "Male2_bvh.zip": (149, 7242625, "f30dc56f8557d4b9f1bd6b8b0c4218c076471de0a9c1a20f3bc4a8b776cc14b1", "development"),
}
CHANNELS = {"Xposition", "Yposition", "Zposition", "Xrotation", "Yrotation", "Zrotation"}


def inspect_clip(raw):
    lines = [line.strip() for line in raw.decode("ascii").splitlines() if line.strip()]
    if lines[0] != "HIERARCHY" or lines.count("MOTION") != 1:
        raise ValueError("unexpected BVH section markers")
    motion = lines.index("MOTION")
    if motion + 3 >= len(lines):
        raise ValueError("missing BVH motion rows")
    frames = re.fullmatch(r"Frames:\s*([1-9]\d*)", lines[motion + 1])
    interval = re.fullmatch(r"Frame Time:\s*(0\.\d+)", lines[motion + 2])
    if frames is None or interval is None:
        raise ValueError("unexpected BVH frames or sample interval")
    count, step = int(frames.group(1)), Decimal(interval.group(1))
    if count != len(lines) - motion - 3 or step <= 0:
        raise ValueError("frame count or interval differs")
    width = 0
    for line in lines[1:motion]:
        fields = line.split()
        if fields[0] != "CHANNELS":
            continue
        if len(fields) < 3 or int(fields[1]) != len(fields) - 2 or not set(fields[2:]) <= CHANNELS:
            raise ValueError("unexpected BVH channel declaration")
        width += len(fields) - 2
    if width == 0:
        raise ValueError("BVH channel declarations absent")
    for row in lines[motion + 3:]:
        values = row.split()
        if len(values) != width or not all(isfinite(float(value)) for value in values):
            raise ValueError("motion data extent or finite range differs")
    nominal_30 = abs(step * 30 - 1) <= Decimal("0.00001")
    return {
        "frames": count, "frame_period_seconds": str(step), "channel_count": width,
        "duration_seconds": str((count - 1) * step), "nominal_30fps": nominal_30,
        "contains_128_native_30fps_frames": nominal_30 and count >= 128,
    }


def audit(directory, output):
    private = []
    complete = set()
    summary = {}
    for name, (members, size, archive_sha, split) in ARCHIVES.items():
        archive = directory / name
        with archive.open("rb") as stream:
            raw = stream.read(size + 1)
        if len(raw) != size or sha256(raw).hexdigest() != archive_sha:
            raise ValueError(f"original archive differs: {name}")
        actor = name.split("_", 1)[0]
        with ZipFile(BytesIO(raw)) as source:
            entries = source.infolist()
            if len(entries) != members or len({m.filename for m in entries}) != members:
                raise ValueError(f"original archive membership differs: {name}")
            counts, feature_widths = Counter(), Counter()
            for item in entries:
                if (not item.filename.startswith(actor + "_") or "/" in item.filename
                        or "\\" in item.filename or not item.filename.lower().endswith(".bvh")
                        or S_ISLNK(item.external_attr >> 16) or not 0 < item.file_size <= 2 * (1 << 20)):
                    raise ValueError(f"invalid archive member: {name}")
                body = source.read(item)
                record = inspect_clip(body)
                body_hash = sha256(body).hexdigest()
                if body_hash in complete:
                    raise ValueError("byte-identical motion clip crosses the candidate inventory")
                complete.add(body_hash)
                group = re.search(r"_([A-Z])\d", item.filename)
                locomotion = bool(group and group.group(1) in {"B", "C"})
                qualifies = record["contains_128_native_30fps_frames"]
                private.append({
                    **record, "member": item.filename, "bytes": len(body), "SHA256": body_hash,
                    "archive": name, "source_subject": actor, "split": split,
                    "locomotion_B_or_C": locomotion,
                })
                counts["count"] += 1
                counts["nominal_30fps"] += record["nominal_30fps"]
                counts["native_128"] += qualifies
                counts["locomotion_B_or_C"] += locomotion
                counts["locomotion_B_or_C_native_128"] += locomotion and qualifies
                feature_widths[record["channel_count"]] += 1
            summary[actor] = {
                **dict(counts), "split": split, "BVH_channel_counts": dict(feature_widths),
                "verified_archive": {"file": name, "bytes": size, "sha256": archive_sha},
            }
    doc = {
        "schema": "bw1000.hymotion.accad.public-bvh-inventory.v1",
        "summary": summary, "clips": private, "exact_body_hash_overlap": 0,
        "native_motion_201_features_built": False, "gpu_steps": 0,
    }
    contents = (json.dumps(doc, sort_keys=True, indent=2) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(contents)
    return {"summary": summary, "private_receipt_SHA256": sha256(contents).hexdigest(), "clips": len(private)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive_directory, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
