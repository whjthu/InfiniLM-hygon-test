"""Verify private held-out prediction and failure bodies without publishing them."""

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import struct

from training_material_mae import MATERIAL_FIELDS, QUERY_ARTIFACT, RGBA_ARTIFACT


JSONL_METRICS = {"TextVQA": "question_id", "COCO CIDEr": "image_id",
                 "AUC@5": "sample_id", "AUC@15": "sample_id", "AUC@30": "sample_id"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_location(location, roots):
    path, marker, number = location.rpartition("#line=")
    require(not marker or re.fullmatch(r"[1-9][0-9]*", number),
            "invalid held-out artifact line locator")
    path = Path(path if marker else location)
    if not path.is_absolute():
        path = roots[0] / path
    resolved = path.resolve(strict=True)
    require(any(resolved.is_relative_to(root) for root in roots) and resolved.is_file(),
            "held-out artifact is outside the declared roots or is not a regular file")
    return resolved, int(number) if marker else None


def verify_line(raw, expected, identity_key):
    require(sha256(raw).hexdigest() == expected["artifact"]["SHA256"], "held-out artifact line hash differs")
    if identity_key is None:
        return
    row = json.loads(raw)
    identifier = row[identity_key]
    if identity_key in ("question_id", "image_id"):
        require(type(identifier) is int, "held-out artifact ID has the wrong type")
        identifier = str(identifier)
    status = row["status"] if identity_key == "sample_id" else row.get("status", "scored")
    require(identifier == expected["sample_id"] and status == expected["status"],
            "held-out artifact line belongs to a different sample or status")


def file_digest(path):
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_lines(path, wanted, identity_key, declared_file_sha):
    digest = sha256()
    # Universal line boundaries retain the original CR/LF bytes when newline is empty.
    remaining = set(wanted)
    count = 0
    with path.open("r", encoding="utf-8", errors="surrogateescape", newline="") as handle:
        for number, line in enumerate(handle, 1):
            count = number
            raw = line.encode("utf-8", errors="surrogateescape")
            if declared_file_sha is not None:
                digest.update(raw)
            if number in wanted:
                verify_line(raw, wanted[number], identity_key)
                remaining.remove(number)
            if not remaining and declared_file_sha is None:
                break
    require(not remaining, "held-out artifact line is missing")
    require(declared_file_sha is None or digest.hexdigest() == declared_file_sha,
            "held-out prediction file hash differs")
    if identity_key is not None:
        headers = 1 if identity_key == "sample_id" else 0
        require(count == len(wanted) + headers and (not headers or 1 not in wanted),
                "held-out prediction file includes rows outside the planned sample inventory")


def verifier(repo, private_roots=()):
    roots = tuple(dict.fromkeys(path.resolve(strict=True) for path in (repo, *private_roots)))
    require(all(root.is_dir() for root in roots), "each held-out artifact root must be a directory")

    def verify(samples, metric, declared_file_sha=None):
        if metric == QUERY_ARTIFACT:
            counts = {}
            for sample in samples:
                path, line = resolve_location(sample["artifact"]["location"], roots)
                limit = 12 * 16_384
                with path.open("rb") as handle:
                    raw = handle.read(limit + 1)
                require(line is None and sample["sample_id"] not in counts and 0 < len(raw) <= limit
                        and len(raw) % 12 == 0 and sha256(raw).hexdigest() == sample["artifact"]["SHA256"],
                        "SF3D query position bytes differ")
                require(all(all(map(math.isfinite, point)) for point in struct.iter_unpack("<fff", raw)),
                        "nonfinite SF3D query position")
                counts[sample["sample_id"]] = len(raw) // 12
            return counts
        if metric in MATERIAL_FIELDS:
            records = {}
            locations = set()
            for sample in samples:
                path, line = resolve_location(sample["artifact"]["location"], roots)
                require(line is None and path not in locations and sample["sample_id"] not in records,
                        "expected a distinct SF3D material record for each sample")
                locations.add(path)
                with path.open("rb") as handle:
                    raw = handle.read(4 * 1024 ** 2 + 1)
                require(0 < len(raw) <= 4 * 1024 ** 2 and sha256(raw).hexdigest() == sample["artifact"]["SHA256"],
                        "SF3D material artifact size or hash differs")
                record = json.loads(raw)
                require(record["sample_id"] == sample["sample_id"] and record["status"] == sample["status"],
                        "SF3D material artifact belongs to another sample or status")
                records[sample["sample_id"]] = record
            return records
        by_path = {}
        identity_key = JSONL_METRICS.get(metric)
        for sample in samples:
            path, line = resolve_location(sample["artifact"]["location"], roots)
            require(metric != RGBA_ARTIFACT or line is None, "expected the original SF3D RGBA file")
            require(identity_key is None or line is not None, "expected a line in the pinned prediction file")
            entries = by_path.setdefault(path, {})
            require(line not in entries, "held-out prediction locator reused within one arm")
            entries[line] = sample
        require(identity_key is None or (len(by_path) == 1 and declared_file_sha is not None),
                "held-out arm requires one bound prediction file")

        for path, entries in by_path.items():
            whole = entries.pop(None, None)
            require(whole is None or not entries, "one held-out arm mixes a whole artifact with its lines")
            if whole is not None:
                require(file_digest(path) == whole["artifact"]["SHA256"], "held-out artifact body hash differs")
            else:
                verify_lines(path, entries, identity_key, declared_file_sha)

    return verify
