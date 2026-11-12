"""CPU input identity checks for the native VGGT engineering recipe."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Callable


FIELDS = {"images", "depths", "extrinsics", "intrinsics", "point_masks"}


def checked_files(root: Path, rows: list[dict], deadline: Callable[[], None]) -> dict:
    """Read only manifest members, rejecting aliases, escape and changing files."""
    if root.is_symlink() or not root.is_dir() or not rows:
        raise ValueError("invalid input root or empty manifest")
    base = root.resolve()
    seen = set()
    total = 0
    for row in rows:
        name = row["name"]
        parts = PurePosixPath(name)
        if (not name or parts.is_absolute() or parts.as_posix() != name
                or ".." in parts.parts or "\\" in name or name in seen
                or ":" in name):
            raise ValueError("invalid or duplicate manifest member")
        seen.add(name)
        path = root.joinpath(*parts.parts)
        if any(root.joinpath(*parts.parts[:i]).is_symlink()
               for i in range(1, len(parts.parts) + 1)):
            raise ValueError("symlink manifest member")
        if not path.resolve().is_relative_to(base) or not path.is_file():
            raise ValueError("missing or escaping manifest member")
        before = path.stat()
        if type(row["bytes"]) is not int or before.st_size != row["bytes"]:
            raise ValueError("manifest member size mismatch")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                deadline()
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        after = path.stat()
        identity = lambda s: (s.st_size, s.st_mtime_ns, s.st_dev, s.st_ino)
        if identity(before) != identity(after) or digest.hexdigest() != row["SHA256"]:
            raise ValueError("manifest member identity mismatch")
        total += before.st_size
    return {"files": len(seen), "bytes": total}


def check_batches(
    dataset: Any, config: dict, reference: dict, *, make_ids: Callable,
    make_batch: Callable, valid_depth: Callable, describe: Callable,
    batch_hash: Callable, selected_hash: Callable, deadline: Callable,
    progress: Callable,
) -> dict:
    """Mirror native candidate selection; compare every attempt and tensor field.

    Callbacks keep the planning/checking code importable without Torch. Production
    callers supply the unchanged backend conversion, validity and hash functions.
    All returned identities and progress records belong in private experiment storage.
    """
    expected = reference["expected_batches"]
    attempts = reference["attempts"]
    count = config["cached_batches"]
    if (type(count) is not int or count <= 0 or len(expected) != count
            or [row["batch"] for row in expected] != list(range(count))
            or [row["candidate"] for row in attempts] != list(range(len(attempts)))
            or any(type(row["valid"]) is not bool for row in attempts)
            or sum(row["valid"] for row in attempts) != count
            or any(set(row["fields"]) != FIELDS for row in expected)):
        raise ValueError("incomplete reference denominator")
    names = sorted(dataset.sequence_list)
    if not names or len(names) != reference["source_sequence_count"]:
        raise ValueError("sequence count mismatch")
    if (reference["source_valid_batches"] != count
            or reference["source_rejected_batches"] != len(attempts) - count):
        raise ValueError("reference rejection count mismatch")
    progress({"event": "plan", "planned_attempts": len(attempts),
              "planned_valid_batches": count, "planned_tensor_fields": count * 5})
    batches, observed, fields, mismatches = [], [], [], []
    candidate = 0
    try:
        while len(batches) < count and candidate < count * len(names) * 8:
            deadline()
            name = names[candidate % len(names)]
            frame_count = len(dataset.data_store[name])
            ids = make_ids([(candidate * 17 + i * 37) % frame_count
                            for i in range(config["frames_per_step"])])
            raw = dataset.get_data(seq_name=name, ids=ids, aspect_ratio=1.0)
            batch = make_batch(raw)
            valid = bool(valid_depth(batch["point_masks"]))
            row = {"candidate": candidate, "valid": valid,
                   "seq_name": batch["seq_name"], "ids": list(batch["ids"])}
            observed.append(row)
            if candidate >= len(attempts) or row != attempts[candidate]:
                mismatches.append({"candidate": candidate, "kind": "candidate_identity"})
            if valid:
                index = len(batches)
                found = describe(batch)
                if set(found) != FIELDS:
                    raise ValueError("observed tensor field set mismatch")
                for key in sorted(FIELDS):
                    if found[key] != expected[index]["fields"][key]:
                        mismatches.append({"batch": index, "field": key, "kind": "tensor_identity"})
                digest = batch_hash(batch)
                if digest != expected[index]["content_SHA256"]:
                    mismatches.append({"batch": index, "kind": "batch_identity"})
                fields.append({"batch": index, "fields": found, "content_SHA256": digest})
                batches.append(batch)
            candidate += 1
            deadline()
            progress({"event": "candidate_complete", **row,
                      "completed_valid_batches": len(batches), "mismatches": len(mismatches)})
        if len(batches) != count or len(observed) != len(attempts):
            mismatches.append({"kind": "attempt_denominator"})
        digest = selected_hash(batches)
        if digest != reference["expected_selected_batch_content_SHA256"]:
            mismatches.append({"kind": "selected_batch_identity"})
        deadline()
        return {"passed": not mismatches, "attempts": observed, "batches": fields,
                "candidate_batches": len(observed), "valid_batches": len(batches),
                "rejected_batches": sum(not row["valid"] for row in observed),
                "tensor_fields_checked": len(fields) * 5, "mismatches": mismatches,
                "selected_batch_content_SHA256": digest}
    except BaseException as error:
        progress({"event": "failed", "error_type": type(error).__name__,
                  "completed_attempts": len(observed), "completed_valid_batches": len(batches),
                  "planned_attempts": len(attempts), "planned_valid_batches": count})
        raise
