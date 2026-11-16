"""Verify a frozen development set before dispatching a paired VGGT evaluator."""

import json
import re

from score_vggt_tum_pair import PILOT_GEOMETRY_SHA
from tum_vggt_pair import prepare_pair, read_pinned


SEALED_SEQUENCES = {"freiburg3_structure_texture_near", "freiburg3_long_office_household"}


def load_development_set(root, manifest, expected_sha):
    rows = json.loads(read_pinned(manifest, expected_sha))["pairs"]
    case_ids, seen_shas, matches = set(), set(), set()
    cases, evidence = [], []
    for row in rows:
        case_id, sequence, digest = row["id"], row["sequence"], row["manifest_sha256"]
        if (not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", case_id) or case_id in case_ids
                or not sequence.startswith("freiburg3_") or sequence in SEALED_SEQUENCES
                or digest in seen_shas or digest == PILOT_GEOMETRY_SHA):
            raise ValueError("repeated, invalid or masked development case")
        directory = root / case_id
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("missing exact development directory")
        data = json.loads(read_pinned(directory / "pair.json", digest))
        if data["sequence"] != sequence:
            raise ValueError("development pair sequence/file identity differs")
        batch, geometry = prepare_pair(directory, data)
        batch["frame_identity"] = [item["rgb"] for item in geometry["matches"]]
        frame_pair = (sequence, *batch["frame_identity"])
        if frame_pair in matches:
            raise ValueError("same frame pair appears more than once")
        case_ids.add(case_id)
        seen_shas.add(digest)
        matches.add(frame_pair)
        cases.append((batch, sequence, digest))
        evidence.append(dict(case_id=case_id, manifest_sha256=digest, geometry=geometry))
    if not cases:
        raise ValueError("development cases are absent")
    return cases, evidence
