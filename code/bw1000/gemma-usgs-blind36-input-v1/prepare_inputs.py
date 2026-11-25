"""Assemble the preregistered Gemma pilot's inputs without opening answers."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import struct


REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "local-only/training-USGS-multievent-render144-v2-original/dataset"
OUTPUT = REPO / "local-only/gemma-usgs-blind36-input-v1"
PLAN_SHA256 = "0993b5b287d5b0a90d450310d6b6e0105b6ebf99589cf6e14a0e04d1556770ef"
DISPLAYED_EVENTS = ["E01", "E02", "E03", "E04", "E05", "E06"]
KINDS = {
    "depth_change",
    "maximum_depth",
    "maximum_reported_number",
    "minimum_depth",
    "minimum_reported_number",
    "reported_number_change",
}


def digest(raw):
    return sha256(raw).hexdigest()


def exact_read(path, pin, cap):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.resolve(strict=True) != path:
        raise ValueError("Expected an exact regular input file")
    if not 0 < before.st_size == pin["bytes"] <= cap:
        raise ValueError("Input extent differs from the sealed manifest")
    with path.open("rb") as stream:
        first_fd = os.fstat(stream.fileno())
        raw = stream.read(cap + 1)
        last_fd = os.fstat(stream.fileno())
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    compare = lambda info: tuple(getattr(info, field) for field in fields)
    if not compare(before) == compare(first_fd) == compare(last_fd) == compare(after):
        raise ValueError("Input file identity changed during the read")
    if (
        before.st_ctime_ns != after.st_ctime_ns
        or first_fd.st_ctime_ns != last_fd.st_ctime_ns
        or len(raw) != pin["bytes"]
        or digest(raw) != pin["SHA256"]
    ):
        raise ValueError("Input bytes differ from the sealed manifest")
    return raw


def read_plan():
    path = Path(__file__).with_name("PLAN.json")
    raw = path.read_bytes()
    if digest(raw) != PLAN_SHA256:
        raise ValueError("Pilot protocol differs")
    plan = json.loads(raw)
    for relative, expected in plan["public_proof_and_helpers_SHA256"].items():
        if digest((REPO / relative).read_bytes()) != expected:
            raise ValueError("A frozen proof or scorer changed")
    return plan


def collect(plan):
    manifest_raw = exact_read(SOURCE / "manifest-private.json", plan["sealed_dataset_manifest"], 32768)
    manifest = json.loads(manifest_raw)
    blind_charts = {chart["id"]: chart for chart in manifest["charts"] if chart["split"] == "blind"}
    blind_events = {event for chart in blind_charts.values() for event in chart["event_ids"]}
    blind_groups = {chart["group"] for chart in blind_charts.values()}
    media = {chart["image"] for chart in blind_charts.values()}
    if len(blind_charts) != 6 or len(blind_events) != 36 or len(blind_groups) != 2 or len(media) != 6:
        raise ValueError("Original blind group, chart or event count differs")

    # Select the prompt file and chart names from the frozen manifest. Do not
    # iterate over the split directory or its manifest entry for the labels.
    inputs = "blind/inputs.jsonl"
    for relative in media:
        parts = relative.split("/")
        if len(parts) != 3 or parts[:2] != ["blind", "media"] or not parts[2].endswith(".png"):
            raise ValueError("Unexpected blind media member")
        if not parts[2][:-4].isdigit():
            raise ValueError("Unexpected chart filename")
    data = {
        relative: exact_read(SOURCE / relative, manifest["outputs"][relative], 131072)
        for relative in sorted(media | {inputs})
    }
    counts = Counter()
    ids = set()
    rows = [json.loads(line) for line in data[inputs].splitlines()]
    for row in rows:
        chart = blind_charts[row["chart_id"]]
        messages = row["messages"]
        if (
            row["id"] in ids
            or row["group"] != chart["group"]
            or row["event_ids"] != DISPLAYED_EVENTS
            or row["images"] != [chart["image"]]
            or row["labels"] is not None
            or len(messages) != 1
            or messages[0]["role"] != "user"
            or not messages[0]["content"].startswith("<image>\n")
        ):
            raise ValueError("A row differs from the answer-free chart binding")
        ids.add(row["id"])
        counts[row["chart_id"], row["kind"]] += 1
    if len(rows) != 36 or counts != Counter((chart, kind) for chart in blind_charts for kind in KINDS):
        raise ValueError("The complete six-chart, six-kind denominator is required")
    for relative in media:
        raw = data[relative]
        if raw[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR" or struct.unpack(">II", raw[16:24]) != (512, 512):
            raise ValueError("Original 512 by 512 PNG input required")
    return data


def put(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    plan = read_plan()
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError("The original input bundle already exists")
    data = collect(plan)
    OUTPUT.mkdir()
    for relative, raw in data.items():
        path = OUTPUT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        put(path, raw)
    record = {
        "state": "ORIGINAL_ANSWERS_EXCLUDED_INPUT_BUNDLE_VERIFIED",
        "protocol_SHA256": PLAN_SHA256,
        "dataset_manifest_SHA256": plan["sealed_dataset_manifest"]["SHA256"],
        "questions": 36,
        "charts": 6,
        "event_groups": 2,
        "objects": {relative: {"bytes": len(raw), "SHA256": digest(raw)} for relative, raw in data.items()},
        "gold_files_opened": 0,
        "optimizer_updates": 0,
        "model_calls": 0,
    }
    receipt = (json.dumps(record, sort_keys=True) + "\n").encode()
    put(OUTPUT / "INPUT_FILES.json", receipt)
    print(json.dumps({
        "state": record["state"],
        "questions": 36,
        "charts": 6,
        "event_groups": 2,
        "input_objects": len(data),
        "bytes": sum(map(len, data.values())),
        "protocol_SHA256": PLAN_SHA256,
        "private_input_manifest_SHA256": digest(receipt),
        "gold_files_opened": 0,
        "model_calls": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
