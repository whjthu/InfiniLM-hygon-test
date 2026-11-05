"""Freeze C1 routed component operands from the original fixed-teacher capture."""

import argparse
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT.parent
ARCHIVE = LOCAL / "Deep438-1378-terminal-evidence-v1.tar.gz"
RECEIPT = LOCAL / "Deep438-diagnostic-v2-CPU911-original-receipt.json"
READER = ROOT / ("code/bw1000/Deep-T1-MoE-activation-quant-r438/model-diagnostic-v2/"
                 "analysis-device-1378/analyze_archive.py")
ARCHIVE_SHA = "c51c493632f2a1b64f7a396db59a85a12e06f76a507e94cd82aab511b91daed2"
RECEIPT_SHA = "cbf33261cf3d64aa496a4b8ad0fd2a3c2fcefa62ad36465e9168a0475920896d"
READER_SHA = "51d39885c9098450c8fbbb31fea67f1c91115753642dd8266220a810d661d2f2"
OBJECT_SHA = "08ed7c8bfb7fe56cee66a6d6bdc5a5d569b05574ae68b2e7741d755a563ea756"
CAPTURE_NATIVE_SHA = "1b81667f28892a840c8396c57d9d8134ca40f94f6d0fa7c7c2deab5d1782643f"
LAYER = 3
FIELDS = {
    "hidden": ("BF16", [1, 4096], 8192),
    "route_ids": ("I64", [1, 6], 48),
    "route_weights": ("F32", [1, 6], 24),
    "reference_I8": ("I8", [6, 256], 1536),
    "reference_scale": ("F32", [6, 1], 24),
}


def digest(raw):
    return sha256(raw).hexdigest()


def member_bytes(files, folder, summary, tag):
    owner = summary["owner"]
    metadata = files[folder + f"owner{owner}.json"]
    if digest(metadata) != summary["metadata_SHA256"]:
        raise ValueError("owner metadata differs")
    meta = json.loads(metadata)
    raw = files[folder + meta["binary"]]
    if (meta["owner"] != owner or meta["tag"] != tag or len(meta["records"]) != 86
            or not meta["Native_forward_completed_before_parent_read"]
            or len(raw) != meta["bytes"] or digest(raw) != meta["SHA256"]
            or digest(raw) != summary["SHA256"]):
        raise ValueError("owner capture identity differs")
    record, = [r for r in meta["records"] if r["header"][:6] == [*tag, LAYER, owner, 0]]
    selected = {}
    for name, (dtype, shape, size) in FIELDS.items():
        field = record["tensors"][name]
        start = field["offset"]
        value = raw[start:start + size]
        if ((field["dtype"], field["shape"], field["bytes"]) != (dtype, shape, size)
                or start < 0 or len(value) != size or digest(value) != field["SHA256"]):
            raise ValueError(f"bad captured operand: {name}")
        selected[name] = value
    if any(w & 0x7f80 == 0x7f80 for w, in struct.iter_unpack("<H", selected["hidden"])):
        raise ValueError("non-finite BF16 input")
    ids = struct.unpack("<6q", selected["route_ids"])
    weights = struct.unpack("<6f", selected["route_weights"])
    scales = struct.unpack("<6f", selected["reference_scale"])
    if (any(not 0 <= v < 256 for v in ids)
            or not all(math.isfinite(v) and v >= 0 for v in weights)
            or not all(math.isfinite(v) and v > 0 for v in scales)
            or 128 in selected["reference_I8"]):
        raise ValueError("bad route or quantization values")
    return selected, ids


def freeze(destination, *, archive=ARCHIVE, receipt_path=RECEIPT, reader_path=READER):
    receipt = receipt_path.read_bytes()
    if digest(receipt) != RECEIPT_SHA or digest(reader_path.read_bytes()) != READER_SHA:
        raise ValueError("original proof pins differ")
    proof = json.loads(receipt)
    object_path = "/refactor/candidates/models-5f91ad7-r3675/Deep/helpers/routed.o"
    if (proof["library_SHA256"] != CAPTURE_NATIVE_SHA
            or proof["all_objects_SHA256"][object_path] != OBJECT_SHA):
        raise ValueError("capture Native object differs")
    spec = importlib.util.spec_from_file_location("original1378_reader", reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    files, archive_manifest, archive_sha = reader.load(archive)
    if archive_sha != ARCHIVE_SHA or not archive_manifest["all_passes_or_terminal"]:
        raise ValueError("original terminal archive differs")
    report = json.loads(files["run/typed/report.json"])
    first, = [v for v in report["sequences"] if v["sequence"] == 0]
    if not (report["completed"] and report["passed"] and first["completed"] and first["mode"] == 0):
        raise ValueError("original baseline pass incomplete")
    inputs = []
    for output, past in ((82, 8570), (83, 8571)):
        tag = [0, output, 9 + output]
        window, = [v for v in first["captures"] if v["tag"] == tag]
        if first["decisions"][output]["input_past"] != past or first["decisions"][output]["input_count"] != 1:
            raise ValueError("expected original C1 input")
        if sorted(v["owner"] for v in window["owners"]) != list(range(8)):
            raise ValueError("incomplete original owners")
        for summary in sorted(window["owners"], key=lambda v: v["owner"]):
            payload, ids = member_bytes(files, f"run/typed/pass0-output{output}-owners/", summary, tag)
            inputs.append((past, summary["owner"], payload, ids))

    destination.mkdir()
    rows = []
    experts_by_rank = {rank: set() for rank in range(8)}
    for past, rank, payload, ids in inputs:
        prefix = f"past{past}-rank{rank}"
        descriptors = {}
        for name, raw in payload.items():
            filename = f"{prefix}-{name}.bin"
            with (destination / filename).open("xb") as out:
                out.write(raw)
            dtype, shape, size = FIELDS[name]
            descriptors[name] = dict(file=filename, SHA256=digest(raw), dtype=dtype, shape=shape, bytes=size)
        experts_by_rank[rank].update(ids)
        rows.append(dict(past=past, rank=rank, layer=LAYER, distinct_experts=len(set(ids)), fields=descriptors))
    result = {
        "archive_SHA256": ARCHIVE_SHA, "original_step": "910374.1378",
        "capture_Native_SHA256": CAPTURE_NATIVE_SHA, "shared_routed_object_SHA256": OBJECT_SHA,
        "capture_pass": 0, "activation_quant_candidate_mode": 0,
        "rows": rows, "payload_bytes": sum(v["bytes"] for r in rows for v in r["fields"].values()),
        "required_expert_counts_by_rank": {str(rank): len(ids) for rank, ids in experts_by_rank.items()},
        "weight_contract": {
            "model": "hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8",
            "key_template": "layers.3.ffn.experts.{id}.{projection}.{field}",
            "w1_w3": "rank slice on output axis, 256 contiguous rows for each weight and scale",
            "w2": "weight rank slice on input axis, 256 contiguous columns; scale is replicated",
            "load_using": "derive selected expert IDs from the sealed input bytes for each rank; preserve original ID slots",
        },
        "scope": "Private fixed-teacher inputs with mode OFF. Same routed object as a314; capture Context differs. "
                 "No new checkpoint payload read, GPU execution, free-generation quality, or performance evidence.",
    }
    raw = (json.dumps(result, indent=2) + "\n").encode()
    with (destination / "FIXTURE_MANIFEST.json").open("xb") as out:
        out.write(raw)
    print(json.dumps(dict(samples=len(rows), bytes=result["payload_bytes"], manifest_SHA256=digest(raw),
                          distinct_experts=list(result["required_expert_counts_by_rank"].values()))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--reader", type=Path, default=READER)
    args = parser.parse_args()
    freeze(args.destination, archive=args.archive, receipt_path=args.receipt, reader_path=args.reader)
