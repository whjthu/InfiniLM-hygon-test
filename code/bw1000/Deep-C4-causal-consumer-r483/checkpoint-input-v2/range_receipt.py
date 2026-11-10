"""Validate the original Deep483 range inspector receipt without opening weights."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

METADATA_SHA256 = "f56b6d38bdf6097488e707b200ea2cc8b4466ce27750de1c038a6be987a4cf00"
MAX_RECEIPT_BYTES = 16 * 1024
EXPECTED_RANGE_BYTES = 13_144_320


def sha(data):
    return sha256(data).hexdigest()


def digest_text(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate checkpoint receipt field")
        result[key] = value
    return result


def validate_receipt(raw, receipt_sha, metadata_raw):
    """Return typed-loader pins only from a separately identified receipt."""
    if len(raw) > MAX_RECEIPT_BYTES or not digest_text(receipt_sha) or sha(raw) != receipt_sha:
        raise ValueError("original range receipt size or digest differs")
    if sha(metadata_raw) != METADATA_SHA256:
        raise ValueError("frozen checkpoint metadata digest differs")
    metadata = json.loads(metadata_raw)
    receipt = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(receipt, dict):
        raise ValueError("range receipt object required")
    fields = {
        "schema": 1, "layer": metadata["layer"],
        "original_config_SHA256": metadata["original_config_SHA256"],
        "original_index_SHA256": metadata["original_index_SHA256"],
        "total_selected_weight_bytes": EXPECTED_RANGE_BYTES,
        "header_SHA256": {name: row["header_SHA256"] for name, row in metadata["headers"].items()},
        "whole_shard_hashed": False, "Native_dlopen": False, "Runtime": False, "GPU": False,
        "weight_values_exported": False,
    }
    for name, expected in fields.items():
        if type(receipt.get(name)) is not type(expected) or receipt[name] != expected:
            raise ValueError(f"range receipt field differs: {name}")
    ranges = receipt.get("weights")
    if not isinstance(ranges, dict) or set(ranges) != set(metadata["weights"]):
        raise ValueError("exact seven original checkpoint ranges required")
    pins = {}
    for name, expected in sorted(metadata["weights"].items()):
        value = ranges[name]
        if (not isinstance(value, dict) or set(value) != {"bytes", "SHA256"}
                or type(value["bytes"]) is not int or value["bytes"] != expected["bytes"]
                or not digest_text(value["SHA256"])):
            raise ValueError(f"invalid original range pin: {name}")
        pins[name] = dict(bytes=value["bytes"], SHA256=value["SHA256"])
    return dict(**fields, weights=pins)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--receipt-sha", required=True)
    args = parser.parse_args()
    with args.receipt.open("rb") as file:
        raw = file.read(MAX_RECEIPT_BYTES + 1)
    metadata = Path(__file__).with_name("CHECKPOINT_METADATA.json").read_bytes()
    pins = validate_receipt(raw, args.receipt_sha, metadata)
    # Keep the actual pins in the caller. The CLI emits only receipt identity and counts.
    print(json.dumps(dict(receipt_SHA256=args.receipt_sha, ranges=len(pins["weights"]),
                          range_bytes=pins["total_selected_weight_bytes"],
                          request_execution_inferred=False, weight_bytes_read_by_reader=0)))


if __name__ == "__main__":
    main()
