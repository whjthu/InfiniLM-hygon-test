"""Bind two disjoint original range receipts to one exact CSA checkpoint map."""

import argparse
from base64 import b64decode
from hashlib import sha256
import json
from pathlib import Path
import re
from types import ModuleType


BASE = Path(__file__).resolve().parents[1]
METADATA_SHA = "39e2af65e0f30502d4a0d1f97df54fc27f0023f2ba1861810a634b1065a7e4f7"
NON_INDEXER_SHA = "be180a2fa0ce240e7b88079c4d417effba37f5ecc7a440f18541a32c511859fd"
INDEXER_REQUEST = "7ae20a906e2646b1bbcea3e712cb6bba"
ACCOUNTING_STEP = "1194900.540"
DEP = {
    "csa-identity-swa-checkpoint-v1/inspect_checkpoint.py": "6260ca808fc5650d3c269de3fef9d219797220009173b8347d15cfdac565a88d",
    "csa-non-indexer-range-pins-v1/read_ranges.py": "6af479a6bd0351a1ab9b0c5defb80acce61dff329b3ca986e33205614e72d212",
    "checkpoint-input-v2/range_receipt.py": "fe7e96d235e8d3b891692b9dea2a985715bbc883f3c248daf0623332f43bbff1",
    "checkpoint-input-v2/CHECKPOINT_METADATA.json": "f56b6d38bdf6097488e707b200ea2cc8b4466ce27750de1c038a6be987a4cf00",
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def pinned(path, cap, wanted):
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    need(0 < len(raw) <= cap and digest(raw) == wanted, "pinned evidence bytes differ: " + path.name)
    return raw


def module(relative):
    path = BASE / relative
    parsed = ModuleType(path.stem)
    parsed.__file__ = str(path)
    exec(compile(pinned(path, 65536, DEP[relative]), str(path), "exec"), vars(parsed))
    return parsed


def bind(metadata_raw, non_indexer_raw, original_indexer=None):
    """Return a coverage report and a full consumable receipt only for 23 verified ranges."""
    need(digest(metadata_raw) == METADATA_SHA and digest(non_indexer_raw) == NON_INDEXER_SHA,
         "metadata or original .540 receipt digest differs")
    csa = module("csa-identity-swa-checkpoint-v1/inspect_checkpoint.py")
    selector = module("csa-non-indexer-range-pins-v1/read_ranges.py")
    indexer = module("checkpoint-input-v2/range_receipt.py")
    old_path = "checkpoint-input-v2/CHECKPOINT_METADATA.json"
    indexer_meta_raw = pinned(BASE / old_path, 65536, DEP[old_path])
    indexer_meta = csa.parse(indexer_meta_raw)
    metadata, non = csa.parse(metadata_raw), csa.parse(non_indexer_raw)
    selected = {name for _, _, name in selector.selection(metadata)}
    header_pins = {name: row["header_SHA256"] for name, row in metadata["headers"].items()}
    shared = {key: metadata[key] for key in ("original_config_SHA256", "original_index_SHA256")}
    indexer_names = {"indexer." + name for name in indexer_meta["weights"]}
    rows = metadata["weights"]
    need(not selected & indexer_names and set(rows) == selected | indexer_names
         and metadata["layer"] == indexer_meta["layer"] == 2 and metadata["TP"] == indexer_meta["TP"] == 1
         and all(indexer_meta[key] == val for key, val in shared.items())
         and {name: row["header_SHA256"] for name, row in indexer_meta["headers"].items()} == header_pins
         and all(rows["indexer." + name] == spec for name, spec in indexer_meta["weights"].items()),
         "original indexer metadata differs from the CSA layer-2 rows")
    need(non["schema"] == "Deep-CSA-layer2-non-indexer-range-pins-v1" and non["step"] == ACCOUNTING_STEP
         and non["node"] == selector.NODE and non["metadata_SHA256"] == METADATA_SHA
         and all(non[key] == val for key, val in shared.items()) and non["header_SHA256"] == header_pins
         and non["full_checkpoint_bound"] is False and non["actual_GPU_calls"] == non["indexer_payload_bytes_read"] == 0
         and non["actual_value_bytes_read"] == selector.EXPECTED_BYTES and set(non["weights"]) == selected,
         "original .540 range receipt scope differs")
    weights = non["weights"]
    for name, pin in weights.items():
        need(set(pin) == {"bytes", "SHA256"} and type(pin["bytes"]) is int and pin["bytes"] == rows[name]["bytes"]
             and indexer.digest_text(pin["SHA256"]), "invalid non-indexer range pin")
    missing = sorted(indexer_names)
    proof = dict(schema="bw1000.deep.csa.split-range-coverage.v1", layer=2, TP=1,
                 metadata_SHA256=METADATA_SHA, non_indexer_SHA256=NON_INDEXER_SHA, non_indexer_step=ACCOUNTING_STEP,
                 original_indexer_request=INDEXER_REQUEST, validated_non_indexer_ranges=len(weights),
                 non_indexer_payload_bytes=selector.EXPECTED_BYTES, validated_indexer_ranges=0,
                 missing_indexer_names=missing, missing_indexer_payload_bytes=sum(rows[n]["bytes"] for n in missing),
                 planned_CSA_payload_bytes=metadata["selected_payload_bytes"], full_payload_pins_complete=False,
                 checkpoint_weights_read_by_binder=0, actual_GPU_calls=0, load_or_GPU_qualified=False)
    if original_indexer is None:
        return proof, None
    indexer_raw, indexer_sha = original_indexer
    accepted = indexer.validate_receipt(indexer_raw, indexer_sha, indexer_meta_raw)["weights"]
    merged = canonical(dict(schema="Deep-CSA-layer2-complete-range-bindings-v1", metadata_SHA256=METADATA_SHA,
                            **shared, header_SHA256=header_pins, weights={**weights, **{
                                "indexer." + name: pin for name, pin in accepted.items()}},
                            non_indexer_original_SHA256=NON_INDEXER_SHA, indexer_original_SHA256=indexer_sha))
    merged_sha = digest(merged)
    complete = csa.bind_range_pins(metadata_raw, METADATA_SHA, merged, merged_sha)
    need(len(complete) == 23 and sum(pin["bytes"] for pin in complete.values()) == selector.SOURCE_PAYLOAD,
         "complete CSA payload scope differs")
    return dict(proof, validated_indexer_ranges=7, missing_indexer_names=[], missing_indexer_payload_bytes=0,
                full_payload_pins_complete=True, merged_receipt_SHA256=merged_sha), merged


def inspect_private_wrapper(raw, original_indexer_sha=None):
    receipt = json.loads(raw)
    seen = receipt["observed"]
    need(receipt["rc"] == 0 and receipt["sent"] is True and receipt["authenticated"] is True
         and seen["old_request"] == INDEXER_REQUEST and seen["checkpoint_payload_reads"] == seen["executed_remote_inspectors"] == 0,
         "authenticated read-only wrapper is required")
    rows = seen["files"]

    def read(name):
        row = rows[name]
        payload = b64decode(row["evidence_base64"], validate=True)
        need(row["exists"] is True and 0 < len(payload) == row["bytes"] <= 16384
             and re.fullmatch(r"[a-f0-9]{64}", row["SHA256"]) and digest(payload) == row["SHA256"],
             "returned private file binding differs")
        return payload, row["SHA256"]

    visible = rows["original_seven"]["exists"] is True
    original = None
    status = "UNKNOWN_RESULT_ABSENT"
    if original_indexer_sha is not None:
        need(visible, "independent indexer digest supplied but original receipt absent")
        payload, enclosed_sha = read("original_seven")
        need(enclosed_sha == original_indexer_sha, "independent original indexer digest differs")
        original = payload, original_indexer_sha
        status = "RECEIPT_BOUND"
    elif visible:
        status = "VISIBLE_INDEPENDENT_PIN_PENDING"
    report, full = bind(read("metadata")[0], read("non_indexer")[0], original)
    return dict(report, observed_epoch=seen["observed_epoch"], private_wrapper_SHA256=digest(raw),
                original_indexer_result_visible=visible, original_indexer_status=status), full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_wrapper", type=Path)
    parser.add_argument("--full-receipt", type=Path)
    parser.add_argument("--original-indexer-sha", help="independently recorded digest of the original seven-range receipt")
    args = parser.parse_args()
    with args.private_wrapper.open("rb") as stream:
        raw = stream.read(65537)
    need(0 < len(raw) <= 65536, "bounded private wrapper required")
    report, full = inspect_private_wrapper(raw, args.original_indexer_sha)
    if args.full_receipt:
        need(full is not None, "missing original seven-range receipt; no full receipt can be written")
        with args.full_receipt.open("xb") as stream:
            stream.write(full)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
