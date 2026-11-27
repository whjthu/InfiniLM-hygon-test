"""Read an authenticated GQA preflight into the existing T03 capture roster."""
from hashlib import sha256
import json
from pathlib import Path
import re


BEFORE = ("input_ids", "labels", "pixel_values")
AFTER = ("attention_mask", "labels", "logits_to_keep")
DATASET_SHA256 = "0ccbe304ad1d7e686bcfcf8b81bf17ef43bab4c4fcaeef5410fc7010bf170752"
PROVIDERS = {
    "swift_mixin": "6b6de2e7503ce414cb837961e1355be41a56787ecca502e8570bfcfa36c15ab9",
    "swift_prepare_inputs": "28fe9cb927390ad423add8b98411a90fb2d1c801ead124b354802bdd0779ecf6",
    "framework_fingerprint": "816ef694537273ac04d8f7d40df5694a16e8a869797d2d2f65664d21e6a0ea49",
}
HEX = re.compile("[0-9a-f]{64}")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def digest_value(value):
    return isinstance(value, str) and HEX.fullmatch(value) is not None


def build_roster(receipt, staged_train, request):
    """Bind each prepared input tuple to one source row and one post-mask tuple."""
    require(isinstance(request, str) and re.fullmatch("[0-9a-f]{32}", request) is not None,
            "exact fingerprint request required")
    require(receipt["request"] == request and receipt["state"] == "NATIVE_FINGERPRINT_RECEIPT"
            and receipt["GPU"] == receipt["client_rc"] == 0, "successful zero-GRES receipt required")
    account = receipt["accounting"]
    require(len(account) == 9 and re.fullmatch(r"1103606\.\d+", account[0]) is not None
            and account[1:4] == ["training-internvl-gqa-logit64-cpu-" + request[:12], "COMPLETED", "0:0"]
            and account[5:7] == ["f13r1n14", "cpu=2,mem=6G,node=1"], "terminal fingerprint accounting required")
    w = receipt["worker"]
    require(w["fingerprint_providers"] == PROVIDERS and w["meta_model_only"] is w["passed"] is True
            and w["gpu_initialized"] is w["training_ready"] is False
            and w["model_weights_read"] == w["optimization_updates"] == 0, "native fingerprint provider differs")
    lines = staged_train.splitlines()
    require(w["source_train_SHA256"] == DATASET_SHA256
            and digest(staged_train) == w["staged_train_SHA256"]
            and len(lines) == w["native_rows"] == w["native_forward_fingerprint_rows"] == len(w["rows"]) == 64,
            "exact candidate training rows required")
    roster = {}
    row_digests = set()
    for ordinal, (line, row) in enumerate(zip(lines, w["rows"], strict=True), 1):
        item = row["native_logit_fingerprints"]
        row_digest = digest(line)
        require(row["ordinal"] == ordinal and row["staged_row_SHA256"] == row_digest
                and row_digest not in row_digests and row["original_prefix_masked"] is True
                and 0 < row["supervised_eos"] <= row["supervised_tokens"] < row["full_tokens"] <= 4096
                and item["selected_logit_rows"] == row["supervised_tokens"] + 1,
                "candidate row identity or native supervision differs")
        before, after = item["before"], item["after"]
        require(set(before) == set(BEFORE) and set(after) == set(AFTER)
                and all(digest_value(value) for value in (*before.values(), *after.values())),
                "native input fingerprint fields differ")
        key = tuple(before[name] for name in BEFORE)
        require(key not in roster, "candidate pre-forward tuple is ambiguous")
        roster[key] = {"ordinal": ordinal, "row_SHA256": row_digest, "after": after}
        row_digests.add(row_digest)
    return roster


def load_roster(receipt_path, receipt_sha, dataset_path, expected_extent, request):
    receipt_path, dataset_path = Path(receipt_path), Path(dataset_path)
    require(all(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
                for path in (receipt_path, dataset_path)),
            "canonical trained input artefacts required")
    with receipt_path.open("rb") as stream:
        raw = stream.read(100_001)
    require(0 < len(raw) <= 100_000 and digest(raw) == receipt_sha, "exact CPU receipt bytes required")
    require(0 < expected_extent <= 128_000, "bounded candidate training input bytes required")
    with dataset_path.open("rb") as stream:
        data = stream.read(expected_extent + 1)
    require(len(data) == expected_extent, "candidate dataset extent differs")
    return build_roster(json.loads(raw), data, request)
