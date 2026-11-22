"""Compare the original T07 continuation traces without loading tensor payloads."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
import sys


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/"
            "training-zimage-COCO20-native25-resume20-hip-v2")
TRACE_PINS = {
    "continuous": (21546, "34a669bd9ccd2c3cde26d4bbcb661e5a595b7451e40d4895d49e51adb485ce0a"),
    "resume20": (4305, "5abb4d8be0d38a9ed62d7c5d7215ccb22372065e8503ff50b5ff52d4e95f8924"),
}
INPUT_FIELDS = ("timestep_id", "cast_timestep_SHA256", "input_latent_SHA256",
                "noise_SHA256", "noised_latent_SHA256", "training_target_SHA256")
STAGES = ("input_identity", "training_operands", "prediction", "loss", "preclip_norm")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def trace(raw, expected):
    rows = [json.loads(line) for line in raw.splitlines()]
    need([row["global_step"] for row in rows] == list(expected), "Original update denominator differs")
    for row in rows:
        step = row["global_step"]
        need(type(step) is int and type(row["ordinal"]) is int and row["ordinal"] == (step - 1) % 20,
             "Original data cursor differs")
        need(type(row["loss"]) in (int, float) and isfinite(row["loss"]) and row["loss"] >= 0
             and type(row["preclip_norm"]) in (int, float) and isfinite(row["preclip_norm"])
             and row["preclip_norm"] > 0, "Invalid original loss or clipping norm")
        operands = row["operands"]
        need(set(operands) == {*INPUT_FIELDS, "prediction_SHA256"}
             and type(operands["timestep_id"]) is int and 0 <= operands["timestep_id"] < 1000
             and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                     for name, value in operands.items() if name.endswith("_SHA256")),
             "Incomplete original timestep, operand or prediction identity")
        need(isinstance(row["input_identity"], dict) and bool(row["input_identity"]), "Missing original input")
    return rows


def compare(continuous, resumed):
    outcomes = []
    for left, right in zip(trace(continuous, range(1, 26))[20:], trace(resumed, range(21, 26)), strict=True):
        a, b = left["operands"], right["operands"]
        equal = {
            "input_identity": left["input_identity"] == right["input_identity"],
            "training_operands": all(a[field] == b[field] for field in INPUT_FIELDS),
            "prediction": a["prediction_SHA256"] == b["prediction_SHA256"],
            "loss": left["loss"] == right["loss"],
            "preclip_norm": left["preclip_norm"] == right["preclip_norm"],
        }
        outcomes.append({"global_step": right["global_step"], "exact": equal,
                         "loss_absolute_difference": abs(left["loss"] - right["loss"]),
                         "preclip_norm_absolute_difference": abs(left["preclip_norm"] - right["preclip_norm"])})
    return {"paired_updates": len(outcomes), "steps": outcomes,
            "first_difference_by_observed_stage": {
                stage: next((row["global_step"] for row in outcomes if not row["exact"][stage]), None)
                for stage in STAGES
            }, "gradient_tensors_per_step_available": False,
            "optimizer_tensors_per_step_available": False, "original_recovery_passed": False}


def read_original(arm):
    name = ROOT / arm / "steps.jsonl"
    size, digest = TRACE_PINS[arm]
    need(name.resolve(strict=True) == name and name.is_file(), "Original trace escaped its run directory")
    with name.open("rb") as stream:
        raw = stream.read(size + 1)
    need(len(raw) == size and sha256(raw).hexdigest() == digest, "Original trace differs from its frozen receipt")
    return raw


def main():
    if len(sys.argv) != 1:
        raise SystemExit("This reader accepts only the two original pinned traces")
    result = compare(read_original("continuous"), read_original("resume20"))
    result["source_trace_pins"] = {arm: {"bytes": size, "SHA256": digest}
                                   for arm, (size, digest) in TRACE_PINS.items()}
    result["raw_trace_payload_transferred"] = False
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
