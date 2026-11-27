"""Bounded read-only terminal/metadata verifier for a single native OFF step."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"


def verify_row(row, pid, label, node):
    if len(row) != 9 or not re.fullmatch(re.escape(pid) + r"\.[0-9]+", row[0]) or row[1] != label or row[5] != node:
        raise ValueError("OFF_TERMINAL_IDENTITY")
    tres = dict(x.split("=", 1) for x in row[6].split(","))
    if set(tres) != {"cpu", "mem", "node"} or tres["cpu"] != "8" or tres["mem"] not in ("12288M", "12G") or tres["node"] != "1":
        raise ValueError("OFF_TERMINAL_TRES")
    return row[2:4] == ["COMPLETED", "0:0"] and row[4].isdigit() and 0 < int(row[4]) <= 240 and row[7] < row[8]


def read(run, manifest_sha, account):
    run = run.resolve(strict=True)
    def load(name, expected=None):
        path = run / name
        if path.resolve(strict=True) != path or not path.is_file() or not 0 < path.stat().st_size <= 4 * 1024**2:
            raise ValueError("OFF_METADATA_TYPE")
        data = path.read_bytes()
        if expected is not None and hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("OFF_METADATA_SHA")
        return json.loads(data)
    source = load("src/SOURCE_FILES.json", manifest_sha)
    intent = load("intent.json")
    claim = load("src/CLAIM.json", source["files"]["CLAIM.json"]["SHA256"])
    request = claim["request"]
    label = f"training-minicpm-off-native6-cpu-{request[:12]}"
    if not (re.fullmatch(r"[0-9a-f]{32}", request) and run.name == label == claim["label"]
            and intent["claim_SHA256"] == source["files"]["CLAIM.json"]["SHA256"]
            and intent["source_manifest_SHA256"] == manifest_sha
            and all(intent[k] == claim[k] for k in ("request", "publication", "parent", "node", "input_manifest_SHA256"))
            and intent["deadline"] - intent["epoch"] == 300):
        raise ValueError("OFF_TERMINAL_SCOPE")
    rows = [x.split("|") for x in account(claim["parent"]).splitlines()]
    matches = [row for row in rows if len(row) > 1 and row[1] == label]
    if len(matches) != 1:
        return dict(request=request, passed=False, state="OFF_STEP_MISSING_OR_AMBIGUOUS", planned=6, completed=None)
    row = matches[0]
    success = verify_row(row, claim["parent"], label, claim["node"])
    result = dict(request=request, passed=False, state="OFF_FAILURE_OR_INCOMPLETE", accounting=row, planned=6, completed=None)
    if not (run / "outcome.json").is_file():
        return result
    out = load("outcome.json")
    if out["request"] != request or out["publication"] != claim["publication"]:
        raise ValueError("OFF_OUTCOME_SCOPE")
    result["completed"] = out["completed_rows"]
    if not (success and out["passed"] is True and out["srun_returncode"] == 0 and out["srun_reaped"] is True
            and out["terminal"]["row"] == row[:7]):
        return result
    worker = load("worker-result.json", out["worker_result_SHA256"])
    suffix = load("native-suffix-private.json", worker["native_suffix"]["SHA256"])
    records = load("native-identities-private.json", worker["native_identities"]["SHA256"])
    providers = load("providers-private.json")
    need = (worker["passed"] is True and worker["completed_rows"] == worker["planned_rows"] == 6
            and worker["encode_calls"] == 12 and worker["collator_calls"] == 6 and worker["unique_media"] == 4
            and worker["source_dataset_SHA256"] == claim["input_manifest_SHA256"]
            and worker["EOS_supervised"] == worker["full_untruncated"] == 6
            and worker["GPU_allocation"] == worker["actual_optimizer_updates"] == 0
            and worker["model_loaded"] is False and worker["dev_or_blind_read"] is False
            and worker["own_device_FDs_before"] == worker["own_device_FDs_after"] == []
            and suffix["template_suffix"] == ["<|im_end|>\n"] and suffix["EOS_offsets"]
            and suffix["native_suffix_metadata_encode_calls"] == 2 and suffix["full_and4096_suffix_equal"] is True
            and [x["ordinal"] for x in records] == list(range(1, 7))
            and all(x["source_id_SHA256"] == hashlib.sha256(f"off-train-{i:02d}".encode()).hexdigest()
                    and 1 < x["length"] <= 4096 and x["shifted_supervised_tokens"] > 0
                    and x["EOS_supervised"] is True and x["full_untruncated"] is True
                    for i, x in enumerate(records, 1))
            and all(providers[role] and providers[role][-1]["official_wheel_bytes_equal"] is True
                    for role in ("get_model_processor", "get_template", "encode", "data_collator", "set_mode", "encode_context", "tokenize")))
    if not need:
        raise ValueError("OFF_NATIVE_RESULT_CONTRACT")
    return dict(result, passed=True, state="OFF6_NATIVE_ZERO_GRES_INGRESS_PASS", completed=6,
                model_loaded=False, actual_optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{32}", args.request) or not re.fullmatch(r"[0-9a-f]{64}", args.manifest):
        raise ValueError("OFF_READER_ARGV")
    def account(parent):
        return subprocess.check_output(["sacct", "-j", parent, "-nP", "-o", COLUMNS], timeout=12, text=True)
    result = read(ROOT / f"training-minicpm-off-native6-cpu-{args.request[:12]}", args.manifest, account)
    if result["request"] != args.request:
        raise ValueError("OFF_READER_REQUEST")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
