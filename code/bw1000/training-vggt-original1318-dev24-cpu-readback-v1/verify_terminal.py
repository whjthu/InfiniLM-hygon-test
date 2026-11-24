"""Independently qualify numeric two-CPU zero-GRES accounting and retain provider evidence."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import host
import worker


EXACT = ({"cpu=2", "mem=2G", "node=1"}, {"cpu=2", "mem=2048M", "node=1"})
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
EPOCH_HANDOFF_SECONDS = 5


def stamp(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def exact_accounting(intent, actual_step, raw):
    worker.need(re.fullmatch(re.escape(intent["parent"]) + r"\.[0-9]+", actual_step)
                and intent["label"] == host.CPU_STAGE + intent["request"] and intent["replay_allowed"] is False
                and intent["deadline"] == intent["epoch"] + worker.LIMIT
                and intent["epoch"] <= intent["submission_epoch"] < intent["epoch"] + 25, "ORIGINAL_INTENT_BINDING")
    rows = [line.split("|") for line in raw.splitlines() if line.split("|", 1)[0] == actual_step]
    worker.need(len(rows) == 1 and len(rows[0]) == 9, "EXACT_NINE_COLUMN_NUMERIC_STEP")
    row, = rows
    resources = row[6].split(",")
    state = row[2].split()[0].rstrip("+")
    worker.need(row[1] == intent["label"] and state in TERMINAL and re.fullmatch(r"[0-9]+:[0-9]+", row[3])
                and row[4].isascii() and row[4].isdecimal() and row[5] == intent["node"]
                and len(resources) == 3 and set(resources) in EXACT, "EXACT_INDEPENDENT_ZERO_GRES_2CPU_2G")
    start, end = stamp(row[7]), stamp(row[8])
    worker.need(math.floor(intent["submission_epoch"]) <= start <= end <= math.ceil(intent["deadline"] + EPOCH_HANDOFF_SECONDS)
                and int(row[4]) <= worker.LIMIT + EPOCH_HANDOFF_SECONDS and abs(end-start-int(row[4])) <= 1,
                "EXACT_ORIGINAL_EPOCH_TERMINAL_BUDGET")
    return row


def verify(intent, actual_step, accounting, result_raw=None, provider_raw=None):
    row = exact_accounting(intent, actual_step, accounting)
    expected = {key: intent[key] for key in ("request", "parent", "node", "epoch", "deadline")}
    expected.update(step=actual_step, cpus=2, GPU=0)
    out = {**worker.DENOM, "request": intent["request"], "accounting": row, "actual_target_inputs_qualified": 0,
           "independent_provider_identity_bound": False, "state": "TERMINAL_WITHOUT_WORKER_RESULT"}
    if provider_raw is not None:
        provider_doc = json.loads(provider_raw)
        worker.need(provider_doc.get("schema") == "vggt.1318.cpu-provider.v1"
                    and provider_doc["binding"] == expected and all(provider_doc[key] == val for key, val in worker.DENOM.items()),
                    "EXACT_INDEPENDENT_PROVIDER_SCOPE")
        provider = provider_doc["provider"]
        worker.need(provider["loaded_torch"] is False and all(isinstance(provider[key], str) and provider[key] for key in
                    ("version", "module_path", "core_extension_path", "python_executable", "python_extension_suffix"))
                    and (provider["python_SOABI"] is None or isinstance(provider["python_SOABI"], str))
                    and all(type(provider[key]["bytes"]) is int and provider[key]["bytes"] > 0
                            and host.SHA.fullmatch(provider[key]["SHA256"]) for key in ("core_extension", "module")),
                    "EXACT_INDEPENDENT_NUMPY_PROVIDER_PINS")
        out.update(independent_provider_identity_bound=True, provider_artifact=worker.pin(provider_raw), provider=provider)
    if result_raw is not None:
        result = json.loads(result_raw)
        worker.need(result["binding"] == expected and all(result[key] == val for key, val in worker.DENOM.items())
                    and result["actual_provider_succeeded"] == (provider_raw is not None)
                    and result["actual_target_inputs_qualified"] in (0, 22), "EXACT_INDEPENDENT_WORKER_BINDING")
        if provider_raw is not None:
            worker.need(result["provider_receipt"] == worker.pin(provider_raw), "ORIGINAL_INDEPENDENT_PROVIDER_RECEIPT")
        out.update(state=result["state"], result_artifact=worker.pin(result_raw))
        if result["state"] == "PROVIDER_AND_SAME_22_OF_24_PROVENANCE_CHECKED":
            audit = result["audit_result"]
            worker.need(row[2:4] == ["COMPLETED", "0:0"] and provider_raw is not None
                        and result["actual_target_inputs_qualified"] == 22 and audit["qualified_common_windows"] == 22
                        and audit["original_windows"] == 24 and audit["actual_weight_bytes_read"] == 0
                        and audit["predictions"] == 0, "SUCCESS_REQUIRES_SCHEDULER_AND_AUDIT_IDENTITY")
            out["actual_target_inputs_qualified"] = 22
        else:
            worker.need(result["actual_target_inputs_qualified"] == 0, "FAILED_WORKER_CANNOT_QUALIFY_INPUTS")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("request", "step", "intent-sha", "accounting-sha"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = host.PROJECT / "incoming" / host.scope_name(args.request) / "evidence"
    intent = json.loads(host.read(evidence/"INTENT.json", 16384, args.intent_sha))
    worker.need(intent["request"] == args.request and host.SHA.fullmatch(args.intent_sha)
                and host.SHA.fullmatch(args.accounting_sha) and args.output.parent == evidence, "EXACT_PRIVATE_INDEPENDENT_OUTPUT")
    accounting = host.read(args.accounting, 131072, args.accounting_sha).decode("ascii")
    artifacts = [host.read(path, 16384) if path.exists() else None
                 for path in (evidence/"RESULT.json", evidence/"PROVIDER.json")]
    result = verify(intent, args.step, accounting, *artifacts)
    result["independent_accounting_SHA256"] = args.accounting_sha
    print(json.dumps({"receipt": worker.save(args.output, result), "qualified": result["actual_target_inputs_qualified"],
                      "provider_bound": result["independent_provider_identity_bound"]}, sort_keys=True))


if __name__ == "__main__":
    main()
