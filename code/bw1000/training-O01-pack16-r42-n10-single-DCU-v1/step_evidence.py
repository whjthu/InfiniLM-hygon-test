"""Validate the actual Slurm step record before any HIP work."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


PARENT = "1103606"
NODE = "f12r2n10"
CAP = 16384
FIELDS = ("StepId", "Name", "NodeList", "State", "TRES")
BOUNDARY = re.compile(r"^\s*StepId=(?P<id>[1-9][0-9]*\.(?:[0-9]+|batch|extern|interactive))(?=\s|$)")
KEYS = re.compile(r"(?:^|\s)(StepId|Name|NodeList|State|TRES)=([^\s]+)")


def require(condition, code):
    if not condition:
        raise ValueError(code)


def exact_record(raw, step, label, node, resource_gate, evidence):
    require(type(raw) is bytes and 0 < len(raw) <= CAP, "O01_QUERY_EXTENT")
    require(re.fullmatch(r"[1-9][0-9]*\.[0-9]+", step), "O01_EXPECTED_NUMERIC_STEP")
    require(raw.endswith(b"\n"), "O01_QUERY_INCOMPLETE_LINE")
    lines = raw.decode("ascii").splitlines()
    records = []
    evidence.update(physical_lines=len(lines), controller_records=None, matching_records=None)
    for line in lines:
        if not line.strip():
            continue
        boundary = BOUNDARY.match(line)
        tokens = KEYS.findall(line)
        if boundary is not None:
            records.append((boundary["id"], []))
        require(records and (boundary is not None or not any(key == "StepId" for key, _ in tokens))
                and (boundary is not None or line[0].isspace()
                     or re.match(r"^[A-Za-z][A-Za-z0-9_]*=", line)),
                "O01_UNCLASSIFIED_SLURM_RECORD_LINE")
        records[-1][1].extend(tokens)
    matched = [tokens for identity, tokens in records if identity == step]
    evidence.update(controller_records=len(records), matching_records=len(matched))
    require(len(matched) == 1, "O01_UNIQUE_EXACT_NUMERIC_SLURM_RECORD")
    row = dict(matched[0])
    require(len(row) == len(matched[0]), "O01_DUPLICATE_STRICT_SLURM_FIELDS")
    expected = {"StepId": step, "Name": label, "NodeList": node, "State": "RUNNING"}
    evidence["fields"] = {field: {"present": field in row, "matches": row.get(field) == value}
                          for field, value in expected.items()}
    require(all(row.get(field) == value for field, value in expected.items()), "O01_EXACT_STEP_NAME_NODE_STATE")
    evidence["TRES"] = {"present": "TRES" in row, "validator_called": False, "matches": False}
    require("TRES" in row, "O01_TRES_MISSING")
    evidence["TRES"]["validator_called"] = True
    resource_gate(row["TRES"])
    evidence["TRES"]["matches"] = True
    return row["TRES"]


def query_step(step, label, node, resource_gate, evidence, query=subprocess.check_output):
    try:
        raw = query(["scontrol", "show", "step", step, "--oneliner"], timeout=5)
    except subprocess.TimeoutExpired as error:
        evidence.update(query="TIMEOUT")
        raw = error.output
    except subprocess.CalledProcessError as error:
        evidence.update(query="NONZERO", returncode=error.returncode)
        raw = error.output
    except (OSError, subprocess.SubprocessError):
        evidence.update(query="PROCESS_ERROR")
        raw = None
    else:
        evidence.update(query="EXIT_ZERO", returncode=0)
    if type(raw) is bytes:
        evidence.update(stdout_extent=len(raw), stdout_SHA256=sha256(raw).hexdigest())
    require(evidence["query"] == "EXIT_ZERO", "O01_SLURM_QUERY_NOT_SUCCESSFUL")
    return exact_record(raw, step, label, node, resource_gate, evidence)


def cpu_tres(value):
    parts = value.split(",")
    require(len(parts) == 3 and set(parts) in (
        {"cpu=4", "mem=8G", "node=1"}, {"cpu=4", "mem=8192M", "node=1"}),
        "O01_ZERO_GRES_4CPU_8G_TRES")


def probe(epoch, label, result):
    require(re.fullmatch(r"O01-pack16-r42-cpu-[0-9a-f]{12}", label), "O01_NEW_CPU_PROBE_LABEL")
    require(math.isfinite(epoch) and 0 <= time.time() - epoch < 120, "O01_EXACT_120S_EPOCH")
    require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,
            "O01_ISOLATED_PROBE_PYTHON")
    env = os.environ
    numeric = env.get("SLURM_STEP_ID", "")
    node = socket.gethostname().split(".")[0]
    require(env.get("SLURM_JOB_ID") == PARENT and re.fullmatch(r"[0-9]+", numeric)
            and node == NODE and env.get("SLURM_CPUS_PER_TASK") == "4"
            and len(os.sched_getaffinity(0)) == 4, "O01_REAL_ZERO_GRES_NUMERIC_STEP")
    groups = Path("/proc/self/cgroup").read_text("ascii").splitlines()
    require(any({"job_" + PARENT, "step_" + numeric} <= set(line.split(":", 2)[-1].split("/")) for line in groups),
            "O01_REAL_JOB_STEP_CGROUP")
    require(env.get("SLURM_STEP_GPUS", "") in ("", "(null)", "none", "N/A")
            and all(env.get(key, "0") in ("", "0", "(null)") for key in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_TASK"))
            and not any(key in env.get("SLURM_TRES_PER_TASK", "").lower() for key in ("gres/", "gpu", "dcu"))
            and all(env.get(key) == "-1" for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")),
            "O01_NO_STEP_DEVICE_VISIBILITY")
    step = PARENT + "." + numeric
    require(step not in ("1103606.3431", "1103606.3432", "1103606.3433"), "O01_3432_FAMILY_REPLAY_FORBIDDEN")
    result.update(step=step, node=node, epoch=epoch, label=label)
    query_step(step, label, node, cpu_tres, result)
    require(time.time() < epoch + 120, "O01_CPU_EPOCH_EXPIRED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    result = {"schema": "O01_R42_ZERO_GRES_STEP_RECORD_V1", "passed": False,
              "HIP_imports": 0, "measurement_calls": 0, "actual_target_GPU": 0}
    try:
        probe(args.epoch, args.label, result)
        result["passed"] = True
    except (OSError, ValueError, UnicodeError, AttributeError) as error:
        code = str(error) if type(error) is ValueError else ""
        result.update(error_class=type(error).__name__,
                      predicate=code if re.fullmatch(r"O01_[A-Z0-9_]+", code) else None)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


from dataclasses import dataclass


FILENAME="START_STEP_QUERY_EVIDENCE.json"
FIELDS=("StepId","Name","NodeList","State")


def require(ok,code):
    if not ok:
        raise ValueError(code)


@dataclass(frozen=True)
class ExpectedStep:
    parent:str
    step:str
    node:str
    label:str
    intent_SHA256:str

    def __post_init__(self):
        require(re.fullmatch(r"[1-9][0-9]+",self.parent) and re.fullmatch(self.parent+r"\.[0-9]+",self.step)
                and re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}",self.node)
                and re.fullmatch(r"O01-pack16-[a-z0-9-]+-[0-9a-f]{12}",self.label)
                and re.fullmatch(r"[0-9a-f]{64}",self.intent_SHA256),"ORIGINAL_NUMERIC_STEP_IDENTITY_REQUIRED")


def evidence(scope):
    return {"schema":"O01_PACK16_PORTABLE_STRICT_STEP_EVIDENCE_V1","parent":scope.parent,"step":scope.step,"node":scope.node,
            "intent_SHA256":scope.intent_SHA256,"query":{"class":"NOT_INVOKED","timeout_s":5,"returncode":None,
            "stdout_extent":None,"stdout_SHA256":None,"complete":None,"extent_ok":None,"ascii":None},
            "row_count":None,"fields":{name:{"present":None,"matches":None} for name in FIELDS},
            "TRES":{"gate_reached":False,"present":None,"validator_called":False,"matches":None},"result":"NO_QUERY_RESULT"}


def persist(directory,receipt):
    raw=(json.dumps(receipt,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    require(directory.resolve(strict=True)==directory and directory.is_dir() and len(raw)<=4096,"PRIVATE_BOUNDED_OUTPUT_REQUIRED")
    path=directory/FILENAME
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def observe(scope,receipt,query,resource_gate):
    receipt["result"]="STRICT_NUMERIC_RECORD_OR_TRES_REJECTED"
    record={}
    try:
        tres=query_step(scope.step,scope.label,scope.node,resource_gate,record,query)
        receipt["result"]="STRICT_NUMERIC_RECORD_AND_TRES_PASSED"
        return tres
    finally:
        receipt["record_evidence"]=record
