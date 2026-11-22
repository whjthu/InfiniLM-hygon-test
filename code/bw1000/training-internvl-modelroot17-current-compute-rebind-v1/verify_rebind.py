"""Bind the sealed InternVL inventory to the current T03 compute node."""
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def fields(line):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", line))


def query(*command):
    return subprocess.check_output(command, text=True, timeout=8).strip().splitlines()


def pinned(path, expected, cap):
    need(path.resolve(strict=True) == path and path.is_file(), "Original evidence path differs")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    need(0 < len(raw) <= cap and sha256(raw).hexdigest() == expected, "Original evidence bytes differ")
    return raw


def main():
    claim = json.loads((HERE / "CLAIM.json").read_bytes())
    base = HERE / "original_compute_verifier.py"
    pinned(base, claim["original_compute_verifier_SHA256"], 65536)
    original = runpy.run_path(str(base), run_name="internvl_rebind")
    bindings = original["node"].__globals__
    bindings.update(OUT=HERE, REQUEST=claim["request"], PARENT=claim["parent"], NODE=claim["node"])
    auditor = original["load_assembler"]()
    need(os.getuid() == auditor.OWNER and HERE.name == claim["label"], "Unexpected owner or scope")
    if sys.argv[1:] == ["node"]:
        original["node"](auditor)
        raw_proof = (HERE / "computed-proof.json").read_bytes()
        print(json.dumps({"proof": json.loads(raw_proof), "SHA256": sha256(raw_proof).hexdigest()}))
        return 0
    need(sys.argv[1:] == [], "Unexpected invocation")

    raw = pinned(ROOT / "results/training-internvl-modelroot17-compute-CPU-v1/export-pins.json",
                 claim["original_export_pin_SHA256"], 65536)
    anchors = json.loads(raw)
    need(anchors["request"] == claim["original_mount_request"] and
         anchors["plan_SHA256"] == original["PLAN_SHA"] and len(anchors["files"]) == 20,
         "Original lease export identity differs")
    old = pinned(ROOT / "results/training-internvl-modelroot17-compute-CPU-v1/outcome.json",
                 claim["original_mount_outcome_SHA256"], 65536)
    need(json.loads(old)["state"] == "VERIFIED_ALL17_READONLY_ON_T03_COMPUTE_NODE", "Original compute proof differs")

    parent, node, label = claim["parent"], claim["node"], claim["label"]
    [job] = [fields(line) for line in query("scontrol", "show", "job", parent, "-o")]
    need(job["JobState"] == "RUNNING" and job["UserId"].endswith(f"({os.getuid()})") and
         "gres/dcu=320" in job["AllocTRES"].split(",") and
         node in query("scontrol", "show", "hostnames", job["NodeList"]), "Original T03 ownership differs")
    for line in query("scontrol", "show", "step", parent, "-o"):
        step = fields(line)
        if re.fullmatch(parent + r"\.\d+", step.get("StepId", "")):
            need(node not in query("scontrol", "show", "hostnames", step["NodeList"]), "T03 node is occupied")

    def accounting():
        rows = [line.split("|") for line in query("sacct", "-nP", "-j", parent, "--format=" + COLUMNS)]
        matches = [row for row in rows if row[1] == label]
        need(len(matches) <= 1, "Duplicate scoped compute reader")
        return matches[0] if matches else None

    need(accounting() is None, "Original reader already entered Slurm")
    anchors["request"] = claim["request"]
    auditor.write_once(HERE / "export-pins.json", auditor.serialized(anchors))
    argv = ["srun", "--jobid=" + parent, "--job-name=" + label, "--exact", "--immediate=5", "--nodes=1",
            "--ntasks=1", "--nodelist=" + node, "--cpus-per-task=1", "--mem=2048M", "--gres=none",
            "--time=1", sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()), "node"]
    with (HERE / "step.log").open("xb") as stdout:
        child = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=subprocess.STDOUT, timeout=42)
    deadline = time.monotonic() + 9
    row = accounting()
    while (not row or row[2] in {"RUNNING", "PENDING", "COMPLETING"}) and time.monotonic() < deadline:
        time.sleep(.4)
        row = accounting()
    result = {"request": claim["request"], "accounting": row, "step_client_rc": child.returncode,
              "current_compute_mount_verified": False, "training_READY": False, "weight_body_bytes": 0}
    if child.returncode == 0 and row and row[2:4] == ["COMPLETED", "0:0"] and row[5] == node and "gres/" not in row[6]:
        [receipt] = [json.loads(line) for line in (HERE / "step.log").read_bytes().splitlines()
                     if line.startswith(b"{")]
        proof = receipt["proof"]
        need(proof["step"] == row[0] and proof["request"] == claim["request"] and proof["read_only_members"] == 17,
             "Compute proof identity differs")
        need(receipt["SHA256"] == sha256(auditor.serialized(proof)).hexdigest(), "Compute receipt digest differs")
        result.update(proof, current_compute_mount_verified=True, compute_proof_SHA256=receipt["SHA256"])
    body = auditor.serialized(result)
    auditor.write_once(HERE / "outcome.json", body)
    print(json.dumps(result | {"outcome_SHA256": sha256(body).hexdigest()}))
    return int(not result["current_compute_mount_verified"])


if __name__ == "__main__":
    raise SystemExit(main())
