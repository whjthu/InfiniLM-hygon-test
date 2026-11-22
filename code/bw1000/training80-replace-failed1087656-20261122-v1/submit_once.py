"""Submit one bounded replacement for the failed forty-node project allocation."""

from hashlib import sha256
from pathlib import Path
import json
import os
import pwd
import re
import subprocess
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")

def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def query(args, timeout=12):
    return subprocess.run(args, check=True, text=True, capture_output=True, timeout=timeout).stdout.splitlines()


def record(path, data):
    raw = (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    return sha256(raw).hexdigest()


def scheduled_count(tres, name):
    vals = dict(x.split("=", 1) for x in tres.split(",") if "=" in x)
    gpu = vals.get(name)
    typed = [int(v) for k, v in vals.items() if k.startswith(name + ":")]
    if gpu is None:
        return sum(typed)
    count = int(gpu)
    need(not typed or sum(typed) == count, "Generic and typed DCU totals disagree")
    return count


def info(job_id):
    [row] = query(["scontrol", "show", "job", "-o", job_id])
    return dict(x.split("=", 1) for x in row.split() if "=" in x)


def owned(job):
    return job.get("UserId", "").endswith(f"({os.getuid()})")


def validate(c, node_name):
    name = c["label"]
    need(c["nodes"] == 40 and c["cards_per_node"] == 8 and c["max_total_project_DCU"] == 640,
         "Original replacement size differs")
    need(c["survivor"] == "1103606" and c["failed_parent"] == "1087656", "Original parents differ")
    keep = info(c["survivor"])
    need(owned(keep) and keep["JobState"] == "RUNNING" and keep["NumNodes"] == "40"
         and scheduled_count(keep["AllocTRES"], "gres/dcu") == 320 and keep["NumCPUs"] == "2560"
         and keep["Partition"] == c["partition"] and keep["QOS"] == c["qos"] and keep["Account"] == c["account"],
         "Original surviving assignment changed")
    nodes = query(["scontrol", "show", "hostnames", keep["NodeList"]])
    need(len(set(nodes)) == len(nodes) == 40 and all(re.fullmatch(r"[A-Za-z0-9-]+", n) for n in nodes),
         "Original surviving node identities differ")
    archive = query(["sacct", "-n", "-P", "-X", "-u", node_name, "-S", "2026-11-22T16:00:00",
                     "-o", "JobIDRaw,JobName%180,State%24,End,AllocNodes,AllocTRES%120"])
    terminal = [r for line in archive if len(r := line.split("|")) == 6 and r[0] == c["failed_parent"]]
    if not terminal:
        raw_history = query(["sacct", "-n", "-P", "-X", "-j", c["failed_parent"],
                             "-o", "JobIDRaw,JobName%180,State%24,End,AllocNodes,AllocTRES%120"])
        terminal = [r for line in raw_history if len(r := line.split("|")) == 6 and r[0] == c["failed_parent"]]
    need(len(terminal) == 1 and terminal[0][2] == "NODE_FAIL" and terminal[0][3] not in ("", "Unknown")
         and terminal[0][4] == "40" and scheduled_count(terminal[0][5], "gres/dcu") == 320,
         "Original forty-node allocation has no authoritative NODE_FAIL terminal")
    need(not any(len(r := line.split("|")) == 6 and r[1] == name for line in archive),
         "Replacement label is present in accounting")
    live = query(["squeue", "-u", node_name, "-h", "-o", "%i|%j|%T"])
    total = 0
    keep_count = 0
    for line in live:
        ident, label, _ = line.split("|")
        need(re.fullmatch(r"[0-9]+", ident) and label != name, "Unexpected live job identity or duplicate replacement")
        row = info(ident)
        need(owned(row), "Queued job does not belong to authenticated owner")
        total += scheduled_count(row["ReqTRES"], "gres/dcu")
        keep_count += ident == c["survivor"]
    need(keep_count == 1 and total == 320, "Other live or pending GPU allocations change the 640-card cap")
    return terminal[0], nodes, total


def submit(scope):
    claim = json.loads((scope / "CLAIM.json").read_bytes())
    need(re.fullmatch(r"[a-f0-9]{32}", claim["request"]) and scope.name == claim["label"],
         "Invalid replacement request or namespace")
    need(claim["max_sbatch_invocations"] == 1 and claim["wall_limit"] == "30-00:00:00"
         and claim["parent_within_current_project"] is True, "Unrecognized replacement claim")
    owner = pwd.getpwuid(os.getuid()).pw_name
    terminal, nodes, current = validate(claim, owner)
    epoch = time.time()
    intent = {"request": claim["request"], "label": claim["label"], "epoch": epoch, "deadline": epoch+140,
              "failed_parent_accounting": terminal, "preserved_parent": claim["survivor"],
              "survivor_node_count": len(nodes), "requested_additional_nodes": 40,
              "current_DCU_requested_by_live_jobs": current, "cap_after_submission_DCU": current+320}
    intent_sha = record(scope / "intent.json", intent)
    argv = ["sbatch", "--parsable", "--job-name="+claim["label"], "--partition="+claim["partition"],
            "--account="+claim["account"], "--qos="+claim["qos"], "--nodes=40", "--ntasks=320",
            "--ntasks-per-node=8", "--cpus-per-task=8", "--mem=240000M", "--gres=dcu:8", "--exclusive",
            "--time="+claim["wall_limit"], "--exclude="+",".join(nodes),
            "--output="+str(scope/"batch-%j.log"), "--wrap", "exec /usr/bin/sleep infinity"]
    try:
        result = subprocess.run(argv, capture_output=True, timeout=min(40, max(1,int(intent["deadline"]-time.time()))))
    except (subprocess.TimeoutExpired, OSError) as fault:
        doc = {"request":claim["request"], "intent_SHA256":intent_sha, "remote_outcome":"UNKNOWN",
               "failure_type":type(fault).__name__, "replay_allowed":False}
        record(scope / "submission-return.json", doc)
        print(json.dumps(doc))
        return
    text = result.stdout.decode("utf-8", "replace").strip()
    match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?", text)
    job = match[1] if result.returncode == 0 and match else None
    doc = {"request":claim["request"], "intent_SHA256":intent_sha, "sbatch_rc":result.returncode, "job_id":job,
           "raw_stdout_SHA256":sha256(result.stdout).hexdigest(), "raw_stderr_SHA256":sha256(result.stderr).hexdigest(),
           "remote_outcome": "JOB_ID_RETURNED" if job else ("SUBMISSION_REJECTED" if result.returncode else "UNKNOWN"),
           "replay_allowed":False}
    if job is None:
        error = result.stderr.decode("utf-8", "replace").strip().lower()
        doc["error_category"] = next((tag for tag, phrase in [
            ("QOS_OR_ASSOCIATION", "qos"), ("INVALID_RESOURCE", "invalid"),
            ("UNAVAILABLE_CONFIGURATION", "node configuration is not available"),
            ("ALLOCATION_LIMIT", "limit"), ("ACCESS_DENIED", "permission denied")
        ] if phrase in error), None)
    record(scope / "submission-return.json", doc)
    print(json.dumps(doc))


if __name__ == "__main__":
    submit(Path(__file__).resolve().parent)
