"""Launch three independent single-card arm controllers under one original epoch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from scope import PACKAGE, SRC, CPU, ARMS, LABEL, COLUMNS, require, read, pin, seconds_left, verify_prerequisites
from verify_package import verify


def save(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(publication, claim_sha, request):
    epoch, monotonic_start = time.time(), time.monotonic()
    deadline, monotonic_end = epoch + 1800, monotonic_start + 1800
    os.umask(0o077)
    require(re.fullmatch(r"[0-9a-f]{40}", publication) and re.fullmatch(r"[0-9a-f]{64}", claim_sha)
            and re.fullmatch(r"[0-9a-f]{32}", request), "invalid original launch identity")
    require(Path(__file__).resolve().parent == SRC and not PACKAGE.is_symlink(), "wrong installed scope")
    require(pin(SRC / "CLAIM.json", 1024**2)["SHA256"] == claim_sha, "published claim changed")
    verify(SRC)
    require(not any((PACKAGE / name).exists() for name in ("intent.json", "arms", "controllers.json", "outcome.json")),
            "original namespace requires reconciliation; no replay")
    def query(*args):
        return subprocess.check_output(args, text=True, timeout=min(12, seconds_left(deadline, time.time(), 12),
            monotonic_end-time.monotonic())).strip().splitlines()
    def accounting(parent, name):
        rows = [row for line in query("sacct", "-n", "-P", "-j", parent, "-o", COLUMNS)
                if len(row := line.split("|")) == 7 and row[1] == name]
        require(len(rows) <= 1, "duplicate original label")
        return rows[0] if rows else None
    verify_prerequisites(accounting)
    for arm, spec in ARMS.items():
        job_lines = query("scontrol", "show", "job", spec["parent"], "--oneliner")
        require(len(job_lines) == 1, "parent allocation unavailable")
        job = dict(x.split("=",1) for x in job_lines[0].split() if "=" in x)
        require(job["JobState"] == "RUNNING" and job["NumNodes"] == "40"
                and "gres/dcu=320" in job["AllocTRES"].split(",")
                and spec["node"] in query("scontrol", "show", "hostnames", job["NodeList"]), "assigned parent/node changed")
        for phase in ("pre", "gpu", "post"):
            require(accounting(spec["parent"], LABEL + "-" + arm + "-" + phase) is None, "original numbered step exists")
    binding = read(SRC / "INPUT_BINDING.json")
    require(pin(CPU / "dev/visual.jsonl", 100000)["SHA256"] == binding["metadata_files"]["visual.jsonl"]["SHA256"],
            "frozen revised dev input changed")
    inputs = [json.loads(line) for line in (CPU / "dev/visual.jsonl").read_bytes().splitlines()]
    ids = [r["id"] for r in inputs]
    require(len(ids) == len(set(ids)) == 108, "dev108 denominator changed")
    common = dict(epoch=epoch, deadline=deadline, request=request, publication=publication, claim_SHA256=claim_sha,
                  planned_rows_per_arm=108, per_topic_denominator=36, all_arm_denominator=324, max_new_tokens=64,
                  max_GPU_invocations=3, maximum_GPU_card_seconds=2880, planned_ids=ids,
                  no_distributed_training_or_collective=True)
    save(PACKAGE / "intent.json", common)
    (PACKAGE / "arms").mkdir()
    for arm, spec in ARMS.items():
        folder = PACKAGE / "arms" / arm
        folder.mkdir()
        save(folder / "intent.json", dict(common, arm=arm, parent=spec["parent"], node=spec["node"]))
    children, handles, streams = {}, {}, []
    try:
        for arm in ARMS:
            folder = PACKAGE / "arms" / arm
            stream = (folder / "host-controller.log").open("xb"); streams.append(stream)
            child = subprocess.Popen(["/usr/local/python3.12/bin/python3", "-B", "-u", str(SRC / "arm_controller.py"), arm, "host"],
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children[arm] = child
            identity = dict(PID=child.pid)
            try:
                fields = Path("/proc", str(child.pid), "stat").read_text().rsplit(") ",1)[1].split()
                identity.update(PGID=int(fields[2]), startticks=int(fields[19]))
            except FileNotFoundError:
                identity["exited_before_identity_observation"] = True
            handles[arm] = identity
            save(folder / "host-controller-handle.json", identity)
        save(PACKAGE / "controllers.json", handles)
        outcomes = {}
        for arm, child in children.items():
            try:
                rc = child.wait(timeout=min(seconds_left(deadline,time.time(),1800), monotonic_end-time.monotonic()))
            except (ValueError, subprocess.TimeoutExpired):
                rc = None
            folder = PACKAGE / "arms" / arm
            receipt = folder / "controller-outcome.json"
            outcomes[arm] = dict(rc=rc, original_outcome_present=receipt.is_file(),
                                 passed=read(receipt)["passed"] if receipt.is_file() else False,
                                 planned_rows=108, replay_allowed=False)
        result = dict(request=request, original_epoch=epoch, deadline=deadline, arms=outcomes,
                      all_arm_denominator=324, passed=len(outcomes)==3 and all(x["rc"]==0 and x["passed"] for x in outcomes.values()),
                      formal_quality=False, formal_performance=False, effective_busy_qualified=False,
                      equal_budget_causal_victory=False)
        save(PACKAGE / "outcome.json", result)
        print(json.dumps(result, allow_nan=False))
        return 0 if result["passed"] else 1
    finally:
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication",required=True)
    parser.add_argument("--claim-sha256",required=True)
    parser.add_argument("--request-id",required=True)
    args=parser.parse_args()
    raise SystemExit(main(args.publication,args.claim_sha256,args.request_id))
