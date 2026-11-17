"""Start exactly one published explicit-input Treasury continuous controller."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from verify_package import verify

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RUN = ROOT / "results/training-qwen-treasury-explicit-sft-v1"
LABEL = "training-qwen-treasury-explicit-continuous-hip-v1"
HOST_PYTHON = "/usr/local/python3.12/bin/python3"
PARENT, NODE = "1087656", "a01r4n16"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def save(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def budget(epoch, monotonic_start, now, monotonic_now, reserve=0):
    available = min(epoch + 1620 - now, monotonic_start + 1620 - monotonic_now) - reserve
    require(all(math.isfinite(x) for x in (epoch, monotonic_start, now, monotonic_now, available))
            and epoch <= now and monotonic_start <= monotonic_now and available > 0,
            "original training epoch exhausted or invalid")
    return available


def main(publication, claim_sha256, request):
    epoch, started = time.time(), time.monotonic()
    os.umask(0o077)
    require(re.fullmatch(r"[0-9a-f]{40}", publication) is not None
            and re.fullmatch(r"[0-9a-f]{64}", claim_sha256) is not None
            and re.fullmatch(r"[0-9a-f]{32}", request) is not None, "invalid published identity")
    source = Path(__file__).resolve().parent
    require(source == RUN / "src" and not RUN.is_symlink(), "unexpected installed namespace")
    require(hashlib.sha256((source / "CLAIM.json").read_bytes()).hexdigest() == claim_sha256,
            "published claim differs")
    verify(source)
    require(not any((RUN / name).exists() for name in (
        "intent.json", "launch-receipt.json", "launch-outcome.json", "controller.log",
        "controller-outcome.json", "framework", "continuous")), "original namespace requires reconciliation")
    output = subprocess.check_output(["sacct", "-n", "-P", "-j", PARENT, "-o", COLUMNS],
                                     text=True, timeout=min(14, budget(epoch, started, time.time(), time.monotonic())))
    rows = [line.split("|") for line in output.splitlines() if line]
    require(all(len(row) == 7 for row in rows), "accounting schema changed")
    require(not any(row[1] in {LABEL + "-" + suffix for suffix in ("pre", "gpu", "post")} for row in rows),
            "original numbered step requires reconciliation")
    for number, suffix in ((284, "pre"), (285, "gpu"), (286, "post")):
        old = [row for row in rows if row[1] == "training-qwen-treasury-continuous-hip-v3-" + suffix]
        require(len(old) == 1 and old[0][0] == PARENT + "." + str(number)
                and old[0][2:4] == ["COMPLETED", "0:0"] and old[0][5] == NODE,
                "original comparable continuous scope is not terminal")
    cfg = dict(epoch=epoch, deadline=epoch + 1620, node=NODE, parent=PARENT, arm="continuous",
               request_id=request, publication=publication, claim_SHA256=claim_sha256,
               maximum_GPU_invocations=1, planned_optimizer_updates=5)
    save(RUN / "intent.json", cfg)
    child = None
    rc, timeout = None, False
    with (RUN / "controller.log").open("xb") as stream:
        child = subprocess.Popen([HOST_PYTHON, "-B", "-u", str(source / "treasury_continuous.py"), "host"],
                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        receipt = dict(cfg, PID=child.pid)
        try:
            fields = Path("/proc", str(child.pid), "stat").read_text().rsplit(") ", 1)[1].split()
            receipt.update(PGID=int(fields[2]), startticks=int(fields[19]))
        except FileNotFoundError:
            receipt["exited_before_identity_observation"] = True
        save(RUN / "launch-receipt.json", receipt)
        try:
            rc = child.wait(timeout=budget(epoch, started, time.time(), time.monotonic()))
        except subprocess.TimeoutExpired:
            timeout = True
    result = dict(request_id=request, publication=publication, epoch=epoch, deadline=epoch + 1620,
                  rc=rc, client_timeout=timeout,
                  controller_outcome_present=(RUN / "controller-outcome.json").is_file(),
                  controller_handle_preserved=True, replay_allowed=False,
                  recovery_qualified=False, formal_T01_accepted=False, formal_performance=False)
    save(RUN / "launch-outcome.json", result)
    print(json.dumps(result, allow_nan=False))
    return 0 if rc == 0 and not timeout else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", required=True)
    parser.add_argument("--claim-sha256", required=True)
    parser.add_argument("--request-id", required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.publication, args.claim_sha256, args.request_id))
