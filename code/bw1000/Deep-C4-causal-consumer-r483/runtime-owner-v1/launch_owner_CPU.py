"""Submit one active-load provenance check on the same published I03 allocation."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import socket
import subprocess
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
PACKAGE = Path(__file__).resolve().parent
LABEL = "Deep-C4-causal-consumer-r483-runtime-owner-CPU-v1"
PREVIOUS_PACKAGE = ROOT / "incoming/Deep-C4-causal-consumer-r483/CPU-v2"
PARENT, NODE = "1103606", "f13r2n01"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def verify(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("package identity differs: " + path.name)
    return raw


def main(publication, claim_sha):
    if not re.fullmatch("[0-9a-f]{40}", publication) or not re.fullmatch("[0-9a-f]{64}", claim_sha):
        raise ValueError("published launch identity missing")
    claim = json.loads(verify(PACKAGE / "CLAIM.json", claim_sha))
    if (claim["parent"], claim["node"], claim["GPU"], claim["attempts"]) != (PARENT, NODE, 0, 1):
        raise ValueError("unexpected resource scope")
    for name, digest in claim["files"].items():
        verify(PACKAGE / name, digest)
    verify(PREVIOUS_PACKAGE / "CPU_outer.sh", claim["parent_outer_SHA256"])
    verify(PREVIOUS_PACKAGE / "launch_CPU.py", claim["parent_launcher_SHA256"])
    launcher = runpy.run_path(str(PREVIOUS_PACKAGE / "launch_CPU.py"), run_name="prior_readonly_resource")
    launcher["verify_membership"]()
    raw = subprocess.check_output(["sacct", "-n", "-P", "-j", PARENT, "-o", COLUMNS], timeout=15, text=True)
    rows = [p for line in raw.splitlines() if len(p := line.split("|")) == 7]
    prior = [p for p in rows if p[0] == "1103606.156"]
    if len(prior) != 1 or prior[0][2:4] != ["COMPLETED", "0:0"] or any(p[1] == LABEL for p in rows):
        raise ValueError("prior build is not terminal or this original scope already exists")
    epoch = time.time()
    deadline = epoch + claim["seconds"]
    intent = ROOT / "results" / (LABEL + "-intent.json")
    receipt = ROOT / "results" / (LABEL + "-launch.json")
    log = ROOT / "logs" / (LABEL + "-submit.log")
    if any(p.exists() or p.is_symlink() for p in (intent, receipt, log)):
        raise ValueError("reconcile original namespace instead of replay")
    command = ["srun", "--jobid=" + PARENT, "--exact", "--immediate=5", "--cpu-bind=none", "-N1", "-n1",
               "-c4", "--mem=8192M", "--gres=none", "-w", NODE, "-t", "00:03:00", "--job-name=" + LABEL,
               "--output=" + str(ROOT / "logs" / (LABEL + "-%J.log")), "env",
               "RUNTIME_OWNER_DEADLINE=" + repr(deadline),
               "WAVE_INSIDE_R2323=Deep-C4-causal-consumer-r483/runtime-owner-v1/CPU_inside.sh",
               "bash", str(PREVIOUS_PACKAGE / "CPU_outer.sh"), "probe"]
    original = dict(command=command, publication=publication, claim_SHA256=claim_sha,
                    host=socket.gethostname(), epoch=epoch, deadline=deadline)
    with intent.open("x") as stream:
        json.dump(original, stream)
    with log.open("xb") as stream:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    fields = (Path("/proc") / str(process.pid) / "stat").read_text().rsplit(") ", 1)[1].split()
    evidence = dict(original, owned_srun_PID=process.pid, PGID=int(fields[2]), startticks=int(fields[19]))
    with receipt.open("x") as stream:
        json.dump(evidence, stream)
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", required=True)
    parser.add_argument("--claim-sha", required=True)
    args = parser.parse_args()
    main(args.publication, args.claim_sha)
