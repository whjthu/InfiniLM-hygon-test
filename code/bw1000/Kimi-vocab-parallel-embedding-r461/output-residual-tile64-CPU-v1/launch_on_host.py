"""Submit the claimed one-TU build from the Hygon login host."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from CPU_build import LABEL, ROOT, pins_for


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
INCOMING = "Kimi-vocab-parallel-embedding-r461/output-residual-tile64-CPU-v1"
ACCOUNT_FORMAT = "JobIDRaw,JobName%100,State%35,ExitCode,ElapsedRaw,NodeList,AllocTRES%160"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def query(*args, timeout=8):
    return subprocess.check_output(args, text=True, timeout=timeout).splitlines()


def fields(line):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", line))


def expand(expression):
    need(expression not in ("", "(null)", "None"), "missing Slurm node expression")
    rows = query("scontrol", "show", "hostnames", expression)
    need(rows and len(rows) == len(set(rows)), "invalid Slurm node list")
    return set(rows)


def check_resources(claim):
    parent, node = claim["parent"], claim["node"]
    row, = query("scontrol", "show", "job", parent, "--oneliner")
    job = fields(row)
    owner = re.fullmatch(r"[^()]+\((\d+)\)", job.get("UserId", ""))
    need(job.get("JobId") == parent and job.get("JobState") == "RUNNING"
         and job.get("NumNodes") == "40" and owner and int(owner[1]) == os.getuid()
         and "gres/dcu=320" in job.get("AllocTRES", "").split(","),
         "current parent ownership, state or TRES differs")
    hosts = expand(job.get("NodeList", ""))
    need(len(hosts) == 40 and set(claim["I01_pool"]) <= hosts and node in claim["I01_pool"],
         "current I01 membership differs")
    inventory = query("scontrol", "show", "step", parent, "--oneliner")
    need(inventory, "active-step inventory is unavailable")
    numbered = set()
    for line in inventory:
        step = fields(line)
        handle = step.get("StepId", "")
        if handle in (parent + ".batch", parent + ".extern"):
            continue
        need(re.fullmatch(re.escape(parent) + r"\.\d+", handle) and handle not in numbered
             and step.get("Nodes", "").isdigit(), "ambiguous numbered Slurm step")
        numbered.add(handle)
        members = expand(step.get("NodeList", ""))
        need(len(members) == int(step["Nodes"]) and members <= hosts, "inconsistent step membership")
        need(node not in members, "I01 build node has an active step")


def accounting(claim, timeout=8):
    lines = query("sacct", "-nP", "-j", claim["parent"], "-S", claim["accounting_start"],
                  "-o", ACCOUNT_FORMAT, timeout=timeout)
    rows = [line.split("|") for line in lines if line.strip()]
    need(all(len(row) == 7 for row in rows), "unexpected Slurm accounting schema")
    need(any(row[0] == claim["parent"] for row in rows), "parent missing from Slurm accounting")
    return [row for row in rows if row[1] == LABEL]


def exclusive_json(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def main(publication, manifest):
    need(re.fullmatch(r"[0-9a-f]{40}", publication) and re.fullmatch(r"[0-9a-f]{64}", manifest),
         "missing publication or source manifest")
    need(ROOT == PROJECT / "incoming" / INCOMING, "package not at its claimed installation path")
    claim = pins_for(ROOT, manifest)
    need((claim["parent"], claim["node"], claim["cpu_cores"], claim["mem_MiB"]) ==
         ("1103606", "f13r1n01", 4, 8192), "dispatcher resource claim differs")
    result_dir = PROJECT / "results"
    intent = result_dir / (LABEL + "-intent.json")
    launch = result_dir / (LABEL + "-launch.json")
    log = PROJECT / "logs" / (LABEL + "-srun.log")
    output = PROJECT / "candidates" / LABEL
    need(not any(p.exists() or p.is_symlink() for p in (intent, launch, log, output)),
         "original namespace already exists")
    need(not accounting(claim), "original candidate already has Slurm accounting")
    check_resources(claim)
    epoch = time.time()
    deadline = epoch + 300
    args = [
        "srun", "--jobid=" + claim["parent"], "--exact", "--immediate=5", "-N1", "-n1",
        "-c4", "--mem=8192M", "--gres=none", "--nodelist=" + claim["node"],
        "--time=00:05:00", "--job-name=" + LABEL,
        "--output=" + str(PROJECT / "logs" / (LABEL + "-%J.log")),
        "env", "KIMI_TILE_EPOCH=" + repr(epoch), "KIMI_TILE_INNER_DEADLINE=" + repr(epoch + 260),
        "KIMI_TILE_MANIFEST_SHA256=" + manifest, "WAVE_INSIDE_R2323=" + INCOMING + "/CPU_inside.sh",
        "/bin/bash", str(ROOT / "CPU_outer.sh"), "probe",
    ]
    context = dict(publication=publication, source_manifest_SHA256=manifest, epoch=epoch, deadline=deadline,
                   login_host=socket.gethostname(), parent=claim["parent"], node=claim["node"],
                   gpu=0, argv=args)
    exclusive_json(intent, context)
    result = dict(context, dispatch_outcome="UNKNOWN", srun_returncode=None,
                  accounting_columns=ACCOUNT_FORMAT.split(","), accounting_rows=None)
    try:
        with log.open("xb") as stream:
            child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            result["owned_srun_PID"] = child.pid
            try:
                stat = Path("/proc", str(child.pid), "stat").read_text().rsplit(") ", 1)[1].split()
                result.update(owned_srun_PGID=int(stat[2]), owned_srun_startticks=int(stat[19]))
            except FileNotFoundError:
                pass
            try:
                result["srun_returncode"] = child.wait(timeout=max(1, deadline - time.time() - 15))
                result["dispatch_outcome"] = "SRUN_RETURNED"
            except subprocess.TimeoutExpired:
                # The kernel wait returned no terminal result. Preserve the original step and intent.
                result["dispatch_outcome"] = "UNKNOWN"
        remaining = deadline - time.time()
        if remaining > 1:
            result["accounting_rows"] = accounting(claim, timeout=min(8, remaining - 0.5))
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        result["observer_error"] = type(error).__name__
    finally:
        exclusive_json(launch, result)
    print(json.dumps({k: result[k] for k in ("srun_returncode", "dispatch_outcome", "accounting_rows")}))
    return 0 if result["srun_returncode"] == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", required=True)
    parser.add_argument("--manifest", required=True)
    options = parser.parse_args()
    raise SystemExit(main(options.publication, options.manifest))
