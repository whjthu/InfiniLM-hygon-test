"""Run three zero-GPU HOST snapshot transport gates in a unique bounded namespace."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from snapshot_io import separate_logs, read_host_snapshot
from slurm_accounting import wait_terminal
from verify_package import verify

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "training-qwen-treasury-snapshot-io-v1"
RUN = ROOT / "results" / LABEL
SRC = RUN / "source"
ARMS = {"base": ("1087656", "a01r4n14"), "old_domain": ("1087656", "a01r4n16"),
        "domain": ("1103606", "f13r1n10")}
COLS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def require(ok, why):
    if not ok:
        raise ValueError(why)


def save(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024**2, "unsafe small artifact")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def budgets(deadline, now):
    require(math.isfinite(deadline) and math.isfinite(now), "invalid deadline")
    slurm = 60 * (int(min(120, deadline - now - 45)) // 60)
    require(slurm >= 60, "original CPU epoch exhausted")
    work = min(60, slurm - 15)
    return dict(Slurm_seconds=slurm, inside_seconds=work, client_seconds=min(slurm + 5, deadline-now-35))


def main(publication, claim_sha, request):
    epoch, mono = time.time(), time.monotonic()
    deadline, mono_end = epoch + 600, mono + 600
    os.umask(0o077)
    require(re.fullmatch(r"[0-9a-f]{40}", publication) and re.fullmatch(r"[0-9a-f]{64}", claim_sha)
            and re.fullmatch(r"[0-9a-f]{32}", request), "invalid unique identity")
    require(Path(__file__).resolve().parent == SRC and not RUN.is_symlink(), "unexpected source namespace")
    require(sha(SRC / "CLAIM.json") == claim_sha, "claim differs")
    verify(SRC)
    require(not any((RUN / n).exists() for n in ("intent.json", "arms", "outcome.json")), "original namespace exists; no replay")
    def query(*args, own_end=None):
        left = min(deadline-time.time(), mono_end-time.monotonic(), 9)
        if own_end is not None:
            left = min(left, own_end-time.monotonic())
        require(left > 0, "own observation deadline exhausted")
        return subprocess.check_output(args, text=True, timeout=left).strip().splitlines()
    def accounting(parent, label, own_end=None):
        rows = [row for line in query("sacct", "-n", "-P", "-j", parent, "-o", COLS, own_end=own_end)
                if len(row := line.split("|")) == 7 and row[1] == label]
        require(len(rows) <= 1, "duplicate exact label")
        return rows[0] if rows else None
    for arm, (parent, node) in ARMS.items():
        require(accounting(parent, LABEL+"-"+arm) is None, "original CPU step exists")
        lines = query("squeue", "-h", "-j", parent, "-o", "%T|%D|%N")
        require(len(lines) == 1, "parent unavailable")
        state, count, nodes = lines[0].split("|")
        require((state, count) == ("RUNNING", "40") and node in query("scontrol", "show", "hostnames", nodes), "parent/node differs")
    save(RUN/"intent.json", dict(request=request, publication=publication, claim_SHA256=claim_sha, epoch=epoch,
         deadline=deadline, arms=ARMS, actual_expected_GPU_cards=0, GPU_GO=False, maximum_CPU_steps=3))
    (RUN/"arms").mkdir()
    results = {}
    # Three serial CPU steps reduce scheduler request pressure; no GPU follow-on branch exists.
    for arm, (parent, node) in ARMS.items():
        folder = RUN/"arms"/arm
        folder.mkdir()
        label = LABEL+"-"+arm
        rc, error = None, None
        try:
            limit = budgets(deadline, time.time())
            require(accounting(parent,label) is None, "original CPU label appeared")
            command = ["srun", "--jobid="+parent, "--job-name="+label, "--exact", "--immediate=5", "-N1", "-n1", "-w", node,
                       "-c2", "--mem=4096M", "--gres=none", "--time=00:%02d:00" % (limit["Slurm_seconds"]//60),
                       "/usr/bin/timeout", "--kill-after=5s", str(limit["inside_seconds"])+"s",
                       "/usr/local/python3.12/bin/python3", "-B", "-u", str(SRC/"probe.py"), parent, node]
            require(limit["Slurm_seconds"] + 45 <= min(deadline-time.time(),mono_end-time.monotonic()), "preparation consumed reserve")
            save(folder/"step-intent.json",dict(epoch=epoch,deadline=deadline,limits=limit,command=command))
            with separate_logs(folder/"snapshot.stdout",folder/"snapshot.stderr") as (stdout,stderr):
                child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
                owned = dict(PID=child.pid)
                try:
                    fields=Path("/proc",str(child.pid),"stat").read_text().rsplit(") ",1)[1].split()
                    owned.update(PGID=int(fields[2]),startticks=int(fields[19]))
                except FileNotFoundError:
                    owned["exited_before_identity_observation"]=True
                save(folder/"owned-srun.json",owned)
                try:
                    rc=child.wait(timeout=min(limit["client_seconds"],deadline-time.time()-35,mono_end-time.monotonic()-35))
                except subprocess.TimeoutExpired:
                    pass
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            error=type(exc).__name__
        own_end=min(mono_end-5,time.monotonic()+30, time.monotonic()+max(0,deadline-time.time()-5))
        errors=[]
        def observe():
            try:
                return accounting(parent,label,own_end)
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                errors.append(type(exc).__name__)
                return None
        visible=wait_terminal(observe,own_end,interval=2,max_observations=14)
        row=visible["row"]
        value=None
        if rc==0 and row and row[2:4]==["COMPLETED","0:0"] and row[5]==node and "gres/dcu=" not in row[6]:
            try:
                value=read_host_snapshot(folder/"snapshot.stdout",node,row[0])
                require(b"snapshot-io-gate: retained independent stderr marker" in (folder/"snapshot.stderr").read_bytes(), "independent stderr marker missing")
            except (OSError,ValueError,KeyError,TypeError) as exc:
                error=type(exc).__name__
        passed=value is not None and error is None
        item=dict(arm=arm,passed=passed,rc=rc,error=error,accounting_columns=COLS,accounting=visible,accounting_errors=errors,
             stdout_SHA256=sha(folder/"snapshot.stdout") if (folder/"snapshot.stdout").is_file() else None,
             stderr_SHA256=sha(folder/"snapshot.stderr") if (folder/"snapshot.stderr").is_file() else None,
             stderr_bytes=(folder/"snapshot.stderr").stat().st_size if (folder/"snapshot.stderr").is_file() else None,
             snapshot=value,actual_GPU_cards=0 if row and "gres/dcu=" not in row[6] else None,replay_allowed=False)
        save(folder/"outcome.json",item)
        results[arm]=item
    passed=len(results)==3 and all(v["passed"] for v in results.values())
    save(RUN/"outcome.json",dict(request=request,epoch=epoch,deadline=deadline,arms=results,passed=passed,
         GPU_GO=False,actual_GPU_steps=0,automatic_GPU_followon=False,previous_v2_324missing_preserved=True))
    print(json.dumps(dict(request=request,passed=passed,GPU_GO=False,actual_GPU_steps=0)))
    return 0 if passed else 1


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publication",required=True)
    p.add_argument("--claim-sha256",required=True)
    p.add_argument("--request-id",required=True)
    a=p.parse_args()
    raise SystemExit(main(a.publication,a.claim_sha256,a.request_id))
