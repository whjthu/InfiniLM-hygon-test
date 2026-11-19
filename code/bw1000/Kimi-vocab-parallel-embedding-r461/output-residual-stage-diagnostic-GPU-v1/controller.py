"""Launch exactly one prepared residual comparison and retain its original terminal evidence."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


DIR = Path(__file__).resolve().parent
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
INCOMING = "Kimi-vocab-parallel-embedding-r461/output-residual-stage-diagnostic-GPU-v1"
ACCOUNT = "JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
            "BOOT_FAIL", "DEADLINE", "REVOKED"}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query(*args, timeout=8):
    return subprocess.check_output(args, text=True, timeout=timeout).splitlines()


def fields(line):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", line))


def verify_allocation(claim):
    parent, node = claim["parent"], claim["node"]
    row, = query("scontrol", "show", "job", parent, "--oneliner")
    job = fields(row)
    owner = re.fullmatch(r"[^()]+\((\d+)\)", job.get("UserId", ""))
    need(job.get("JobId") == parent and job.get("JobState") == "RUNNING" and owner
         and int(owner[1]) == os.getuid() and job.get("NumNodes") == "40"
         and "gres/dcu=320" in job.get("AllocTRES", "").split(","), "wrong Slurm owner or allocation")
    expand = lambda expression: set(query("scontrol", "show", "hostnames", expression))
    nodes = expand(job["NodeList"])
    need(len(nodes) == 40 and node in nodes and node in claim["I01_pool"], "wrong I01 membership")
    steps = query("scontrol", "show", "step", parent, "--oneliner")
    need(steps, "active step inventory unavailable")
    for line in steps:
        step = fields(line)
        handle = step.get("StepId", "")
        if handle in (parent + ".batch", parent + ".extern"):
            continue
        need(re.fullmatch(re.escape(parent)+r"\.\d+", handle) and step.get("NodeList"), "ambiguous active step")
        members = expand(step["NodeList"])
        need(members <= nodes and len(members) == int(step["Nodes"]) and node not in members, "assigned node active")


def accounting(claim, timeout=8):
    parent, label = claim["parent"], claim["label"]
    records = [row.split("|") for row in query("sacct", "-nP", "-j", parent, "-S", "2026-11-13",
                                              "-o", ACCOUNT, timeout=timeout) if row.strip()]
    need(all(len(row) == 7 for row in records) and any(row[0] == parent for row in records), "wrong accounting schema")
    return [row for row in records if row[1] in (label, label+"-pre", label+"-post")]


def save(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def main(publication, manifest):
    need(DIR == ROOT / "incoming" / INCOMING and re.fullmatch("[a-f0-9]{40}", publication), "unexpected location or source commit")
    mf = DIR / "SOURCE_FILES.json"
    need(sha(mf) == manifest, "published manifest differs")
    for name, d in json.loads(mf.read_bytes())["files"].items():
        need(Path(name).name == name and sha(DIR/name) == d, "published member differs")
    claim = json.loads((DIR / "CLAIM.json").read_bytes())
    label, parent, node = (claim[k] for k in ("label", "parent", "node"))
    need((parent, node, claim["GPU_cards"], claim["max_launches"]) == ("1103606","f13r1n01",1,1),
         "wrong residual experiment scope")
    library = ROOT / Path(claim["component"]["path"]).relative_to("/refactor")
    need(sha(library) == claim["component"]["SHA256"], "CPU331 artifact differs")
    out = ROOT / "results" / label
    need(not out.exists() and not out.is_symlink() and not accounting(claim), "same scope already exists")
    verify_allocation(claim)
    out.mkdir(mode=0o700)
    epoch = time.time()
    deadline = epoch + claim["whole_seconds"]
    remaining = lambda: deadline - time.time()
    save(out/"intent.json", dict(publication=publication, source_manifest_SHA256=manifest, epoch=epoch,
                                deadline=deadline, GPU_cards=1, parent=parent, node=node))
    env = {k:v for k,v in os.environ.items() if not k.startswith(("SLURM_", "CUDA_", "HIP_", "ROCR_",
            "GPU_", "INFINILM_", "INFINICORE_", "BASH_ENV", "PYTHONPATH", "LD_PRELOAD", "LD_AUDIT"))}
    env.update(KIMI_RESIDUAL_EPOCH=repr(epoch), WAVE_INSIDE_R2323=INCOMING+"/inside.sh")
    base = ["srun","--jobid="+parent,"--exact","--immediate=8","--kill-on-bad-exit=1",
            "-N1","-n1","-w",node]
    result = dict(GPU_launches=0, GPU_result=None, post=None, accounting=None, successful=False)

    def probe(suffix, owners):
        args = [*base,"-c1","--mem=512M","--gres=none","-t","00:00:25","--job-name="+label+"-"+suffix,
                "/usr/local/python3.12/bin/python3","-I","-S","-B",str(DIR/"probe.py"),json.dumps(owners)]
        receipt = json.loads(subprocess.check_output(args, env=env, timeout=min(36,remaining()-2)))
        need(receipt["node"] == node and receipt["actual_CPU"].startswith(parent+"."), "metadata on wrong node")
        return receipt

    def consume_terminal():
        end = min(deadline-40, time.time()+25)
        while time.time() < end:
            matching = [row for row in accounting(claim, timeout=min(7,end-time.time())) if row[1] == label]
            need(len(matching) <= 1, "ambiguous original GPU step")
            if matching:
                row = matching[0]
                result["last_GPU_accounting"] = row
                need(re.fullmatch(re.escape(parent)+r"\.\d+", row[0]) and row[5] == node
                     and "gres/dcu=1" in row[6].split(","), "wrong actual GPU handle")
                if row[2].split()[0].rstrip("+") in TERMINAL:
                    return row
            time.sleep(.65)
        return None

    try:
        pre = probe("pre", [])
        save(out/"pre.json", pre)
        need(remaining() > 218, "expired before GPU launch")
        argv = [*base, "-c4","--mem=8192M","--gres=dcu:1","-t","00:03:00","--job-name="+label,
                "--output="+str(ROOT/"logs"/(label+"-%J.log")), "/bin/bash", str(DIR/"outer.sh"),"probe"]
        with (out/"srun.log").open("xb") as log:
            child = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            result["GPU_launches"] = 1
            save(out/"srun-owner.json", dict(PID=child.pid, argv=argv, epoch=epoch))
            try:
                result["srun_rc"] = child.wait(timeout=min(190, remaining()-48))
            except subprocess.TimeoutExpired:
                result["srun_timeout"] = True
        row = consume_terminal()
        result["accounting"] = row
        if row is not None:
            path = out/"worker-result.json"
            owned = []
            if path.is_file():
                worker = json.loads(path.read_bytes())
                need(worker["step"] == row[0] and worker["boot_id"] == pre["boot_id"], "worker identity differs")
                result["GPU_result"] = dict(bytes=path.stat().st_size, SHA256=sha(path))
                owned += [{k: r[k] for k in ("PID","startticks") if k in r}
                          for r in worker["records"] if "PID" in r]
            post = probe("post", owned)
            need((pre["PCI"],pre["boot_id"]) == (post["PCI"],post["boot_id"]), "pre/post device or boot changed")
            result["post"] = post
            result["successful"] = (result["GPU_result"] is not None and worker["complete"] is True
                                    and row[2:4] == ["COMPLETED","0:0"] and result.get("srun_rc") == 0)
    except Exception as error:
        result["failure"] = dict(kind=type(error).__name__, reason=str(error)[:180])
    finally:
        save(out/"controller-result.json", result)
    print(json.dumps(result, allow_nan=False))
    return 0 if result["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
