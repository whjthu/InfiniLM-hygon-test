"""Run the published two-update tensor audit on one CPU allocation."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from slurm_accounting import wait_terminal

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
SRC = ROOT / "src"
ORIGINAL = PROJECT / "results/training-qwen-native-qproj-lt-first-two-v1"
PARENT = "1103606"
NODE = "f13r1n11"
LABEL = "training-qwen-qproj-first-two-capture-cpu-v1"
PYTHON = "/usr/local/python3.12/bin/python3"
COLUMNS = "JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def save(name, value):
    with (ROOT / name).open("x", encoding="utf8") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    intent = json.loads((ROOT / "intent.json").read_bytes())
    epoch, deadline = intent["epoch"], intent["deadline"]
    entry = time.monotonic() + deadline-time.time()

    def left(cap=0):
        return min(deadline-time.time(), entry-time.monotonic())-cap

    def query(*args):
        need(left(5)>0, "CPU reader deadline exhausted")
        return subprocess.check_output(args, timeout=min(10,left(5)), text=True).strip()

    def accounting():
        return [line.split("|")[:7] for line in query("sacct","-n","-P","-j",PARENT,"--format="+COLUMNS).splitlines()]

    def owned():
        rows = [row for row in accounting() if row[1]==LABEL]
        need(len(rows)<=1, "Duplicate CPU step")
        return rows[0] if rows else None

    original = json.loads((SRC / "TERMINAL_BINDING.json").read_bytes())
    need(digest(SRC / "SOURCE_FILES.json")==intent["source_manifest_SHA256"], "Published manifest differs")
    for name, pin in json.loads((SRC / "SOURCE_FILES.json").read_bytes())["files"].items():
        path=SRC/name
        need(path.stat().st_size==pin["bytes"] and digest(path)==pin["SHA256"], "Source member differs")
    for name, hash_value in original["original_files"].items():
        need(digest(ORIGINAL/name)==hash_value, "Original GPU receipt changed")
    rows=accounting()
    for expected in original["original_accounting"]:
        need(rows.count(expected)==1, "Original GPU accounting changed")
    need(not any(row[1]==LABEL for row in rows),"Existing CPU request")
    parent=query("squeue","-h","-j",PARENT,"-o","%T|%D|%N").split("|")
    need(parent[:2]==["RUNNING","40"] and NODE in query("scontrol","show","hostnames",parent[2]).splitlines(),
         "Frozen CPU node unavailable")
    for row in query("squeue","--steps","-h","-j",PARENT,"-o","%i|%N").splitlines():
        handle, hosts=row.split("|",1)
        if not handle.endswith((".batch",".extern")):
            need(hosts not in ("","(null)") and NODE not in query("scontrol","show","hostnames",hosts).splitlines(),
                 "CPU node has active or unknown step")
    evidence=ROOT/"evidence"
    evidence.mkdir(mode=0o700)
    framework=PROJECT/"results/training-qwen-continuous-branches21-hip-v2/framework"
    assets=PROJECT/"results/training-qwen-a68952c-swift-initial-CPU-v1/source"
    command=["/public/software/apps/apptainer/1.3.4/bin/apptainer","exec","--no-home","--pwd","/"]
    for source,dest,mode in [
        (ORIGINAL,"/original","ro"),(SRC,"/fix","ro"),(evidence,"/evidence","rw"),(framework,"/framework","ro"),
        (assets,"/stage","ro"),(Path("/opt/hyhal"),"/opt/hyhal","ro"),(PROJECT/"results/training-vlm-swift-offline-runtime-v1","/swift-runtime","ro"),
        (PROJECT/"results/training-minicpm-hygon-processor-r2/cpu/runtime-site","/minicpm-transformers","ro"),
    ]:
        command += ["--bind",f"{source}:{dest}:{mode}"]
    command += [str(PROJECT/"containers/dtk26-sandbox-r1"),"/bin/bash","--noprofile","--norc","-c",
                "set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
                "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 "
                "HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 "
                "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/framework:/minicpm-transformers:/swift-runtime/site "
                "REFERENCE_READER_EPOCH="+shlex.quote(str(epoch))+" REFERENCE_READER_DEADLINE="+shlex.quote(str(epoch+870))
                +"; exec python3 -B -u /fix/compute.py"]
    minutes=int(left(45)//60)
    need(14<=minutes<=16 and time.time()<epoch+45,"CPU execution budget unavailable")
    step=["srun","--jobid="+PARENT,"--job-name="+LABEL,"--exact","--nodes=1","--ntasks=1","--nodelist="+NODE,
          "--cpus-per-task=4","--mem=16384M","--gres=none","--immediate=5","--time="+str(minutes),*command]
    with (ROOT/"CPU.log").open("xb") as log:
        child=subprocess.Popen(step,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        proc=Path("/proc",str(child.pid),"stat").read_text().rsplit(") ",1)[1].split()
        save("owned-srun.json",dict(PID=child.pid,PGID=int(proc[2]),startticks=int(proc[19])))
        forced=False
        try:
            child.wait(timeout=left(93))
        except subprocess.TimeoutExpired:
            forced=True
        finally:
            if child.poll() is None:
                for sig in (signal.SIGTERM,signal.SIGKILL):
                    try:
                        os.killpg(child.pid,sig)
                        child.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                    except ProcessLookupError:
                        break
    observation=wait_terminal(owned,time.monotonic()+min(40,max(1,left(10))),interval=2,max_observations=20)
    row=observation["row"]
    computed=evidence/"compute.json"
    passed=bool(child.poll()==0 and not forced and observation["terminal_visible"] and row[2:4]==["COMPLETED","0:0"]
                and row[5]==NODE and "gres/dcu" not in row[6] and computed.is_file())
    save("outcome.json",dict(passed=passed,request=intent["request_id"],terminal=observation,rc=child.poll(),
                            epoch=epoch,deadline=deadline,compute_SHA256=digest(computed) if computed.is_file() else None,
                            log_SHA256=digest(ROOT/"CPU.log"),original_GPU_resubmissions=0,GPU_cards=0))
    return 0 if passed else 1


if __name__=="__main__":
    raise SystemExit(main())
