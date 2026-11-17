"""Submit only three bounded zero-GPU input-entry gates; never invoke inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from scope import ROOT, RESULTS, LABEL, RUN, SRC, PREP, MEDIA, MODEL, ARMS, ADAPTERS, COLS
from scope import require, read, sha, remaining, stage_budget, runtime_bind_source
from slurm_accounting import wait_terminal
from verify_package import verify


def save(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def container(arm, report):
    runtime = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--contain", "--rocm", "--no-home", "--pwd", "/"]
    arm_run = RUN / "arms" / arm
    binds = [(arm_run, arm_run, "rw"), (SRC, SRC, "ro"), (MODEL, MODEL, "ro"), (PREP, PREP, "ro"),
             (MEDIA, PREP / "dev/media", "ro"),
             (RESULTS / "training-vlm-swift-offline-runtime-v1", "/swift-runtime", "ro"),
             (RESULTS / "training-qwen-vision014-CPU-v2/vision", "/vision", "ro"),
             (RESULTS / "training-minicpm-hygon-processor-r2/cpu/runtime-site", "/minicpm-transformers", "ro"),
             (Path("/opt/hyhal"), "/opt/hyhal", "ro")]
    if ADAPTERS[arm] is not None:
        binds.append((Path(ADAPTERS[arm]), Path(ADAPTERS[arm]), "ro"))
    for host, target, mode in binds:
        runtime.extend(("--bind", f"{runtime_bind_source(host)}:{target}:{mode}"))
    args = ["python3", "-B", "-u", str(SRC / "probe.py"), "--arm", arm, "--output", str(report)]
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT; "
             "export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= "
             "PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 "
             "TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
             "PYTHONPATH=/vision:/minicpm-transformers:/swift-runtime/site:" + str(SRC) + "; "
             "exec /usr/bin/timeout --kill-after=5s 90s " + shlex.join(args))
    return runtime + [str(ROOT / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", shell]


def main(publication, claim_sha, request):
    epoch, monotonic_start = time.time(), time.monotonic()
    deadline, mono_end = epoch + 600, monotonic_start + 600
    os.umask(0o077)
    require(re.fullmatch(r"[0-9a-f]{40}", publication) and re.fullmatch(r"[0-9a-f]{64}", claim_sha)
            and re.fullmatch(r"[0-9a-f]{32}", request), "invalid publication/request identity")
    require(Path(__file__).resolve().parent == SRC and not RUN.is_symlink(), "unexpected installed namespace")
    require(sha(SRC / "CLAIM.json") == claim_sha, "claim changed")
    verify(SRC)
    require(not any((RUN / name).exists() for name in ("intent.json", "arms", "outcome.json")), "original namespace exists; no replay")
    def query(*args, own_end=None):
        timeout = remaining(deadline, mono_end, time.time(), time.monotonic(), 12)
        if own_end is not None:
            timeout = min(timeout, own_end-time.monotonic())
        require(timeout > 0, "own terminal observation window exhausted")
        return subprocess.check_output(args, text=True, timeout=timeout).strip().splitlines()
    def accounting(parent, label, own_end=None):
        rows = [row for line in query("sacct", "-n", "-P", "-j", parent, "-o", COLS, own_end=own_end)
                if len(row := line.split("|")) == 7 and row[1] == label]
        require(len(rows) <= 1, "duplicate original step label")
        return rows[0] if rows else None
    for arm, (parent, node) in ARMS.items():
        require(accounting(parent, LABEL + "-" + arm) is None, "original CPU label exists")
        jobs = query("scontrol", "show", "job", parent, "--oneliner")
        require(len(jobs) == 1, "parent unavailable")
        job = dict(x.split("=",1) for x in jobs[0].split() if "=" in x)
        require(job["JobState"] == "RUNNING" and job["NumNodes"] == "40"
                and node in query("scontrol", "show", "hostnames", job["NodeList"]), "wrong CPU parent/node")
    save(RUN / "intent.json", dict(publication=publication, claim_SHA256=claim_sha, request=request, epoch=epoch,
         deadline=deadline, arms={k:dict(parent=v[0],node=v[1]) for k,v in ARMS.items()}, maximum_CPU_steps=3,
         GPU_GO=False, actual_expected_GPU_cards=0, common_inside_seconds=90, common_Slurm_seconds=120))
    (RUN / "arms").mkdir()
    for arm in ARMS:
        (RUN / "arms" / arm).mkdir()
    def run_arm(arm):
        parent, node = ARMS[arm]
        folder = RUN / "arms" / arm
        label = LABEL + "-" + arm
        rc, failure = None, None
        limits = None
        try:
            limits = stage_budget(deadline, mono_end, time.time(), time.monotonic())
            args = container(arm, folder / "probe.json")
            require(accounting(parent,label) is None, "original step appeared; no replay")
            require(120 <= remaining(deadline,mono_end,time.time(),time.monotonic(),120,45), "container preparation consumed reserved budget")
            command = ["srun", "--jobid="+parent, "--job-name="+label, "--exact", "--immediate=5", "-N1", "-n1",
                       "-w", node, "-c2", "--mem=4096M", "--gres=none", "--time=00:02:00", *args]
            save(folder/"step-intent.json",dict(epoch=epoch,deadline=deadline,limits=limits,command=command))
            with (folder/"step.log").open("xb") as stream:
                child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                owned=dict(PID=child.pid)
                try:
                    fields=Path("/proc",str(child.pid),"stat").read_text().rsplit(") ",1)[1].split()
                    owned.update(PGID=int(fields[2]),startticks=int(fields[19]))
                except FileNotFoundError:
                    owned["exited_before_identity_observation"]=True
                save(folder/"owned-srun.json",owned)
                try:
                    rc=child.wait(timeout=remaining(deadline,mono_end,time.time(),time.monotonic(),125,35))
                except subprocess.TimeoutExpired:
                    pass
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            failure=type(error).__name__
        errors=[]
        own_end=min(mono_end-5,time.monotonic()+30)
        def observe():
            try:return accounting(parent,label,own_end)
            except (OSError,ValueError,subprocess.SubprocessError) as error:
                errors.append(type(error).__name__);return None
        visibility=wait_terminal(observe,own_end,interval=2,max_observations=14)
        row=visibility["row"]
        actual=None
        if (folder/"probe.json").is_file():
            try:
                actual=read(folder/"probe.json")
            except (OSError,ValueError,TypeError) as error:
                failure=failure or type(error).__name__
        passed=bool(rc==0 and failure is None and row and row[2:4]==["COMPLETED","0:0"] and row[5]==node
                    and "gres/dcu=" not in row[6] and actual and actual["passed"] is True
                    and actual["actual_node"]==node and actual["actual_step"]==row[0]
                    and actual["resolved"]["file_count"]==39 and actual["visual_rows"]==actual["text_only_rows"]==108
                    and actual["actual_GPU_cards"]==0 and actual["torch_imported"] is False)
        result=dict(arm=arm,passed=passed,rc=rc,failure=failure,limits=limits,accounting_columns=COLS,
                    accounting=visibility,accounting_errors=errors,probe_SHA256=sha(folder/"probe.json") if (folder/"probe.json").is_file() else None,
                    actual_GPU_cards=0 if row and "gres/dcu=" not in row[6] else None,replay_allowed=False)
        save(folder/"outcome.json",result)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=dict(zip(ARMS,pool.map(run_arm,ARMS)))
    passed=all(value["passed"] for value in results.values())
    result=dict(request=request,epoch=epoch,deadline=deadline,passed=passed,arms=results,
                all_three_actual_entry_gates_passed=passed,GPU_GO=False,
                next_GPU_requires_independent_terminal_audit_and_new_v2_claim=True,
                default_CWD_cover_mechanism_proved=False)
    save(RUN/"outcome.json",result)
    print(json.dumps(dict(request=request,passed=passed,all_three_actual_entry_gates_passed=passed,GPU_GO=False)))
    return 0 if passed else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication",required=True)
    parser.add_argument("--claim-sha256",required=True)
    parser.add_argument("--request-id",required=True)
    args=parser.parse_args()
    raise SystemExit(main(args.publication,args.claim_sha256,args.request_id))
