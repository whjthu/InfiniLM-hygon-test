"""One bounded CPU-only step, reusing cached model metadata/runtime/data in place."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time

R=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL="training-minicpm-usgs-native88-CPU-v1"
REQUEST="7f9a0e2809bf4f53a7294fdc3dc74225"
RUN=R/"results"/LABEL
PARENT,NODE="1103606","f13r1n19"
COLS="JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def need(ok,why):
    if not ok:
        raise ValueError(why)


def save(name,value):
    with (RUN/name).open("x",encoding="utf8") as f:
        json.dump(value,f,sort_keys=True,allow_nan=False)
        f.flush();os.fsync(f.fileno())


def file_sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024**2),b""):
            h.update(b)
    return h.hexdigest()


def asset(p,pin):
    before=p.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_size==pin["bytes"],"Original asset size/type differs")
    need(file_sha(p)==pin["SHA256"],"Original asset SHA differs")
    after=p.lstat()
    need(all(getattr(before,k)==getattr(after,k) for k in ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns")),"Original asset changed")


def main(payload):
    os.umask(0o077)
    epoch,deadline=payload["epoch"],payload["deadline"]
    need(all(type(x) in (int,float) and math.isfinite(x) for x in (epoch,deadline)) and deadline-epoch==300
         and 0<epoch<=time.time()<deadline-40,"Original CPU epoch invalid or exhausted")
    def left(reserve=0):
        value=deadline-time.time()-reserve
        need(value>0,"Original CPU epoch exhausted")
        return value
    def query(*args,reserve=15):
        return subprocess.check_output(args,text=True,timeout=min(9,left(reserve))).strip()
    source={n:base64.b64decode(v,validate=True) for n,v in payload["files"].items()}
    mf=source["SOURCE_FILES.json"]
    need(hashlib.sha256(mf).hexdigest()==payload["source_manifest_SHA256"],"Source manifest differs")
    pins=json.loads(mf)["files"]
    need(set(source)==set(pins)|{"SOURCE_FILES.json"} and sum(map(len,source.values()))<1024**2,"Packet scope differs")
    for name,pin in pins.items():
        need(Path(name).name==name and len(source[name])==pin["bytes"] and hashlib.sha256(source[name]).hexdigest()==pin["SHA256"],"Source member differs")
    claim=json.loads(source["CLAIM.json"])
    need(claim["request"]==payload["request"]==REQUEST and claim["label"]==LABEL
         and claim["CPU_GO"] is True and claim["root_GO"] is True
         and all(claim[k] is False for k in ("GPU_GO","training_GO","Model_GO")),"Unadopted CPU-only claim")
    RUN.mkdir(mode=0o700)
    (RUN/"src").mkdir()
    for name,raw in source.items():
        with (RUN/"src"/name).open("xb") as f:
            f.write(raw)
    save("intent.json",dict(request=REQUEST,publication=payload["publication"],epoch=epoch,deadline=deadline,
         source_manifest_SHA256=payload["source_manifest_SHA256"],claim_SHA256=hashlib.sha256(source["CLAIM.json"]).hexdigest(),
         parent=PARENT,node=NODE,planned_rows=88,planned_encode=176,planned_collator=88,actual_optimizer_updates=0))
    sys.path.insert(0,str(RUN/"src"))
    from slurm_accounting import wait_terminal
    def accounting():
        lines=query("sacct","-nP","-j",PARENT,"-o",COLS,reserve=3).splitlines()
        rows=[x.split("|") for x in lines if len(x.split("|"))==7 and x.split("|")[1]==LABEL]
        need(len(rows)<=1,"Duplicate exact CPU step label")
        return rows[0] if rows else None
    need(accounting() is None,"Original namespace already has a step; reconcile, do not replay")
    job=dict(x.split("=",1) for x in query("scontrol","show","job",PARENT,"-o").split() if "=" in x)
    need(job["JobState"]=="RUNNING" and job["NumNodes"]=="40" and job["UserId"].endswith(f"({os.getuid()})")
         and "gres/dcu=320" in job["AllocTRES"].split(",") and NODE in query("scontrol","show","hostnames",job["NodeList"]).splitlines(),"Current owned node allocation differs")
    for line in query("scontrol","show","step",PARENT,"-o").splitlines():
        fields=dict(x.split("=",1) for x in line.split() if "=" in x)
        suffix=fields["StepId"].removeprefix(PARENT+".")
        if suffix in ("batch","extern"):
            continue
        need(suffix.isdecimal() and NODE not in query("scontrol","show","hostnames",fields["NodeList"]).splitlines(),"Current step occupies selected CPU node")
    assets=json.loads(source["ASSETS.json"])
    initial=assets["initial"]
    asset(Path(initial["path"]),initial)
    (RUN/"metadata-only/MiniCPM-V-4.6").mkdir(parents=True)
    model=Path(assets["model_root"])
    for name,pin in assets["metadata"].items():
        asset(model/name,pin)
        shutil.copyfile(model/name,RUN/"metadata-only/MiniCPM-V-4.6"/name)
    data=R/"results/training-domain-usgs-table-reading-v1/data/train"
    need(file_sha(data/"swift.jsonl")==assets["dataset_SHA256"],"Existing USGS train source changed")
    for name,pin in assets["media"].items():
        asset(data/"media"/name,pin)
    framework=R/"results/training-qwen-a68952c-swift-initial-CPU-v1/source/FRAMEWORK_FILES.json"
    need(file_sha(framework)=="e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41","Common framework registration manifest differs")
    save("asset-admission.json",dict(model_metadata_files=9,train_rows=88,media_files=22,
         initial_file_SHA256=initial["SHA256"],initial_named_SHA256=initial["named_SHA256"],initial_file_hashed=True,
         initial_tensor_state_deserialized=False,framework_manifest_only=True,full393_reverified=False,
         model_weight_body_read=False,new_assets_transferred=False,dev_or_blind_read=False))
    runtime_alias=Path("/opt/hyhal").resolve(strict=True)
    need(runtime_alias==Path("/usr/local/hyhal") and runtime_alias.is_dir() and not runtime_alias.is_symlink(),"Qualified driver alias differs")
    exe=Path("/public/software/apps/apptainer/1.3.4/bin/apptainer")
    need(exe.is_file() and os.access(exe,os.X_OK),"Qualified Apptainer executable unavailable")
    seconds=min(240,int(left(35)))
    need(seconds>30,"Insufficient epoch for new CPU step")
    container=[str(exe),"exec","--contain","--no-home","--pwd","/"]
    for host,target,mode in ((RUN,"/pilot","rw"),(data,data,"ro"),
        (R/"results/training-vlm-swift-offline-runtime-v1","/swift-runtime","ro"),
        (R/"results/training-minicpm-hygon-processor-r2/cpu/runtime-site","/minicpm-transformers","ro"),
        (R/"results/training-vlm-swift-public-artifacts-v1/packages","/wheels/swift","ro"),
        (R/"results/training-minicpm-hygon-processor-r2/packages","/wheels/transformers","ro"),
        (runtime_alias,"/opt/hyhal","ro")):
        need(Path(host).is_dir(),"Declared existing readonly runtime absent")
        container.extend(["--bind",f"{host}:{target}:{mode}"])
    shell=("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
       "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 "
       "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 "
       "CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; "
       "export PYTHONPATH=/minicpm-transformers:/swift-runtime/site:/pilot/src; "
       "exec python3 -B -u /pilot/src/worker.py")
    container.extend([str(R/"containers/dtk26-sandbox-r1"),"/bin/bash","--noprofile","--norc","-c",shell])
    command=["srun","--jobid="+PARENT,"--job-name="+LABEL,"--exact","--immediate=5","-N1","-n1",
       "--nodelist="+NODE,"--cpus-per-task=8","--mem=12288M","--gres=none",
       "--time=00:%02d:%02d"%(seconds//60,seconds%60),*container]
    save("step-intent.json",dict(command=command,slurm_seconds=seconds,epoch=epoch,deadline=deadline,
         CPU=8,memory_MiB=12288,GRES=0,whole_seconds=300,worker_max_seconds=210))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",HIP_VISIBLE_DEVICES="-1",ROCR_VISIBLE_DEVICES="-1")
    child=None
    rc=None
    error=None
    started=time.monotonic()
    try:
        with (RUN/"step.log").open("xb") as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            fields=Path(f"/proc/{child.pid}/stat").read_text().rsplit(") ",1)[1].split()
            save("srun-owner.json",dict(pid=child.pid,startticks=int(fields[19]),host=os.uname().nodename,
                 boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip()))
            until=time.monotonic()+seconds
            while child.poll() is None:
                need(time.monotonic()<until and time.time()<deadline-25,"Original CPU step timeout")
                need((RUN/"step.log").stat().st_size<=16*1024**2,"CPU log cap exceeded")
                time.sleep(.25)
            rc=child.returncode
    except BaseException as exc:
        error=type(exc).__name__
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill();child.wait(timeout=3)
    accounting_errors=[]
    def observe_terminal():
        try:
            return accounting()
        except (ValueError,subprocess.SubprocessError) as exc:
            accounting_errors.append(type(exc).__name__)
            return None
    terminal=wait_terminal(observe_terminal,time.monotonic()+min(20,max(0,deadline-time.time()-3)),interval=1,max_observations=15)
    row=terminal["row"]
    proper=False
    if row:
        tres=dict(x.split("=",1) for x in row[6].split(","))
        proper=(row[2:4]==["COMPLETED","0:0"] and row[5]==NODE and tres.get("cpu")=="8"
                and tres.get("node")=="1" and tres.get("mem") in ("12G","12288M")
                and not any(k.startswith("gres/") for k in tres))
    result_path=RUN/"worker-result.json"
    result=json.loads(result_path.read_bytes()) if result_path.is_file() else None
    passed=rc==0 and error is None and proper and result and result.get("passed") is True
    save("outcome.json",dict(request=REQUEST,publication=payload["publication"],epoch=epoch,deadline=deadline,
         passed=bool(passed),srun_returncode=rc,error_type=error,terminal=terminal,accounting_columns=COLS,
         srun_reaped=child is not None and child.poll() is not None,step_client_wall_seconds=time.monotonic()-started,accounting_errors=accounting_errors,
         planned_rows=88,planned_encode=176,planned_collator=88,worker_result_SHA256=file_sha(result_path) if result else None,
         completed_rows=result.get("completed_rows",result.get("progress",{}).get("completed_rows",0)) if result else None,
         actual_optimizer_updates=0,GPU_GO=False,training_GO=False))
    print(json.dumps(dict(request=REQUEST,passed=bool(passed),execution_state="TERMINAL" if terminal["terminal_visible"] else "UNKNOWN",
                    planned_rows=88,actual_optimizer_updates=0)),flush=True)
    return 0 if passed else 1


if __name__=="__main__":
    raise SystemExit(main(PAYLOAD))
