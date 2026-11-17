"""One HOST plus original-container metadata comparison inside one zero-GPU Slurm step."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from scope import RUN,SRC,ARMS,ROOT,FLAGS,IMAGE,original_binds,runtime_bind_source,require,read,pin
from snapshot_io import separate_logs,unique_object,reject_constant


def save(path,value):
    with path.open("x",encoding="utf8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")


def read_report(path,arm,context):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=1024**2,"unsafe metadata report")
    value=json.loads(path.read_bytes(),object_pairs_hook=unique_object,parse_constant=reject_constant)
    require(value["kind"]=="training_qwen_capacity_metadata" and value["arm"]==arm and value["context"]==context
            and value["completed"] is True and value["Torch_imported"] is False and value["SDK_explicit_calls"]==0
            and value["LLVM_explicit_calls"]==value["model_or_weight_reads"]==0 and value["canary_written"] is False
            and value["GPU_GO"] is False,"metadata report scope differs")
    return value


def container_command(arm,probe_deadline,seconds):
    args=list(FLAGS)
    for host,target,mode in original_binds(arm):
        args.extend(("--bind",runtime_bind_source(host)+":"+target+":"+mode))
    command=["python3","-I","-S","-B","-","--arm",arm,"--context","container","--deadline",str(probe_deadline)]
    # Keep the original v3 env/RO project binds. Metadata Python ignores site/PYTHONPATH; no Torch/SDK import.
    shell=("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh >&2; unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT; "
           "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 "
           "TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/vision:/minicpm-transformers:/swift-runtime/site:"+str(ROOT/"results/training-qwen-treasury-explicit-dev3-gpu-v3/source")+"; "
           "exec /usr/bin/timeout --kill-after=3s "+str(seconds)+"s "+shlex.join(command))
    return args+[str(IMAGE),"/bin/bash","--noprofile","--norc","-c",shell]


def save_child(folder,context,child):
    owned=dict(PID=child.pid)
    try:
        fields=Path("/proc",str(child.pid),"stat").read_text().rsplit(") ",1)[1].split()
        owned.update(PGID=int(fields[2]),startticks=int(fields[19]))
    except FileNotFoundError:
        owned["exited_before_identity_observation"]=True
    save(folder/(context+"-owned-process.json"),owned)


def stage(arm,deadline):
    started=time.monotonic();spec=ARMS[arm];folder=RUN/"arms"/arm
    require(os.environ.get("SLURM_JOB_ID")==spec["parent"] and os.uname().nodename.split(".")[0]==spec["node"],"wrong CPU step/node")
    require(os.environ.get("SLURM_STEP_GPUS") in (None,"","(null)"),"capacity stage unexpectedly owns GPU")
    intent=read(RUN/"intent.json")
    require(time.time()<deadline<=intent["deadline"] and deadline-time.time()<=50,"capacity stage exceeds original budget")
    probe=(SRC/"capacity_probe.py").read_bytes()
    reports={};commands={};failure=None
    try:
        host_deadline=min(deadline-5,time.time()+10)
        command=["/usr/local/python3.12/bin/python3","-I","-S","-B","-","--arm",arm,"--context","host","--deadline",str(host_deadline)]
        commands["host"]=command
        with separate_logs(folder/"host.stdout",folder/"host.stderr") as (out,err):
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=out,stderr=err,start_new_session=True)
            save_child(folder,"host",child)
            child.communicate(input=probe,timeout=min(12,deadline-time.time()-5))
            require(child.returncode==0,"HOST capacity probe failed")
        reports["host"]=read_report(folder/"host.stdout",arm,"host")
        available=int(min(35,deadline-time.time()-5));require(available>=5,"no remaining container metadata budget")
        probe_deadline=time.time()+available
        command=container_command(arm,probe_deadline,available);commands["container"]=command
        save(folder/"container-command.json",dict(original_epoch=intent["epoch"],deadline=deadline,probe_deadline=probe_deadline,
              original_v3_runtime_flags=FLAGS,original_v3_binds=original_binds(arm),command=command,no_new_bind=True))
        with separate_logs(folder/"container.stdout",folder/"container.stderr") as (out,err):
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=out,stderr=err,start_new_session=True)
            save_child(folder,"container",child)
            child.communicate(input=probe,timeout=min(available+4,deadline-time.time()))
            require(child.returncode==0,"original container metadata probe failed")
        reports["container"]=read_report(folder/"container.stdout",arm,"container")
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        failure=type(error).__name__
    pins={}
    for context in ("host","container"):
        for channel in ("stdout","stderr"):
            p=folder/(context+"."+channel)
            if p.is_file():
                pins[context+"."+channel]=pin(p)
    passed=failure is None and set(reports)=={"host","container"}
    result=dict(arm=arm,passed=passed,failure_class=failure,node=spec["node"],step=spec["parent"]+"."+os.environ["SLURM_STEP_ID"],
        original_epoch=intent["epoch"],deadline=deadline,wall_seconds=time.monotonic()-started,reports=reports,stream_pins=pins,
        GPU_GO=False,actual_allocated_GPU_cards=0,canary_written=False,exact_ENOSPC_writer_path_proved=False,
        capacity_comparison_complete=passed,source_only_cache_default_candidates=True)
    save(folder/"stage-outcome.json",result)
    print(json.dumps(dict(arm=arm,passed=passed,GPU_GO=False)))
    return 0 if passed else 1


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--arm",required=True,choices=ARMS);p.add_argument("--deadline",required=True,type=float)
    args=p.parse_args();raise SystemExit(stage(args.arm,args.deadline))
