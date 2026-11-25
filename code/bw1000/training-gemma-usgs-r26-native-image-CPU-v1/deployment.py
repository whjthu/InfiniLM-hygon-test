"""Render a one-step candidate command and verify separate accounting. It never submits."""

from datetime import datetime,timezone
import math
from pathlib import PurePosixPath
import re
import shlex

from admission import FRAMEWORK_HOST, FRAMEWORK_MANIFEST_HOST, LABEL, PROCESSOR_HOST, RESULTS, WORK_SECONDS, need


RUNS=RESULTS/LABEL
APPTAINER="/public/software/apps/apptainer/1.3.4/bin/apptainer"
SANDBOX=str(RESULTS.parent/"containers/dtk26-sandbox-r1")


def argv(parent,node,request,publication,epoch,source_sha):
    need(re.fullmatch(r"[0-9]+",parent) and re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}",node)
         and re.fullmatch(r"[0-9a-f]{32}",request) and re.fullmatch(r"[0-9a-f]{40}",publication)
         and re.fullmatch(r"[0-9a-f]{64}",source_sha) and math.isfinite(epoch) and 0<epoch<10**10,
         "EXPLICIT_NEW_CPU_SCOPE")
    root=RUNS/request
    name=LABEL+"-"+request[:12]
    bindings=[
        (root/"src","/audit-src","ro"),(root/"data","/r26-train","ro"),(root/"out","/audit-out","rw"),
        (FRAMEWORK_HOST,"/input-framework","ro"),(FRAMEWORK_MANIFEST_HOST,"/qualification/fresh-framework-files.json","ro"),
        (PROCESSOR_HOST,"/gemma-processor","ro"),(RESULTS/"training-vlm-swift-offline-runtime-v1","/swift-runtime","ro"),
        (RESULTS/"training-minicpm-hygon-processor-r2/cpu/runtime-site","/minicpm-transformers","ro"),
        (PurePosixPath("/usr/local/hyhal"),"/opt/hyhal","ro"),
    ]
    cmd=["srun","--jobid="+parent,"--job-name="+name,"--exact","--immediate=5","-N1","-n1","--nodelist="+node,
         "--cpus-per-task=4","--mem=8192M","--gres=none","--time=00:03:00",
         APPTAINER,"exec","--contain","--no-home","--pwd","/"]
    for host,target,mode in bindings:
        cmd+=["--bind",f"{host}:{target}:{mode}"]
    env={"CUDA_VISIBLE_DEVICES":"-1","HIP_VISIBLE_DEVICES":"-1","ROCR_VISIBLE_DEVICES":"-1","HF_HUB_OFFLINE":"1",
         "TRANSFORMERS_OFFLINE":"1","HF_HUB_DISABLE_TELEMETRY":"1","PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1",
         "OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1","PYTHONPATH":"/minicpm-transformers:/swift-runtime/site:/input-framework"}
    arguments=["python3","-B","-u","/audit-src/native_image.py","--parent",parent,"--node",node,"--request",request,
               "--publication",publication,"--epoch",str(epoch),"--source-manifest-sha256",source_sha]
    shell=("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; export "+
           " ".join(f"{key}={shlex.quote(value)}" for key,value in env.items())+"; exec "+shlex.join(arguments))
    return cmd+[SANDBOX,"/bin/bash","--noprofile","--norc","-c",shell]


def terminal(row,result,parent,node,request,publication,source_sha,epoch):
    need(len(row)==9 and re.fullmatch(re.escape(parent)+r"\.[0-9]+",row[0])
         and row[1]==LABEL+"-"+request[:12] and row[2:4]==["COMPLETED","0:0"]
         and row[4].isdecimal() and 0<=int(row[4])<=WORK_SECONDS and row[5]==node
         and row[6] in ("cpu=4,mem=8G,node=1","cpu=4,mem=8192M,node=1")
         and all(value not in ("","Unknown","None") for value in row[7:]),"ACTUAL_INDEPENDENT_ZERO_GRES_TERMINAL")
    def utc(value):
        return datetime.strptime(value,"%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    start,end=map(utc,row[7:])
    need(epoch-1<=start<=end<=epoch+WORK_SECONDS+1
         and abs((end-start)-int(row[4]))<=1,"SLURM_UTC_ACCOUNTING_EPOCH")
    need(result["step"]==row[0] and result["node"]==node and result["request"]==request
         and result["publication"]==publication and result["source_manifest_SHA256"]==source_sha
         and result["epoch"]==epoch and result["deadline"]==epoch+WORK_SECONDS,"TERMINAL_BOUND_TO_NATIVE_RECEIPT")
    need(result["passed"] is True and (result["planned_rows"],result["planned_pairs"],result["stable_pair_ids"])==(12,6,6)
         and all(result["progress"][key]==12 for key in ("full_completed","capped_completed","collator_completed","verified_rows"))
         and result["actual_optimizer_updates"]==0 and result["model_loaded"] is False
         and result["training_ready"] is False,"NATIVE_CPU_RECEIPT_DENOMINATORS")
    return {"passed":True,"step":row[0],"GPU_steps":0,"native_image_rows":12,"paired_identities":6,
            "new_optimizer_updates":0,"training_ready":False}


def accounting_read_argv(parent):
    need(re.fullmatch(r"[0-9]+",parent),"BOUND_ACCOUNTING_PARENT")
    return ["env","TZ=UTC","SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S","sacct","-nP","-j",parent,
            "--format","JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"]
