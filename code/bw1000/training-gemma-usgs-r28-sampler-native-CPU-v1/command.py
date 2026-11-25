"""Freeze the source-only R28 sampler packet and render a zero-GRES command."""

import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
import math
from pathlib import Path,PurePosixPath
import re
import shlex


HERE=Path(__file__).resolve().parent
ROOT=PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS=ROOT/"results"
BASE=RESULTS/"training-gemma-usgs72-fresh-initial-domain25-v1"
OLD_MEDIA=RESULTS/"training-gemma-usgs-render72-native-CPU-v1/data/train/media"
LABEL="training-gemma-usgs-r28-sampler-native-CPU-v1"
RUNTIME=RESULTS/"training-vlm-swift-offline-runtime-v1"
TRANSFORMERS=RESULTS/"training-minicpm-hygon-processor-r2/cpu/runtime-site"
COLUMNS="JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"


def need(value,reason):
    if not value:
        raise ValueError(reason)


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def pin(raw):
    return {"bytes":len(raw),"SHA256":sha256(raw).hexdigest()}


def source_manifest():
    raw=(HERE/"probe.py").read_bytes()
    compile(raw,"frozen_compute_probe","exec")
    return canonical({"schema":"GEMMA_R28_NATIVE_SAMPLER_CPU_ONLY_SOURCE_V1","files":{"probe.py":pin(raw)}})


def runner(parent,node,request,epoch,manifest):
    need(re.fullmatch(r"[0-9]+",parent) and re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}",node)
         and re.fullmatch(r"[0-9a-f]{32}",request) and re.fullmatch(r"[0-9a-f]{64}",manifest)
         and math.isfinite(epoch) and epoch>0,"FRESH_CPU_SCOPE_STEP_OWNER")
    run=RESULTS/LABEL/request
    binds=[
        (run/"src","/r28-src","ro"),(run/"out","/r28-out","rw"),
        (BASE/"view/train800-swift.jsonl","/r28-original/train800-swift.jsonl","ro"),
        (BASE/"continuous-plan.json","/r28-original/continuous-plan.json","ro"),
        (BASE/"runtime-binding.json","/r28-original/runtime-binding.json","ro"),
        (BASE/"fresh-framework-files.json","/r28-original/fresh-framework-files.json","ro"),
        (BASE/"framework","/input-framework","ro"),(OLD_MEDIA,str(OLD_MEDIA),"ro"),
        (RUNTIME,"/swift-runtime","ro"),(TRANSFORMERS,"/minicpm-transformers","ro"),
        (PurePosixPath("/usr/local/hyhal"),"/opt/hyhal","ro")]
    argv=["srun","--jobid="+parent,"--job-name="+LABEL+"-"+request[:12],"--exact","--immediate=5","-N1","-n1",
          "--nodelist="+node,"--cpus-per-task=4","--mem=8192M","--gres=none","--time=00:03:00",
          "/public/software/apps/apptainer/1.3.4/bin/apptainer","exec","--contain","--no-home","--pwd","/"]
    for local,target,mode in binds:
        argv.extend(["--bind",f"{local}:{target}:{mode}"])
    env={"CUDA_VISIBLE_DEVICES":"-1","HIP_VISIBLE_DEVICES":"-1","ROCR_VISIBLE_DEVICES":"-1",
         "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","HF_HUB_DISABLE_TELEMETRY":"1","HF_DATASETS_OFFLINE":"1",
         "PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1","OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1",
         "PYTHONPATH":"/minicpm-transformers:/swift-runtime/site:/input-framework"}
    cmd=["python3","-B","-u","/r28-src/probe.py","--parent",parent,"--node",node,"--request",request,
         "--epoch",str(epoch),"--manifest",manifest]
    shell=("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; unset PYTHONPYCACHEPREFIX; export "+
           " ".join(f"{key}={shlex.quote(value)}" for key,value in env.items())+"; exec "+shlex.join(cmd))
    return argv+[str(ROOT/"containers/dtk26-sandbox-r1"),"/bin/bash","--noprofile","--norc","-c",shell]


def accounting_argv(parent):
    need(re.fullmatch(r"[0-9]+",parent),"EXACT_PARENT_IDENTITY")
    return ["env","TZ=UTC","SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S","sacct","-nP","-j",parent,"--format",COLUMNS]


def independent(row,result,projection_bytes,parent,node,request,epoch,manifest):
    """Read real target accounting independently before consuming the CPU projection."""
    need(len(row)==9 and re.fullmatch(re.escape(parent)+r"\.[0-9]+",row[0])
         and row[1]==LABEL+"-"+request[:12] and row[2:4]==["COMPLETED","0:0"]
         and row[4].isdecimal() and 0<=int(row[4])<=120 and row[5]==node
         and row[6] in ("cpu=4,mem=8G,node=1","cpu=4,mem=8192M,node=1"),"EXACT_NUMERIC_CPU_TERMINAL")
    def parse(value):
        return datetime.strptime(value,"%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    start,end=map(parse,row[7:])
    need(epoch-1<=start<=end<=epoch+121 and abs(end-start-int(row[4]))<=1,"SAME_ACTUAL_CPU_EPOCH")
    need(result["schema"]=="R28_NATIVE_CPU_COMPONENT_SAMPLER_RESULT_V1" and result["passed"] is True
         and (result["step"],result["node"],result["request"],result["epoch"],result["source_manifest_SHA256"])==
         (row[0],node,request,epoch,manifest)
         and result["actual_gpu_steps"]==result["actual_optimizer_updates"]==result["actual_live_training_ranks"]==0
         and result["actual_native_rank_projections"]==8 and result["actual_distinct_physical_slots"]==800
         and result["private_projection"]==pin(projection_bytes),"EXACT_TARGET_CPU_PROJECTION_RESULT")
    projection=json.loads(projection_bytes)
    ranks=projection["ranks"]
    need(projection["schema"]=="R28_NATIVE_CPU_COMPONENT_800_ORDER_V1" and projection["production_sampler_adoption"] is False
         and projection["step"]==row[0] and projection["request"]==request
         and set(ranks)=={str(i) for i in range(8)} and all(len(rank)==100 for rank in ranks.values())
         and sorted(i for rank in ranks.values() for i in rank)==list(range(1,801)),"EXACT_800_PRIVATE_PROJECTION")
    return {"state":"INDEPENDENT_ZERO_GRES_CPU_COMPONENT_PROJECTION_ONLY","step":row[0],
            "private_projection":pin(projection_bytes),"actual_native_projections":8,"actual_slots":800,
            "actual_training_rank_processes":0,"actual_optimizer_updates":0,
            "permitted_to_generate_production_800_view":False,"training_ready":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest",action="store_true")
    parser.add_argument("--parent")
    parser.add_argument("--node")
    parser.add_argument("--request")
    parser.add_argument("--epoch",type=float)
    args=parser.parse_args()
    source=source_manifest()
    if args.write_manifest:
        with (HERE/"SOURCE_FILES.json").open("xb") as stream:
            stream.write(source)
    answer={"source_manifest":pin(source),"source_members":["probe.py","SOURCE_FILES.json"],
            "compute_work_seconds":120,"slurm_limit_seconds":180,"CPU":4,"memory_MiB":8192,"GRES":"none",
            "stage_new_raw_files":0,"training_ready":False}
    if any(value is not None for value in (args.parent,args.node,args.request,args.epoch)):
        need(all(value is not None for value in (args.parent,args.node,args.request,args.epoch)),"EXPLICIT_FULL_PLACEMENT")
        answer["srun_argv"]=runner(args.parent,args.node,args.request,args.epoch,pin(source)["SHA256"])
        answer["independent_accounting_argv"]=accounting_argv(args.parent)
    print(json.dumps(answer,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
