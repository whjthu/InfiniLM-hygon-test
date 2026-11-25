"""Emit exact source-only staging and a separate bounded R29 CPU command."""

import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
import math
from pathlib import Path,PurePosixPath
import re
import shlex


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
LABEL="training-gemma-usgs-r32-ipc-boundary-CPU-v1"
ROOT=PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS=ROOT/"results"
BASE=RESULTS/"training-gemma-usgs72-fresh-initial-domain25-v1"
MEDIA=RESULTS/"training-gemma-usgs-render72-native-CPU-v1/data/train/media"
PROCESSOR=RESULTS/"training-gemma-processor-assets-v1/models/gemma-3-4b-it"
PREVIOUS="f51e026158944d79ace5cf3ce54d8241"
PREVIOUS_MANIFEST="45c52a9ea4841097f050c27887a6be8e9b838bda1812c5cb168cc9c0b36ae749"
OLD_FILES={
 "probe.py":("code/bw1000/training-gemma-usgs-r28-sampler-native-CPU-v1/probe.py",16655,
             "c0bc5e03e4ab14e1cf02cff62843a6a133257751005a73015859d5ef27850594"),
 "MODEL_PROCESSOR_FILES.json":("code/bw1000/training-gemma-native-inputs-HIP-CPU-v1/MODEL_PROCESSOR_FILES.json",2107,
             "eaab7b89d89e5cbe27391c3601ab36acfa5aa35f8c0ea18f6f1730d5424d228b")}
ACCOUNTING="JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def pin(raw):
    return {"SHA256":sha256(raw).hexdigest(),"bytes":len(raw)}


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def source_files():
    manifest=(HERE/"SOURCE_FILES.json").read_bytes()
    entries=json.loads(manifest)["files"]
    need(set(entries)=={'probe.py', 'probe_r32.py', 'r30_cache_gate.py', 'r31_modelscope_cache.py', 'MODEL_PROCESSOR_FILES.json', 'ipc_witness.py'},"EXACT_R32_COMPUTE_MEMBERS")
    data={name:(HERE/name).read_bytes() for name in entries}
    need({name:pin(raw) for name,raw in data.items()}==entries,"EXACT_R32_COMPUTE_BYTES")
    return {**data,"SOURCE_FILES.json":manifest}


def argv(parent,node,request,epoch,source,fallback=False):
    need(re.fullmatch(r"[0-9]+",parent) and re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}",node)
         and re.fullmatch(r"[0-9a-f]{32}",request) and request!=PREVIOUS and math.isfinite(epoch) and epoch>0
         and re.fullmatch(r"[0-9a-f]{64}",source),"FRESH_R29_SCOPE_NO_ORIGINAL_REPLAY")
    run=RESULTS/LABEL/request
    binds=[(run/"src","/r32-src","ro"),(run/"out","/r32-out","rw"),(run/"scratch","/r32-scratch","rw"),
           (BASE/"view/train800-swift.jsonl","/r28-original/train800-swift.jsonl","ro"),
           (BASE/"continuous-plan.json","/r28-original/continuous-plan.json","ro"),
           (BASE/"runtime-binding.json","/r28-original/runtime-binding.json","ro"),
           (BASE/"fresh-framework-files.json","/r28-original/fresh-framework-files.json","ro"),
           (BASE/"framework","/input-framework","ro"),(MEDIA,str(MEDIA),"ro"),(PROCESSOR,"/r29-processor","ro"),
           (RESULTS/"training-vlm-swift-offline-runtime-v1","/swift-runtime","ro"),
           (RESULTS/"training-minicpm-hygon-processor-r2/cpu/runtime-site","/minicpm-transformers","ro"),
           (PurePosixPath("/usr/local/hyhal"),"/opt/hyhal","ro")]
    result=["srun","--jobid="+parent,"--job-name="+LABEL+"-"+request[:12],"--exact","--immediate=5","-N1","-n1",
          "--nodelist="+node,"--cpus-per-task=4","--mem=8192M","--gres=none","--time=00:03:00",
          "/public/software/apps/apptainer/1.3.4/bin/apptainer","exec","--contain","--no-home","--pwd","/"]
    for local,mounted,access in binds:
        result.extend(["--bind",f"{local}:{mounted}:{access}"])
    env={"CUDA_VISIBLE_DEVICES":"-1","HIP_VISIBLE_DEVICES":"-1","ROCR_VISIBLE_DEVICES":"-1", "HF_HUB_OFFLINE":"1",
          "TRANSFORMERS_OFFLINE":"1","HF_HUB_DISABLE_TELEMETRY":"1","HF_DATASETS_OFFLINE":"1",
          "PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1","OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1",
          "PYTHONPATH":"/minicpm-transformers:/swift-runtime/site:/input-framework"}
    compute=["python3","-B","-u","/r32-src/probe_r32.py","--parent",parent,"--node",node,"--request",request,
             "--epoch",str(epoch),"--manifest",source]
    if fallback:
        compute.append("--allow-component-fallback")
    shell=("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; unset PYTHONPYCACHEPREFIX; export "+
           " ".join(f"{key}={shlex.quote(value)}" for key,value in env.items())+"; exec "+shlex.join(compute))
    return result+[str(ROOT/"containers/dtk26-sandbox-r1"),"/bin/bash","--noprofile","--norc","-c",shell]


def independent(row,report,parent,node,request,epoch,manifest):
    need(len(row)==9 and re.fullmatch(re.escape(parent)+r"\.[0-9]+",row[0])
         and row[1]==LABEL+"-"+request[:12] and row[2] in ("COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY")
         and row[4].isdecimal() and 0<=int(row[4])<=181 and row[5]==node
         and row[6] in ("cpu=4,mem=8G,node=1","cpu=4,mem=8192M,node=1"),"ACTUAL_INDEPENDENT_ZERO_GRES_NUMERIC_STEP")
    def utc(value):
        return datetime.strptime(value,"%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    a,b=map(utc,row[7:])
    need(epoch-1<=a<=b<=epoch+181 and abs(b-a-int(row[4]))<=1,"ORIGINAL_R29_ACCOUNTING_EPOCH")
    if report is not None:
        need(report["step"]==row[0] and report["epoch"]==epoch and report["request"]==request
             and report["source_manifest_SHA256"]==manifest
             and not any(report[key] for key in ("actual_gpu","actual_optimizer_updates","actual_training_ranks",
                                                 "training_ready","production_adoption")),
             "EXACT_R29_PRIVATE_WORKER_INDEPENDENT_RESULT")
        need(len(report.get("diagnostic",{}).get("frames",[]))<=6,"BOUNDED_DIAGNOSTIC")
    return {"accounting_step":row[0],"accounting_state":row[2],"worker_result_present":report is not None,
            "provider_projection_only":True,"training_ready":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-source-dir",action="store_true")
    parser.add_argument("--parent")
    parser.add_argument("--node")
    parser.add_argument("--request")
    parser.add_argument("--epoch",type=float)
    parser.add_argument("--allow-component-fallback",action="store_true")
    opts=parser.parse_args()
    files=source_files()
    if opts.write_source_dir:
        folder=HERE/"SOURCE_PACKET"
        folder.mkdir()
        for name,raw in files.items():
            with (folder/name).open("xb") as stream:
                stream.write(raw)
    mf=pin(files["SOURCE_FILES.json"])
    summary={"source_manifest":mf,"source_files":{name:pin(raw) for name,raw in files.items()},
        "stage_raw_inputs":0,"memory_MiB":8192,"CPU":4,"GRES":"none","worker_seconds":120,"slurm_seconds":180,
        "original_replay_allowed":False,"training_ready":False}
    placement=(opts.parent,opts.node,opts.request,opts.epoch)
    if any(value is not None for value in placement):
        need(all(value is not None for value in placement),"FULL_FRESH_PLACEMENT_REQUIRED")
        summary["argv"]=argv(*placement,mf["SHA256"],opts.allow_component_fallback)
        summary["independent_read_argv"]=["env","TZ=UTC","SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S","sacct",
                "-nP","-j",opts.parent,"--format",ACCOUNTING]
    print(json.dumps(summary,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
