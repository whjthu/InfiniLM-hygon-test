"""One zero-GRES step in the exact original DTK image, with independent accounting."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time

from worker import (CPU, FRAMEWORK, FRAMEWORK_MANIFEST, MEM_LIMIT_MIB, NATIVE, NATIVE_ALIAS,
                    NODE, ORIGINAL, PARENT, ROOT, digest, need, read, save)

HERE=Path(__file__).resolve().parent
LABEL="t09-abo-4560-full-cpu-reload-tres-v2"
PY="/usr/local/python3.12/bin/python3"
APPTAINER="/public/software/apps/apptainer/1.3.4/bin/apptainer"
STAGE=ROOT/"incoming/t09-private-framework393-ingress-v1/b1a692f520b94308a36818d2e1db252"
RUNTIME=ROOT/"results/training-t06-sdxl-runtime-v1"
STANDBY="1103606.4564"
STANDBY_LABEL="training80-dcu-standby-4e26863e6e6c-f13r3n14"
STANDBY_TRES={"cpu=8","gres/dcu=8","mem=16G","node=1"}
COLUMNS="JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"
MEM={f"mem={MEM_LIMIT_MIB}M","mem=62.50G"}
ORIGINAL_FAILED_REQUEST="8775fd46a6da47839fb405f986265812"
TERMINAL={"COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED","BOOT_FAIL"}
CAP=480
V3_NAME="training-hunyuanvideo-t09-abo-domain-n14-fd-v3"
HISTORICAL_ROWS=(
    ["1103606.4560",V3_NAME+"-gpu","COMPLETED","0:0","251",NODE,
     "cpu=8,gres/dcu=1,mem=125G,node=1","2026-11-27T11:44:50","2026-11-27T11:49:01"],
    ["1103606.4561",V3_NAME+"-post","COMPLETED","0:0","0",NODE,
     "cpu=1,mem=1G,node=1","2026-11-27T11:49:03","2026-11-27T11:49:03"])


def scope(request):
    need(re.fullmatch("[0-9a-f]{32}",request)
         and request not in ("1b22ddd9e5854853a43e646807e92b3f",ORIGINAL_FAILED_REQUEST),"NEW_CPU_RELOAD_REQUEST")
    return ROOT/"incoming"/(LABEL+"-"+request), LABEL+"-"+request[:12]


def query(*argv):
    out=subprocess.run(argv,check=True,capture_output=True,timeout=8,env=dict(os.environ,TZ="UTC",LC_ALL="C"))
    need(len(out.stdout)<=2*1024**2 and len(out.stderr)<=65536,"BOUNDED_SCHEDULER_OUTPUT")
    return out.stdout.decode("ascii").splitlines()


def accounting(handle, runner=query):
    rows=[x.split("|") for x in runner("sacct","-nP","-j",handle,"-o",COLUMNS)]
    need(all(len(row)==9 for row in rows),"NINE_UTC_ACCOUNTING_COLUMNS")
    return rows


def fields(raw):
    return dict(x.split("=",1) for x in raw.split() if "=" in x)


def exact_tres(text):
    if type(text) is not str:
        return False
    tokens=text.split(",")
    return len(tokens)==len(set(tokens))==3 and any(
        set(tokens)=={f"cpu={CPU}",memory,"node=1"} for memory in MEM)


def controller_resources(row, env):
    raw=row.get("TRES",row.get("AllocTRES",""))
    need(type(raw) is str and len(raw)<=256,"BOUNDED_CONTROLLER_TRES")
    return dict(raw_controller_TRES=raw,controller_TRES_exact=exact_tres(raw),
                step_GPUs_empty=not env.get("SLURM_STEP_GPUS"),
                GPUs_on_node_zero_or_unset=env.get("SLURM_GPUS_ON_NODE") in (None,"","0"))


def standby_tres(text):
    tokens=text.split(",")
    return len(tokens)==4 and set(tokens) in (STANDBY_TRES, (STANDBY_TRES-{"mem=16G"})|{"mem=16384M"})


def parent_gate(runner=query):
    def covers(spec):
        need(re.fullmatch(r"f[0-9rnf,\[\]-]{1,8192}",spec),"EXACT_JOB_NODESET")
        return NODE in runner("scontrol","show","hostnames",spec)
    j,=map(fields,runner("scontrol","show","job",PARENT,"--oneliner"))
    q,=runner("squeue","-h","-j",PARENT,"-o","%i|%T|%N")
    job,state,nodes=q.split("|")
    need(j["JobId"]==job==PARENT and j["JobState"]==state=="RUNNING"
         and j["UserId"].endswith(f"({os.getuid()})") and covers(j["NodeList"]) and covers(nodes),"EXACT_DUAL_OWNED_PARENT")
    active=[]
    for line in runner("scontrol","show","step",PARENT,"--oneliner"):
        row=fields(line)
        if re.fullmatch(PARENT+r"\.[0-9]+",row["StepId"]) and row["State"] not in TERMINAL and covers(row["NodeList"]):
            active.append(row)
    queue=[]
    for line in runner("squeue","--steps","-h","-j",PARENT,"-o","%i|%N"):
        handle,spec=line.split("|")
        if re.fullmatch(PARENT+r"\.[0-9]+",handle) and covers(spec):
            queue.append(handle)
    original,=[row for row in accounting(STANDBY,runner) if row[0]==STANDBY]
    need(original[1]==STANDBY_LABEL and original[5]==NODE and standby_tres(original[6]),"EXACT_STANDBY_ACCOUNTING")
    if active:
        row,=active
        need(row["StepId"]==STANDBY and row.get("Name",row.get("JobName"))==STANDBY_LABEL and row["NodeList"]==NODE
             and row["State"]==original[2]=="RUNNING" and original[3]=="0:0"
             and standby_tres(row.get("TRES",row.get("AllocTRES",""))) and queue in ([],[STANDBY]),
             "ONLY_EXACT_4564_CORESIDENT_OWNER")
        held_cpu,held_mem=8,16384
    else:
        need(not queue and original[2] in TERMINAL and original[8] not in ("","Unknown"),
             "STANDBY_UNRECONCILED_OR_OTHER_QUEUE_STEP")
        held_cpu=held_mem=0
    detailed=[fields(x) for x in runner("scontrol","-d","show","job",PARENT)]
    resources,=[x for x in detailed if {"Nodes","CPU_IDs","Mem"}<=set(x) and covers(x["Nodes"])]
    cpus=set()
    for atom in resources["CPU_IDs"].split(","):
        need(re.fullmatch(r"[0-9]{1,5}(?:-[0-9]{1,5})?",atom),"PARENT_CPU_ID_SYNTAX")
        number=atom.split("-")
        left,right=int(number[0]),int(number[-1])
        span=set(range(left,right+1))
        need(0<=left<=right<=16384 and not cpus&span,"UNIQUE_CPU_SLOTS")
        cpus.update(span)
    mem=re.fullmatch(r"([0-9]{1,12})([MG]?)",resources["Mem"])
    need(mem is not None and len(cpus)-held_cpu>=CPU
         and int(mem[1])*(1024 if mem[2]=="G" else 1)-held_mem>=MEM_LIMIT_MIB,"CPU_MEMORY_AFTER_EXACT_STANDBY")
    return original


def source(folder, expected):
    raw=read(folder/"SOURCE_FILES.json",32768,expected)
    files=json.loads(raw)["files"]
    need(set(files)=={"worker.py","host.py","check_local.py","HANDOFF.md"},"EXACT_RELOAD_SOURCE_SET")
    member={n:read(folder/n,p["bytes"],p["SHA256"]) for n,p in files.items()}
    need(all(len(member[n])==p["bytes"] for n,p in files.items()),"EXACT_SOURCE_MEMBER_EXTENT")
    return member|{"SOURCE_FILES.json":raw}


def install(args):
    here,name=scope(args.request)
    members=source(HERE,args.manifest)
    need(re.fullmatch("[0-9a-f]{40}",args.publication),"EXACT_SOURCE_PUBLICATION")
    here.mkdir(mode=0o700)
    for name_dir in ("src","home","tmp","var-tmp"):
        (here/name_dir).mkdir(mode=0o700)
    for n,raw in members.items():
        with (here/"src"/n).open("xb") as handle:
            handle.write(raw);handle.flush();os.fsync(handle.fileno())
    return dict(request=args.request,scope=name,pin=save(here/"STAGE.json",dict(request=args.request,
                 publication=args.publication,manifest=args.manifest,actual_GPU=0,actual_Slurm_steps=0)))


def job_argv(here, name, request, manifest):
    return ["srun","--jobid="+PARENT,"--job-name="+name,"--exact","--immediate=5","--nodes=1","--ntasks=1",
            "--nodelist="+NODE,"--cpus-per-task="+str(CPU),"--mem="+str(MEM_LIMIT_MIB)+"M","--gres=none","--time=00:08:00",
            "--kill-on-bad-exit=1",PY,"-B",str(here/"src/host.py"),"compute","--request",request,"--manifest",manifest]


def container(here, request, gate):
    argv=[APPTAINER,"exec","--rocm","--no-mount","home,cwd","--pwd","/"]
    original_elf=ORIGINAL/"miopen-private/provider/libMIOpen.so.1.0"
    binds=((ROOT,ROOT,"ro"),(RUNTIME,"/training-runtime","ro"),(NATIVE,NATIVE_ALIAS,"ro"),
           (ORIGINAL/"overlay/flash_attn_no_pad.py",NATIVE_ALIAS/"hyvideo/utils/flash_attn_no_pad.py","ro"),
           (STAGE/"framework",FRAMEWORK,"ro"),(STAGE/"FRAMEWORK_FILES.json",FRAMEWORK_MANIFEST,"ro"),
           (STAGE,"/t09-stage-ro","ro"),("/opt/hyhal","/opt/hyhal","ro"),
           (original_elf,"/opt/dtk/lib/libMIOpen.so.1.0","ro"),(here,here,"rw"),(here/"src",here/"src","ro"),
           (here/"home","/cpu-reload-home","rw"),(here/"tmp","/tmp","rw"),(here/"var-tmp","/var/tmp","rw"))
    for source_path,target,mode in binds:
        argv.extend(("--bind",f"{source_path}:{target}:{mode}"))
    options=("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh >&2; unset PYTHONPYCACHEPREFIX; "
             "export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; "
             "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/training-runtime/site; "
             "export HOME=/cpu-reload-home OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 "
             "TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1; exec ")
    worker=shlex.join(["python3","-B","-u",str(here/"src/worker.py"),"--request",request,
                      "--gate",str(here/"compute-gate.json"),"--gate-sha",gate,"--output",str(here/"result.json")])
    return [*argv,str(ROOT/"containers/dtk26-sandbox-r1"),"/bin/bash","--noprofile","--norc","-c",options+worker]


def compute(args):
    here,name=scope(args.request)
    source(here/"src",args.manifest)
    intent=json.loads(read(here/"intent.json",32768))
    step=os.environ.get("SLURM_STEP_ID","")
    need(os.environ.get("SLURM_JOB_ID")==PARENT and step.isdecimal() and socket.gethostname().split(".")[0]==NODE
         and intent["request"]==args.request and intent["manifest"]==args.manifest and time.time()<intent["epoch"]+CAP,
         "EXACT_RELOAD_COMPUTE_SCOPE")
    need(not any(os.environ.get(x) for x in ("APPTAINER_BIND","APPTAINER_BINDPATH","SINGULARITY_BIND","SINGULARITY_BINDPATH",
                                            "LD_PRELOAD","LD_AUDIT","LD_PROFILE")),"UNEXPECTED_CONTAINER_OR_LOADER_ENV")
    handle=PARENT+"."+step
    owned,=[row for row in map(fields,query("scontrol","show","step",handle,"--oneliner")) if row.get("StepId")==handle]
    observed=controller_resources(owned,os.environ)
    save(here/"controller-tres-observation.json",dict(request=args.request,step=handle,**observed))
    need(owned.get("Name",owned.get("JobName"))==name and owned["State"]=="RUNNING" and owned["NodeList"]==NODE
         and observed["controller_TRES_exact"] and observed["step_GPUs_empty"] and observed["GPUs_on_node_zero_or_unset"],
         "EXACT_NUMERIC_ZERO_GRES")
    pin=save(here/"compute-gate.json",dict(request=args.request,step=handle,node=NODE,CPU=CPU,
             memory_MiB=MEM_LIMIT_MIB,zero_GRES=True,verified_epoch=time.time()))
    os.chdir("/")
    argv=container(here,args.request,pin["SHA256"])
    os.execv(argv[0],argv)


def start(args):
    here,name=scope(args.request)
    mark=json.loads(read(here/"STAGE.json",8192,args.stage_sha))
    auth=json.loads(read(args.authority,16384,args.authority_sha))
    now=time.time()
    need(mark["request"]==args.request and mark["manifest"]==args.manifest and mark["publication"]==args.publication
         and auth["request"]==args.request and auth["parent"]==PARENT and auth["node"]==NODE
         and auth["stage_SHA256"]==args.stage_sha and auth["manifest"]==args.manifest and auth["publication"]==args.publication
         and auth["root_GO"] is True and auth["CPU_GO"] is True and auth["GPU_GO"] is False and auth["zero_GRES_once"] is True
         and auth["allowed_standby"]==STANDBY and auth["observed_epoch"]<=now<auth["expires_epoch"]<=auth["observed_epoch"]+65,
         "EXACT_ROOT_CPU_ONLY_AUTHORITY")
    source(here/"src",args.manifest)
    for recorded in HISTORICAL_ROWS:
        step=recorded[0]
        need([r for r in accounting(step) if r[0]==step]==[recorded],"ORIGINAL_4560_4561_SETTLED")
    prior=parent_gate()
    need(not [r for r in accounting(PARENT) if r[1]==name] and time.time()<auth["expires_epoch"],"ORIGINAL_SCOPE_UNUSED_AND_FRESH")
    argv=job_argv(here,name,args.request,args.manifest)
    save(here/"intent.json",dict(request=args.request,epoch=time.time(),argv=argv,manifest=args.manifest,
                               authority_sha=args.authority_sha,standby_accounting=prior,planned_adapter_tensors=216,no_replay=True))
    outcome=dict(request=args.request,rc=None,state="RECONCILE_ORIGINAL_NUMERIC_STEP",planned_adapter_tensors=216)
    try:
        with (here/"worker.stdout.private").open("xb") as out,(here/"worker.stderr.private").open("xb") as err:
            outcome["rc"]=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=CAP+8).returncode
    except (OSError,subprocess.SubprocessError) as error:
        outcome["error_type"]=type(error).__name__
    finally:
        save(here/"owner-outcome.json",outcome)
    return outcome


def terminal(args):
    here,name=scope(args.request)
    rows=[r for r in accounting(PARENT) if r[1]==name]
    want=dict(request=args.request,qualified=False,state="UNKNOWN",accounting=rows,planned_adapter_tensors=216)
    need(len(rows)<=1,"ORIGINAL_CPU_ROW_DUPLICATE")
    if not rows or rows[0][2] not in TERMINAL:
        return want
    row=rows[0]
    need(re.fullmatch(PARENT+r"\.[0-9]+",row[0]) and row[5]==NODE and exact_tres(row[6])
         and row[4].isdigit() and int(row[4])<=CAP and row[7]<=row[8] and row[8] not in ("","Unknown"),
         "EXACT_TERMINAL_ZERO_GRES_ACCOUNTING")
    want["state"]="ORIGINAL_CPU_TERMINAL"
    want["raw_accounting_AllocTRES"]=row[6]
    observation=here/"controller-tres-observation.json"
    if observation.is_file():
        raw=read(observation,8192)
        controller=json.loads(raw)
        need(controller["request"]==args.request and controller["step"]==row[0]
             and controller["controller_TRES_exact"] is exact_tres(controller["raw_controller_TRES"]),
             "EXACT_CONTROLLER_OBSERVATION_SCOPE")
        want.update(controller_observation=controller,controller_observation_SHA256=digest(raw))
    if (here/"result.json").is_file():
        raw=read(here/"result.json",32768)
        worker=json.loads(raw)
        need(worker["request"]==args.request and worker.get("actual_step") in (None,row[0])
             and worker["planned_adapter_tensors"]==216,
             "EXACT_TERMINAL_WORKER_SCOPE")
        want.update(worker=worker,worker_SHA256=digest(raw),qualified=row[2:4]==["COMPLETED","0:0"]
                    and worker.get("actual_step")==row[0] and worker["passed"] is True)
    return want


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode",choices=("install","start","compute","terminal"))
    cli.add_argument("--request",required=True)
    for field in ("manifest","stage-sha","publication","authority-sha"):
        cli.add_argument("--"+field)
    cli.add_argument("--authority",type=Path)
    args=cli.parse_args()
    os.umask(0o077)
    result={"install":install,"start":start,"compute":compute,"terminal":terminal}[args.mode](args)
    print(json.dumps(result,allow_nan=False,sort_keys=True),flush=True)
