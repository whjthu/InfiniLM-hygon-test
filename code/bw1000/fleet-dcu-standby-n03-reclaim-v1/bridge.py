"""Dedicated one-shot compute bridge for the n03 CPU-only timeout reclaim."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time


RESULTS=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
HERE=Path(__file__).resolve().parent
PARENT,TARGET_PARENT,NODE,CPU_NODE="1103606","1194900","f13r4n03","f12r2n13"
PYTHON="/usr/local/python3.12/bin/python3"
BRIDGE_CAP=11520
MEMBERS={"bridge.py","controller.py","original_bridge.py","standby_controller.py","standby_worker.py",
         "collect_training80_dispatch_snapshot.py","training_dispatch_allocation.py","tuning_queue.py",
         "read_training_slurm_membership.py","sample_training80_busy.py","ledger.json","pci.json","historical-2443.json"}


def need(ok,code):
    if not ok:
        raise ValueError(code)


def pin(path,expected,cap=262144):
    need(path.resolve(strict=True)==path and stat.S_ISREG(path.lstat().st_mode),"EXACT_CANONICAL_STAGE_FILE")
    with path.open("rb") as stream:
        raw=stream.read(cap+1)
    need(0<len(raw)<=cap and sha256(raw).hexdigest()==expected,"EXACT_STAGE_MEMBER_SHA")
    return raw


def write(path,value):
    with path.open("x",encoding="utf8") as file:
        file.write(json.dumps(value,allow_nan=False,sort_keys=True)+"\n")
        file.flush()
        os.fsync(file.fileno())


def source(request,wanted):
    need(re.fullmatch("[0-9a-f]{32}",request) and request!="998fbfec65b4480f880aa3ebb39b630a" and
         HERE==RESULTS/f"training80-n03-reclaim-bridge-{request}" and
         HERE.stat().st_uid==os.getuid() and stat.S_IMODE(HERE.stat().st_mode)==0o700,
         "EXACT_NEW_OWNED_BRIDGE_SCOPE")
    manifest=json.loads(pin(HERE/"SOURCE_FILES.json",wanted,16384))
    need(set(manifest["files"])==MEMBERS and manifest["request"]==request and
         manifest["scope"]=="f13r4n03-original-189-cpu-only","EXACT_SINGLETON_SOURCE_CLOSURE")
    raw={}
    for name,data in manifest["files"].items():
        value=pin(HERE/name,data["SHA256"])
        need(len(value)==data["bytes"],"EXACT_SINGLETON_SOURCE_EXTENT")
        raw[name]=value
    return raw


def load(raw,name):
    env={"__name__":"pinned_"+name.removesuffix(".py"),"__file__":str(HERE/name)}
    exec(compile(raw[name],str(HERE/name),"exec"),env)
    return env


def runtime(raw):
    sys.path.insert(0,str(HERE))
    import collect_training80_dispatch_snapshot as collector
    import read_training_slurm_membership as membership
    import training_dispatch_allocation as allocation
    import sample_training80_busy
    import tuning_queue

    modules=(collector,membership,allocation,sample_training80_busy,tuning_queue)
    need(all(Path(mod.__file__)==HERE/(mod.__name__+".py") for mod in modules),"EXACT_COLLECTOR_PROVIDER_ORIGINS")
    kind,groups,owners=allocation.allocation(raw["ledger.json"])
    pins=json.loads(raw["pci.json"])
    sample_training80_busy.require_pins(owners,pins)
    need(kind==allocation.REPLACEMENT_KIND and owners[NODE][0]==TARGET_PARENT and owners[CPU_NODE][0]==PARENT,
         "EXACT_FROZEN_SINGLETON_AND_BRIDGE_NODE")
    return dict(groups=groups,pci=pins[NODE],allocation=allocation,membership=membership,collector=collector,
                standby=load(raw,"standby_controller.py"),worker_SHA256=sha256(raw["standby_worker.py"]).hexdigest())


def owner(raw,request,wanted,authority):
    value=json.loads(pin(HERE/"owner.json",authority,8192))
    publication=value["publication"]
    need(re.fullmatch("[0-9a-f]{40}",publication),"EXACT_SOURCE_PUBLICATION")
    controller=load(raw,"controller.py")
    controller["evidence"](raw["historical-2443.json"])
    pci=json.loads(raw["pci.json"])[NODE]
    controller["validate_owner"](value,request,publication,pci,time.time())
    return value


def control_argv(request,manifest_sha,authority):
    return ["srun","--jobid="+PARENT,"--job-name=training80-reclaim-n03-bridge-"+request[:12],
            "--exact","--immediate=8","--threads=1","--nodes=1","--ntasks=1","--nodelist="+CPU_NODE,
            "--cpus-per-task=2","--gres=none","--mem=2048M","--time=03:12:00","--kill-on-bad-exit=1",
            PYTHON,"-I","-S","-B",str(HERE/"bridge.py"),"compute",request,manifest_sha,authority]


def start(request,manifest_sha,authority):
    need(not any(os.environ.get(key) for key in ("SLURM_OVERLAP","SRUN_OVERLAP")),"NO_INHERITED_STEP_OVERLAP")
    raw=source(request,manifest_sha)
    approve=owner(raw,request,manifest_sha,authority)
    need(not os.path.lexists(RESULTS/f"training80-dcu-standby-{request}"),"NO_PRIOR_STANDBY_INTENT")
    ctrl=load(raw,"controller.py")
    ctrl["preexisting"](ctrl["live_query"],RESULTS/f"training80-dcu-standby-{request}",
        f"training80-dcu-standby-{request[:12]}-{NODE}",
        "training80-admit-"+sha256(f"{request}:same-lock-n03-pre".encode()).hexdigest()[:32])
    label="training80-reclaim-n03-bridge-"+request[:12]
    recent=ctrl["live_query"]("sacct","-nP","-S","2026-11-27T00:00:00","-j",PARENT,"-o","JobIDRaw,JobName%128")
    need(all(row.split("|")[1]!=label for row in recent),"NO_PRIOR_COMPUTE_LABEL")
    epoch=time.time()
    write(HERE/"intent.json",dict(request=request,manifest_sha=manifest_sha,authority=authority,epoch=epoch,
        deadline_epoch=epoch+BRIDGE_CAP,publication=approve["publication"],replay_allowed=False))
    argv=[PYTHON,"-I","-S","-B","-c","import os,sys,time;time.sleep(0.75);os.execvp(sys.argv[1],sys.argv[1:])",
          *control_argv(request,manifest_sha,authority)]
    with (HERE/"stdout.private").open("xb") as out,(HERE/"stderr.private").open("xb") as err:
        process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    try:
        ticks=int(Path(f"/proc/{process.pid}/stat").read_text().rsplit(")",1)[1].split()[19])
    except FileNotFoundError:
        ticks=None
    write(HERE/"launcher.json",dict(request=request,pid=process.pid,startticks=ticks,control_label=label))
    print(json.dumps(dict(request=request,bridge_launch_pending=True,control_label=label,planned_DCU=0)),flush=True)
    return 0


def compute(request,manifest_sha,authority):
    raw=source(request,manifest_sha)
    approve=owner(raw,request,manifest_sha,authority)
    record=json.loads((HERE/"intent.json").read_bytes())
    step=os.environ.get("SLURM_STEP_ID","")
    cgroups=Path("/proc/self/cgroup").read_text().splitlines()
    need((record["request"],record["manifest_sha"],record["authority"],record["publication"])==
         (request,manifest_sha,authority,approve["publication"]) and record["replay_allowed"] is False and
         0<=time.time()-record["epoch"]<=65 and record["deadline_epoch"]==record["epoch"]+BRIDGE_CAP and
         socket.gethostname().split(".")[0]==CPU_NODE and os.environ.get("SLURM_JOB_ID")==PARENT and
         re.fullmatch("[0-9]+",step) and not os.environ.get("SLURM_STEP_GPUS") and
         os.environ.get("SLURM_GPUS_ON_NODE") in (None,"","0") and
         any({"job_"+PARENT,"step_"+step}<=set(line.split(":")[-1].split("/")) for line in cgroups),
         "EXACT_ZERO_GRES_COMPUTE_BRIDGE_CONTEXT")
    controller=load(raw,"controller.py")
    handle=PARENT+"."+step
    rows=controller["live_query"]("scontrol","show","step",handle,"--oneliner")
    records=[dict(term.split("=",1) for term in row.split() if "=" in term) for row in rows]
    [current]=[row for row in records if row.get("StepId")==handle]
    need((current["Name"],current["State"],current["NodeList"],current["TRES"])==
         ("training80-reclaim-n03-bridge-"+request[:12],"RUNNING",CPU_NODE,"cpu=2,mem=2G,node=1"),
         "EXACT_BRIDGE_RUNNING_ZERO_GRES_RESOURCES")
    for key in tuple(os.environ):
        if key.startswith(("SLURM_","SRUN_","PMI_","PMIX_")):
            os.environ.pop(key,None)
    write(HERE/"compute.json",dict(request=request,node=CPU_NODE,step=handle,target=[NODE]))
    children=load(raw,"original_bridge.py")["StandbyChildren"]()
    try:
        output=controller["execute"](HERE,request,approve["publication"],approve,runtime(raw),children)
        print(json.dumps(output,sort_keys=True),flush=True)
    except (OSError,ValueError,subprocess.SubprocessError,KeyError,TypeError) as error:
        msg=str(error)
        write(HERE/"compute-error.json",dict(request=request,error=type(error).__name__,
              code=msg if re.fullmatch("[A-Z_0-9]{1,100}",msg) else None,
              GPU_launcher_children=len(children.children),replay_allowed=False))
    finally:
        exits=children.wait(record["deadline_epoch"]-10)
        write(HERE/"child-terminal.json",dict(request=request,children=len(exits),observed_launcher_exit_codes=exits))
    return 0 if len(exits)==1 and exits==[0] and not (HERE/"compute-error.json").exists() else 2


if __name__=="__main__":
    os.umask(0o077)
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase",choices=("start","compute"))
    cli.add_argument("request")
    cli.add_argument("manifest_SHA256")
    cli.add_argument("owner_SHA256")
    args=cli.parse_args()
    raise SystemExit((start if args.phase=="start" else compute)(args.request,args.manifest_SHA256,args.owner_SHA256))
