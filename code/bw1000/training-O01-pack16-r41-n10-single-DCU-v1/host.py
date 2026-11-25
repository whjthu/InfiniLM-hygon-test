"""One new Pack16 n10 owner. Keep the ordinary lock only until numeric-step visibility."""

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from gate import APPTAINER, HEX, NODE, PARENT, PCI, ROOT, RUNTIME_SHA, canonical, digest, label, need, read, resources, save


UID=154459
HOST_PY="/usr/local/python3.12/bin/python3"
RESULTS=ROOT/"results"
RUNS=RESULTS/'training-O01-pack16-r41-n10-single-DCU-v1'
LOCK=RESULTS/".training80-dcu-standby.lock"
SITE=RESULTS/"training-t06-sdxl-runtime-v1"
SANDBOX=ROOT/"containers/dtk26-sandbox-r1"
OLD_EMPTY=None
OLD_SAMPLE='f9ce84341eb8f879b38b0166ace3d1a89b32bca81491c4f975c0517cd3f17726'
PRIOR=None
PREVIOUS_STEP="1103606.3352"
PREVIOUS_LABEL="training80-dcu-standby-e26b0d27c777-f12r2n10"
PREVIOUS_WORK_SECONDS=10547
CLOSED_STEPS={"1103606.3405","1103606.3407","1103606.3423"}
COLS="JobIDRaw,JobName%110,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"
TERMINAL={"COMPLETED","FAILED","CANCELLED","TIMEOUT","NODE_FAIL","OUT_OF_MEMORY","PREEMPTED","BOOT_FAIL","DEADLINE","REVOKED"}
INTENT="GPU_INTENT.json"


def utc(value):
    return datetime.strptime(value,"%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def only(rows):
    need(len(rows)==1,"N10_EXACTLY_ONE_SCHEDULER_ROW")
    return rows[0]


def fields(row):
    return dict(part.split("=",1) for part in row.split() if "=" in part)


def accounting(query,job):
    values=[line.split("|") for line in query("sacct","-nP","-j",job,"-o",COLS)]
    need(all(len(row)==9 for row in values),"N10_EXACT_NINE_ACCOUNTING_COLUMNS")
    return values


def cpu_observation(payload,query,now,after=None):
    need(payload["parent"]==PARENT and payload["node"]==NODE and payload["zero_GRES_invoked"] is True
         and payload["GPU_steps_requested"]==0 and payload["exclusivity_lease"] is False and payload["launch_rc"]==0
         and payload["state"]=="ORIGINAL_ZERO_GRES_PHYSICAL_SAMPLE_CAPTURED"
         and payload["other_UID_device_FDs"] in (0,"UNOBSERVED") and HEX.fullmatch(payload["boot_SHA256"]),
         "N10_PHYSICAL_ONLY_NO_LEASE")
    request=payload["request"]
    step=payload["step"]
    need(step not in CLOSED_STEPS,"N10_NEW_PHYSICAL_NUMERIC_STEP")
    row=only(payload["Slurm_accounting_observed"])
    live=only([item for item in accounting(query,step) if item[0]==step])
    need(live==row and row[:4]==[step,"training80-o01-n10-physical-"+request[:12],"COMPLETED","0:0"]
         and re.fullmatch(r"1103606\.[0-9]+",step) and step!=PRIOR[0]
         and not any(token in row[6].lower() for token in ("gpu","dcu","gres/"))
         and row[5]==NODE and row[4].isdecimal() and 0<=int(row[4])<=60,"N10_PHYSICAL_ZERO_GRES_C0")
    first,second=payload["sample"]
    pcis=first["PCIs"]
    need(len(pcis)==len(set(pcis))==8 and all(PCI.fullmatch(pci) for pci in pcis)
         and pcis==second["PCIs"]
         and all(item["kfd_process_count"]==item["own_device_fd_process_count"]==0
                 and not item["occupied_requested_ports"] for item in (first,second)),"N10_TWO_CLEAN_SAME_BOOT_SAMPLES")
    end=utc(row[8])
    if after is None:
        need(0<=now-end<=60,"N10_STALE_OR_FUTURE_ZERO_GRES_PHYSICAL_SAMPLE")
    else:
        need(utc(row[7])>=after and end<=now,"N10_POST_INDEPENDENT_AND_AFTER_GPU")
    return {"step":step,"PCIs":pcis,"boot_SHA256":payload["boot_SHA256"],"accounting_end":end,
            "other_UID_FD_coverage":payload["other_UID_device_FDs"]}


def idle(query,wanted,now):
    def hosts(spec):
        need(spec not in ("","(null)","Unknown","N/A"),"N10_EXACT_SCHEDULER_NODELIST")
        if re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}",spec):
            return {spec}
        return set(query("scontrol","show","hostnames",spec))
    parent=fields(only(query("scontrol","show","job",PARENT,"-o")))
    active_parent,st,where=only(query("squeue","-h","-w",NODE,"-o","%i|%T|%N")).split("|")
    need(parent["JobId"]==active_parent==PARENT and parent["JobState"]==st=="RUNNING"
         and parent["UserId"].endswith(f"({UID})") and NODE in hosts(parent["NodeList"]) and NODE in hosts(where)
         and "gres/dcu=320" in parent["AllocTRES"].split(",") and utc(parent["EndTime"])>=now+240,
         "N10_EXACT_OWNED_PARENT_AND_FULL_WINDOW")
    previous=only([row for row in accounting(query,PRIOR[0]) if row[0]==PRIOR[0]])
    need(previous==PRIOR,"N10_EXACT_ORIGINAL_3352_C0")
    scopes=[]
    for line in query("scontrol","-d","show","job",PARENT):
        for nodes,cpus,mem in re.findall(r"\bNodes=(\S+)\s+CPU_IDs=(\S+)\s+Mem=(\S+)",line):
            if NODE in hosts(nodes):
                scopes.append((cpus,mem))
    cpus,mem=only(scopes)
    slots=set()
    for atom in cpus.split(","):
        match=re.fullmatch(r"([0-9]+)(?:-([0-9]+))?",atom)
        need(match is not None,"N10_PARENT_CPU_ID_MAP")
        a,b=int(match[1]),int(match[2] or match[1])
        need(0<=a<=b<=16384,"N10_PARENT_CPU_ID_MAP")
        slots.update(range(a,b+1))
    m=re.fullmatch(r"([0-9]+)([MGT]?)",mem)
    need(m is not None and len(slots)>=4 and int(m[1])*{"":1,"M":1,"G":1024,"T":1048576}[m[2]]>=8192,
         "N10_ALLOCATED_CPU_MEMORY")
    active=[]
    for line in query("scontrol","show","step",PARENT,"-o"):
        row=fields(line)
        if row.get("State","").rstrip("+") not in TERMINAL:
            active.append((row["StepId"],row.get("Name",""),row["NodeList"]))
    active.extend(tuple(line.split("|")) for line in query("squeue","--steps","-h","-j",PARENT,"-o","%i|%j|%N"))
    for step,name,node_list in active:
        need(name!=wanted and re.fullmatch(PARENT+r"\.(?:[0-9]+|batch|extern)",step),"N10_UNCLASSIFIED_STEP_OR_OLD_LABEL")
        if step.rsplit(".",1)[1] not in ("batch","extern"):
            need(NODE not in hosts(node_list),"N10_NOT_EMPTY_IN_BOTH_SCHEDULER_VIEWS")
    history=accounting(query,PARENT)
    need(not any(row[1]==wanted for row in history),"N10_NO_EXISTING_EXACT_LABEL")
    need(not any(re.fullmatch(r"1103606\.[0-9]+",row[0]) and row[5]==NODE
         and row[2].split()[0].rstrip("+") not in TERMINAL for row in history),"N10_ACCOUNTING_ACTIVE_STEP_VETO")


def runtime_closure(runtime):
    mf=json.loads(read(runtime/"SCOPED_FILES.json",8192,RUNTIME_SHA))
    need(set(mf["files"])=={"baseline.py","contract.py","hipprof_trace.py","candidate.py","worker.py"},
         "N10_SIX_FROZEN_PACK16_MEMBERS")
    for name,item in mf["files"].items():
        raw=read(runtime/name,65536,item["SHA256"])
        need(len(raw)==item["bytes"],"N10_FROZEN_PACK16_MEMBER_LENGTH")


def authority(a,owner,pub,now):
    need(a["schema"]=="O01_R41_N10_NEW_DIRECT_OWNER_V1" and (a["request"],a["publication"],a["node"],a["parent"])==
         (owner,pub,NODE,PARENT) and a["previous_step"]==PRIOR[0] and a["historical_release_SHA256"]==OLD_EMPTY
         and a["historical_zero_GRES_SHA256"]==OLD_SAMPLE
         and a["root_approved_one_DCU"] is True and a["sole_dispatcher_n10_until_independent_post"] is True
         and a["automatic_ordinary_refill_allowed"] is False and a["borrowed_ordinary_grant"] is False
         and 0<=now-a["observed_epoch"]<=60 and now+240<=a["valid_until_epoch"]
         and "PCI" not in a,"N10_ROOT_NEW_DIRECT_OWNER_AND_ORDINARY_EXCLUSION")


def argv(scope,intent_sha,root,runtime,gate_path):
    out=root/"compute"
    intent=root/INTENT
    inside=["python3","-B","-u","/pack16-r29-gate.py","--inside","--intent","/pack16-r29-intent.json",
            "--intent-sha256",intent_sha,"--out","/pack16-output"]
    shell=("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; unset PYTHONPYCACHEPREFIX ROCBLAS_LAYER "
           "HIPBLASLT_LOG_LEVEL HIPBLASLT_LOG_MASK; export PYTHONPATH=/training-runtime/site "
           "PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "
           "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; exec "+shlex.join(inside))
    args=["srun","--jobid="+PARENT,"--job-name="+scope["label"],"--exact","--immediate=5","--nodes=1","--ntasks=1",
          "--nodelist="+NODE,"--cpus-per-task=4","--threads-per-core=1","--cpu-bind=threads","--mem=8192M","--gres=dcu:1",
          "--time=00:04:00","--kill-on-bad-exit=1",HOST_PY,"-I","-S","-B",str(gate_path),
          "--intent",str(intent),"--intent-sha256",intent_sha,"--out",str(out),"--",APPTAINER,"exec","--rocm","--no-home","--pwd","/"]
    for src,dst,mode in ((runtime,"/pair-package","ro"),(gate_path,"/pack16-r29-gate.py","ro"),
                         (intent,"/pack16-r29-intent.json","ro"),(out,"/pack16-output","rw"),
                         (SITE,"/training-runtime","ro"),("/usr/local/hyhal","/opt/hyhal","ro")):
        args.extend(["--bind",f"{src}:{dst}:{mode}"])
    return args+[str(SANDBOX),"/bin/bash","--noprofile","--norc","-c",shell]


def launch(args,root,runtime,gate_path,query,clock,popen):
    import fcntl
    need(not (root/INTENT).exists() and not any((root/"compute").iterdir()),"N10_EXISTS_RECONCILE_ONLY")
    auth=json.loads(read(args.authority,16384,args.authority_pin))
    authority(auth,args.request,args.publication,clock())
    pre_path=RESULTS/("training80-o01-n10-physical-"+auth["pre_request"])/"accounting-read.json"
    need(re.fullmatch(r"[0-9a-f]{32}",auth["pre_request"]) and HEX.fullmatch(auth["pre_sha256"]),"N10_EXACT_PRE_PROBE_REFERENCE")
    pre=json.loads(read(pre_path,16384,auth["pre_sha256"]))
    need(pre["request"]==auth["pre_request"]=="17518359ac254a4b979220fb17953bf6" and pre["physical_mode"]=="pre"
         and pre["owner_request"]==args.request,"N10_FRESH_PHYSICAL_PROBE_REQUEST")
    child=None
    seen=None
    with LOCK.open("rb") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        standby=cpu_observation(pre,query,clock())
        wanted=label(args.request)
        idle(query,wanted,clock())
        epoch=clock()
        authority(auth,args.request,args.publication,epoch)
        need(0<=epoch-standby["accounting_end"]<=45,"N10_PHYSICAL_FRESHNESS_AT_INTENT")
        scope={"request":args.request,"publication":args.publication,"label":wanted,"parent":PARENT,"node":NODE,
               "epoch":epoch,"deadline_epoch":epoch+240,"runtime_SHA256":RUNTIME_SHA,"pre_SHA256":auth["pre_sha256"],
               "root_authority_SHA256":args.authority_pin,"pre_step":standby["step"],"PCIs":standby["PCIs"],
               "boot_SHA256":standby["boot_SHA256"],"only_new_gpu_srun":1,"ordinary_refill_allowed":False,
               "other_UID_FD_coverage":standby["other_UID_FD_coverage"]}
        pin=save(root/INTENT,scope)
        command=argv(scope,pin,root,runtime,gate_path)
        save(root/"ARGV_PIN.json",{"SHA256":digest(canonical(command)),"request":args.request})
        with (root/"GPU.stdout").open("xb") as out,(root/"GPU.stderr").open("xb") as err:
            try:
                child=popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=shell_env())
                save(root/"SPAWN.json",{"request":args.request,"client_PID":child.pid,"intent_SHA256":pin})
                observe_end=time.monotonic()+8
                while time.monotonic()<observe_end:
                    matches=[fields(row) for row in query("scontrol","show","step",PARENT,"-o")
                             if "Name="+wanted+" " in row+" "]
                    if matches:
                        item=only(matches)
                        resources(item["TRES"])
                        need(re.fullmatch(r"1103606\.[0-9]+",item["StepId"]) and item["NodeList"]==NODE,
                             "N10_ACTIVE_SPAWN_IDENTITY")
                        seen=item["StepId"]
                        break
                    if child.poll() is not None:
                        break
                    time.sleep(.2)
            except (OSError,ValueError,subprocess.SubprocessError):
                pass
    # A missing visible step is UNKNOWN. The durable owner and ordinary exclusion
    # are held by the current coordinator, and no entry ever resubmits.
    client_timeout=False
    if child is None:
        rc=None
    else:
        try:
            rc=child.wait(timeout=max(.1,epoch+240-clock()-8))
        except subprocess.TimeoutExpired:
            client_timeout=True
            rc=None
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
    report={"request":args.request,"state":"ONE_SRUN_RETURNED_AWAITING_INDEPENDENT_POST" if rc is not None else
            "ONE_SRUN_OUTCOME_UNKNOWN_AWAITING_RECONCILIATION","visible_step_hint":seen,"srun_rc":rc,
            "intent_SHA256":pin,"srun_client_timed_out":client_timeout,
            "replay":False,"refill_permitted":False,"other_devices_claimed":0}
    save(root/"HOST_RETURN.json",report)
    return report


def shell_env():
    env=dict(os.environ,TZ="UTC",SLURM_TIME_FORMAT="%Y-%m-%dT%H:%M:%S")
    return {key:value for key,value in env.items() if not key.endswith("PYTHONPYCACHEPREFIX")}


def terminal(root,owner,query):
    value=json.loads(read(root/INTENT,16384))
    need(value["request"]==owner and value["label"]==label(owner),"N10_EXACT_ORIGINAL_OWNER")
    matches=[row for row in accounting(query,PARENT) if row[1]==value["label"]]
    need(len(matches)<=1,"N10_UNIQUE_NUMERIC_GPU_RESULT")
    if not matches:
        return {"state":"NO_AUTHORITATIVE_NUMERIC_STEP_UNKNOWN","refill_permitted":False,"replay":False}
    [row]=matches
    resources(row[6])
    need(re.fullmatch(r"1103606\.[0-9]+",row[0]) and row[0] not in CLOSED_STEPS|{PRIOR[0],value["pre_step"]}
         and row[5]==NODE,"N10_EXACT_NUMERIC_GPU_RESULT")
    final=row[2].split()[0].rstrip("+") in TERMINAL
    if final:
        need(utc(row[7])>=value["epoch"]-1 and utc(row[8])<=value["deadline_epoch"]+1
             and row[4].isdecimal() and 0<=int(row[4])<=240 and utc(row[7])<=utc(row[8])
             and abs(utc(row[8])-utc(row[7])-int(row[4]))<=1,"N10_GPU_RESULT_EPOCH")
    return {"state":"AUTHORITATIVE_GPU_TERMINAL_AWAITING_INDEPENDENT_POST" if final else "EXACT_STEP_ACTIVE",
            "row":row,"refill_permitted":False,"replay":False}


def post_release(root,owner,post_request,post_sha,query,now):
    result=terminal(root,owner,query)
    need(result["state"]=="AUTHORITATIVE_GPU_TERMINAL_AWAITING_INDEPENDENT_POST",
         "N10_ACTUAL_GPU_TERMINAL_BEFORE_POST")
    intent=json.loads(read(root/INTENT,16384))
    need(re.fullmatch(r"[0-9a-f]{32}",post_request) and HEX.fullmatch(post_sha),
         "N10_PRIVATE_INDEPENDENT_POST_IDENTITY")
    path=RESULTS/("training80-o01-n10-physical-"+post_request)/"accounting-read.json"
    raw=read(path,16384,post_sha)
    evidence=json.loads(raw)
    need(evidence["request"]==post_request=="0ee7849706d24ada90d2ee4d43e3bcc1" and evidence["physical_mode"]=="post"
         and evidence["owner_request"]==owner,"N10_EXACT_REQUEST_POST")
    ended=utc(result["row"][8])
    post=cpu_observation(evidence,query,now,ended)
    need(post["step"] not in (intent["pre_step"],result["row"][0])
         and post["boot_SHA256"]==intent["boot_SHA256"] and post["PCIs"]==intent["PCIs"],
         "N10_SAME_BOOT_AND_EIGHT_BOARDS_AFTER_TERMINAL")
    return {"state":"TERMINAL_AND_INDEPENDENT_PHYSICAL_CLEANUP_VERIFIED","row":result["row"],
            "post_step":post["step"],"post_SHA256":post_sha,"other_UID_FD_coverage":post["other_UID_FD_coverage"],
            "refill_permitted":True,"replay":False,"component_performance_claimed":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("launch","terminal","post"))
    for name in ("request","publication"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--authority",type=Path)
    p.add_argument("--authority-pin")
    p.add_argument("--runtime",type=Path)
    p.add_argument("--source-manifest-sha256")
    p.add_argument("--post-request")
    p.add_argument("--post-sha256")
    a=p.parse_args()
    need(os.getuid()==UID and HEX.fullmatch(a.source_manifest_sha256 or "")
         and re.fullmatch(r"[0-9a-f]{40}",a.publication) and label(a.request),"N10_ROOT_AUTHENTICATED_EXACT_SCOPE")
    root=RUNS/a.request
    need(root.resolve(strict=True)==root,"N10_EXACT_SCOPED_TARGET_STAGE")
    manifest=json.loads(read(root/"src/SOURCE_FILES.json",16384,a.source_manifest_sha256))
    need(set(manifest["files"])=={"host.py","gate.py","step_evidence.py"},"N10_EXACT_NEW_WRAPPER_SOURCE")
    for name,item in manifest["files"].items():
        need(len(read(root/"src"/name,65536,item["SHA256"]))==item["bytes"],"N10_SOURCE_MEMBER_EXTENT")
    need(Path(__file__).resolve(strict=True)==root/"src/host.py","N10_EXACT_RUNNING_SOURCE")
    def query(*cmd):
        need(cmd[0] in ("sacct","scontrol","squeue"),"N10_READ_ONLY_SCHEDULER_QUERY")
        result=subprocess.check_output(cmd,stdin=subprocess.DEVNULL,timeout=4,env=shell_env())
        need(len(result)<=1048576,"N10_BOUNDED_READ_ONLY_QUERY")
        return result.decode("ascii").splitlines()
    global PRIOR, OLD_EMPTY
    previous_raw=read(root/"PREDECESSOR.json",16384)
    release=json.loads(previous_raw)
    PRIOR=release["incumbent_accounting"]
    need(release["old_counter_SHA256"]==OLD_SAMPLE and release["owner_request"]==a.request
         and release["publication"]==a.publication and PRIOR[:4]==[PREVIOUS_STEP,PREVIOUS_LABEL,"COMPLETED","0:0"]
         and len(PRIOR)==9 and PRIOR[4].isdecimal() and int(PRIOR[4])>=PREVIOUS_WORK_SECONDS
         and PRIOR[5:7]==[NODE,"cpu=8,gres/dcu=8,mem=16G,node=1"],
         "N10_EXACT_3352_NATURAL_RELEASE_CARRIER")
    OLD_EMPTY=digest(previous_raw)
    if a.mode=="launch":
        need(a.authority and a.runtime and HEX.fullmatch(a.authority_pin or "")
             and a.authority==root/"ROOT_DIRECT_AUTHORITY.json" and a.runtime.is_relative_to(ROOT/"incoming")
             and a.runtime.resolve(strict=True)==a.runtime,"N10_EXACT_PUBLICATION_AUTHORITY_AND_OLD_RUNTIME")
        runtime_closure(a.runtime)
        os.umask(0o077)
        output=launch(a,root,a.runtime,root/"src/gate.py",query,time.time,subprocess.Popen)
    elif a.mode=="terminal":
        output=terminal(root,a.request,query)
    else:
        output=post_release(root,a.request,a.post_request,a.post_sha256,query,time.time())
    print(json.dumps(output,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    try:
        sys.exit(main())
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as error:
        token=str(error) if isinstance(error,ValueError) else ""
        print(json.dumps({"state":"BLOCKED_OR_RECONCILE","type":type(error).__name__,
                          "gate":token if re.fullmatch(r"N10_[A-Z0-9_]+",token) else None,
                          "replay":False,"refill_permitted":False},sort_keys=True),flush=True)
        sys.exit(2)
