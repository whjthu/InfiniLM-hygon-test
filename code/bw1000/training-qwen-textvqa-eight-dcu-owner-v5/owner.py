"""Install handoff consumer and finite single-GPU-step .220/.1231 HOST owner."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import runpy
import shlex
import signal
import socket
import subprocess
import sys
import time

from contract import (APPTAINER, CPU_STEP_GRES, CPUS, DCU, FRAMEWORK, FRAMEWORK_SHA, GPU, IMPORT_CPUS,
    IMPORT_LABEL, IMPORT_MEMORY, IMPORT_MODULE_NAMES, IMPORT_SECONDS, INITIAL, INPUT, INSIDE, MEMORY, MODEL, NATIVE, PREFIX, PROCESSOR,
    PYHOST, REQUEST, RESULTS, SANDBOX, SHA64, STAGE, SWIFT, VISION, WHOLE, cache_prefix, digest, encode,
    import_scope, inventory, need, original_evidence, placement, save, small, utc)
from device import scope_module


COLUMNS="JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"
PHASES=("pre","gpu","post")
ACCOUNTING_SHA="15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b"


def query(*args):
    raw=subprocess.check_output(args,stdin=subprocess.DEVNULL,timeout=9,
        env=dict(os.environ,TZ="UTC",SLURM_TIME_FORMAT="%Y-%m-%dT%H:%M:%S"))
    need(len(raw)<=1048576,"EXACT_BOUNDED_SCHEDULER_READ")
    return raw.decode("ascii").splitlines()


def accounting(root):
    source=root/"src/accounting1231.py"
    namespace={"__name__":"_frozen_accounting"}
    exec(compile(small(source,ACCOUNTING_SHA),str(source),"exec"),namespace)
    return namespace


def config(run, source_sha):
    need(SHA64.fullmatch(source_sha) and run.is_dir() and run.resolve(strict=True)==run and not run.is_symlink(),
         "EXACT_OWNER_SOURCE_DIRECTORY")
    src=run/"src"
    raw=small(src/"PACKET.json",source_sha)
    pins=json.loads(raw)["files"]
    from packet_spec import MEMBERS
    need(set(pins)==set(MEMBERS) and
         {file.relative_to(src).as_posix() for file in src.rglob("*") if file.is_file()}=={"PACKET.json",*pins},
         "EXACT_SOURCE_ONLY_PACKET_MEMBERSHIP")
    for name,pin in pins.items():
        need(len(small(src/name,pin["SHA256"]))==pin["bytes"],"EXACT_SOURCE_ONLY_PACKET_FILE")
    return pins


def prepare_assets(run,source_sha):
    need(not (run/"assets-rehashed.json").exists() and not (run/"assets-rehashed.json").is_symlink(),
         "ASSET_VERIFICATION_ALREADY_RECORDED")
    config(run,source_sha)
    original_evidence(run/"src")
    python=sys.executable
    need(not any(p.is_symlink() for p in (Path(APPTAINER),SANDBOX)) and Path(APPTAINER).is_file() and SANDBOX.is_dir(),
         "EXACT_ORIGINAL_CONTAINER_PATH")
    # This runs only the existing read-only stage validator, in a fresh Python so
    # same-named helper modules from this outer owner cannot shadow the trainer.
    gate="import sys;sys.path.insert(0,sys.argv[1]);import prepare;assert len(prepare.stage_proof())==41"
    subprocess.run([python,"-I","-B","-X","pycache_prefix="+str(cache_prefix(run)),
                   "-c",gate,str(run/"src/trainer")],timeout=25,
                   stdin=subprocess.DEVNULL,check=True,capture_output=True)
    result={**inventory(run,hash_bytes=True),"source_manifest_SHA256":source_sha}
    save(run/"assets-rehashed.json",result)
    return {"state":result["state"],"model_files":14,"framework_files":393,
            "model_bytes":result["model_bytes"],"receipt_SHA256":digest(encode(result)),"GPU":0}


def verify_prepared_assets(run,source_sha,asset_sha):
    need(SHA64.fullmatch(asset_sha),"EXACT_ASSET_RECEIPT_SHA")
    prior=json.loads(small(run/"assets-rehashed.json",asset_sha,524288))
    now={**inventory(run,hash_bytes=False),"source_manifest_SHA256":source_sha}
    need(prior==now,"EXACT_ASSETS_SAME_HOST_AND_FILE_IDENTITIES")


def fields(line):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)",line))


def one(lines):
    need(len(lines)==1,"EXACT_ONE_READ_ONLY_SCHEDULER_ROW")
    return lines[0]


def hosts(query_scheduler, text):
    need(re.fullmatch(r"[A-Za-z0-9,\[\]._-]+",text) and text not in {"None","Unknown","null"},
         "EXACT_SCHEDULER_HOSTLIST")
    nodes=query_scheduler("scontrol","show","hostnames",text)
    need(nodes and len(nodes)==len(set(nodes)),"EXACT_SCHEDULER_HOST_SET")
    return set(nodes)


def memory(value):
    match=re.fullmatch(r"([0-9]+)([MGT]?)",value)
    need(match is not None,"EXACT_SLURM_MEMORY")
    return int(match[1])*{"":1,"M":1,"G":1024,"T":1048576}[match[2]]


def cpu_count(value):
    items=set()
    for raw in value.split(","):
        match=re.fullmatch(r"([0-9]+)(?:-([0-9]+))?",raw)
        need(match is not None,"EXACT_PARENT_CPU_IDS")
        low,high=int(match[1]),int(match[2] or match[1])
        need(0<=low<=high<=16384,"EXACT_PARENT_CPU_ID_RANGE")
        items.update(range(low,high+1))
    return len(items)


def allocation(intent, query_scheduler, now, uid):
    parent,node=intent["parent"],intent["node"]
    job=fields(one(query_scheduler("scontrol","show","job",parent,"-o")))
    p,live,nodes=one(query_scheduler("squeue","-h","-j",parent,"-o","%i|%T|%N")).split("|")
    membership=hosts(query_scheduler,job["NodeList"])
    need(job["JobId"]==p==parent and job["JobState"]==live=="RUNNING"
         and job["NumNodes"]=="40" and len(membership)==40 and node in membership
         and membership==hosts(query_scheduler,nodes) and "gres/dcu=320" in job["AllocTRES"].split(",")
         and job["UserId"].endswith(f"({uid})") and now<intent["deadline"]<=utc(job["EndTime"]),
         "EXACT_LIVE_OWNED_40_NODE_PARENT_AND_WINDOW")
    node_info=fields(one(query_scheduler("scontrol","show","node",node,"-o")))
    need(node_info["NodeName"]==node and int(node_info["CPUTot"])>=CPUS and int(node_info["RealMemory"])>=MEMORY
         and not re.search(r"DOWN|DRAIN|FAIL|MAINT|NO_RESPOND",node_info["State"]),"EXACT_8_DCU_NODE_CPU_MEMORY_STATE")
    scoped=[]
    for row in query_scheduler("scontrol","-d","show","job",parent):
        for names,cpus,mem in re.findall(r"\bNodes=(\S+)\s+CPU_IDs=(\S+)\s+Mem=(\S+)",row):
            if node in hosts(query_scheduler,names):
                scoped.append((cpus,mem))
    cpus,mem=one(scoped)
    need(cpu_count(cpus)>=CPUS and memory(mem)>=MEMORY,"EXACT_ALLOCATED_PARENT_CPU_AND_MEMORY")
    jobsteps=query_scheduler("scontrol","show","step",parent,"-o")
    need(jobsteps,"EXACT_ACTIVE_STEP_READ")
    active=[]
    for row in map(fields,jobsteps):
        if row.get("State") not in {"COMPLETED","FAILED","CANCELLED","TIMEOUT","NODE_FAIL"}:
            active.append((row["StepId"],row["NodeList"]))
    active.extend(row.split("|") for row in query_scheduler("squeue","--steps","-h","-j",parent,"-o","%i|%N"))
    for step,where in active:
        suffix=step.removeprefix(parent+".")
        if suffix in {"batch","extern"}:
            continue
        need(step==parent+"."+suffix and suffix.isdecimal() and node not in hosts(query_scheduler,where),
             "ACTIVE_OR_UNCLASSIFIED_STEP_CONFLICT")


def rows(intent, query_scheduler):
    names={intent["label"]+"-"+phase:phase for phase in PHASES}
    observed={}
    for raw in query_scheduler("sacct","-nP","-j",intent["parent"],"-o",COLUMNS):
        row=raw.split("|")
        if len(row)>1 and row[1] in names:
            phase=names[row[1]]
            need(len(row)==9 and phase not in observed and re.fullmatch(intent["parent"]+r"\.[0-9]+",row[0]),
                 "EXACT_UNAMBIGUOUS_NUMERIC_ACCOUNTING")
            observed[phase]=row
    return observed


def accountable(row, intent, phase):
    cpu,mem,gres=(CPUS,MEMORY,DCU) if phase=="gpu" else (2,4096,0)
    elements=row[6].split(",")
    tres=dict(item.split("=",1) for item in elements)
    expected={"gres/dcu":str(gres)} if gres else {}
    need(row[5]==intent["node"] and len(tres)==len(elements) and tres["cpu"]==str(cpu)
         and tres["node"]=="1" and memory(tres["mem"])==mem
         and {key:value for key,value in tres.items() if key.startswith("gres/") and value!="0"}==expected
         and 0<=int(row[4])<=intent["deadline"]-intent["epoch"]
         and intent["epoch"]-1<=utc(row[7])<=utc(row[8])<=intent["deadline"]+1,
         "EXACT_NUMERIC_STEP_ALLOCATION_AND_EPOCH")
    return row


def container(run,intent=None,probe=None):
    env=("/vision:/minicpm-transformers:/swift-runtime/site:"+str(FRAMEWORK)+":"+str(run/"src"))
    body=("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
          "unset PYTHONPYCACHEPREFIX; "
          "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1; "
          "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="+shlex.quote(env)+"; ")
    binds=[(run,run,"rw"),(FRAMEWORK,FRAMEWORK,"ro"),(STAGE,STAGE,"ro"),(MODEL,MODEL,"ro"),
           (INITIAL.parent,INITIAL.parent,"ro"),(INPUT,INPUT,"ro"),(VISION,"/vision","ro"),
           (SWIFT,"/swift-runtime","ro"),(PROCESSOR,"/minicpm-transformers","ro"),("/opt/hyhal","/opt/hyhal","ro")]
    argv=[APPTAINER,"exec"]
    if probe is None:
        argv.append("--rocm")
        body+=("LEFT=$(python3 -B -c "+shlex.quote(f"import time; print(int({intent['epoch']+INSIDE!r}-time.time()))")+
               '); test "$LEFT" -gt 60; exec /usr/bin/timeout --kill-after=10s "$LEFT" python3 -B -u '+
               shlex.join([str(run/"src/device.py"),str(run),"gpu"]))
    else:
        request,parent,node,source_sha=probe
        binds.append((run/"src",run/"src","ro"))
        body+=("export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=; "
               "exec /usr/bin/timeout --kill-after=5s 230s python3 -B -u "+
               shlex.join([str(run/"src/device.py"),str(run),"probe-imports","--probe-request",request,
                           "--parent",parent,"--node",node,"--source-sha",source_sha]))
    argv.extend(("--no-home","--pwd","/"))
    for src,dest,mode in binds:
        argv.extend(("--bind",f"{src}:{dest}:{mode}"))
    return [*argv,str(SANDBOX),"/bin/bash","--noprofile","--norc","-c",body]


def import_argv(run,probe,parent,node,source_sha):
    import_scope(run,probe,parent,node)
    need(SHA64.fullmatch(source_sha),"EXACT_IMPORT_SOURCE_MANIFEST")
    need(not any(path.exists() or path.is_symlink() for path in
                 (run/("py310-import-"+probe+suffix) for suffix in ("-intent.json",".json"))),
         "ZERO_GRES_IMPORT_REQUEST_ALREADY_USED")
    return ["srun","--jobid="+parent,"--job-name="+IMPORT_LABEL+probe,"--exact","--nodes=1","--ntasks=1",
            "--nodelist="+node,"--cpus-per-task="+str(IMPORT_CPUS),"--mem="+str(IMPORT_MEMORY)+"M",
            "--gres=none","--immediate=5","--time=4",*container(run,probe=(probe,parent,node,source_sha))]


def validate_import_terminal(run,manifest,proof):
    probe=proof["request"]
    row=proof["terminal"]
    need(proof["source_manifest_SHA256"]==manifest and proof["independent_accounting_read"] is True
         and proof["replay_allowed"] is False and SHA64.fullmatch(proof["result_SHA256"])
         and len(row)==9 and row[1]==IMPORT_LABEL+probe and row[2:4]==["COMPLETED","0:0"]
         and re.fullmatch(r"(?:1103606|1194900)\.[0-9]+",row[0])
         and 0<=int(row[4])<=IMPORT_SECONDS and utc(row[7])<=utc(row[8])
         and utc(row[8])-utc(row[7])<=IMPORT_SECONDS,
         "EXACT_INDEPENDENT_ZERO_GRES_IMPORT_TERMINAL")
    owner_request=import_scope(run,probe,row[0].split(".")[0],row[5])
    tres=dict(item.split("=",1) for item in row[6].split(","))
    need(set(tres)=={"cpu","mem","node"} and tres["cpu"]=="4" and memory(tres["mem"])==8192
         and tres["node"]=="1","EXACT_INDEPENDENT_IMPORT_ZERO_GRES_4_CPU_8_GIB")
    raw=small(run/("py310-import-"+probe+".json"),proof["result_SHA256"],262144)
    result=json.loads(raw)
    before=result["framework_before"]
    need(result["request"]==probe and result["owner_request"]==owner_request and result["numeric_step"]==row[0]
         and result["node"]==row[5] and result["source_manifest_SHA256"]==manifest
         and result["state"]=="EXACT_PYTHON310_ZERO_GRES_IMPORT_PASS" and result["python"]=="3.10"
         and result["GPU"]==result["optimizer_updates"]==0 and result["no_gpu_approval"] is True
         and result["cache_prefix_is_none"] is result["bytecode_writes_disabled"] is True
         and result["cuda_initialized"] is False and result["framework_unchanged"] is True
         and result["framework_after"]==before and len(before["members"])==393
         and before["manifest_SHA256"]==FRAMEWORK_SHA
         and result["module_sources"].keys()==IMPORT_MODULE_NAMES
         and all(0<item["bytes"]<=2*1024**2 and SHA64.fullmatch(item["SHA256"])
                 and isinstance(item["path"],str) and item["path"].startswith("/")
                 for item in result["module_sources"].values()),
         "EXACT_PYTHON310_IMPORT_PROOF_REQUIRED_BEFORE_GPU_OWNER")
    return proof["result_SHA256"]


def step_argv(run,intent,phase,minutes):
    gpu=phase=="gpu"
    work=container(run,intent) if gpu else [PYHOST,"-E","-s","-B","-X","pycache_prefix="+str(cache_prefix(run)),
                                          str(run/"src/device.py"),str(run),phase]
    return ["srun","--jobid="+intent["parent"],"--job-name="+intent["label"]+"-"+phase,
            "--exact","--nodes=1","--ntasks=1","--nodelist="+intent["node"],
            "--cpus-per-task="+str(CPUS if gpu else 2),"--mem="+str(MEMORY if gpu else 4096)+"M",
            "--gres="+("dcu:8" if gpu else CPU_STEP_GRES),"--immediate=5","--time="+str(minutes),*work]


def execute(argv,run,phase,budget):
    with (run/(phase+".log")).open("xb") as log:
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        try:
            stat=Path("/proc",str(child.pid),"stat").read_text().rsplit(") ",1)[1].split()
            save(run/(phase+"-owner.json"),dict(pid=child.pid,pgid=int(stat[2]),startticks=int(stat[19])))
        except FileNotFoundError:
            pass
        try:
            return child.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            for kind in (signal.SIGTERM,signal.SIGKILL):
                if child.poll() is not None:
                    break
                try:
                    os.killpg(child.pid,kind)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            return None


class Owner:
    def __init__(self,run,intent,query_scheduler=query,send=execute,clock=time.time,mono=time.monotonic):
        self.run,self.intent,self.query,self.send,self.clock,self.mono=run,intent,query_scheduler,send,clock,mono
        scope=scope_module(run)
        self.clock_scope=scope["EpochClock"](intent["epoch"],wall=clock,monotonic=mono)
        self.minutes=scope["gpu_minutes"]
        self.accounting=accounting(run)

    def phase(self,phase):
        need(phase not in rows(self.intent,self.query),"EXACT_PHASE_ALREADY_SEEN_NO_REPLAY")
        left=self.clock_scope.left
        if phase=="gpu":
            minutes=self.minutes(self.intent["epoch"],self.clock())
            budget=left(GPU,8)
        else:
            budget=min(60 if phase=="pre" else 120,left(WHOLE,25))
            minutes=max(1,math.ceil(budget/60))
        argv=step_argv(self.run,self.intent,phase,minutes)
        marker=self.run/(phase+"-intent.json")
        save(marker,{"phase":phase,"request":self.intent["request"],"epoch":self.intent["epoch"],
                     "label":self.intent["label"]+"-"+phase,"argv_SHA256":digest(encode(argv)),"replay_allowed":False})
        error=None
        try:
            rc=self.send(argv,self.run,phase,budget)
        except (OSError,ValueError,subprocess.SubprocessError) as failure:
            rc,error=None,type(failure).__name__
        wait=self.accounting["wait_terminal"](lambda:rows(self.intent,self.query).get(phase),
                   self.mono()+min(25,left(WHOLE,5)),interval=1,max_observations=25)
        row=wait["row"]
        if row is not None:
            accountable(row,self.intent,phase)
        result=dict(row=row,launch_returncode=rc,error_type=error,replay_allowed=False)
        save(self.run/(phase+"-terminal.json"),result)
        return result

    def snapshot(self,phase,result):
        row=result["row"]
        need(row and row[2:4]==["COMPLETED","0:0"] and result["launch_returncode"]==0,
             "EXACT_ZERO_GRES_SNAPSHOT_TERMINAL")
        raw=small(self.run/(phase+".log"),limit=131072)
        values=[json.loads(line) for line in raw.splitlines() if line.startswith(b"{")]
        observed=one(values)
        validate_physical(observed,self.intent,row,phase)
        save(self.run/("before.json" if phase=="pre" else "after.json"),observed)
        return observed


def validate_physical(observed,intent,row,phase):
    need(observed["step"]==row[0] and observed["node"]==intent["node"] and observed["phase"]==phase
         and observed["request"]==intent["request"] and len(observed["pci"])==len(set(observed["pci"]))==8
         and observed["full_fd_inventory_performed"] is True and observed["kfd_holder_count"]==0
         and observed["master_port"]==intent["port"] and observed["master_port_available"] is True
         and (phase!="post" or observed["independent_owned_process_scan"] is True
              and observed["owned_processes_remaining"]==[]),"EXACT_PHYSICAL_PRE_OR_POST")


def native_receipts(run,intent,gpu):
    worker=run/"native-outcome.json"
    if not worker.is_file():
        return {"qualified_optimizer_updates":0,"first_nonzero_update":None,"native_rank_receipts":0}
    raw=small(worker,limit=131072)
    hashes={"native_outcome_SHA256":digest(raw),"qualified_optimizer_updates":0,"first_nonzero_update":None,
            "native_rank_receipts":0}
    try:
        producer=json.loads(raw)
        sources=run/"src/trainer"
        sys.path.insert(0,str(sources))
        try:
            read=runpy.run_path(str(sources/"native_entry.py"),run_name="_verified_independent_native_read")["capture"]
            observed=read(run/"capture")
        finally:
            sys.path.pop(0)
        hashes["native_rank_receipts"]=observed["rank_receipts"]
        hashes["independent_observed_native_counts_SHA256"]=digest(encode(observed))
        need(producer["request"]==intent["request"] and producer["numeric_step"]==gpu[0]
             and producer["planned_optimizer_updates"]==2 and producer["Model_GO"] is producer["promotion"] is False
             and producer["replay_allowed"] is False and producer["observed"]==observed,
             "INDEPENDENT_ORIGINAL_NATIVE_COUNT_IDENTITY")
        if (gpu[2:4]==["COMPLETED","0:0"] and producer["child_returncode"]==0 and producer["error_type"] is None
                and producer["cleanup"]=={"owned_processes_remaining":0,"termination_needed":False}
                and observed["rank_completion_receipts"]==observed["rank_initialization_receipts"]==8
                and not observed["rank_read_errors"] and not observed["rank_failure_receipts"]
                and observed["native_sampler_verified_all_original_rows"] and observed["qualified_nonzero_update"] in (1,2)):
            hashes.update(first_nonzero_update=observed["qualified_nonzero_update"],
                qualified_optimizer_updates=sum(row["ranks_confirmed_optimizer_advance"]==8 for row in observed["updates"]))
    except (OSError,ValueError,TypeError,KeyError,ImportError) as err:
        hashes["native_audit_error_type"]=type(err).__name__
    return hashes


def reconcile(run,intent,query_scheduler=query):
    actual=rows(intent,query_scheduler)
    is_terminal=accounting(run)["is_terminal"]
    terminal={phase:accountable(row,intent,phase) for phase,row in actual.items() if is_terminal(row)}
    out=dict(request=intent["request"],GPU_intent=(run/"gpu-intent.json").is_file(),terminal=terminal,
             state="GPU_TERMINAL_UNKNOWN",release=False,replay_allowed=False,planned_optimizer_updates=2,
             Model_GO=False,blind=False,new_scheduling_operations=0,qualified_optimizer_updates=0)
    if "gpu" not in terminal:
        if not out["GPU_intent"]:
            out["state"]="NO_GPU_INTENT_RECONCILIATION_ONLY"
        return out
    out["state"]="GPU_TERMINAL_POST_EVIDENCE_PENDING"
    if set(terminal)!=set(PHASES) or not (run/"after.json").is_file():
        return out
    pre,after=[json.loads(small(run/name,limit=131072)) for name in ("before.json","after.json")]
    validate_physical(pre,intent,terminal["pre"],"pre")
    validate_physical(after,intent,terminal["post"],"post")
    need(pre["pci"]==after["pci"] and pre["boot_id"]==after["boot_id"]
         and pre["monotonic_ns"]<after["monotonic_ns"]
         and terminal["pre"][2:4]==terminal["post"][2:4]==["COMPLETED","0:0"]
         and utc(terminal["pre"][8])<=utc(terminal["gpu"][7])<=utc(terminal["gpu"][8])<=utc(terminal["post"][7]),
         "EXACT_NUMERIC_GPU_THEN_SAME_BOOT_CLEAN_POST")
    out.update(state="GPU_TERMINAL_AND_INDEPENDENT_CLEAN_POST",release=True,
               post_SHA256=digest(small(run/"after.json")))
    out.update(native_receipts(run,intent,terminal["gpu"]))
    return out


def drive(owner, authority, verify_allocation=True, uid=None):
    run,intent=owner.run,owner.intent
    clock=owner.clock
    error=None
    try:
        if verify_allocation:
            allocation(intent,owner.query,clock(),os.getuid() if uid is None else uid)
        owner.snapshot("pre",owner.phase("pre"))
        placement(intent,authority,clock())
        if verify_allocation:
            allocation(intent,owner.query,clock(),os.getuid() if uid is None else uid)
        placement(intent,authority,clock())
        owner.phase("gpu")
    except (ValueError,OSError,RuntimeError,KeyError,subprocess.SubprocessError) as failure:
        error=type(failure).__name__
    if (run/"gpu-intent.json").is_file():
        try:
            gpu=rows(intent,owner.query).get("gpu")
            need(owner.accounting["is_terminal"](gpu),"GPU_TERMINAL_UNKNOWN")
            accountable(gpu,intent,"gpu")
            owner.snapshot("post",owner.phase("post"))
        except (ValueError,OSError,RuntimeError,KeyError,subprocess.SubprocessError) as failure:
            error=error or type(failure).__name__
    out=reconcile(run,intent,owner.query)
    save(run/"owner-outcome.json",dict(out,controller_error_type=error))
    return out


def prepared(run,scope_path,scope_sha,authority_path,authority_sha,coordinator_path,asset_sha,source_sha,clock,import_proof):
    need(all(SHA64.fullmatch(key) for key in (scope_sha,authority_sha,asset_sha,source_sha)),"EXACT_OWNER_BINDINGS")
    scope=json.loads(small(scope_path,scope_sha))
    authority=json.loads(small(authority_path,authority_sha))
    placement(scope,authority,clock())
    small(coordinator_path,authority["coordinator_evidence_SHA256"])
    cache_prefix(run)
    need(scope["source_manifest_SHA256"]==source_sha and run.name==PREFIX+scope["request"]
         and not (run/"intent.json").exists() and not (run/"intent.json").is_symlink(),"EXCLUSIVE_FRESH_OWNER")
    config(run,source_sha)
    verify_prepared_assets(run,source_sha,asset_sha)
    need(import_proof is not None and SHA64.fullmatch(scope.get("import_proof_SHA256","")),
         "EXACT_PYTHON310_IMPORT_PROOF_REQUIRED_BEFORE_GPU_OWNER")
    importer=json.loads(small(import_proof,scope["import_proof_SHA256"],8192))
    validate_import_terminal(run,source_sha,importer)
    placement(scope,authority,clock())
    return scope,authority


def keeper_identity(pid):
    fields=Path("/proc",str(pid),"stat").read_text().rsplit(") ",1)[1].split()
    return dict(PID=pid,PGID=int(fields[2]),startticks=int(fields[19]),hostname=socket.gethostname(),
                boot_SHA256=digest(Path("/proc/sys/kernel/random/boot_id").read_bytes()))


def owner_intent(scope,epoch,scope_sha,authority_sha,asset_sha,authority):
    return {**scope,"label":PREFIX+scope["request"],"epoch":epoch,"deadline":epoch+WHOLE,
            "authority_SHA256":authority_sha,"scope_SHA256":scope_sha,"assets_SHA256":asset_sha,
            "planned_optimizer_updates":2,"coordinator_evidence_SHA256":authority["coordinator_evidence_SHA256"]}


def start(run,scope_path,scope_sha,authority_path,authority_sha,coordinator_path,asset_sha,source_sha,
          *,clock=time.time,keeper_sha=None,import_proof=None):
    scope,authority=prepared(run,scope_path,scope_sha,authority_path,authority_sha,coordinator_path,asset_sha,source_sha,clock,import_proof)
    marker=run/"keeper-intent.json"
    if keeper_sha is not None:
        need(SHA64.fullmatch(keeper_sha),"EXACT_DETACHED_START_INTENT_SHA")
        saved=json.loads(small(marker,keeper_sha))
        epoch=saved["epoch"]
        need(saved==owner_intent(scope,epoch,scope_sha,authority_sha,asset_sha,authority)
             and 0<=clock()-epoch<30,"SAME_DETACHED_OWNER_EPOCH_AND_INTENT")
    else:
        need(not marker.exists() and not marker.is_symlink(),"DETACHED_OWNER_ALREADY_OWNS_REQUEST")
        epoch=clock()
    placement(scope,authority,clock())
    intent=owner_intent(scope,epoch,scope_sha,authority_sha,asset_sha,authority)
    need(not rows(intent,query),"ORIGINAL_REQUEST_ACCOUNTING_ALREADY_EXISTS")
    save(run/"intent.json",intent)
    return drive(Owner(run,intent),authority)


def detach(run,scope_path,scope_sha,authority_path,authority_sha,coordinator_path,asset_sha,source_sha,*,
           clock=time.time,launch=subprocess.Popen,identity=keeper_identity,read_scheduler=query,import_proof=None):
    marker=run/"keeper-intent.json"
    need(not marker.exists() and not marker.is_symlink(),"DETACHED_OWNER_ALREADY_OWNS_REQUEST")
    scope,authority=prepared(run,scope_path,scope_sha,authority_path,authority_sha,coordinator_path,asset_sha,source_sha,clock,import_proof)
    epoch=clock()
    intent=owner_intent(scope,epoch,scope_sha,authority_sha,asset_sha,authority)
    need(not rows(intent,read_scheduler),"ORIGINAL_REQUEST_ACCOUNTING_ALREADY_EXISTS")
    placement(scope,authority,clock())
    save(marker,intent)
    intent_sha=digest(encode(intent))
    remaining=int(intent["deadline"]-clock())-10
    need(remaining>=WHOLE-30,"EXACT_DETACHED_KEEPER_SINGLE_EPOCH_BOUND")
    argv=["/usr/bin/timeout","--kill-after=10s",str(remaining)+"s",PYHOST,"-E","-s","-B",
          "-X","pycache_prefix="+str(cache_prefix(run)),"-u",
          str(run/"src/owner.py"),"start","--run",str(run),"--source-sha",source_sha,
          "--scope",str(scope_path),"--scope-sha",scope_sha,"--authority",str(authority_path),
          "--authority-sha",authority_sha,"--coordinator-proof",str(coordinator_path),"--assets-sha",asset_sha,
          "--import-proof",str(import_proof),"--from-keeper",intent_sha]
    result=dict(request=intent["request"],epoch=epoch,deadline=intent["deadline"],
                keeper_intent_SHA256=intent_sha,argv_SHA256=digest(encode(argv)),
                state="DETACHED_OWNER_START_UNKNOWN",replay_allowed=False,owner=None)
    try:
        with (run/"keeper.stdout").open("xb") as out, (run/"keeper.stderr").open("xb") as err:
            process=launch(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err,cwd=run,start_new_session=True)
        observed=identity(process.pid)
        need(observed["PID"]==observed["PGID"]==process.pid and observed["startticks"]>0,
             "EXACT_DETACHED_OWNER_PROCESS_IDENTITY")
        result.update(state="EXACT_DETACHED_OWNER_STARTED",owner=observed)
    except (OSError,ValueError,subprocess.SubprocessError) as err:
        result["error_type"]=type(err).__name__
    save(run/"keeper-owner.json",result)
    return result


def status(run,source_sha,query_scheduler=query):
    config(run,source_sha)
    intent=run/"intent.json"
    if intent.is_file():
        return reconcile(run,json.loads(small(intent)),query_scheduler)
    marker=run/"keeper-intent.json"
    need(marker.is_file(),"NO_EXACT_OWNER_INTENT")
    saved=json.loads(small(marker))
    return dict(request=saved["request"],state="DETACHED_OWNER_PRESTART_OR_TRANSPORT_UNKNOWN",
                keeper_intent_SHA256=digest(small(marker)),release=False,replay_allowed=False,
                planned_optimizer_updates=2,qualified_optimizer_updates=0,new_scheduling_operations=0)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode",choices=("assets","start","detach","reconcile","import-argv"))
    cli.add_argument("--run",type=Path,required=True)
    cli.add_argument("--source-sha",required=True)
    for name in ("scope","authority"):
        cli.add_argument("--"+name,type=Path)
        cli.add_argument("--"+name+"-sha")
    cli.add_argument("--assets-sha")
    cli.add_argument("--coordinator-proof",type=Path)
    cli.add_argument("--import-proof",type=Path)
    cli.add_argument("--probe-request")
    cli.add_argument("--parent")
    cli.add_argument("--node")
    cli.add_argument("--from-keeper")
    args=cli.parse_args()
    run=args.run
    need(sys.platform=="linux" and run.parent==RESULTS and re.fullmatch(PREFIX+r"[0-9a-f]{32}",run.name),
         "ACTUAL_SHARED_HOST_EXACT_ONE_REQUEST_ROOT")
    need(sys.dont_write_bytecode and sys.pycache_prefix==str(cache_prefix(run)),
         "EXACT_ISOLATED_SHARED_OWNER_PYTHON_CACHE")
    if args.mode=="assets":
        result=prepare_assets(run,args.source_sha)
    elif args.mode=="start":
        result=start(run,args.scope,args.scope_sha,args.authority,args.authority_sha,args.coordinator_proof,
                     args.assets_sha,args.source_sha,keeper_sha=args.from_keeper,import_proof=args.import_proof)
    elif args.mode=="detach":
        result=detach(run,args.scope,args.scope_sha,args.authority,args.authority_sha,args.coordinator_proof,
                      args.assets_sha,args.source_sha,import_proof=args.import_proof)
    elif args.mode=="import-argv":
        config(run,args.source_sha)
        result={"argv":import_argv(run,args.probe_request,args.parent,args.node,args.source_sha),
                "submitted":False,"GPU":0,"owner_GPURight":False}
    else:
        result=status(run,args.source_sha)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
