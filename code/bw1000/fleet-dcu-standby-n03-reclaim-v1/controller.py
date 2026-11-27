"""Shared-lock singleton admission and v2 standby handoff for original n03 CPU-only .189."""
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time


RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
PARENT, NODE, OLD = "1194900", "f13r4n03", "1194900.189"
CPU_NODE = "f12r2n13"
PORTS = [43843,43844]
PYTHON = "/usr/local/python3.12/bin/python3"
ORIGINAL_PROOF = "5e9403b627d39ed6b0ef08a9945cdf98d2fad7ac508a6dabbe7c8d991107b894"
ORIGINAL_PUBLICATION = "661739fdf596d1304bbf5a2e9e0f93d9c90b9569"
OLD_PROBE = ["1194900.2443","training80-admit-f643366d0c99480eb76de27aed5cab10","COMPLETED","0:0","0",NODE,
             "cpu=1,mem=256M,node=1","2026-11-27T16:49:27","2026-11-27T16:49:27"]
PROOF_FIELDS = ("controller_exact_timeout","queue_clear","statistics_empty","query_complete","raw_singleton_timeout")
COLUMNS = "JobIDRaw,JobName%128,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"
ENV = ("SLURM_OVERLAP","SRUN_OVERLAP")


def need(ok,message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def read(path,cap,wanted=None):
    need(path.resolve(strict=True)==path and stat.S_ISREG(path.lstat().st_mode),"CANONICAL_PINNED_FILE")
    with path.open("rb") as stream:
        raw=stream.read(cap+1)
    need(0<len(raw)<=cap and (wanted is None or digest(raw)==wanted),"EXACT_SOURCE_OR_EVIDENCE")
    return raw


def evidence(raw):
    need(digest(raw)==ORIGINAL_PROOF,"EXACT_HISTORICAL_2443_PROOF")
    envelope=json.loads(raw)
    response=envelope["response"]
    selected=response["nodes"][NODE]
    row,witness=selected["device"],selected["witness"]
    proof=all(witness[phase].get(field) is True for phase in ("initial","final") for field in PROOF_FIELDS)
    need(envelope["request"]==response["request"]=="998fbfec65b4480f880aa3ebb39b630a" and
         all(envelope[k] is True for k in ("authority_verified","payload_flushed","remote_received")) and
         envelope["rc"]==response["rc"]==response["new_GPU_steps_authorized"]==0 and
         response["publication"]==ORIGINAL_PUBLICATION and response["raw_snapshot_preserved"] is True and
         witness["handle"]==OLD and witness["probe_accounting"]==OLD_PROBE and proof and
         witness["physical_probe_witness_complete"] is True and witness["probe_returned_zero"] is True and
         row["active_steps"]==[OLD] and row["device_probe_step"]==OLD_PROBE[0] and
         row["kfd_process_count"]==row["own_device_fd_process_count"]==0 and row["ports_clean"] is True and
         row["KFD_clean"] is row["PID_clean"] is row["device_probe_observed"] is True and row["probe_error"] is None,
         "ORIGINAL_PROBE_HISTORICAL_ONLY")


def validate_owner(owner,request,publication,pci,now):
    need(set(owner)=={"request","publication","parent","node","predecessor","ports","queue_excluded_nodes",
                      "original_hold_conflicts","issued","expires","historical_SHA256","pci","seconds"} and
         owner["request"]==request and owner["publication"]==publication and
         (owner["parent"],owner["node"],owner["predecessor"])==(PARENT,NODE,OLD) and owner["ports"]==PORTS and
         owner["queue_excluded_nodes"]==[NODE] and owner["original_hold_conflicts"]==[] and
         owner["historical_SHA256"]==ORIGINAL_PROOF and owner["pci"]==pci and owner["seconds"]==10800 and
         type(owner["issued"]) in (int,float) and type(owner["expires"]) in (int,float) and
         0<=now-owner["issued"]<=65 and now<owner["expires"]<=owner["issued"]+65,
         "EXACT_FRESH_SINGLETON_OWNER_INPUT")


def admission(query,groups,publish_source):
    module=publish_source["membership"]
    allocation=publish_source["allocation"]
    raw=module.snapshot(groups,owner_uid=os.getuid(),run_query=query)
    need(all(item["owned_by_observer"] is item["ledger_exact"] is item["step_locations_exact"] is True
             and item["state"]=="RUNNING" for item in raw["parents"].values()),"FROZEN_CURRENT_PARENT_ROSTER")
    proof=allocation.confirm_stale_step(query,OLD)
    proof["raw_singleton_timeout"]=allocation.stale_step_candidate(raw,OLD)
    need(all(proof.get(field) is True for field in PROOF_FIELDS),"N03_ONLY_FROZEN_CPU_TIMEOUT_MAY_REMAIN")
    return proof


def preexisting(query,root,gpu_label,cpu_label,*,final=False):
    need(not os.path.lexists(root),"ORIGINAL_NEW_STANDBY_NAMESPACE_EXISTS")
    found=[]
    for line in query("sacct","-nP","-S","2026-11-27T00:00:00","-j",PARENT,"-o","JobIDRaw,JobName%128"):
        key,name=line.split("|")
        if name in (gpu_label,cpu_label):
            found.append((key,name))
    need(all(name==cpu_label for _,name in found) if final else not found,"ORIGINAL_LABEL_ALREADY_USED")
    return found


def account_probe(query,handle,label,*,started):
    found=[line.split("|") for line in query("sacct","-nP","-j",handle,"-o",COLUMNS) if line.split("|")[0]==handle]
    if not found:
        return None
    need(len(found)==1 and len(found[0])==9,"EXACT_PRE_ACCOUNTING_RECORD")
    row,=found
    if row[2] in ("PENDING","RUNNING","COMPLETING"):
        return None
    need(row[1:4]==[label,"COMPLETED","0:0"] and re.fullmatch("[0-9]+",row[4]) and int(row[4])<=15 and
         row[5:7]==[NODE,"cpu=1,mem=256M,node=1"],"EXACT_ZERO_GRES_PREFLIGHT_TERMINAL")
    start,end=(datetime.fromisoformat(value+"+00:00").timestamp() for value in row[7:9])
    need(started-2<=start<=end<=time.time()+2,"EXACT_NEW_PREFLIGHT_CHRONOLOGY")
    return row


def physical(directory,request,pci,query,source,*,run_cpu=subprocess.run):
    nonce=digest(f"{request}:same-lock-n03-pre".encode())[:32]
    label="training80-admit-"+nonce
    argv=["srun","--threads=1","--jobid="+PARENT,"--job-name="+label,"--exact","--immediate=5","--nodes=1",
          "--ntasks=1","--nodelist="+NODE,"--cpus-per-task=1","--mem=256M","--gres=none","--time=00:01:00",
          "--kill-on-bad-exit=1",PYTHON,"-B",str(directory/"collect_training80_dispatch_snapshot.py"),
          "node",nonce,PARENT,json.dumps(PORTS)]
    begin=time.time()
    result=run_cpu(argv,stdin=subprocess.DEVNULL,capture_output=True,timeout=12,check=False)
    need(result.returncode==0 and not result.stderr and 0<len(result.stdout)<=4096,"EXACT_PHYSICAL_CPU_STEP_RETURN")
    [row]=source["collector"].read_worker_rows(result.stdout,nonce,PARENT,[NODE],PORTS,os.getuid()).values()
    need("probe_error" not in row and sorted(row["PCIs"])==sorted(pci) and
         type(row["kfd_process_count"]) is type(row["own_device_fd_process_count"]) is int and
         row["kfd_process_count"]==row["own_device_fd_process_count"]==0 and row["occupied_requested_ports"]==[],
         "FRESH_SAME_LOCK_PCI_KFD_FD_PORTS")
    deadline=time.monotonic()+6
    while True:
        account=account_probe(query,row["step"],label,started=begin)
        if account is not None:
            return dict(request=nonce,probe=row,accounting=account,stdout_SHA256=digest(result.stdout))
        need(time.monotonic()<deadline,"MISSING_FRESH_ZERO_GRES_TERMINAL")
        time.sleep(0.3)


def live_query(*argv):
    env={**os.environ,"TZ":"UTC","LC_ALL":"C","SLURM_TIME_FORMAT":"standard"}
    result=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,timeout=5,check=False,env=env)
    need(result.returncode==0 and not result.stderr and len(result.stdout)<=2*1024*1024,"EXACT_QUIET_SLURM_QUERY")
    return result.stdout.decode().strip().splitlines()


def visible_gpu(query,label):
    deadline=time.monotonic()+8
    while True:
        raw=query("scontrol","show","step",PARENT,"--oneliner")
        rows=[dict(field.split("=",1) for field in line.split() if "=" in field) for line in raw]
        matches=[row for row in rows if row.get("Name")==label]
        if matches:
            need(len(matches)==1,"AMBIGUOUS_STANDBY_GPU_STEP")
            row,=matches
            need(re.fullmatch(PARENT+r"\.[0-9]+",row.get("StepId","")) and row.get("NodeList")==NODE and
                 row.get("State")=="RUNNING" and row.get("TRES")=="cpu=8,gres/dcu=8,mem=16G,node=1",
                 "VISIBLE_STANDBY_GPU_STEP_MISMATCH")
            return row["StepId"]
        need(time.monotonic()<deadline,"SPAWNED_BUT_NUMERIC_STEP_UNOBSERVED_UNKNOWN")
        time.sleep(0.35)


def sync_directory(directory):
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def execute(directory,request,publication,owner,source,children,*,query=live_query,run_cpu=subprocess.run,lock_factory=None):
    from contextlib import contextmanager
    @contextmanager
    def real_lock():
        import fcntl
        fd=os.open(RESULTS/".training80-dcu-standby.lock",os.O_RDWR|os.O_NOFOLLOW)
        with os.fdopen(fd,"rb+") as lock:
            need(stat.S_ISREG(os.fstat(lock.fileno()).st_mode),"ORIGINAL_SHARED_LOCK_REGULAR_FILE")
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            yield

    worker=read(directory/"standby_worker.py",8192,source["worker_SHA256"])
    evidence(read(directory/"historical-2443.json",8192))
    groups,pci=source["groups"],source["pci"]
    validate_owner(owner,request,publication,pci,time.time())
    root=RESULTS/f"training80-dcu-standby-{request}"
    labels=(f"training80-dcu-standby-{request[:12]}-{NODE}",
            "training80-admit-"+digest(f"{request}:same-lock-n03-pre".encode())[:32])
    old=source["standby"]
    with (real_lock if lock_factory is None else lock_factory)():
        need(not any(os.environ.get(name) for name in ENV),"NO_INHERITED_STEP_OVERLAP")
        preexisting(query,root,*labels)
        before=admission(query,groups,source)
        check=physical(directory,request,pci,query,source,run_cpu=run_cpu)
        final=admission(query,groups,source)
        seen=preexisting(query,root,*labels,final=True)
        need(seen==[(check["accounting"][0],labels[1])],"EXACT_ONLY_FRESH_CPU_LABEL")
        validate_owner(owner,request,publication,pci,time.time())
        root.mkdir(mode=0o700)
        old["write"](root,"intent.json",dict(request=request,publication=publication,parent=PARENT,node=NODE,
            nodes=[NODE],seconds=owner["seconds"],synthetic=True,replay_allowed=False,predecessor=OLD,
            historical_proof_SHA256=ORIGINAL_PROOF,initial_scheduler=before,final_scheduler=final,
            physical=check,epoch=time.time()))
        with (root/"worker.py").open("xb") as stream:
            stream.write(worker)
            stream.flush()
            os.fsync(stream.fileno())
        sync_directory(root)
        sync_directory(RESULTS)
        need(not children.children,"SOLE_ORIGINAL_GPU_CHILD_REQUIRED")
        old["subprocess"]=children
        record=old["spawn"](root,NODE,{"parent":PARENT,"pci":pci},owner["seconds"],request)
        need(len(children.children)==1 and record["label"]==labels[0],"EXACT_ONCE_V2_CHILD")
        handle=visible_gpu(query,labels[0])
        result=dict(request=request,publication=publication,synthetic=True,submit_records=[record],
                    launch_failures=[],changed_before_launch=[],card_steps_observed=1,actual_GPU_step=handle,
                    pre=check["accounting"][0])
        old["write"](root,"submission-receipt.json",result)
    return result
