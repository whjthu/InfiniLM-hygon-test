"""Bind T09 science to a separately authorized current placement."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time


D=Path(__file__).resolve().parent
PLACEMENT=json.loads((D/"PLACEMENT.json").read_bytes())
PARENT=PLACEMENT["parent"]
NODE=PLACEMENT["node"]
WINDOW=1440
ALLOWED={"1103606":{"f13r3n13","f13r3n14"}}
TERMINAL={"COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED",
          "BOOT_FAIL","DEADLINE","REVOKED"}


def need(ok,code):
    if not ok:
        raise ValueError(code)


def fields(row):
    return dict(item.split("=",1) for item in row.split() if "=" in item)


def utc(value):
    return datetime.strptime(value,"%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def step_tres(raw,phase):
    tokens=raw.split(",")
    want={"cpu=8","node=1","gres/dcu=1"} if phase=="gpu" else {"cpu=1","node=1"}
    memory=({"mem=125G","mem=128000M"} if phase=="gpu" else {"mem=1G","mem=1024M"})
    return len(tokens)==len(want)+1 and any(set(tokens)==want|{item} for item in memory)


def placement_config():
    need(PARENT in ALLOWED and NODE in ALLOWED[PARENT] and PARENT!="1087656"
         and all(type(PLACEMENT[k]) is int and PLACEMENT[k]>=0 for k in ("host_model_dev","execution_model_dev"))
         and PLACEMENT["device_architecture"]=="gfx936"
         and PLACEMENT["host_model_observed_devs"]==[47,53] and PLACEMENT["host_model_dev"] in (47,53)
         and PLACEMENT["execution_model_dev"]==49,
         "T09_SCOPE_NO_EXACT_CURRENT_PLACEMENT")
    return PLACEMENT


def bind_authority(authority, *, request, publication, epoch, route):
    p=placement_config()
    now=time.time()
    need(authority["request_id"]==request and authority["publication"]==publication
         and authority["parent"]==PARENT and authority["node"]==NODE and authority["epoch"]==epoch
         and 0<=now-authority["observed_epoch"]<=90 and authority["valid_through_epoch"]>=epoch+WINDOW
         and authority["root_sole_owner"] is authority["ordinary_node_exclusion_until_independent_post"] is True
         and authority["no_prior_unknown_conflict"] is authority["one_DCU_once"] is True
         and authority["qualified_original_unprofiled_route"] is True
         and authority["boot_id"]==route["boot_id"] and authority["physical_preflight_SHA256"]==route["receipt_SHA256"]
         and route["scope"]==request and route["parent"]==PARENT and route["node"]==NODE
         and route["same_image_provider"] is True and route["device_architecture"]==p["device_architecture"]
         and route["runtime_single_DCU_path_qualified"] is True and route["passive_inventory_complete"] is True
         and route["kfd_holder_count"]==route["same_UID_drm_or_kfd_fd_owner_count"]==0
         and len(route["pci"])==len(set(route["pci"]))==8
         and all(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",pci) for pci in route["pci"])
         and 0<=now-route["observed_epoch"]<=90,"T09_INDEPENDENT_CURRENT_SINGLE_DCU_ROUTE_AND_CUSTODY")
    need(authority['fresh_prefix7_diagnostic_GO'] is True
         and authority['current_GPU_capability_gate_before_any_updates'] is True
         and authority['prior_3660_receipt_SHA256']=='c9f071765f0085e141d6e7b7fee5fa6252a3f72d42c36b485d196bbb2b3d6fc2'
         and authority['prior_3640_archive_SHA256']=='34d113536c88f7ebc9fa06c3218324f4cb5f31e89cc382077c6e032f01fc46e4',
         'T09_EXPLICIT_ROOT_PREFIX_RECOMPUTE_SCOPE')
    need(all(authority.get(k)==v for k,v in {'prior_3665_archive_SHA256': '4ac8f320ad512760f61fce6b31a3c048fd52271e83dc38f084e7be5d41c230e4', 'prior_3670_cpu_archive_SHA256': '19404bd4fa6d539ab302825b4245a79c1905470a6e741ed1e425a5d493468071'}.items()) and
         authority.get('checkpoint_recompute_diagnostic_GO') is True,
         'T09_EXACT_3665_3670_AND_ROOT_RECOMPUTE_AUTHORITY')
    return authority


@contextmanager
def shared_lock(project):
    import fcntl
    lock=project/"results/.training80-dcu-standby.lock"
    fd=os.open(lock,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
    try:
        original=os.fstat(fd)
        need(stat.S_ISREG(original.st_mode),"T09_CANONICAL_EXISTING_SHARED_LOCK")
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=lock.lstat()
        need((original.st_dev,original.st_ino)==(current.st_dev,current.st_ino),"T09_SHARED_LOCK_CHANGED")
        yield
    finally:
        os.close(fd)


def live_views(read, uid, epoch):
    placement_config()
    def nodes(spec):
        if re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}",spec):
            return {spec}
        need(re.fullmatch(r"f[0-9rnf,\[\]-]{1,8192}",spec) is not None,"T09_UNCLASSIFIED_NODELIST")
        out=read(["scontrol","show","hostnames",spec])
        need(0<len(out)<=640 and len(out)==len(set(out)) and all(re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}",node)
                                                              for node in out),"T09_HOST_EXPANSION")
        return set(out)
    raw,=read(["scontrol","show","job",PARENT,"--oneliner"])
    job=fields(raw)
    need(job["JobId"]==PARENT and job["JobState"]=="RUNNING" and job["UserId"].endswith(f"({uid})")
         and NODE in nodes(job["NodeList"]) and utc(job["EndTime"])>=epoch+WINDOW
         and any(re.fullmatch(r"gres/dcu=[1-9][0-9]*",item) for item in job["AllocTRES"].split(",")),
         "T09_EXACT_LIVE_PARENT_OWNER_AND_BUDGET")
    raw,=read(["squeue","-h","-j",PARENT,"-o","%i|%T|%N"])
    parent,state,spec=raw.split("|")
    need(parent==PARENT and state=="RUNNING" and NODE in nodes(spec),"T09_LIVE_PARENT_QUEUE")
    matches=[]
    for line in read(["scontrol","-d","show","job",PARENT]):
        item=fields(line)
        if {"Nodes","CPU_IDs","Mem"}<=item.keys() and NODE in nodes(item["Nodes"]):
            matches.append(item)
    item,=matches
    cpus=set()
    for atom in item["CPU_IDs"].split(","):
        need(re.fullmatch(r"[0-9]{1,5}(?:-[0-9]{1,5})?",atom) is not None,"T09_CPU_LIST_FORMAT")
        ends=atom.split("-")
        a,b=int(ends[0]),int(ends[-1])
        need(0<=a<=b<=16384,"T09_CPU_RANGE")
        slot=set(range(a,b+1))
        need(not cpus&slot,"T09_DUPLICATE_CPU_SLOT")
        cpus.update(slot)
    memory=re.fullmatch(r"([0-9]{1,12})([MG]?)",item["Mem"])
    need(len(cpus)>=8 and memory is not None and int(memory[1])*(1024 if memory[2]=="G" else 1)>=128000,
         "T09_PER_NODE_CPU_MEMORY")
    for raw in read(["scontrol","show","step",PARENT,"--oneliner"]):
        item=fields(raw)
        handle=item["StepId"]
        need(re.fullmatch(PARENT+r"\.(?:[0-9]+|batch|extern)",handle) is not None,"T09_EXACT_CONTROLLER_STEP")
        if handle.rsplit(".",1)[1].isdecimal() and item["State"].rstrip("+") not in TERMINAL:
            need(NODE not in nodes(item["NodeList"]),"T09_EXISTING_CONTROLLER_OWNER")
    for raw in read(["squeue","--steps","-h","-j",PARENT,"-o","%i|%N"]):
        handle,spec=raw.split("|")
        need(re.fullmatch(PARENT+r"\.(?:[0-9]+|batch|extern)",handle) is not None,"T09_EXACT_QUEUE_STEP")
        if handle.rsplit(".",1)[1].isdecimal():
            need(NODE not in nodes(spec),"T09_EXISTING_QUEUE_OWNER")
    return True


def numeric_cgroup(environ=None, chunks=None):
    env=os.environ if environ is None else environ
    step=env.get("SLURM_STEP_ID","")
    need(env.get("SLURM_JOB_ID")==PARENT and re.fullmatch(r"[0-9]+",step) is not None,"T09_NUMERIC_STEP")
    text=Path("/proc/self/cgroup").read_text("ascii").splitlines() if chunks is None else chunks
    need(any({"job_"+PARENT,"step_"+step}<=set(line.split(":",2)[-1].split("/")) for line in text),
         "T09_EXACT_NUMERIC_STEP_CGROUP")
    return PARENT+"."+step


def authentic_step(phase,label):
    identity=numeric_cgroup()
    need(socket.gethostname().split(".",1)[0]==NODE,"T09_EXACT_COMPUTE_NODE")
    raw=subprocess.check_output(["scontrol","show","step",identity,"--oneliner"],timeout=7,text=True)
    found=[item for line in raw.splitlines() if (item:=fields(line)).get("StepId")==identity]
    need(len(found)==1,"T09_ONE_EXACT_LIVE_STEP_RECORD")
    item=found[0]
    need(item.get("Name",item.get("JobName"))==label and item.get("State")=="RUNNING"
         and item.get("NodeList")==NODE and step_tres(item.get("TRES",item.get("AllocTRES","")),phase),
         "T09_NATIVE_STEP_RESOURCE_PROOF")
    return identity
