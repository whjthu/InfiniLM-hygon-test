"""Bind the actual allocated Pack16 DCU, then enter the unchanged measurement worker."""

import argparse
import ctypes
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
from types import ModuleType


PARENT,NODE="1103606","f12r2n10"
OLD_REQUEST="a31421758fc74d319923dbce7045cc12"
OLD_LABEL="O01-zimage-square-rhs-pack16-v1"
OLD_PUB="c4da359b16d2eea35300547f42e9967dd3199b86"
PREFIX="O01-pack16-r40-n10-"
R40_OWNER="0d0cea41f5124c719f122ffb481a12a3"
HELPER_SHA="05113bd795355ac88851efbc6b870d9adfbbed3347746b4f2bb96267c810cf98"
RUNTIME_SHA="999e985cfe5066e6520c6a713d45d0240851a07a04b712246e39d0bce006b74f"
ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
HIP=Path("/opt/dtk/hip/lib/libgalaxyhip.so.5.2.26113.2065-6699a1f0")
HIP_SHA="b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b"
APPTAINER="/public/software/apps/apptainer/1.3.4/bin/apptainer"
PCI=re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
HEX=re.compile(r"[0-9a-f]{64}")


def need(ok,code):
    if not ok:
        raise ValueError(code)


def digest(raw):
    return sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def read(path,cap,expected=None):
    a=path.lstat()
    need(stat.S_ISREG(a.st_mode) and path.resolve(strict=True)==path and 0<a.st_size<=cap,"N10_CANONICAL_PINNED_SOURCE")
    with path.open("rb") as stream:
        raw=stream.read(cap+1)
    b=path.lstat()
    key=lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
    need(len(raw)==a.st_size and key(a)==key(b) and (expected is None or digest(raw)==expected),"N10_EXACT_FILE_PIN")
    return raw


def save(path,obj):
    raw=canonical(obj)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return digest(raw)


def label(owner):
    need(owner==R40_OWNER and owner!=OLD_REQUEST,"N10_FRESH_OUTER_SCOPE")
    return PREFIX+owner[:12]


def intent(raw,now):
    value=json.loads(raw)
    need(value["request"]!=OLD_REQUEST and value["label"]==label(value["request"])
         and (value["parent"],value["node"])==(PARENT,NODE)
         and re.fullmatch(r"[0-9a-f]{40}",value["publication"])
         and len(value["PCIs"])==len(set(value["PCIs"]))==8 and all(PCI.fullmatch(item) for item in value["PCIs"])
         and HEX.fullmatch(value["boot_SHA256"]) and HEX.fullmatch(value["pre_SHA256"])
         and HEX.fullmatch(value["root_authority_SHA256"]) and value["runtime_SHA256"]==RUNTIME_SHA
         and re.fullmatch(r"1103606\.[0-9]+",value["pre_step"]) and value["pre_step"]!="1103606.3352"
         and math.isfinite(value["epoch"]) and value["deadline_epoch"]==value["epoch"]+240
         and value["epoch"]<=now<value["deadline_epoch"]-90 and "PCI" not in value,
         "N10_FRESH_EPOCH_WITH_NO_PCI_GUESS")
    return value


def resources(tres):
    parts=tres.split(",")
    need(len(parts)==len(set(parts))==4 and set(parts) in (
        {"cpu=4","gres/dcu=1","mem=8G","node=1"},
        {"cpu=4","gres/dcu=1","mem=8192M","node=1"}),"N10_4CPU_8G_1DCU_TRES")


def allocation(value,env,where,affinity,cgroup,boot):
    numeric=env.get("SLURM_STEP_ID","")
    need(env.get("SLURM_JOB_ID")==PARENT and re.fullmatch(r"[0-9]+",numeric) and where==NODE
         and env.get("SLURM_CPUS_PER_TASK")=="4" and affinity==4,"N10_EXACT_NUMERIC_ASSIGNED_STEP")
    step=PARENT+"."+numeric
    need(step not in ("1103606.3352",value["pre_step"]) and digest(boot)==value["boot_SHA256"],
         "N10_SAME_BOOT_AND_DISTINCT_STEP")
    matches=[]
    for row in cgroup.splitlines():
        _,controllers,path=row.split(":",2)
        parts=set(path.strip("/").split("/"))
        if {"job_"+PARENT,"step_"+numeric}<=parts:
            matches.append(controllers)
    need(matches,"N10_ACTUAL_SLURM_JOB_STEP_CGROUP")
    # Parent GPU variables may describe its eight-card reservation. A populated step
    # or process visibility variable must itself denote at most the assigned device.
    for key in ("SLURM_STEP_GPUS","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES"):
        item=env.get(key,"")
        if item not in ("","(null)","N/A"):
            need(re.fullmatch(r"[0-9]+|GPU-[a-zA-Z0-9-]+",item),"N10_SINGLE_STEP_VISIBILITY")
    return step


def hip_device(library=HIP):
    raw=read(library,12_349_016,HIP_SHA)
    need(len(raw)==12_349_016,"N10_EXACT_OLD_GALAXYHIP_EXTENT")
    hip=ctypes.CDLL(str(library),mode=os.RTLD_NOW|os.RTLD_LOCAL)
    count=ctypes.c_int()
    get_count=hip.hipGetDeviceCount
    get_count.argtypes,get_count.restype=[ctypes.POINTER(ctypes.c_int)],ctypes.c_int
    need(get_count(ctypes.byref(count))==0 and count.value==1,"N10_EXACT_ONE_REAL_HIP_DEVICE")
    get_pci=hip.hipDeviceGetPCIBusId
    get_pci.argtypes,get_pci.restype=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int],ctypes.c_int
    address=ctypes.create_string_buffer(32)
    need(get_pci(address,len(address),0)==0,"N10_REAL_ASSIGNED_HIP_PCI_QUERY")
    got=address.value.decode("ascii").lower()
    return ("0000:"+got if len(got)==7 else got)


def worker_scope(value,step,pci,boot):
    need(PCI.fullmatch(pci) and pci in value["PCIs"] and digest(boot)==value["boot_SHA256"],
         "N10_REAL_PCI_IN_SAME_BOOT_EIGHT_BOARD_WHITELIST")
    return {"request":OLD_REQUEST,"label":OLD_LABEL,"source_publication":OLD_PUB,"runtime_SHA256":RUNTIME_SHA,
            "root_admitted":True,"parent":PARENT,"node":NODE,"gfx":"gfx936","torch_version":"2.10.0",
            "torch_hip":"6.3.26113","epoch":value["epoch"],"deadline_epoch":value["deadline_epoch"],
            "pre_step":value["pre_step"],"pre_SHA256":value["pre_SHA256"],
            "protection_authority_SHA256":value["root_authority_SHA256"],
            "boot_id":boot.decode("ascii").strip(),"PCI":pci,
            "outer_request":value["request"],"actual_outer_step":step,"claim_from_old_grant":False}


def state(value):
    boot=Path("/proc/sys/kernel/random/boot_id").read_bytes()
    group=Path("/proc/self/cgroup").read_text("ascii")
    return allocation(value,os.environ,socket.gethostname().split(".")[0],len(os.sched_getaffinity(0)),group,boot),boot


def pinned_step_evidence():
    # -I -S drops the script directory. Compile exactly the bytes already
    # verified above, with no second open, search path change, or pycache use.
    gate=Path(__file__).absolute()
    need(gate==gate.resolve(strict=True) and gate.is_file(),"R40_CANONICAL_ABSOLUTE_GATE_PATH")
    helper=gate.with_name("step_evidence.py")
    raw=read(helper,16384,HELPER_SHA)
    name="_o01_pack16_r40_pinned_step_evidence"
    need(name not in sys.modules,"R40_FRESH_PRIVATE_HELPER_MODULE")
    module=ModuleType(name)
    module.__file__=str(helper)
    sys.modules[name]=module
    try:
        exec(compile(raw,str(helper),"exec",dont_inherit=True),module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def main():
    if sys.argv[1:]==["--check-pinned-step-evidence"]:
        need(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,"R40_IDENTICAL_ISOLATED_NO_SITE_START")
        original=tuple(sys.path)
        helper=pinned_step_evidence()
        sample=helper.ExpectedStep(PARENT,PARENT+".0",NODE,label(R40_OWNER),"0"*64)
        need(tuple(sys.path)==original and callable(helper.observe) and callable(helper.persist),"R40_NO_IMPORT_PATH_OR_DEVICE_SIDE_EFFECT")
        print(canonical({"request":R40_OWNER,"helper_SHA256":HELPER_SHA,"check_only":True,"scheduler_queries":0,
                         "worker_or_HIP_entry":False,"module_loaded":sample.parent==PARENT,"sys_path_unchanged":True}).decode(),end="")
        return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--intent",required=True,type=Path)
    p.add_argument("--intent-sha256",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--inside",action="store_true")
    p.add_argument("command",nargs=argparse.REMAINDER)
    a=p.parse_args()
    require=a.out.resolve(strict=True)==a.out and a.out.is_dir()
    need(require,"N10_EXACT_NEW_OUTPUT")
    value=intent(read(a.intent,16384,a.intent_sha256),time.time())
    step,boot=state(value)
    if not a.inside:
        argv=a.command
        need(argv[:3]==["--",APPTAINER,"exec"],"N10_EXACT_SINGLE_APPTAINER_ENTRY")
        helper=pinned_step_evidence()
        expected=helper.ExpectedStep(PARENT,step,NODE,value["label"],a.intent_sha256)
        audit=helper.evidence(expected)
        try:
            tres=helper.observe(expected,audit,subprocess.check_output,resources)
        finally:
            helper.persist(a.out,audit)
        save(a.out/"STEP_GATE.json",{"request":value["request"],"step":step,"assigned_TRES":tres,
                 "intent_SHA256":a.intent_sha256,"same_boot":True,"actual_cgroup":True})
        os.execv(APPTAINER,argv[1:])
    gate=json.loads(read(a.out/"STEP_GATE.json",8192))
    need((gate["request"],gate["step"],gate["intent_SHA256"])==(value["request"],step,a.intent_sha256)
         and gate["actual_cgroup"] is gate["same_boot"] is True,"N10_SAME_REAL_STEP_INSIDE_CONTAINER")
    resources(gate["assigned_TRES"])
    pci=hip_device()
    scoped=worker_scope(value,step,pci,boot)
    scope_path=a.out/"PACK16_COMPONENT_SCOPE.json"
    checksum=save(scope_path,scoped)
    save(a.out/"ACTUAL_PCI_ADMISSION.json",{"request":value["request"],"step":step,"PCI":pci,
        "scope_SHA256":checksum,"hip_provider_SHA256":HIP_SHA,"visible_devices":1,"numeric_TRES":gate["assigned_TRES"],
        "actual_cgroup":True,"same_boot":True,"other_devices_claimed":0})
    remaining=int(value["deadline_epoch"]-time.time())-38
    need(remaining>=75,"N10_RESERVED_TERMINAL_CLEANUP_BUDGET")
    executable="/usr/bin/timeout"
    os.execv(executable,[executable,"--kill-after=5s",str(remaining),sys.executable,"-B","-u","/pair-package/worker.py",
            "--scope",str(scope_path),"--scope-sha256",checksum,"--out",str(a.out/"worker")])


if __name__=="__main__":
    main()
