"""Wait for eight durable local diagnostic receipts before aborting training."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import time


DIAGNOSTIC="live-sampler-diagnostic.json"
WAIT="live-sampler-rendezvous.json"
LIMIT=32768
WORLD=8


def need(value,message):
    if not value:
        raise ValueError(message)


def private_directory(p,rank_leaf):
    info=p.lstat()
    need(p.resolve(strict=True)==p and stat.S_ISDIR(info.st_mode) and info.st_uid==os.getuid()
         and ((stat.S_IMODE(info.st_mode)==0o700) if rank_leaf else (info.st_mode&0o022==0)),
         "R43_EXACT_PRIVATE_DIAGNOSIS_DIRECTORY")


def activation(epoch,deadline):
    config=os.environ["NATIVE_CONTINUOUS_CONFIG_SHA256"]
    step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"]
    need(re.fullmatch(r"[0-9a-f]{64}",config) and re.fullmatch(r"1103606\.[0-9]+",step)
         and epoch<deadline,"R43_EXACT_DIAGNOSIS_ACTIVATION")
    boot=Path("/proc/sys/kernel/random/boot_id").read_bytes()
    pid=Path("/proc/self/stat").read_text().rsplit(") ",1)[1].split()
    raw=f"{config}:{step}:{epoch!r}:{deadline!r}".encode()+b":"+boot
    return {"activation_SHA256":sha256(raw).hexdigest(),"step":step,
            "PID":os.getpid(),"startticks":int(pid[19])}


def write_once(root,name,doc):
    private_directory(root,True)
    need(name in (DIAGNOSTIC,WAIT),"R43_FIXED_DIAGNOSIS_FILENAMES")
    raw=(json.dumps(doc,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    need(0<len(raw)<=LIMIT,"R43_DIAGNOSTIC_OUTPUT_BOUND")
    target=root/name
    temporary=root/("."+name+".tmp")
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,"wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary,target,follow_symlinks=False)
    finally:
        temporary.unlink()
    owner=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        os.fsync(owner)
    finally:
        os.close(owner)


def observe_one(common,rank,shared):
    root=common/f"rank-{rank:02d}"
    path=root/DIAGNOSTIC
    try:
        private_directory(root,True)
        before=path.lstat()
        need(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid()
             and stat.S_IMODE(before.st_mode)==0o600 and 0<before.st_size<=LIMIT,
             "R43_PRIVATE_RANK_RECEIPT_REQUIRED")
        verify=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mode,st.st_uid,st.st_mtime_ns,st.st_ctime_ns)
        flags=os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC
        with os.fdopen(os.open(path,flags),"rb") as readable:
            opened=os.fstat(readable.fileno())
            raw=readable.read(LIMIT+1)
            after=os.fstat(readable.fileno())
        need(verify(before)==verify(opened)==verify(after)==verify(path.lstat()) and len(raw)==before.st_size,
             "R43_STABLE_RANK_RECEIPT_REQUIRED")
        document=json.loads(raw)
        identity=document["rank_activation"]
        need(document["schema"]=="GEMMA_R43_LIVE_TRAIN_BEGIN_SAMPLER_PROJECTION_V1" and type(document["rank"]) is int
             and document["rank"]==rank and identity["activation_SHA256"]==shared["activation_SHA256"]
             and identity["step"]==shared["step"] and type(identity["PID"]) is int and identity["PID"]>0
             and type(identity["startticks"]) is int and identity["startticks"]>0
             and document["optimizer_updates"]==0 and document["diagnostic_only_stops_before_optimizer"] is True,
             "R43_SCOPED_RANK_RECEIPT_REQUIRED")
        # A reader can see the link before its writer reaches directory fsync.
        # Commit the directory itself before counting the receipt as durable.
        owner=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            os.fsync(owner)
        finally:
            os.close(owner)
        need(verify(before)==verify(path.lstat()),"R43_STABLE_RECEIPT_AFTER_DIRECTORY_SYNC")
    except FileNotFoundError:
        return "MISSING",None,None
    except (OSError,TypeError,KeyError,ValueError):
        return "INVALID",None,None
    return "PRESENT",sha256(raw).hexdigest(),(identity["PID"],identity["startticks"])


def await_diagnostics(root,shared,deadline,*,now=time.time,mono=time.monotonic,sleep=time.sleep,read=observe_one):
    common=root.parent
    private_directory(common,False)
    stop=mono()+max(0,min(25,deadline-now()))
    observations={}
    while True:
        for rank in range(WORLD):
            observations[rank]=read(common,rank,shared)
        for rank,value in tuple(observations.items()):
            if value[0]=="PRESENT" and sum(v[0]=="PRESENT" and v[2]==value[2] for v in observations.values())>1:
                duplicates=[r for r,v in observations.items() if v[0]=="PRESENT" and v[2]==value[2]]
                for duplicate in duplicates:
                    observations[duplicate]=("INVALID",None,None)
        if all(value[0]=="PRESENT" for value in observations.values()) or min(stop-mono(),deadline-now())<=0:
            break
        sleep(max(0,min(.2,stop-mono(),deadline-now())))
    return {
        "schema":"GEMMA_R43_BOUNDED_LOCAL_RANK_RECEIPT_RENDEZVOUS_V1",
        "activation_SHA256":shared["activation_SHA256"],"step":shared["step"],
        "expected_ranks":WORLD,"present":{str(k):v[1] for k,v in observations.items() if v[0]=="PRESENT"},
        "missing":[k for k,v in observations.items() if v[0]=="MISSING"],
        "invalid":[k for k,v in observations.items() if v[0]=="INVALID"],
        "optimizer_updates":0,"stop_after_receipt":True,
    }
