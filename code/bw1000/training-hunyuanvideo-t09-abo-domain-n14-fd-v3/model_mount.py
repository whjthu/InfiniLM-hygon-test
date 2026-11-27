"""Observe stable same-FD NFS identity without reading model payload bytes."""
import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time

WEIGHT="/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-hunyuanvideo-original-model-v1/hunyuanvideo-1.5/transformer/480p_t2v/diffusion_pytorch_model.safetensors"
TUPLE=[11025132338,33306632192,1794715911000000000,1794715912000000000]
HISTORIC_FULL_SHA="71f9affa1115fef2b14bd41fba30eab966fe80c9ed98e0fcba495dbc6d8fff86"
HISTORIC_COMPUTE_RECEIPT="8311536400438e4aac50a361b89030633d952ebb5b320ad4e9771b8d3759aff4"
HISTORIC_LOGIN_STAT_RECEIPT="075447f9f21434c9a9d8dccd7607c259757848fb1777ed60139c57a1c68f4611"
HISTORIC_NFS_SOURCE="15f030683563e67ce41ccc4c0a9ea509906aba09337e2126eabaff1298cd7c0c"
AUTHORIZER_RECEIPT="9f02d9465642ffa5561d6c7e204c00225464852b9072711547c8f5594fea49d5"
MOUNT_FIELDS={"fstype","source_SHA256","export_SHA256","root_SHA256","mountpoint_SHA256",
              "mount_relative_SHA256","export_relative_SHA256"}


def need(ok,why):
    if not ok:
        raise ValueError(why)


def digest(value):
    return sha256(value.encode()).hexdigest()


def valid_digest(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}",value) is not None


def unescape(value):
    match={"040":" ","011":"\t","012":"\n","134":"\\"}
    need(re.search(r"\\(?!040|011|012|134)",value) is None,"MODEL_MOUNT_BAD_PATH_ESCAPE")
    return re.sub(r"\\(040|011|012|134)",lambda part:match[part[1]],value)


def posix(value):
    path=PurePosixPath(value)
    need(path.is_absolute() and path.as_posix()==value and ".." not in path.parts
         and not any(ch in value for ch in "\0\r\n\t"),"MODEL_MOUNT_NONCANONICAL_PATH")
    return path


def device_pair(device):
    # Decode the Linux kernel encoding even when reviewing the resulting
    # evidence from a Windows login client.
    need(type(device) is int and 0<=device<2**64,"MODEL_LINUX_DEVICE_NUMBER_BOUNDS")
    major=((device>>8)&0xfff)|((device>>32)&0xfffff000)
    minor=(device&0xff)|((device>>12)&0xffffff00)
    if hasattr(os,"major"):
        need((major,minor)==(os.major(device),os.minor(device)),"MODEL_NATIVE_LINUX_DEVICE_DECODING")
    return f"{major}:{minor}"


def record(mountinfo,fdinfo,path,device):
    ids=re.findall(r"^mnt_id:[ \t]*([0-9]+)$",fdinfo,re.M)
    need(len(ids)==1 and int(ids[0])>0,"MODEL_FD_EXACT_ACTIVE_MOUNT_ID")
    mount_id=ids[0]
    rows=[line for line in mountinfo.splitlines() if line.split(" ",1)[0]==mount_id]
    need(len(rows)==1 and rows[0].count(" - ")==1,"MODEL_MOUNT_ID_AMBIGUOUS_OR_MISSING")
    front,back=(part.split() for part in rows[0].split(" - "))
    need(len(front)>=6 and len(back)==3 and re.fullmatch(r"[0-9]+:[0-9]+",front[2])
         and front[2]==device_pair(device),"MODEL_EFFECTIVE_MOUNT_DEV_MISMATCH")
    kind,source=back[0],unescape(back[1])
    host,sep,export=source.partition(":/")
    need(kind=="nfs" and bool(host) and sep==":/" and digest(source)==HISTORIC_NFS_SOURCE,
         "MODEL_WRONG_OR_UNPROVEN_NFS_SOURCE")
    export=posix("/"+export)
    root,point=(posix(unescape(value)) for value in front[3:5])
    absolute=posix(str(path))
    need(absolute.is_relative_to(point) and absolute!=point,"MODEL_PATH_OUTSIDE_EFFECTIVE_MOUNT")
    relative=absolute.relative_to(point)
    return {"mount_id":int(mount_id),"mount_device":front[2],"binding":{
        "fstype":kind,"source_SHA256":digest(source),"export_SHA256":digest(str(export)),
        "root_SHA256":digest(str(root)),"mountpoint_SHA256":digest(str(point)),
        "mount_relative_SHA256":digest(str(relative)),
        "export_relative_SHA256":digest(str(root/relative))}}


def stamp(info):
    return [info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns]


def stable(before,opened,last,path_after):
    seen=stamp(before)
    need(stat.S_ISREG(before.st_mode) and seen[1:]==TUPLE
         and all(stat.S_ISREG(p.st_mode) and stamp(p)==seen and p.st_mode==before.st_mode for p in
                 (opened,last,path_after)),"MODEL_FD_CHANGED_OR_WRONG_FROZEN_FOUR_TUPLE")
    return seen


def proc(path,cap):
    with open(path,"rb") as f:
        raw=f.read(cap+1)
    need(0<len(raw)<=cap,"MODEL_MOUNT_PROC_EVIDENCE_BOUNDS")
    return raw.decode()


def observe(scope,transport,phase,publication,*,path=None):
    path=Path(WEIGHT if path is None else path)
    check_scope(scope,transport,phase,publication)
    need(path.as_posix()==WEIGHT and path.resolve(strict=True)==path,"MODEL_CANONICAL_ORIGINAL_WEIGHT_ONLY")
    flags=os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK
    before=path.lstat()
    fd=os.open(path,flags)
    try:
        opened=os.fstat(fd)
        first=record(proc("/proc/self/mountinfo",4<<20),proc(f"/proc/self/fdinfo/{fd}",8192),path,opened.st_dev)
        need(os.readlink(f"/proc/self/fd/{fd}")==str(path),"MODEL_OPEN_FD_CANONICAL_PATH_CHANGED")
        last=record(proc("/proc/self/mountinfo",4<<20),proc(f"/proc/self/fdinfo/{fd}",8192),path,opened.st_dev)
        need(path.resolve(strict=True)==path and first==last,"MODEL_EFFECTIVE_FD_MOUNT_CHANGED")
        identity=stable(before,opened,os.fstat(fd),path.lstat())
    finally:
        os.close(fd)
    return {"scope":scope,"transport":transport,"phase":phase,"publication":publication,
            "same_fd_stable":True,"observed_epoch":time.time(),"file_identity":identity,
            "model_path_SHA256":digest(WEIGHT),"payload_bytes_read":0,**first}


def check_scope(scope,transport,phase,publication):
    need(all(type(value) is str and re.fullmatch("[0-9a-f]{32}",value) for value in (scope,transport))
         and type(publication) is str and re.fullmatch("[0-9a-f]{40}",publication)
         and phase in {"authorization","fresh","owner"}
         and ((phase=="owner")==(scope==transport)), "MODEL_WRONG_SCOPE_OR_TRANSPORT")


def check_observation(doc,scope,transport,phase,publication):
    check_scope(scope,transport,phase,publication)
    need((doc["scope"],doc["transport"],doc["phase"],doc["publication"])==(scope,transport,phase,publication)
         and doc["same_fd_stable"] is True and doc["payload_bytes_read"]==0
         and doc["model_path_SHA256"]==digest(WEIGHT) and type(doc["mount_id"]) is int and doc["mount_id"]>0
         and type(doc["observed_epoch"]) in (float,int) and math.isfinite(doc["observed_epoch"]) and doc["observed_epoch"]>0,
         "MODEL_OBSERVATION_FROM_OTHER_OWNER_OR_PHASE")
    ident=doc["file_identity"]
    collected=doc["binding"]
    need(type(ident) is list and len(ident)==5 and all(type(v) is int and v>=0 for v in ident)
         and ident[1:]==TUPLE and doc["mount_device"]==device_pair(ident[0]),
         "MODEL_OBSERVED_FROZEN_FOUR_TUPLE_OR_LOCAL_DEVICE")
    need(set(collected)==MOUNT_FIELDS and collected["fstype"]=="nfs"
         and collected["source_SHA256"]==HISTORIC_NFS_SOURCE
         and all(valid_digest(value) for key,value in collected.items() if key!="fstype"),
         "MODEL_EXACT_NFS_SOURCE_EXPORT_ROOT_AND_RELATIVE_PINS")
    return collected


def authorize(approval,scope):
    need(approval["schema"]=="T09_ABO_V2_ROOT_WEIGHT_MOUNT_BINDING_V1" and approval["scope"]==scope
         and approval["root_observed_same_fd_no_payload"] is True
         and approval["historic_login_stat_receipt_SHA256"]==HISTORIC_LOGIN_STAT_RECEIPT
         and approval["historic_full_weight_receipt_SHA256"]==HISTORIC_COMPUTE_RECEIPT
         and approval["historic_full_weight_SHA256"]==HISTORIC_FULL_SHA
         and approval["authenticated_authorizer_receipt_SHA256"]==AUTHORIZER_RECEIPT,
         "MODEL_SCOPED_ROOT_MOUNT_APPROVAL_REQUIRED")
    pinned=approval["binding"]
    need(set(pinned)==MOUNT_FIELDS and pinned["fstype"]=="nfs" and pinned["source_SHA256"]==HISTORIC_NFS_SOURCE
         and all(valid_digest(value) for key,value in pinned.items() if key!="fstype"),
         "MODEL_ROOT_NFS_EXPORT_ROOT_AND_RELATIVE_PINS")
    return pinned


def compare(doc,approval,scope,transport,phase,publication):
    actual=check_observation(doc,scope,transport,phase,publication)
    need(actual==authorize(approval,scope),"MODEL_CURRENT_NFS_EXPORT_ROOT_OR_RELATIVE_DIFFERS")
    need(doc["observed_epoch"]>=approval["authorization_epoch"],
         "MODEL_CURRENT_WITNESS_PREDATES_AUTHORIZED_BINDING")
    return doc


def execution_identity(value):
    need(value==[49,*TUPLE] and type(value) is list and all(type(x) is int for x in value),
         "MODEL_EXACT_EXECUTION_DEVICE_49")
    return value


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("scope","transport","publication"):
        p.add_argument("--"+key,required=True)
    ns=p.parse_args()
    print(json.dumps(observe(ns.scope,ns.transport,"authorization",ns.publication),sort_keys=True,allow_nan=False))
