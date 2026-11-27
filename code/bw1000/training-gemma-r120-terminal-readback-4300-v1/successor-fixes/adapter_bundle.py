"""Verify the exact three-member native Gemma adapter contract."""

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat


CONFIG="adapter_config.json"
WEIGHT="adapter_model.safetensors"
SIDECAR="additional_config.json"
MEMBERS={CONFIG,WEIGHT,SIDECAR}
CONFIG_PIN={"bytes":1178,"SHA256":"0e4846fd95dcf935bc20e3df098d37feedde78f3572c3417de4c6c64dcf51670"}
SIDECAR_PIN={"bytes":67,"SHA256":"c7799462ebedae6557ffad31566029e2d2f958b7b40e46e972cf901bcaf45733"}
ACTUAL_WEIGHT={"bytes":119280680,"SHA256":"48888329150da76fcd7fb6e53ecf6b7e94c9dfeb26a3f0443231382e3db14d7c"}
ACTUAL_NAMED="5663eaa9445a1c047af864e1d9375e152937fb4334ccab0237c10ceebdbc654e"
ARCHIVE_SHA="a606fa6867c6c18f9f26a9f00b279f79ffa14fd2efb77abcd542241a5067b4fd"
KEYS={"lora_dtype","lorap_lr_ratio","lorap_emb_lr"}


def need(ok,code):
    if not ok:
        raise ValueError(code)


def json_object(raw):
    def pairs(values):
        obj={}
        for key, value in values:
            need(key not in obj,"ADAPTER_DUPLICATE_JSON_KEY")
            obj[key]=value
        return obj
    def constant(_):
        raise ValueError("ADAPTER_NONFINITE_JSON_CONSTANT")
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)
    need(type(value) is dict,"ADAPTER_JSON_OBJECT_REQUIRED")
    return value


def sidecar(raw):
    value=json_object(raw)
    need(set(value)==KEYS and value["lora_dtype"] is None and value["lorap_lr_ratio"] is None
         and type(value["lorap_emb_lr"]) is float and math.isfinite(value["lorap_emb_lr"])
         and value["lorap_emb_lr"]==1e-6,
         "EXACT_SWIFT_ADDITIONAL_CONFIG_SCHEMA")
    need(len(raw)==SIDECAR_PIN["bytes"] and sha256(raw).hexdigest()==SIDECAR_PIN["SHA256"],
         "EXACT_OBSERVED_SWIFT_ADDITIONAL_CONFIG_BYTES")
    return {"keys":sorted(KEYS),"lora_dtype_is_null":True,"lorap_lr_ratio_is_null":True,
            "lorap_emb_lr_type":"finite_float","bytes":len(raw),"SHA256":SIDECAR_PIN["SHA256"]}


def identity(item):
    return tuple(getattr(item,k) for k in ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns","st_mode","st_nlink"))


def file_info(path):
    info=path.lstat()
    need(stat.S_ISREG(info.st_mode) and info.st_nlink==1,"ADAPTER_BUNDLE_SINGLE_REGULAR_MEMBER")
    return info


def bounded(path, *, limit, keep):
    info=file_info(path)
    need(0<info.st_size<=limit,"ADAPTER_MEMBER_BOUNDED_EXTENT")
    flags=os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_BINARY",0)
    body=bytearray()
    pin=sha256()
    extent=0
    with os.fdopen(os.open(path,flags),"rb") as stream:
        before=os.fstat(stream.fileno())
        need(identity(before)==identity(info),"ADAPTER_OPENED_MEMBER_IDENTITY")
        while extent<info.st_size:
            data=stream.read(min(1024**2,info.st_size-extent))
            need(data,"ADAPTER_MEMBER_TRUNCATED")
            extent+=len(data)
            pin.update(data)
            if keep:
                body.extend(data)
        need(not stream.read(1) and identity(before)==identity(os.fstat(stream.fileno()))==identity(file_info(path)),
             "ADAPTER_MEMBER_CHANGED_DURING_READ")
    return bytes(body),{"bytes":extent,"SHA256":pin.hexdigest()}


def prove(directory, *, read_weight=True):
    directory=Path(directory)
    entries=lambda: {p.name for p in directory.iterdir()}
    base=directory.lstat()
    need(stat.S_ISDIR(base.st_mode) and directory.resolve(strict=True)==directory.absolute()
         and entries()==MEMBERS,"EXACT_THREE_NATIVE_ADAPTER_MEMBERS")
    before={name:identity(file_info(directory/name)) for name in MEMBERS}
    raw,config=bounded(directory/CONFIG,limit=8192,keep=True)
    settings=json_object(raw)
    need(config==CONFIG_PIN and settings["r"]==16 and settings["lora_alpha"]==32
         and settings["lora_dropout"]==0.05,"EXACT_ORIGINAL_GEMMA_PEFT_CONFIG")
    raw,pin=bounded(directory/SIDECAR,limit=512,keep=True)
    extra=sidecar(raw)
    if read_weight:
        _,tensor=bounded(directory/WEIGHT,limit=160*1024**2,keep=False)
    else:
        tensor={"bytes":file_info(directory/WEIGHT).st_size}
    need(tensor["bytes"]==ACTUAL_WEIGHT["bytes"] and entries()==MEMBERS
         and identity(base)==identity(directory.lstat())
         and before=={name:identity(file_info(directory/name)) for name in MEMBERS},
         "ADAPTER_BUNDLE_MEMBER_IDENTITY_OR_EXTENT_CHANGED")
    return {"files":{CONFIG:config,WEIGHT:tensor,SIDECAR:pin},"sidecar":extra,"weight_rehashed":read_weight}


def historical(archive_raw):
    """Qualify metadata by the actual independent record, without reopening .4300 materials."""
    need(sha256(archive_raw).hexdigest()==ARCHIVE_SHA,"EXACT_PUBLISHED_4300_4302_ARCHIVE")
    value=json_object(archive_raw)
    reload=value["independent_native_adapter_reload"]
    report=reload["report"]
    ranks=value["actual_independent_rank_readback"]
    receipt=reload["accounting"]
    need(len(receipt)==1 and receipt[0][0]=="1103606.4302" and receipt[0][2:4]==["COMPLETED","0:0"]
         and receipt[0][6]=="cpu=2,mem=8G,node=1" and reload["rc"]==0 and not reload["error"]
         and report["actual_GPU"]==report["actual_optimizer_updates"]==0 and report["tensors_verified"]==476
         and report["exact_rank_value_match"] is report["finite"] is True
         and report["named_value_SHA256"]==ranks["identical_final_SHA256"]==ACTUAL_NAMED
         and report["adapter_config_SHA256"]==CONFIG_PIN["SHA256"]
         and report["additional_config_SHA256"]==SIDECAR_PIN["SHA256"]
         and (report["adapter_bytes"],report["adapter_file_SHA256"])==(ACTUAL_WEIGHT["bytes"],ACTUAL_WEIGHT["SHA256"])
         and ranks["actual_microbatches"]==64 and ranks["optimizer_updates_per_replica"]==2
         and ranks["all_rank_positive_update_step"]==[2] and ranks["supervisor_rank_receipts"]==8
         and value["GPU_terminal_failed"] is True and value["physical_device_release_verified"] is True
         and value["engineering_completed"] is False and value["formally_deliverable_model"] is False,
         "EXACT_INDEPENDENT_RELOAD_AND_RANK_PROVENANCE")
    return {"training_step":"1103606.4300","independent_reload_step":"1103606.4302",
            "actual_eight_rank_completed_updates":2,"actual_forward_inputs":64,
            "tensor_value_match":True,"native_bundle_identity_independently_verified":True,
            "files":{CONFIG:CONFIG_PIN,WEIGHT:ACTUAL_WEIGHT,SIDECAR:SIDECAR_PIN},
            "native_supervisor_terminal_success":False,"unknown_IPC_removed":False,
            "weight_payload_bytes_opened_here":0,"archive_SHA256":ARCHIVE_SHA}
