"""Immutable GLM continuation scope and original container CPU receipt."""
from hashlib import sha256
import json
import os
from pathlib import Path

R=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
D=Path(__file__).resolve().parent
PACKAGE=R/"incoming/GLM952-prefix-continuation/provider"
RUN="prefix-continuation-20261105-a"
LABEL="GLM952-prefix-continuation"
NODES=["a02r3n15","a02r3n16"]
OUTPUT=R/"results/GLM952-prefix-continuation"/RUN
TERMINAL={"COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED","BOOT_FAIL","DEADLINE","REVOKED"}

def sha(path):return sha256(Path(path).read_bytes()).hexdigest()

def clean_environment(base):
    return {k:v for k,v in base.items() if not k.startswith(("SLURM_","PMI_","PMIX_","GLM_"))
            and k not in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES","LD_PRELOAD","LD_AUDIT")}

def publish(path,value):
    path=Path(path)
    with path.open("x") as out:json.dump(value,out,indent=2,allow_nan=False)

def verify(expected,require_ready=True):
    path=D/"GPU_MANIFEST.json"
    assert sha(path)==expected
    m=json.loads(path.read_bytes())
    assert m["GPU_GO"] and m["once"] and not m["retry"] and not m["reload"]
    assert m["scope"]=="GLM952 staged-prefix continuation"
    assert m["nodes"]==NODES and m["cards"]==16 and m["parent_job"]=="910371"
    assert (m["Native_seconds"],m["service_seconds"],m["GPU_seconds"],m["keeper_seconds"])==(3480,3360,3600,3900)
    assert (m["cases"],m["forwards_per_owner"])==(3,55)
    assert sha(PACKAGE/"SOURCE_MANIFEST.json")==m["source_manifest_SHA256"]
    source=json.loads((PACKAGE/"SOURCE_MANIFEST.json").read_bytes())
    for name,pin in source["files_SHA256"].items():assert sha(PACKAGE/name)==pin
    assert sha(D/"config.json")==m["provider_config_SHA256"]
    config=json.loads((D/"config.json").read_bytes())
    assert config["nodes"]==NODES and (config["TP"],config["PP"],config["rows"],config["context"])==(8,2,1,32768)
    assert config["prefix_continuation_execution_authorized"] is True
    assert (config["Native_seconds"],config["service_seconds"])==(3480,3360)
    for name,pin in m["files_SHA256"].items():assert sha(D/name)==pin
    cpu=R/"results/GLM952-prefix-continuation-entry-CPU-v1/result.json"
    assert sha(cpu)==m["entry_CPU_SHA256"]=="129d5a02968c9f353bfa9a83904663ec74344e097a53275c118b86496f198bbf"
    receipt=json.loads(cpu.read_bytes())
    assert receipt["passed"] and receipt["completed"] and receipt["actual_CPU"]=="910371.841"
    assert receipt["package_manifest_SHA256"]==m["source_manifest_SHA256"]
    assert receipt["native_identity"]["library_SHA256"]==m["Native_SHA256"]
    assert sha(R/"incoming/GLM-W4-remapped-final-slots-r469/deployment-v1/idle.py")==m["original_idle_SHA256"]
    assert sha(R/"incoming/GLM-W4-remapped-final-slots-r469/deployment-v1/inside.sh")==m["original_environment_SHA256"]
    assert sha(R/"incoming/Deep-C1-gap-enclosure-r467/profile-source-v1/CPU_outer.sh")==m["outer_SHA256"]
    return m
