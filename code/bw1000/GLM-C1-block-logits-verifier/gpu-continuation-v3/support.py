"""Immutable GLM continuation scope and original container CPU receipt."""
from hashlib import sha256
import json
import os
from pathlib import Path

R=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
D=Path(__file__).resolve().parent
PACKAGE=R/"incoming/GLM952-prefix-continuation/native-admission-v2/provider"
RUN="prefix-continuation-admission-v3-20261105-a"
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
    assert receipt["package_manifest_SHA256"]==m["entry_parent_package_manifest_SHA256"]=="948697298cefb664f206b6a650aa9c50311bf269764ce1a7bcfc9b11bd4b8a61"
    old_package=R/"incoming/GLM952-prefix-continuation/provider"
    assert sha(old_package/"SOURCE_MANIFEST.json")==m["entry_parent_package_manifest_SHA256"]
    old_members=json.loads((old_package/"SOURCE_MANIFEST.json").read_bytes())["files_SHA256"]
    assert set(source["files_SHA256"])==set(old_members)
    changed=[name for name,pin in source["files_SHA256"].items() if old_members[name]!=pin]
    member="public_glm_native/model_admission_r404.py"
    assert changed==[member]
    before=(old_package/member).read_bytes()
    after=(PACKAGE/member).read_bytes()
    assert sha256(before).hexdigest()==m["parent_admission_SHA256"]=="528d53c1ca61ef8066e7d89ffa9c7fb1937c84b72963003267bae87e04b29d16"
    assert sha256(after).hexdigest()==m["fixed_admission_SHA256"]=="25a0c9b97ae3ce3e2194bc36517cc374138224945ea8212a3a2b92bc63eaaafd"
    expected=before.decode().replace("\r\n","\n").replace(
        "/refactor/candidates/GLM-V-on-demand-Model-integration-r403-CPU/libGLM_V_on_demand_Model_r403.so",
        "/refactor/candidates/GLM952-block-prefix-native-v1/libGLM952_block_prefix_v1.so").replace(
        "9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b",
        "35c08ce45814178e3692c55eb13dd5f074643993be1fbe911ecf91a2c200a0cb").encode()
    assert after==expected
    assert receipt["native_identity"]["library_SHA256"]==m["Native_SHA256"]
    assert sha(R/"incoming/GLM-W4-remapped-final-slots-r469/deployment-v1/idle.py")==m["original_idle_SHA256"]
    assert sha(R/"incoming/GLM-W4-remapped-final-slots-r469/deployment-v1/inside.sh")==m["original_environment_SHA256"]
    assert sha(R/"incoming/Deep-C1-gap-enclosure-r467/profile-source-v1/CPU_outer.sh")==m["outer_SHA256"]
    repair=R/"results/GLM952-prefix-continuation-env-CPU-v2/result.json"
    assert sha(repair)==m["environment_CPU_SHA256"]=="d89833f4d94b4b01419e017ff7c4618ae1388233bf4bf6576de44f92dbbd502b"
    q=json.loads(repair.read_bytes())
    assert q["passed"] and q["actual_CPU"]=="910371.845"
    assert q["janus"]["SHA256"]==m["janus_SHA256"]=="4fbcc6e7c58f3c902f140b4d6f449df4f8e5c17dce277dece5a4d531145d1fa4"
    return m
