"""Immutable GLM continuation scope and original container CPU receipt."""
from hashlib import sha256
import json
import os
from pathlib import Path

R=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
D=Path(__file__).resolve().parent
PACKAGE=R/"incoming/GLM952-prefix-logit-retention-v1/provider"
RUN="prefix-logit-retention-20261105-a"
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


PARENT_D=R/'incoming/GLM952-prefix-continuation/gpu-continuation-v5'
PARENT_PACKAGE=R/'incoming/GLM952-prefix-continuation/native-admission-v2/provider'
def verify_parent():
    path=PARENT_D/"GPU_MANIFEST.json"
    assert sha(path)=='c8bc5650d8a135cf76aa87dc1caa50a33205c5f2f01f22e71921f60ba1815b45'
    m=json.loads(path.read_bytes())
    assert m["GPU_GO"] and m["once"] and not m["retry"] and not m["reload"]
    assert m["scope"]=="GLM952 staged-prefix continuation"
    assert m["nodes"]==NODES and m["cards"]==16 and m["parent_job"]=="910371"
    assert (m["Native_seconds"],m["service_seconds"],m["GPU_seconds"],m["keeper_seconds"])==(3480,3360,3600,3900)
    assert (m["cases"],m["forwards_per_owner"])==(3,55)
    assert sha(PARENT_PACKAGE/"SOURCE_MANIFEST.json")==m["source_manifest_SHA256"]
    source=json.loads((PARENT_PACKAGE/"SOURCE_MANIFEST.json").read_bytes())
    for name,pin in source["files_SHA256"].items():assert sha(PARENT_PACKAGE/name)==pin
    assert sha(PARENT_D/"config.json")==m["provider_config_SHA256"]
    config=json.loads((PARENT_D/"config.json").read_bytes())
    assert config["nodes"]==NODES and (config["TP"],config["PP"],config["rows"],config["context"])==(8,2,1,32768)
    assert config["prefix_continuation_execution_authorized"] is True
    assert (config["Native_seconds"],config["service_seconds"])==(3480,3360)
    for name,pin in m["files_SHA256"].items():assert sha(PARENT_D/name)==pin
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
    after=(PARENT_PACKAGE/member).read_bytes()
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
    network=PARENT_D
    for node,pin in m["network_CPU_receipts_SHA256"].items():
        path=network/("CPU853-"+node+".json")
        assert sha(path)==pin
        q=json.loads(path.read_bytes())
        assert q["passed"] and q["actual_CPU"]=="910371.853" and q["node"]==node
        assert q["GPU"] is False and q["candidate_Native_dlopen"] is False
        assert q["janus_and_verbs"]["verbs_devices"]==["shca_0","shca_1","shca_2","shca_3"]
    return m

def verify(expected,require_ready=True):
    old=verify_parent()
    path=D/'GPU_MANIFEST.json';assert sha(path)==expected;m=json.loads(path.read_bytes())
    assert m['scope']=='GLM952 staged-prefix continuation'and m['once']and not m['retry']and not m['reload']
    if require_ready:assert m['GPU_GO']and m['execution_READY']
    assert(m['cases'],m['forwards_per_owner'])==(1,25)
    assert m['nodes']==NODES and m['parent_job']=='910371'and m['cards']==16
    for key in ['Native_SHA256','Native_CPU_receipt_SHA256','entry_CPU_SHA256','entry_parent_package_manifest_SHA256','Native_seconds','service_seconds','GPU_seconds','keeper_seconds']:assert m[key]==old[key],key
    assert sha(D/'config.json')==m['provider_config_SHA256']
    assert json.loads((D/'config.json').read_bytes())==json.loads((PARENT_D/'config.json').read_bytes())
    assert sha(PACKAGE/'SOURCE_MANIFEST.json')==m['source_manifest_SHA256']
    source=json.loads((PACKAGE/'SOURCE_MANIFEST.json').read_bytes());parent=json.loads((PARENT_PACKAGE/'SOURCE_MANIFEST.json').read_bytes())
    expected_changes={'public_glm_native/'+n for n in ['accepted_prefix_continuation.py','accepted_prefix_owner.py','resident_continuation.py','stage_compare.py','retain_logits.py']}
    assert set(source['files_SHA256'])==set(parent['files_SHA256'])|{'public_glm_native/retain_logits.py'}
    assert{k for k,h in source['files_SHA256'].items()if parent['files_SHA256'].get(k)!=h}==expected_changes
    for name,h in source['files_SHA256'].items():assert sha(PACKAGE/name)==h,name
    for name,h in m['files_SHA256'].items():assert sha(D/name)==h,name
    claim_local=json.loads((D/'local-claim-result.json').read_bytes());assert claim_local['passed']and claim_local['actual_require_claim']and claim_local['old_three55_rejected']and not claim_local['Torch_imported']
    local=json.loads((D/'local-entry-result.json').read_bytes());assert local['passed']and local['actual_entry']and local['actual_resident_selector']and not local['Native_execution']and not local['SDK_execution']and local['GPU']==0
    return m
