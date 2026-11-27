"""Enter the vetted eight-rank Swift launch only after actual R39 admissions."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import runpy
import socket
import sys
import time

from assets import BASE, from_target
from short_mount import SHORT, verify_inside
from contract import ASSET, CORE, D, ROOT as RUN, REQUEST, PARENT, NODE, PORT, FRAMEWORK_SHA, STAGE_PIN, RUNTIME_SHA, Clock, stable_json, verify
from prepare import NEW, REQUEST as DATA_REQUEST, PLAN_SHA, digest, emit_file, encoded, need, pin, read


BINDING_SHA="7bd0b8e46b164ef179ac59a2525322dfb50b6802e3a277c45113a4e9bc6cc8a1"
KEY="gemma_usgs_r39_two_updates_witness"
CPU_TERMINAL_SHA="74bac702f6e94de63d3db93ad8744fffdb45b797881510792587ef76e415de02"
R40_ZERO_GRES_MODEL16_IDENTITY="5b2b918aa4b3fed4be6fd54fa8c54240d90a14eb5de36fcce96b4ed37f48f7bf"
PUBLISHED_CPU_MANIFEST="79835e80d07dfa8da194071a5f76a5df7eb3af2d8a2e5409f9ffdc5df091d5b3"


def build_plan(raw,run,src,view):
    plan=json.loads(raw)
    need(digest(raw)==PLAN_SHA and plan["resume_from_checkpoint"] is None,"R39_EXACT_FROZEN_NOMINAL25_PLAN")
    command=plan["command"].copy()
    replacements={"--dataset":view.as_posix(),"--output_dir":(run/"continuous/train").as_posix(),
        "--external_plugins":(src/"gemma_usgs_r39_two_updates.py").as_posix(),"--callbacks":KEY}
    for key,value in replacements.items():
        need(command.count(key)==1,"R39_EXACT_SINGLE_PLAN_FLAG")
        command[command.index(key)+1]=value
    env=dict(plan["environment_overrides"],DATASET=view.as_posix(),OUTPUT_DIR=(run/"continuous/train").as_posix())
    need(command[command.index("--model")+1]==BASE.as_posix() and command[command.index("--max_steps")+1]=="25"
         and "--resume_from_checkpoint" not in command and env["NPROC_PER_NODE"]=="8" and env["MASTER_PORT"]=="29932",
         "R39_PRESERVE_ORIGINAL_BASE_BF16_LORA_ADAMW_NOMINAL25")
    env["MASTER_PORT"]=str(PORT)
    need(not any(flag.split("=",1)[0].replace("-", "_")=="__use_logits_to_keep"
                 for flag in command if flag.startswith("--")), "R120_LOGITS_TO_KEEP_MUST_BE_UNSET_IN_SOURCE_PLAN")
    command.extend(("--use_logits_to_keep","false"))
    return command,env


def trust_cpu_stage():
    view,vpin=stable_json(NEW/"STAGE_VIEW.json")
    index,ipin=stable_json(NEW/"INDEX_RESULT.json")
    terminal,tpin=stable_json(NEW/"INDEX_INDEPENDENT_TERMINAL.json")
    need(tpin["SHA256"]==CPU_TERMINAL_SHA,"R41_EXACT_3406_INDEPENDENT_RECEIPT")
    intent,jpin=stable_json(NEW/"INDEX_INTENT.json")
    row=terminal["accounting"]
    need(terminal["independent"] is True and terminal["result"]==ipin and terminal["request"]==DATA_REQUEST
         and terminal["intent"]==jpin and intent["request"]==DATA_REQUEST and intent["epoch"]==index["epoch"]
         and intent["source_manifest_SHA256"]==PUBLISHED_CPU_MANIFEST and intent["stage_view_SHA256"]==vpin["SHA256"]
         and len(row)==9 and re.fullmatch(PARENT+r"\.[0-9]+",row[0]) and row[0]==index["step"]=="1103606.3406"
         and row[2:4]==["COMPLETED","0:0"] and row[5]==NODE and row[6] in ("cpu=4,mem=8G,node=1","cpu=4,mem=8192M,node=1")
         and row[4].isdigit() and 0<=int(row[4])<=240 and terminal["GPU"]==0,"R39_REAL_CPU_INDEX_TERMINAL_REQUIRED")
    need(view["request"]==index["request"]==DATA_REQUEST and index["passed"] is True
         and index["actual_signatures"]==64 and index["actual_optimizer_updates"]==index["GPU"]==0
         and view["old_bytes_unchanged"]==788 and view["new_train_rows"]==12 and view["new_pairs"]==6,
         "R39_EXACT_CPU_VIEW_AND_SIGNATURE_EVIDENCE")
    for name,record in (("train800-swift.jsonl",view["view"]),("lineage-private.json",view["lineage"]),
                        ("native64-index-private.json",index["native_index"])):
        read(NEW/name,record["SHA256"],size=record["bytes"])
    return view,index


def execute(args):
    operative=verify("GPU")
    verify_inside(check_temp_environment=True,require_marker_only=False)
    original_intent=json.loads(read(Path(args.run),args.run_sha,65536))
    check=Clock(original_intent["epoch"])
    need((original_intent["request_id"],original_intent["parent"],original_intent["node"],original_intent["root"])==
         (REQUEST,PARENT,NODE,str(RUN)) and original_intent["operational_source_manifest"]==operative
         and original_intent["native_deadline"]==original_intent["epoch"]+1980,"R39_EXACT_HELD_CHILD_INTENT")
    step=os.environ.get("SLURM_JOB_ID","")+"."+os.environ.get("SLURM_STEP_ID","")
    need(re.fullmatch(PARENT+r"\.[0-9]+",step) and socket.gethostname().split(".")[0]==NODE
         and not set(os.environ)&{"RANK","LOCAL_RANK","WORLD_SIZE"},"R39_ACTUAL_UNRANKED_NUMERIC_GPU_STEP")
    pre=json.loads(read(Path(args.admission),args.admission_sha,65536))
    need(pre["request_id"]==REQUEST and pre["node"]==NODE and pre["step"].startswith(PARENT+".") and pre["step"]!=step
         and pre["boot_id"]==original_intent["boot_id"] and pre["kfd_holder_count"]==0 and pre["full_fd_inventory_performed"] is True
         and pre["master_port"]==PORT and pre["master_port_available"] is True and len(set(pre["pci"]))==8
         and 0<=time.monotonic_ns()-pre["monotonic_ns"]<=45_000_000_000,"R39_RECENT_8PCI_BEFORE")
    stage,_=stable_json(ASSET/"stage-complete.json",STAGE_PIN)
    need(stage["passed"] is True and stage["training_request_id"]=="247fa3682a114542ac3640fbc726aaae",
         "R39_BOUND_ORIGINAL_845_FRESH_FRAMEWORK")
    manifest=json.loads(read(ASSET/"fresh-framework-files.json",FRAMEWORK_SHA))["files"]
    need(len(manifest)==396,"R39_FRAMEWORK396")
    for name,record in manifest.items():
        need(not Path(name).is_absolute() and ".." not in Path(name).parts,"R39_SAFE_PINNED_FRAMEWORK_MEMBER")
        read(ASSET/"framework"/name,record["SHA256"],16<<20,record["bytes"])
        check.left(1980)
    read(ASSET/"runtime-binding.json",RUNTIME_SHA)
    model_bindings=from_target(lambda:check.left(1980))
    need(len(model_bindings)==16,"R41_EXACT_CURRENT_MODEL16")
    model_binding_sha=digest(encoded(model_bindings).rstrip(b"\n"))
    view,index=trust_cpu_stage()
    snapshot=runpy.run_path(str(D/"snapshot.py"))
    function=snapshot["snapshot"]
    function.__globals__["NODE"]=NODE
    seen=function(True)
    need(seen["pci"]==pre["pci"] and seen["boot_id"]==pre["boot_id"] and seen["kfd_holder_count"]==0,
         "R39_ACTUAL_ALLOCATED_KERNEL_IDENTITIES")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",PORT))
    hip=ctypes.CDLL("libamdhip64.so")
    count=ctypes.c_int()
    need(hip.hipGetDeviceCount(ctypes.byref(count))==0 and count.value==8,"R39_ACTUAL_EIGHT_HIP_DEVICES")
    pcis=[]
    for slot in range(8):
        bus=ctypes.create_string_buffer(32)
        need(hip.hipDeviceGetPCIBusId(bus,32,slot)==0,"R39_REAL_HIP_PCI")
        address=bus.value.decode().lower()
        pcis.append("0000:"+address if len(address)==7 else address)
    need(sorted(pcis)==seen["pci"],"R39_EXACT_HIP_AND_KERNEL_PCI_ORDER")
    read(ASSET/"structural-binding.json",BINDING_SHA,106000)
    command,overrides=build_plan(read(ASSET/"continuous-plan.json",PLAN_SHA),RUN,D,NEW/"train800-swift.jsonl")
    arm=RUN/"continuous"
    arm.mkdir(mode=0o700)
    config={"case":"T02.sft","dataset_SHA256":view["view"]["SHA256"],"view_pin":view["view"],
        "binding":str(ASSET/"structural-binding.json"),"lineage":str(NEW/"lineage-private.json"),"lineage_SHA256":view["lineage"]["SHA256"],
        "native_index":str(NEW/"native64-index-private.json"),"native_index_SHA256":index["native_index"]["SHA256"],
        "initialization_mode":"TARGET_FRESH_DOMAIN_ONLY","seed":42,"data_seed":42}
    raw=encoded(config)
    emit_file(arm/"witness-config.json",raw)
    env=dict(os.environ,**overrides)
    env.pop("PYTHONPYCACHEPREFIX",None)
    env.update(MASTER_ADDR="127.0.0.1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONDONTWRITEBYTECODE="1",
        NATIVE_CONTINUOUS_CONFIG=str(arm/"witness-config.json"),NATIVE_CONTINUOUS_CONFIG_SHA256=digest(raw),
        NATIVE_CONTINUOUS_EPOCH=str(original_intent["epoch"]),NATIVE_CONTINUOUS_DEADLINE=str(original_intent["native_deadline"]),
        NATIVE_CONTINUOUS_MAX_SECONDS="2400",MODELSCOPE_CACHE=str(arm/"private-modelscope-cache"))
    (arm/"private-modelscope-cache").mkdir(mode=0o700)
    for variable in ("HF_HOME","HF_HUB_CACHE","HF_DATASETS_CACHE","XDG_CACHE_HOME","TORCHINDUCTOR_CACHE_DIR"):
        scratch=arm/("private-"+variable.lower())
        scratch.mkdir(mode=0o700)
        env[variable]=str(scratch)
    env["TEMP"]=env["TMP"]=env["TMPDIR"]=str(SHORT)
    emit_file(arm/"entry.json",encoded({"request":REQUEST,"step":step,"view":view["view"],"index":index["native_index"],
        "actual_visible_PCI":pcis,"base_model16_inode_binding_SHA256":model_binding_sha,
        "r40_full_stat_digest_equal":model_binding_sha==R40_ZERO_GRES_MODEL16_IDENTITY,
        "new_shard_payload_bytes_rehashed":0,
        "native_nominal25":True,"actual_optimizer_updates":None,"read_rejected_A100_initial":False}))
    check.left(1980)
    need(0<=time.monotonic_ns()-pre["monotonic_ns"]<=45_000_000_000,"R39_ADMISSION_STALE_BEFORE_NATIVE_EXEC")
    need(re.fullmatch("[0-9a-f]{64}",env.get("GEMMA_IPC_ASSIGNMENTS_SHA256","")),
         "R120_SUPERVISOR_IPC_HANDOFF_REQUIRED")
    claim,_=stable_json(D/"CLAIM.json")
    env["GEMMA_IPC_OPERATION"]=str(RUN)
    env["GEMMA_IPC_RANK_ENTRY"]=claim["native_rank_entry"]["path"]
    os.execve(sys.executable,[sys.executable,"-B",str(D/"elastic_argv.py"),*command],env)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("run","run-sha","admission","admission-sha"):
        p.add_argument("--"+name,required=True)
    execute(p.parse_args())
