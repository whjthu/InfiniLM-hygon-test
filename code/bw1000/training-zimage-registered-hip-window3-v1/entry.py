"""Stage once or dispatch once on the existing Slurm allocation."""

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time


R = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "training-zimage-coco20-registered-hip-window3-v1"
REQUEST = "ef761e04519b422ba7f463ee6f19aa86"
PYTHON = "/usr/local/python3.12/bin/python3"
STAGE = R / "results" / LABEL
TARGET = "f13r2n07"
CONTROL = "f13r2n08"
PARENT = "1103606"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, record):
    with path.open("x", encoding="utf8") as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def verified_source(files):
    manifest_raw = files["SOURCE_FILES.json"]
    pins = json.loads(manifest_raw)["files"]
    need(set(pins) == files.keys() - {"SOURCE_FILES.json"}, "Different execution inventory")
    for name, pin in pins.items():
        need(re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name) and name not in (".", ".."),
             "Unbounded execution member name")
        data = files[name]
        need((len(data),sha(data)) == (pin["bytes"],pin["SHA256"]), "Execution member drift")
    claim = json.loads(files["CLAIM.json"])
    need((claim["label"],claim["request"],claim["parent"],claim["node"],claim["control_node"]) ==
         (LABEL,REQUEST,PARENT,TARGET,CONTROL),"Unexpected controlled GPU scope")
    return claim, sha(manifest_raw)


def stage():
    raw = sys.stdin.buffer.read(600001)
    need(0 < len(raw) <= 600000,"Execution source package exceeded the bound")
    p = json.loads(raw)
    publish = p["publication"]
    need(re.fullmatch(r"[a-f0-9]{40}",publish),"Missing published exact scope")
    files = {name:base64.b64decode(value,validate=True) for name,value in p["files"].items()}
    claim, manifest_sha = verified_source(files)
    need(manifest_sha == p["manifest_sha"],"Staged source manifest differs")
    runner = base64.b64decode(p["entry"],validate=True)
    need(sha(runner)==p["entry_sha"] and len(runner)<=30000,"Controller entry differs")
    common_root = R / "results/training-common-zimage-complete-window-v1"
    scheduler_root = R / "results/training-common-zimage-scheduler-guard-v2"
    binding_root = R / "results/training-common-zimage-registered-hip-v1"
    stage_raw = (common_root/"source-stage-receipt.json").read_bytes()
    cpu_raw = (binding_root/"cpu-outcome.json").read_bytes()
    scheduler_raw = (scheduler_root/"cpu-outcome.json").read_bytes()
    need(sha(scheduler_raw)==claim["scheduler_CPU"]["receipt_SHA256"]
         and json.loads(scheduler_raw)["scheduler_regression"]["changed_timestep_rejected"] is True,
         "Original native scheduler regression receipt differs")
    target_cpu = claim["registered_binding"]
    expected = claim["common_candidate"]
    receipt = json.loads(stage_raw)
    need(sha(stage_raw)==expected["source_stage_sha256"]
         and receipt["CLI_entry_reachable"] is True and sha(cpu_raw)==target_cpu["CPU_receipt_SHA256"]
         and receipt["framework"]==expected["candidate_commit"]
         and receipt["matching_source_files"]==396 and receipt["matching_changed_members"]==3
         and sha((common_root/"candidate-manifest.json").read_bytes())==expected["manifest_sha256"]
         and json.loads(cpu_raw)["rc"] == 0
         and sha((binding_root/"native-binding.json").read_bytes())==target_cpu["SHA256"]
         and sha((common_root/"source-tree/tools/hygon_training_recipe.py").read_bytes())==target_cpu["cli_SHA256"],
         "Corrected 394-file target tree or original native image evidence differs")
    for name,pin in claim["framework_update_members"].items():
        path=common_root/"source-tree/generation/src/gencuda"/name
        code=path.read_bytes()
        need((len(code),sha(code))==(pin["bytes"],pin["SHA256"]),"Pinned registered provider changed")
    for name,pin in claim["framework_changed_members"].items():
        code=(common_root/"source-tree"/name).read_bytes()
        need((len(code),sha(code))==(pin["bytes"],pin["SHA256"]),"Complete-window framework byte drift")
    original = R / "results/training-zimage-COCO20-native-first-step-hip-v2/first-update-private"
    progress = (original/"progress.json").read_bytes()
    binding = claim["fixed_hip_reference"]["progress"]
    need((len(progress),sha(progress))==(binding["bytes"],binding["SHA256"]),"Original first step differs")
    seen=set()
    for pin in json.loads(progress)["files"]:
        if pin["name"] not in {"initial.pt","prepared.pt","target.pt","forward.pt"}:
            continue
        path=original/pin["name"]
        need(not path.is_symlink() and path.is_file() and path.stat().st_size==pin["bytes"],"Original private archive absent")
        seen.add(pin["name"])
    need(len(seen)==4,"Missing exact archived first-step tensors")
    runtime = R / "results/training-zimage-modelscope-whole-runtime-v2/runtime"
    assets=json.loads(files["RUNTIME_EVIDENCE.json"])["independent_runtime"]["files"]
    need(len(assets)==20 and sum(a["bytes"] for a in assets)==33018433789,"Original complete runtime extent differs")
    for pin in assets:
        path=runtime/pin["path"]
        need(path.is_file() and path.stat().st_size==pin["bytes"],"Original corresponding runtime member absent")
    os.umask(0o077)
    STAGE.mkdir()
    for name,data in files.items():
        with (STAGE/name).open("xb") as stream:
            stream.write(data)
    with (STAGE/"entry.py").open("xb") as stream:
        stream.write(runner)
    result=dict(request=REQUEST,publication=publish,manifest_SHA256=manifest_sha,
                entry_SHA256=sha(runner),candidate=expected,asset_count=len(assets),
                original_capture_progress_SHA256=sha(progress),verified_operands=sorted(seen),
                source_files=len(files),gpu_submitted=False,observed_epoch=time.time())
    save(STAGE/"host-preflight.json",result)
    print(json.dumps(result,sort_keys=True),flush=True)


def launch():
    receipt=json.loads((STAGE/"host-preflight.json").read_bytes())
    need(receipt["request"]==REQUEST and receipt["gpu_submitted"] is False,"Original preflight receipt differs")
    need(sha(Path(__file__).read_bytes())==receipt["entry_SHA256"],"Own controller entry changed")
    manifest=(STAGE/"SOURCE_FILES.json").read_bytes()
    pins=json.loads(manifest)["files"]
    files={name:(STAGE/name).read_bytes() for name in pins}
    files["SOURCE_FILES.json"]=manifest
    claim,manifest_sha=verified_source(files)
    need(manifest_sha==receipt["manifest_SHA256"],"GPU execution package changed")
    need(not (STAGE/"intent.json").exists(),"The original launch already has an intent")
    import subprocess
    prior_labels=subprocess.check_output(["sacct","-n","-P","-j",PARENT,"-o","JobName%100"],text=True,timeout=10)
    need(not any(name.startswith(LABEL+"-") for name in prior_labels.splitlines()),"Existing series or conflicting label")
    epoch=time.time()
    save(STAGE/"intent.json",dict(request=REQUEST,epoch=epoch,deadline=epoch+claim["whole_seconds"],
                                manifest_SHA256=manifest_sha,publication=receipt["publication"],
                                parent=PARENT,node=TARGET,control_node=CONTROL))
    args=[
        "srun","--jobid="+PARENT,"--job-name="+LABEL+"-control","--exact","--nodes=1","--ntasks=1",
        "--nodelist="+CONTROL,"--cpus-per-task=2","--mem=4096M","--gres=none","--immediate=5","--threads=1",
        "--time=00:20:00","--kill-on-bad-exit=1",
        "--output="+str(STAGE/"controller.stdout"),"--error="+str(STAGE/"controller.stderr"),
        PYTHON,"-I","-S","-B",str(STAGE/"entry.py"),"control",
    ]
    os.execvp("srun",args)


def control():
    control_step=os.environ["SLURM_STEP_ID"]
    need(socket.gethostname().split(".")[0]==CONTROL and os.environ["SLURM_JOB_ID"]==PARENT
         and control_step.isdecimal(),"Control CPU step escaped original allocation")
    save(STAGE/"controller-owner.json",{"pid":os.getpid(),"step":PARENT+"."+control_step,"node":CONTROL})
    env={k:v for k,v in os.environ.items() if not k.startswith(("SLURM_","PMI_","PMIX_"))}
    os.execve(PYTHON,[PYTHON,"-I","-S","-B",str(STAGE/"controller.py")],env)


if __name__=="__main__":
    {"stage":stage,"launch":launch,"control":control}[sys.argv[1]]()
