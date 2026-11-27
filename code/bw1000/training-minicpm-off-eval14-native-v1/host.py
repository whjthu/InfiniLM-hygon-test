"""Install, check a current parent and render one answerless GPU command."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

from contract import CP2, CP_ADAPTER, CP_FILES, CP_INVENTORY_SHA, CP_RELOAD_SHA, HEX, HEX32, INPUT, MODEL, MODEL_MANIFEST_SHA, PROMPT_SHA, RELOAD_ROW, ROOT, TRAIN, actual_reload, encode, inputs, need, put, small, verify_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = ROOT/"results"
RUNTIME = BASE/"training-vlm-swift-offline-runtime-v1"
TRANSFORMERS = BASE/"training-minicpm-hygon-processor-r2"
VISION = BASE/"training-qwen-vision014-CPU-v2/vision"
HOME_SOURCE = "8bb74a51d200842edbde3cdb93e5f5264eeefa4efab41e985de61ecea522b61"
PROVIDER_SOURCE = "4196a19e4f3fae4eb399ef76f752bbd4f009267abaebd4379237d71373951479"
FRAMEWORK_MANIFEST = "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
RANK_HOME = Path("/mini-off-home")
COLS = "JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"


def label(request):
    need(type(request) is str and HEX32.fullmatch(request), "FRESH_REQUEST")
    return "training-minicpm-off-eval14-native-"+request[:12]


def installation(request, publication, manifest_sha, cp_manifest_sha):
    target = ROOT/"incoming"/label(request)
    need(re.fullmatch(r"[0-9a-f]{40}", publication) and HEX.fullmatch(manifest_sha)
         and cp_manifest_sha==CP_INVENTORY_SHA and not target.exists() and not target.is_symlink(), "EXCLUSIVE_SOURCE_STAGE")
    manifest = small(HERE/"SOURCE_FILES.json", 16384, manifest_sha)
    spec = json.loads(manifest)["files"]
    need(set(spec)=={"contract.py","host.py","native.py","check_local.py","HANDOFF.md"}, "EXACT_PUBLISHED_SOURCE_MEMBERS")
    source = {name: small(HERE/name, 128*1024, pin["SHA256"]) for name,pin in spec.items()}
    need(all(len(raw)==spec[name]["bytes"] for name,raw in source.items()), "SOURCE_SIZE_DIFFERS")
    reload_rows = [x.split("|") for x in query("sacct","-nP","-j",RELOAD_ROW[0],"-o",COLS)]
    reload_rows = [x for x in reload_rows if x[0]==RELOAD_ROW[0]]
    reload = actual_reload(small(TRAIN/"independent-reload-cp2/result.json",32768,CP_RELOAD_SHA), reload_rows)
    prompts = inputs(INPUT)
    cp_inventory = json.loads(small(TRAIN/"continuous-checkpoint-2-files.json", 32768, cp_manifest_sha))
    off = json.loads(small(TRAIN/"off-first-two.json", 32768))
    need(cp_inventory["global_step"]==2 and cp_inventory["member_count"]==14
         and off["request"]=="c6dab1cb20894c45a2bf140a15fb0f2f" and off["adapter"]==CP_ADAPTER
         and off["checkpoint_inventory"]["SHA256"]==cp_manifest_sha and off["native_forwards"]==64
         and off["actual_optimizer_updates"]==2 and off["same_update_on_all_ranks"] is True, "ACTUAL_CP2_INVENTORY_LINK")
    adapter = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        wanted = cp_inventory["files"][name]
        need(wanted==CP_FILES[name], "EXACT_CPU_TESTED_CP2_MEMBERS")
        pin, key = verify_file(CP2/name, 128*1024**2, wanted["SHA256"])
        need(pin==wanted, "ACTUAL_CP2_MEMBER_SIZE")
        adapter[name] = dict(pin, identity=key)
    model_pins = json.loads(small(REPO/"code/bw1000/training-minicpm-final-tail-r2/MODEL_FILES.json", 32768,
                                  MODEL_MANIFEST_SHA))["files"]
    need(len(model_pins)==10 and "model.safetensors" in model_pins, "BASE_MODEL_MANIFEST")
    base = {}
    for name, wanted in model_pins.items():
        need(Path(name).name == name, "BASE_MEMBER_PATH")
        pin, key = verify_file(MODEL/name, wanted["bytes"], wanted["SHA256"])
        need(pin==wanted, "CURRENT_BASE_MEMBER_SHA")
        base[name] = dict(pin, identity=key)
    common = BASE/"training-qwen-a68952c-swift-initial-CPU-v1"
    framework = json.loads(small(common/"source/FRAMEWORK_FILES.json", 1024**2, FRAMEWORK_MANIFEST))["files"]
    framework_helpers = {}
    for name in ("vlm/runtime/first_update_audit.py", "vlm/runtime/first_update_capture.py"):
        pin = framework[name]
        usable = small(common/"framework"/name, 1024**2, pin["SHA256"])
        need(len(usable)==pin["bytes"], "NATIVE_DIGEST_PROVIDER")
        framework_helpers[name] = {"bytes":len(usable),"SHA256":pin["SHA256"]}
    pinned_deps = {
        "internvl_gqa_miopen_home.py": small(REPO/"code/bw1000/training-domain-pilot-v1/internvl_gqa_miopen_home.py",16384,HOME_SOURCE),
        "providers.py": small(REPO/"code/bw1000/training-minicpm-usgs-native88-CPU-v1/providers.py",16384,PROVIDER_SOURCE),
    }
    target.mkdir(mode=0o700)
    membership = {name:put(target/name, raw) for name,raw in {**source,**pinned_deps,"SOURCE_FILES.json":manifest}.items()}
    record = dict(request=request, publication=publication, source_manifest_SHA256=manifest_sha,
                  cp2_manifest_SHA256=cp_manifest_sha, prompt_SHA256=PROMPT_SHA, planned=28, prompt_rows=len(prompts),
                  base=base, adapter=adapter, framework_helpers=framework_helpers, files=membership, independent_reload=reload,
                  state="ANSWERLESS_ONLY_STAGE", actual_GPU=0, gold_read=False)
    return dict(request=request, label=label(request), stage=put(target/"STAGE.json",encode(record)), planned=28)


def query(*argv):
    return subprocess.check_output(argv, text=True, timeout=8, env=dict(os.environ,TZ="UTC")).splitlines()


def record_fields(line):
    return dict(x.split("=",1) for x in line.split() if "=" in x)


def parent_clear(parent, node, runner=query):
    need(re.fullmatch(r"[0-9]+", parent) and re.fullmatch(r"f[0-9]+r[0-9]+n[0-9]+",node), "PARENT_SCOPE")
    cached = {}
    def covers(nodes):
        need(nodes not in ("", "(null)", "None", "N/A"), "UNKNOWN_STEP_NODESET")
        if nodes not in cached:
            cached[nodes] = runner("scontrol","show","hostnames",nodes)
        return node in cached[nodes]
    job, = map(record_fields, runner("scontrol","show","job",parent,"-o"))
    other, = runner("squeue","-h","-j",parent,"-o","%i|%T|%N")
    number, state, nodes = other.split("|")
    need(job["JobId"] == number == parent and job["JobState"] == state == "RUNNING"
         and job["UserId"].endswith(f"({os.getuid()})")
         and covers(nodes) and covers(job["NodeList"]), "PARENT_OWNER_OR_MEMBERSHIP")
    for line in runner("scontrol","show","step",parent,"-o"):
        row = record_fields(line)
        step = row["StepId"]
        if step not in (parent+".extern",parent+".batch"):
            need(re.fullmatch(parent+r"\.[0-9]+",step) and not covers(row["NodeList"]), "NUMBERED_CONTROLLER_STEP")
    for line in runner("squeue","--steps","-h","-j",parent,"-o","%i|%N"):
        step, nodes = line.split("|")
        if step not in (parent+".extern",parent+".batch"):
            need(re.fullmatch(parent+r"\.[0-9]+",step) and not covers(nodes), "NUMBERED_QUEUE_STEP")


def accounting(parent, name, runner=query):
    rows = [x.split("|") for x in runner("sacct","-nP","-j",parent,"-o",COLS)]
    need(all(len(row)==9 for row in rows), "ACCOUNTING_COLUMNS")
    return [x for x in rows if x[1]==name]


def checked_stage(request, digest):
    target = ROOT/"incoming"/label(request)
    stage = json.loads(small(target/"STAGE.json", 256*1024, digest))
    need(stage["request"]==request and stage["prompt_SHA256"]==PROMPT_SHA and stage["planned"]==28
         and stage["state"]=="ANSWERLESS_ONLY_STAGE" and stage["actual_GPU"]==0 and stage["gold_read"] is False, "STAGE_BINDING")
    for name,pin in stage["files"].items():
        need(Path(name).name==name and len(small(target/name,128*1024,pin["SHA256"]))==pin["bytes"], "CURRENT_STAGED_SOURCE")
    return target,stage


def build_command(target, work, request, stage_sha, ready_sha, parent, node):
    common = BASE/"training-qwen-a68952c-swift-initial-CPU-v1/framework"
    command = [APPTAINER,"exec","--rocm","--no-mount","home,cwd","--pwd","/"]
    mounts = ((target,target,"ro"),(INPUT,INPUT,"ro"),(MODEL,MODEL,"ro"),(CP2,CP2,"ro"),(work,work,"rw"),
              (RUNTIME,"/swift-runtime","ro"),(TRANSFORMERS/"cpu/runtime-site","/minicpm-transformers","ro"),
              (VISION,"/vision","ro"),(BASE/"training-vlm-swift-public-artifacts-v1/packages","/wheels/swift","ro"),
              (TRANSFORMERS/"packages","/wheels/transformers","ro"),(Path("/usr/local/hyhal"),"/opt/hyhal","ro"),
              (common,"/mini-framework","ro"),(work/"home",RANK_HOME,"rw"))
    for source,dest,mode in mounts:
        command.extend(("--bind",f"{source}:{dest}:{mode}"))
    inc = ("/vision","/minicpm-transformers","/swift-runtime/site","/mini-framework",str(target))
    prep = [
        "set -euo pipefail", "ulimit -c 0", "source /opt/dtk/env.sh",
        "unset XDG_CONFIG_HOME XDG_CACHE_HOME XDG_DATA_HOME MIOPEN_SYSTEM_DB_PATH MIOPEN_LOCKFILE_PATH MIOPEN_LOCKFILE_DIR",
        "export HOME=/mini-off-home HF_HOME=/mini-off-home/.cache/huggingface MIOPEN_USER_DB_PATH=/mini-off-home/.config/miopen/user-db MIOPEN_CUSTOM_CACHE_DIR=/mini-off-home/.cache/miopen",
        "export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=2",
        "export PYTHONPATH="+shlex.quote(":".join(inc)),
        "exec "+shlex.join(["python3","-B","-u",str(target/"native.py"),"--request",request,"--stage",str(target/"STAGE.json"),
                           "--stage-sha",stage_sha,"--ready",str(work/"READY.json"),"--ready-sha",ready_sha,
                           "--output",str(work/"predictions"),"--home",str(RANK_HOME)]),
    ]
    command += [str(ROOT/"containers/dtk26-sandbox-r1"),"/bin/bash","--noprofile","--norc","-c","; ".join(prep)]
    return ["srun","--jobid="+parent,"--job-name="+label(request),"--exact","--immediate=5","--cpu-bind=none",
            "-N1","-n1","--nodelist="+node,"--cpus-per-task=8","--mem=64000M","--gres=dcu:1","--time=00:15:00",*command]


def preflight(request, stage_sha, parent, node, grant_path, grant_sha):
    target, stage = checked_stage(request,stage_sha)
    grant = json.loads(small(grant_path,16384,grant_sha))
    observed = time.time()
    need(grant["request"]==request and grant["parent"]==parent and grant["node"]==node
         and grant["stage_SHA256"]==stage_sha and grant["root_GO"] is True and grant["ordinary_excluded"] is True
         and grant["physical_clean"] is True and grant["prior_holder_reconciled"] is True
         and grant["observed_epoch"] <= observed < grant["expires_epoch"] <= grant["observed_epoch"] + 60
         and re.fullmatch(r"[0-9a-f-]{36}",grant["boot"]) and len(grant["pci"])==len(set(grant["pci"]))==8
         and all(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",x) for x in grant["pci"]),
         "FRESH_PHYSICAL_AND_ROOT_AUTHORITY")
    need(not accounting(parent,label(request)) and not (BASE/label(request)).exists(), "EVAL_NAMESPACE_NOT_FRESH")
    parent_clear(parent,node)
    need(time.time()<grant["expires_epoch"], "AUTHORITY_EXPIRED_DURING_PREFLIGHT")
    return target,stage,grant


def start(args):
    target, stage, grant = preflight(args.request,args.stage_sha,args.parent,args.node,args.grant,args.grant_sha)
    work = BASE/label(args.request)
    work.mkdir(mode=0o700)
    home = work/"home"
    home.mkdir(mode=0o700)
    for directory in (".config",".config/miopen",".config/miopen/user-db",".config/miopen/miopen-lockfiles",
                      ".cache",".cache/miopen",".cache/huggingface"):
        (home/directory).mkdir(mode=0o700)
    epoch = time.time()
    need(epoch<grant["expires_epoch"], "AUTHORITY_EXPIRED_BEFORE_INTENT")
    ready = dict(request=args.request,parent=args.parent,node=args.node,stage_SHA256=args.stage_sha,input_SHA256=PROMPT_SHA,
                 boot=grant["boot"],pci=grant["pci"],epoch=epoch,deadline=epoch+915,output=str(work/"predictions"))
    pin = put(work/"READY.json",encode(ready))
    argv = build_command(target,work,args.request,args.stage_sha,pin["SHA256"],args.parent,args.node)
    put(work/"intent.json",encode(dict(request=args.request,ready=pin,stage_SHA256=args.stage_sha,source_manifest_SHA256=stage["source_manifest_SHA256"],
                                     root_grant_SHA256=args.grant_sha,planned=28,argv=argv,epoch=epoch,replay=False)))
    result = dict(request=args.request,planned=28,rc=None,disposition="UNKNOWN",terminal=None)
    try:
        need(time.time()<grant["expires_epoch"], "AUTHORITY_EXPIRED_BEFORE_SRUN")
        with (work/"step.private.log").open("xb") as log:
            result["rc"] = subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                         cwd="/",timeout=925).returncode
        rows = accounting(args.parent,label(args.request))
        if len(rows)==1 and rows[0][2] in ("COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL") and rows[0][8] not in ("","Unknown"):
            result.update(disposition="TERMINAL_VISIBLE",terminal=rows[0])
    except BaseException as error:
        result["error_type"] = type(error).__name__
    finally:
        put(work/"host-outcome.json", encode(result))
    return result


def terminal(request, parent, runner=query):
    name = label(request)
    work = BASE/name
    rows = accounting(parent,name,runner)
    result = dict(request=request,accounting_UTC=rows,planned=28,submitted=False,passed=False,seal_present=False)
    if len(rows)!=1:
        return result
    row = rows[0]
    tres = dict(value.split("=",1) for value in row[6].split(","))
    need(len(tres)==len(row[6].split(",")) and re.fullmatch(parent+r"\.[0-9]+",row[0])
         and set(tres)=={"cpu","gres/dcu","mem","node"}
         and tres["cpu"]=="8" and tres["gres/dcu"]=="1" and tres["node"]=="1" and tres["mem"]=="64000M",
         "EXACT_NATIVE_TERMINAL_TRES")
    if not (work/"READY.json").is_file():
        return result
    ready = json.loads(small(work/"READY.json",16384))
    need(ready["request"]==request and ready["parent"]==parent and ready["node"]==row[5], "EXACT_NATIVE_TERMINAL_SCOPE")
    output = work/"predictions"
    if not (output/"SEALED.json").is_file():
        return result
    seal_pin,_ = verify_file(output/"SEALED.json",65536)
    sealed = json.loads(small(output/"SEALED.json",65536,seal_pin["SHA256"]))
    need(sealed["request"]==request and sealed["planned"]==28 and sealed["prompt_SHA256"]==PROMPT_SHA
         and sealed["gold_read"] is False and sealed["source_facts"]["actual_step"]==row[0],"SEALED_TERMINAL_IDENTITY")
    for arm in ("base","adapter"):
        pin = sealed["arms"][arm]
        if pin["bytes"]==0:
            need((output/(arm+".jsonl")).lstat().st_size==0, "EMPTY_PRIMARY")
        else:
            actual,_=verify_file(output/(arm+".jsonl"),256*1024,pin["SHA256"])
            need(actual==pin, "PRIMARY_FILE_PIN")
    ledger,_=verify_file(output/"ledger.private.json",65536,sealed["full_ledger"]["SHA256"])
    need(ledger==sealed["full_ledger"], "FULL_LEDGER_PIN")
    result.update(seal_present=True,seal=seal_pin,counts=sealed["counts"],arms=sealed["arms"])
    result["passed"]=(row[2:4]==["COMPLETED","0:0"] and row[4].isdigit() and 0<int(row[4])<=900
                      and row[7]<row[8] and row[7] not in ("","Unknown") and row[8] not in ("","Unknown")
                      and sealed["passed"] is True)
    result["independent_post_required_for_score"] = True
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("install","preflight","start","terminal"))
    for key in ("request","publication","source-manifest","cp-manifest-sha","stage-sha","parent","node","grant-sha"):
        parser.add_argument("--"+key,required=key=="request")
    parser.add_argument("--grant",type=Path)
    args=parser.parse_args()
    os.umask(0o077)
    if args.mode=="install":
        def expired(*_):
            raise TimeoutError("OFF_EVAL_STAGE_BUDGET")
        signal.signal(signal.SIGALRM,expired)
        signal.alarm(240)
        result=installation(args.request,args.publication,args.source_manifest,args.cp_manifest_sha)
    elif args.mode=="preflight":
        preflight(args.request,args.stage_sha,args.parent,args.node,args.grant,args.grant_sha)
        result=dict(request=args.request,parent=args.parent,node=args.node,placement="PASS_DUAL_CURRENT",submitted=False)
    elif args.mode=="start":
        result=start(args)
    else:
        result=terminal(args.request,args.parent)
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
    return 0 if args.mode!="start" or result["rc"]==0 else 2


if __name__=="__main__":
    raise SystemExit(main())
