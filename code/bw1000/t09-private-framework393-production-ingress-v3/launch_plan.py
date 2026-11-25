"""Render or run one authorized zero-GRES ingress step and reconcile it independently."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import subprocess
import time

from ingress import pinned
from contract import (PROJECT,INVOCATIONS,REQUEST,LABEL,PARENT,NODE,SHARED,MANIFEST,MANIFEST_SHA256,
                      STAGE,STAGE_REQUEST,STAGE_SHA,AUDIT,AUDIT_SHA,encoding,need,write_exclusive)


SCOPE=INVOCATIONS/REQUEST
SOURCE=INVOCATIONS/(REQUEST+"-source")
COLUMNS="JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"
IMAGE=PROJECT/"containers/dtk26-sandbox-r1"
APPTAINER="/public/software/apps/apptainer/1.3.4/bin/apptainer"
HOST_PY="/usr/local/python3.12/bin/python3"
WHOLE=120
TERMINAL={"COMPLETED","FAILED","CANCELLED","TIMEOUT","NODE_FAIL","OUT_OF_MEMORY","BOOT_FAIL","REVOKED"}


def exact_source(source,sha):
    manifest=pinned(source/"SOURCE_FILES.json",sha)
    files=manifest["files"]
    need(manifest["request"]==REQUEST and set(files)=={"framework_binding.py","contract.py","ingress.py","launch_plan.py","host_guard.py"},
         "COMPUTE_SOURCE_SCOPE_EXACT")
    need({path.name for path in source.iterdir()}==set(files)|{"SOURCE_FILES.json"},"COMPUTE_SOURCE_CLOSURE")
    for rel,item in files.items():
        path=source/rel
        need(path.resolve(strict=True)==path and path.is_file() and path.stat().st_size==item["bytes"],
             "COMPUTE_SOURCE_FILE_TYPE")
        need(sha256(path.read_bytes()).hexdigest()==item["SHA256"],"COMPUTE_SOURCE_FILE_SHA")


def check_authority(authority, stage_sha, source_sha, now, *, live):
    parent,node=authority["parent"],authority["node"]
    epoch=authority["epoch"]
    need(authority["request"]==REQUEST and authority["label"]==LABEL
         and (parent,node)==(PARENT,NODE) and authority["stage_request"]==STAGE_REQUEST and authority["stage_audit_SHA256"]==AUDIT_SHA
         and authority["stage_SHA256"]==stage_sha and authority["source_manifest_SHA256"]==source_sha
         and authority["one_zero_GRES_step"] is authority["root_confirmed_current_cpu_placement"] is True
         and type(epoch) in (int,float) and math.isfinite(epoch) and authority["deadline"]==epoch+WHOLE,
         "INDEPENDENT_CURRENT_ONLY_CPU_REQUEST")
    if live:
        need(0<=now-epoch<=20,"ORIGINAL_ZERO_GRES_DISPATCH_EPOCH")
    return parent,node,epoch


def command(parent,node,stage_sha,auth_sha,source_sha,remaining):
    need(all(re.fullmatch("[0-9a-f]{64}",item) is not None for item in (stage_sha,auth_sha,source_sha)),
         "EXACT_ZERO_GRES_TRIGGER_PINS")
    need(type(remaining) is int and 60<=remaining<=WHOLE,"BOUNDED_SINGLE_EPOCH_CPU_ARGV")
    body=('set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; '
          'unset PYTHONPYCACHEPREFIX; export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 '
          f'PYTHONPATH=/training-runtime/site:{SHARED}/generation/src:{SHARED}; '
          'exec python3 -B -u /t09-ingress-source/ingress.py "$@"')
    # Time is in whole Slurm minutes to stay in the original epoch.
    return ["srun","--jobid="+parent,"--exact","--immediate=5","--nodes=1","--ntasks=1","--nodelist="+node,
            "--cpus-per-task=4","--mem=8192M","--gres=none","--time="+str(remaining//60),"--job-name="+LABEL,
            HOST_PY,"-B",str(SOURCE/"host_guard.py"),auth_sha,"--",APPTAINER,"exec","--no-home","--pwd","/",
            "--bind",f"{PROJECT}:{PROJECT}:ro","--bind",f"{STAGE/'framework'}:{SHARED}:ro",
            "--bind",f"{STAGE/'FRAMEWORK_FILES.json'}:{MANIFEST}:ro",
            "--bind",f"{PROJECT/'results/training-t06-sdxl-runtime-v1'}:/training-runtime:ro",
            "--bind",f"{SCOPE}:/t09-ingress-scope:ro","--bind",f"{SOURCE}:/t09-ingress-source:ro",
            "--bind",f"{STAGE}:/t09-stage-ro:ro",
            "--bind","/opt/hyhal:/opt/hyhal:ro",str(IMAGE),"/bin/bash","--noprofile","--norc","-c",body,LABEL,
            "--stage-sha",stage_sha,"--authority-sha",auth_sha,"--source-manifest-sha",source_sha]


def accounting(parent):
    text=subprocess.check_output(["env","TZ=UTC","SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S","sacct","-nP",
                                  "-j",parent,"--format",COLUMNS],timeout=10,text=True)
    need(len(text)<=2<<20,"BOUNDED_SLURM_ACCOUNTING")
    rows=[line.split("|") for line in text.splitlines() if line]
    need(len(rows)<=8192 and all(len(row)==9 for row in rows),"NINE_ACTUAL_ACCOUNTING_COLUMNS")
    return [row for row in rows if row[1]==LABEL]


def terminal(row,authority):
    parent,node,epoch=authority["parent"],authority["node"],authority["epoch"]
    t=lambda x:datetime.strptime(x,"%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    tres=row[6].split(",")
    need(len(row)==9 and row[1]==LABEL and row[5]==node and re.fullmatch(parent+r"\.[0-9]+",row[0]) is not None
         and row[2] in TERMINAL and re.fullmatch(r"[0-9]+:[0-9]+",row[3]) is not None and row[4].isdigit()
         and len(tres)==3 and set(tres) in [{"cpu=4","node=1","mem=8G"},{"cpu=4","node=1","mem=8192M"}],
         "EXACT_TERMINAL_NUMERIC_ZERO_GRES_STEP")
    start,end=t(row[7]),t(row[8])
    need(epoch-1<=start<=end<=authority["deadline"]+1 and 0<=int(row[4])<=WHOLE and abs(end-start-int(row[4]))<=1,
         "INDEPENDENT_EXACT_ONE_EPOCH_TERMINAL")
    return row


def parse_response(data):
    records=[]
    for line in data.splitlines():
        if len(line)>65536:
            continue
        try:
            record=json.loads(line)
        except (ValueError,UnicodeDecodeError):
            continue
        if isinstance(record,dict) and record.get("request")==REQUEST:
            records.append(record)
    need(len(records)==1,"ONE_ACTUAL_PRIVATE_INGRESS_RECEIPT")
    return records[0]


def verified_receipt(rows,authority,owner,intent,host=None):
    completed={"request":REQUEST,"planned":393,"before_verified":0,"after_verified":0,
               "providers_verified":0,"passed":False,"actual_optimizer_updates":0,"actual_GPU_by_consumer":0}
    try:
        row,=rows
        exact=terminal(row,authority)
        completed["exact_accounting"]=exact
        need(owner["request"]==intent["request"]==REQUEST and owner["epoch"]==intent["epoch"]==authority["epoch"]
             and intent["authority_SHA256"]==owner["authority_SHA256"]
             and intent["source_manifest_SHA256"]==owner["source_manifest_SHA256"]==authority["source_manifest_SHA256"]
             and intent["stage_SHA256"]==owner["stage_SHA256"]==authority["stage_SHA256"],
             "CURRENT_OWNER_AND_AUTHORITY_LINKAGE")
        report=owner["ingress"]
        if host is not None:
            value,host_sha=host
            need(value["request"]==REQUEST and value["step"]==exact[0] and value["passed"] is True
                 and value["epoch"]==authority["epoch"] and value["authority_SHA256"]==intent["authority_SHA256"],
                 "INDEPENDENT_SAME_NUMBERED_HOST_CGROUP_TRES")
            completed["host_gate_SHA256"]=host_sha
        if report is not None:
            need(report["request"]==REQUEST and report.get("step")==exact[0] and report["planned"]==393
                 and report["weight_payload_bytes_read"]==report["GPU_SDK_calls"]==report["optimizer_updates"]==0,
                 "EXACT_MOUNTED_INGRESS_RECORD")
            completed.update(before_verified=report["before_verified"],after_verified=report["after_verified"],
                             providers_verified=report["providers_verified"],ingress_phase=report["phase"])
            completed["passed"]=(report["passed"] is True and owner["rc"]==0 and row[2:4]==["COMPLETED","0:0"]
                and report["before_verified"]==report["after_verified"]==393 and report["providers_verified"]==2
                and report["bound_source_manifest_SHA256"]==authority["source_manifest_SHA256"]
                and report["readonly_overlay"] is report["private_same_tree_and_manifest"] is report["actual_same_step_host_gate"] is True
                and report["production_CLI_imported"] is False and host is not None and report["host_gate_SHA256"]==host[1]
                and report["canonical_stage_SHA256"]==STAGE_SHA and report["canonical_stage_audit_SHA256"]==AUDIT_SHA)
    except (KeyError,TypeError,ValueError) as exc:
        completed["error_class"]=type(exc).__name__
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","run-once","verify-terminal"))
    for name in ("stage-sha","authority-sha","source-manifest-sha"):
        p.add_argument("--"+name,required=True)
    a=p.parse_args()
    exact_source(SOURCE,a.source_manifest_sha)
    need(a.stage_sha==STAGE_SHA,"EXACT_ARCHIVED_STAGE_SHA")
    staged=pinned(STAGE/"stage-result.json",a.stage_sha)
    with AUDIT.open("rb") as stream:
        audit=stream.read(65537)
    need(len(audit)<=65536 and sha256(audit).hexdigest()==AUDIT_SHA,"EXACT_INDEPENDENT_PRIOR_STAGE_AUDIT")
    authority=pinned(SCOPE/"authority.json",a.authority_sha)
    need(staged["passed"] is True and staged["request"]==STAGE_REQUEST and staged["verified_original"]==393
         and staged["manifest_SHA256"]==MANIFEST_SHA256,"BIND_SUCCESSFUL_EXACT_PRIVATE_STAGE")
    parent,node,epoch=check_authority(authority,a.stage_sha,a.source_manifest_sha,time.time(),live=a.mode!="verify-terminal")
    common=dict(request=REQUEST,epoch=epoch,authority_SHA256=a.authority_sha,stage_SHA256=a.stage_sha,
                source_manifest_SHA256=a.source_manifest_sha)
    if a.mode=="verify-terminal":
        def get_existing(name):
            path=SCOPE/name
            if not path.is_file():
                return None
            raw=path.read_bytes()
            need(0<len(raw)<=1<<20,"BOUNDED_PRIVATE_OWNER_EVIDENCE")
            return json.loads(raw)
        intent=get_existing("cpu-intent.json")
        owner=get_existing("cpu-outcome.json")
        if intent is not None:
            need(intent["authority_SHA256"]==a.authority_sha,"CURRENT_CPU_INTENT_AUTHORITY_PIN")
        host=None
        if (SCOPE/"host-step.json").is_file():
            raw=(SCOPE/"host-step.json").read_bytes()
            need(0<len(raw)<=16384,"EXACT_BOUNDED_HOST_PROOF")
            host=(json.loads(raw),sha256(raw).hexdigest())
        report=verified_receipt(accounting(parent),authority,owner,intent,host)
        print(json.dumps(report,sort_keys=True))
        return 0 if report["passed"] else 2
    if a.mode=="plan":
        argv=command(parent,node,a.stage_sha,a.authority_sha,a.source_manifest_sha,math.floor(authority["deadline"]-time.time()))
        print(json.dumps(dict(common,argv=argv,submitted=False),sort_keys=True))
        return 0
    need(not accounting(parent),"EXACT_LABEL_ALREADY_EXISTS_RECONCILE")
    remaining=authority["deadline"]-time.time()
    argv=command(parent,node,a.stage_sha,a.authority_sha,a.source_manifest_sha,math.floor(remaining))
    write_exclusive(SCOPE/"cpu-intent.json",encoding(dict(common,argv=argv,repeat_allowed=False)))
    outcome=dict(common,rc=None,error_class=None,ingress=None,planned=393,repeat_allowed=False)
    try:
        remaining=authority["deadline"]-time.time()
        need(remaining>=60,"CPU_EPOCH_EXHAUSTED_BEFORE_SUBMIT")
        need(60*int(next(item.split("=",1)[1] for item in argv if item.startswith("--time=")))<=remaining,
             "LOCKED_CPU_ARGV_STILL_FITS_ORIGINAL_EPOCH")
        proc=subprocess.run(argv,capture_output=True,timeout=remaining)
        need(len(proc.stdout)<=1<<20 and len(proc.stderr)<=1<<20,"BOUNDED_NATIVE_STDIO")
        outcome.update(rc=proc.returncode,stdout_SHA256=sha256(proc.stdout).hexdigest(),
                       stderr_SHA256=sha256(proc.stderr).hexdigest())
        outcome["ingress"]=parse_response(proc.stdout)
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        outcome["error_class"]=type(exc).__name__
    finally:
        write_exclusive(SCOPE/"cpu-outcome.json",encoding(outcome))
    print(json.dumps(outcome,sort_keys=True))
    return 0 if outcome["rc"]==0 and outcome["ingress"] and outcome["ingress"]["passed"] else 2


if __name__=="__main__":
    raise SystemExit(main())
