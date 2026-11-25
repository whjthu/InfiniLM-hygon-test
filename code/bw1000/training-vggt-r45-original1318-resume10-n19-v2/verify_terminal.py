"""Independent readonly reconciliation of the fresh resume10 and physical post."""

import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
import re
import subprocess

from resume_bindings import LABEL,OLD,ROOT,REQUEST,NODE,N19_PCIS,reference,compare,pinned,need


RUN=ROOT/"results"/LABEL
COLS="JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"


def utc(s):
    return datetime.strptime(s,"%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def term(handles):
    raw=subprocess.check_output(["env","TZ=UTC","SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S","sacct",
                                 "-nP","-j",",".join(handles),"--format",COLS],timeout=10)
    need(len(raw)<=65536,"VGGT_TERM_OUTPUT_BOUND")
    lines=[line.split("|") for line in raw.decode("ascii").splitlines() if line]
    need(all(len(line)==9 for line in lines),"VGGT_NINE_COLUMNS")
    return lines


def records(scope,read=term):
    selected=[]
    phases=("pre","gpu","post")
    previous=[scope["stages"][stage]["row"] for stage in phases]
    ids=[r[0] for r in previous]
    need(len(set(ids))==3 and all(re.fullmatch(r"1103606\.[0-9]+",v) and int(v.rsplit(".",1)[1])>1319 for v in ids),
         "VGGT_NEW_NUMERIC_TERMINALS_ONLY")
    got=read(ids)
    for phase,base in zip(phases,previous,strict=True):
        observed=[row for row in got if row[0]==base[0]]
        need(len(observed)==1 and observed[0][:7]==base,"VGGT_EXACT_INDEPENDENT_TERMINAL")
        row=observed[0]
        tres=({"cpu=4","gres/dcu=1","mem=64G","node=1"} if phase=="gpu" else {"cpu=2","mem=2G","node=1"})
        alt=(tres-({"mem=64G"} if phase=="gpu" else {"mem=2G"})) | (
             {"mem=65536M"} if phase=="gpu" else {"mem=2048M"})
        need(row[1]==LABEL+"-"+phase and row[5]==NODE and row[4].isdecimal()
             and int(row[4])<=(1200 if phase=="gpu" else 60)
             and len(row[6].split(","))==len(tres) and set(row[6].split(",")) in (tres,alt)
             and scope["epoch"]-1<=utc(row[7])<=utc(row[8])<=scope["deadline"]+1,
             "VGGT_EXACT_TRES_JOB_NODE_EPOCH")
        selected.append(row)
    a,b,c=selected
    need(a[2:4]==c[2:4]==["COMPLETED","0:0"] and utc(a[8])<=utc(b[7])<=utc(b[8])<=utc(c[7]),
         "VGGT_CONFIRM_INDEPENDENT_POST_AFTER_GPU")
    return dict(zip(phases,selected,strict=True))


def verify(outcome_sha,read=term):
    result=json.loads(pinned(RUN/"outcome.json",outcome_sha,cap=65536))
    response=dict(request=REQUEST,planned_updates=15,actual_updates=None,independent_release=False,
                  ready_supported_by_this_recovery=False,original_CONTINUOUS_replayed=False)
    need(result["request"]==REQUEST and result["planned_updates"]==15 and result["deadline"]==result["epoch"]+1800,
         "VGGT_EXACT_ORIGINAL_OWNER")
    if not all(result["stages"].get(k,{}).get("terminal_visible") for k in ("pre","gpu","post")):
        response["owner_error_class"]=result["error_class"]
        return response
    external=records(result,read)
    before=json.loads(pinned(RUN/"before.json"))
    after=json.loads(pinned(RUN/"after.json"))
    need(result["release_candidate"] is True and before["step"]==external["pre"][0] and after["step"]==external["post"][0]
         and before["boot_id"]==after["boot_id"] and before["pci"]==after["pci"]==list(N19_PCIS)
         and before["actual_host_numeric_cgroup"] is after["actual_host_numeric_cgroup"] is True
         and before["kfd_holder_count"]==after["kfd_holder_count"]==0
         and before["full_fd_inventory_performed"] is after["full_fd_inventory_performed"] is True,
         "VGGT_ACTUAL_INDEPENDENT_PHYSICAL_RELEASE")
    response.update(independent_release=True,steps={k:v[0] for k,v in external.items()},gpu_state=external["gpu"][2])
    if external["gpu"][2:4]!=["COMPLETED","0:0"]:
        return response
    try:
        gpu=json.loads(pinned(RUN/"assignment.json"))
        need(gpu["step"]==external["gpu"][0] and gpu["node"]==NODE and gpu["pci"] in after["pci"],
             "VGGT_ACTUAL_ONE_DEVICE_ASSIGNMENT")
        evidence=json.loads(pinned(RUN/"native-result.json",result["native_result_SHA256"],cap=32768))
        descriptor=evidence["receipts"]["resume10"]
        base=RUN/"arms/resume10"
        raw=pinned(base/"receipt.json",descriptor["SHA256"],cap=32768)
        current=json.loads(raw)
        need(current==descriptor["value"] and current["epoch"]==result["epoch"],"VGGT_NEW_FRESH_NATIVE_PROCESS_RECEIPT")
        tape=pinned(base/"train/steps.ndjson",cap=16384)
        need([json.loads(line) for line in tape.splitlines()]==current["steps"],"VGGT_EXACT_FRESH_RAW_TAPE")
        pinned(base/"native-result.json",current["result_SHA256"],cap=131072)
        for n,item in current["checkpoints"].items():
            pinned(base/"train"/f"checkpoint-{n}.pt",item["file_SHA256"],cap=3<<30,stream=True)
        paired=compare(reference(OLD/"arms/continuous"),current)
        need(paired==evidence["comparison"],"VGGT_INDEPENDENT_COMPARISON_RESULT")
        response.update(actual_updates=15,comparison=paired,ready_supported_by_this_recovery=paired["comparison_passed"],
                        formal_CO3D=False,formal_performance=False)
    except (OSError,ValueError,TypeError,KeyError) as e:
        response["evidence_read_error_class"]=type(e).__name__
    return response


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outcome-sha",required=True)
    result=verify(p.parse_args().outcome_sha)
    print(json.dumps(result,sort_keys=True,separators=(",",":")))
