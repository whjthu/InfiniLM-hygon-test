"""Independent read of the exact R44 owner, numeric steps, and eight receipts."""

import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
import re
import subprocess

from collect_votes import collect
from contract import D,LABEL,NODE,PARENT,REQUEST,ROOT,WHOLE,need,pin,stable_json


COLUMNS="JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"


def utc(value):
    return datetime.strptime(value,"%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def accounting(handles,query=subprocess.check_output):
    raw=query(["env","TZ=UTC","SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S","sacct",
               "-nP","-j",",".join(handles),"--format",COLUMNS],timeout=10)
    need(len(raw)<=65536,"R44_INDEPENDENT_ACCOUNTING_BOUND")
    found=[line.split("|") for line in raw.decode("ascii").splitlines() if line]
    need(all(len(row)==9 for row in found),"R44_EXACT_NINE_ACCOUNTING_COLUMNS")
    return found


def verify(outcome_sha,query=accounting):
    scope,scope_pin=stable_json(ROOT/"outcome.json")
    intent,intent_pin=stable_json(ROOT/"intent.json")
    need(scope_pin["SHA256"]==outcome_sha and scope["request_id"]==intent["request_id"]==REQUEST
         and scope["diagnostic_only"] is True and not scope["engineering_completed"]
         and scope["planned_updates"]==0 and scope["nominal_scheduler_steps"]==25
         and scope["source_manifest"]==intent["source_manifest"]==pin(D/"SOURCE_FILES.json")
         and scope["epoch"]==intent["epoch"] and scope["deadline"]==scope["epoch"]+WHOLE,
         "R44_EXACT_UNIQUE_OWNER_AND_EPOCH")
    output=dict(request_id=REQUEST,independent_release=False,actual_optimizer_updates=None,
                diagnostic_qualified=False,planned_optimizer_updates=0,planned_rank_receipts=8,
                qualified_training=False)
    stages=scope["results"]
    if not all(phase in stages and stages[phase]["terminal"]["terminal_visible"] for phase in ("pre","gpu","post")):
        output["owner_error"]=scope["error"]
        return output
    ids={phase:stages[phase]["terminal"]["row"][0] for phase in ("pre","gpu","post")}
    need(len(set(ids.values()))==3 and all(re.fullmatch(PARENT+r"\.[0-9]+",step) and int(step.rsplit(".",1)[1])>3451
                                         for step in ids.values()),"R44_UNIQUE_NEW_NUMERIC_STEPS")
    rows=query(list(ids.values()))
    exact=[]
    for phase in ("pre","gpu","post"):
        matches=[row for row in rows if row[0]==ids[phase]]
        need(len(matches)==1,"R44_UNIQUE_INDEPENDENT_NUMBERED_STEP")
        row=matches[0]
        prior=stages[phase]["terminal"]["row"]
        need(row[:7]==prior and row[1]==LABEL+"-"+phase and row[5]==NODE and row[4].isdigit()
             and 0<=int(row[4])<=(2100 if phase=="gpu" else 90),"R44_FRESH_STEP_EXACT_IDENTITY")
        wanted={"cpu=64","gres/dcu=8","mem=224000M","node=1"} if phase=="gpu" else {"cpu=2","mem=4G","node=1"}
        alternate=(wanted-({"mem=224000M"} if phase=="gpu" else {"mem=4G"})) | (
                    {"mem=218.75G"} if phase=="gpu" else {"mem=4096M"})
        need(len(row[6].split(","))==len(wanted) and set(row[6].split(",")) in (wanted,alternate),
             "R44_INDEPENDENT_EXACT_TRES")
        exact.append(row)
    pre,gpu,post=exact
    need(pre[2:4]==post[2:4]==["COMPLETED","0:0"] and
         scope["epoch"]-1<=utc(pre[7])<=utc(pre[8])<=utc(gpu[7])<=utc(gpu[8])<=utc(post[7])<=utc(post[8])
         <=scope["deadline"]+1,"R44_INDEPENDENT_BOUNDED_POST_CHRONOLOGY")
    before,_=stable_json(ROOT/"before.json",scope["before"])
    after,_=stable_json(ROOT/"after.json",scope["after"])
    need(before["step"]==pre[0] and after["step"]==post[0] and before["request_id"]==after["request_id"]==REQUEST
         and before["boot_id"]==after["boot_id"] and before["pci"]==after["pci"] and len(set(after["pci"]))==8
         and before["kfd_holder_count"]==after["kfd_holder_count"]==0 and after["owned_processes_remaining"]==[]
         and scope["release_permitted"] is True,"R44_INDEPENDENT_SAME_BOOT_PCI_CLEAN_RELEASE")
    output.update(independent_release=True,steps=ids,gpu_state=gpu[2],gpu_exit=gpu[3])
    try:
        entry,_=stable_json(ROOT/"gpu-step-entry.json")
        need(entry["request_id"]==REQUEST and entry["step"]==gpu[0] and entry["intent"]==intent_pin
             and entry["supervisor"]["boot_id"]==before["boot_id"],"R44_ACTUAL_GPU_ENTRY_BOUND")
        votes=collect(ROOT,gpu[0],scope["epoch"],before["boot_id"])
        output.update(rank_diagnostics=votes)
        native,_=stable_json(ROOT/"native-outcome.json",scope["native_outcome"],cap=2<<20)
        expected=native["diagnostic_receipts"]
        need(type(expected) is dict and all(expected.get(k)==v for k,v in votes.items()),
             "R44_INDEPENDENT_DIAGNOSTIC_VOTE_COMPARISON")
        log=ROOT/"native.stderr"
        need(log.is_file() and not log.is_symlink() and log.stat().st_size<=16<<20,"R44_BOUNDED_ABORT_LOG")
        raw=log.read_bytes()
        planned=bool(re.search(rb"R43_DIAGNOSTIC_(?:COMPLETE_BEFORE_OPTIMIZER|REJECT_[A-Z0-9_]+)",raw))
        qualified=(gpu[2:4]==["FAILED","1:0"] and native["returncode"]==1 and native["diagnosis_only"] is True
                   and planned and expected["planned_abort_marker_observed"] is True
                   and sha256(raw).hexdigest()==expected["native_stderr_SHA256"] and votes["all_eight_clean"]
                   and native["actual_updates"]==scope["actual_updates"]==0
                   and scope["diagnostic_completed"] is True)
        output.update(diagnostic_qualified=bool(qualified),actual_optimizer_updates=0 if qualified else None,
                      planned_abort_observed=planned)
    except (OSError,TypeError,ValueError,KeyError) as error:
        output["diagnostic_read_error_class"]=type(error).__name__
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outcome-sha",required=True)
    a=p.parse_args()
    out=verify(a.outcome_sha)
    print(json.dumps(out,sort_keys=True,separators=(",",":"),allow_nan=False))
