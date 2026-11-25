"""Qualify the actual numeric zero-GRES step on HOST before entering the image."""
import os
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys

from contract import LABEL,NODE,PARENT,REQUEST,INVOCATIONS,encoding,need,write_exclusive
from ingress import establish_step,pinned


def live_record(raw, step):
    need(len(raw)<=2<<20 and len(raw.splitlines())<=8192,"BOUNDED_ACTUAL_CONTROLLER_STEP_QUERY")
    matching=[]
    for line in raw.splitlines():
        atoms=[part.split("=",1) for part in line.split() if "=" in part]
        item=dict(atoms)
        if item.get("StepId")==step:
            need(all(sum(key==atom[0] for atom in atoms)<=1 for key in ("StepId","NodeList","Name","JobName","State","TRES","AllocTRES")),
                 "NO_DUPLICATE_EXACT_SELECTED_STEP_FIELDS")
            matching.append(item)
    row,=matching
    need(not {"Name","JobName"}<=row.keys() or row["Name"]==row["JobName"],"HOST_CONTROLLER_LABEL_VIEWS_AGREE")
    need(not {"TRES","AllocTRES"}<=row.keys() or set(row["TRES"].split(","))==set(row["AllocTRES"].split(",")),
         "HOST_CONTROLLER_TRES_VIEWS_AGREE")
    tres=row.get("TRES",row.get("AllocTRES","")).split(",")
    need(row.get("Name",row.get("JobName"))==LABEL and row["State"]=="RUNNING" and row["NodeList"]==NODE
         and len(tres)==3 and set(tres) in ({"cpu=4","mem=8G","node=1"},{"cpu=4","mem=8192M","node=1"}),
         "HOST_EXACT_ZERO_GRES_CONTROLLER_TRES")
    return {"step":step,"node":NODE,"label":LABEL,"exact_live_TRES":";".join(sorted(tres)),
            "controller_stdout_SHA256":sha256(raw.encode()).hexdigest()}


def main():
    args=sys.argv[1:]
    need(len(args)>2 and args[1]=="--" and Path(args[2]).is_absolute(),"EXACT_HOST_PASS_THROUGH_ARGV")
    scope=INVOCATIONS/REQUEST
    authority=pinned(scope/"authority.json",args[0])
    report=dict(request=REQUEST,step=None,passed=False,phase="NUMERIC_HOST_CGROUP",Slurm_mutations=0,GPU_calls=0)
    try:
        step=establish_step(authority)
        report.update(step=step,phase="CONTROLLER_TRES")
        raw=subprocess.check_output(["scontrol","show","step",step,"--oneliner"],text=True,timeout=6)
        report.update(live_record(raw,step),phase="PRE_IMAGE",passed=True,actual_host_cgroup=True,
                      authority_SHA256=args[0],epoch=authority["epoch"])
    except (OSError,ValueError,TypeError,KeyError,subprocess.SubprocessError) as error:
        report.update(error_class=type(error).__name__,errno=getattr(error,"errno",None))
    write_exclusive(scope/"host-step.json",encoding(report))
    if not report["passed"]:
        print(json.dumps(report,sort_keys=True),flush=True)
        return 2
    os.execv(args[2],args[2:])
    return 2


if __name__=="__main__":
    raise SystemExit(main())
