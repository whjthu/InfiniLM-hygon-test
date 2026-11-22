"""Read only original terminal and process evidence from the unique nine-pair run."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from compare_resume import compare
from preflight_gate import CPU_RESULT_SHA, INSTALL_SHA, REQUEST as PRECHECK, VIEW_SHA


LABEL = "training-vggt-tum-three-recording-native25-resume20-v1"
REQUEST = "7b07ac6c8bd3e86e63e7528fd1c885f0"
PARENT, NODE = "1103606", "f13r4n09"
PHASES = ("pre","gpu","post")


def need(ok, why):
    if not ok:
        raise ValueError(why)


def document(path, pins, key, cap=1<<21):
    need(path.is_absolute() and path.resolve(strict=True)==path and path.is_file(),"unaliasable original evidence required")
    with path.open("rb") as stream:
        raw=stream.read(cap+1)
    need(0<len(raw)<=cap,"bounded original evidence size differs")
    pins[key]=sha256(raw).hexdigest()
    return json.loads(raw)


def disposition(root, accounting, device):
    need(root.name==LABEL,"original training result namespace differs")
    need(set(accounting)==set(PHASES) and len({row[0] for row in accounting.values()})==3,
         "original three independent accounting rows required")
    actual={}
    for phase,row in accounting.items():
        need(type(row) is list and len(row)==7 and all(isinstance(part,str) for part in row)
             and re.fullmatch(PARENT+r"\.[0-9]+",row[0]) and row[1]==LABEL+"-"+phase and row[5]==NODE
             and row[2].split()[0].rstrip("+") in {"COMPLETED","FAILED","CANCELLED","TIMEOUT","NODE_FAIL",
                                                    "OUT_OF_MEMORY","PREEMPTED","REVOKED","BOOT_FAIL","DEADLINE"}
             and re.fullmatch(r"[0-9]+:[0-9]+",row[3]), "unverified original accounting phase")
        resources=dict(item.split("=",1) for item in row[6].split(","))
        need(resources.get("node")=="1" and {k:v for k,v in resources.items() if k.startswith("gres/")}==
             ({"gres/dcu":"1"} if phase=="gpu" else {}),"original phase allocation changed")
        actual[phase]=row[0]
    pins={}
    summary=dict(request=REQUEST,qualified=False,scope="nine-pair one-DCU training and checkpoint20 recovery",
                 planned_updates={"continuous":25,"resume20":5},actual_steps=actual,artifact_SHA256=pins)
    intent=document(root/"intent.json", pins, "intent.json")
    need((intent["request"],intent["parent"],intent["node"])==(REQUEST,PARENT,NODE),
         "original intent changed")
    proof=document(root/"preflight-binding.json",pins,"preflight-binding.json")
    need((proof["preflight_request"],proof["preflight_result_SHA256"],proof["preflight_install_SHA256"],
          proof["view_order_SHA256"])==(PRECHECK,CPU_RESULT_SHA,INSTALL_SHA,VIEW_SHA),"target preflight binding differs")
    before=document(root/"before.json",pins,"before.json")
    after=document(root/"after.json",pins,"after.json")
    verified=device["nodes"][NODE]
    clear_step=verified["device_probe_step"]
    need(len(before["pci"])==8 and before["pci"]==after["pci"]
         and before["boot_id"]==after["boot_id"] and after["kfd_holder_count"]==0
         and after["full_fd_inventory_performed"] is True
         and set(verified["PCIs"])==set(before["pci"]) and verified["parent"]==PARENT
         and verified["active_steps"]==[] and verified["KFD_clean"] is verified["PID_clean"] is True
         and verified["device_probe_observed"] is True and re.fullmatch(PARENT+r"\.[0-9]+",clear_step)
         and clear_step not in actual.values() and int(clear_step.split(".")[-1])>int(actual["gpu"].split(".")[-1]),
         "separate same-parent post did not independently show device clear")
    summary.update(post_clean=True,independent_post_step=clear_step)
    if any(row[2:4]!=["COMPLETED","0:0"] for row in accounting.values()):
        summary["terminal_states"]={name:row[2:4] for name,row in accounting.items()}
        summary["recorded_updates"]={}
        for arm,start in (("continuous",0),("resume20",20)):
            relative="arms/"+arm+"/train/steps.ndjson"
            path=root/relative
            if path.is_file():
                need(path.resolve(strict=True)==path,"partial original step record was aliased")
                with path.open("rb") as stream:
                    raw=stream.read((1<<20)+1)
                need(len(raw)<=1<<20,"partial native step records exceed original scope")
                rows=[json.loads(line) for line in raw.splitlines()]
                need(len(rows)<=25-start and [row["global_optimizer_step"] for row in rows]==
                     list(range(start+1,start+len(rows)+1)), "partial original steps are not an ordered prefix")
                pins[relative]=sha256(raw).hexdigest()
                summary["recorded_updates"][arm]=len(rows)
        return summary
    outcome=document(root/"outcome.json",pins,"outcome.json")
    need(outcome["request"]==REQUEST and outcome["passed"] is True
         and all(outcome["stages"][phase]["row"]==accounting[phase] for phase in PHASES),
         "recorded outcome did not preserve all independent original phases")
    training=document(root/"native-result.json",pins,"native-result.json")
    need(pins["native-result.json"]==outcome["result_SHA256"],"original native result digest moved")
    receipts={}
    for arm in ("continuous","resume20"):
        name="arms/"+arm+"/receipt.json"
        receipt=document(root/name,pins,name)
        need(training["receipts"][arm]=={"SHA256":pins[name],"value":receipt},
             "fresh-process receipt differs from controller evidence")
        receipts[arm]=receipt
    comparison=compare(receipts)
    need(comparison==training["comparison"],"root comparisons disagree on training recovery")
    summary.update(qualified=comparison["comparison_passed"],actual_updates={"continuous":25,"resume20":5},
                   comparison=comparison)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--actual-accounting",type=Path,required=True)
    parser.add_argument("--independent-post",type=Path,required=True)
    args=parser.parse_args()
    values=[json.loads(path.read_bytes()) for path in (args.actual_accounting,args.independent_post)]
    print(json.dumps(disposition(args.root,*values),sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
