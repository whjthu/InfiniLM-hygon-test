"""Build deterministic n03 bridge source bytes. This tool never sends or launches."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
PUBLISHED={
    "original_bridge.py":("code/bw1000/fleet-dcu-standby-compute-bridge-v1/bridge.py",
        "2e796e5a136bb7fcac5892b49a34f0fd81fabd1f6f8774945725a0b199e58360"),
    "standby_controller.py":("code/bw1000/fleet-dcu-standby-v2/controller.py",
        "be504e7670b40d77e503c60c93d76bf2a75003768d764d54884c5f8272976bba"),
    "standby_worker.py":("code/bw1000/fleet-dcu-standby-v2/worker.py",
        "dc8820e17b8d4d2886325c6093ad8884e6a5a1085e0eedd40eb53cfb12235059"),
    "collect_training80_dispatch_snapshot.py":("tools/collect_training80_dispatch_snapshot.py",
        "3e1b069f93219a165cc19431a54b47bc6044cf54c557e941f5bde797f3404078"),
    "training_dispatch_allocation.py":("tools/training_dispatch_allocation.py",
        "bc68718ab63ee923e3a193f62d9e16a599f116c810c39b6f634bee62a5d17e33"),
    "read_training_slurm_membership.py":("tools/read_training_slurm_membership.py",
        "9b3b0ca57c7ade5c28874faf46aa6925178ec0536f666842a085a76825d06de7"),
    "sample_training80_busy.py":("tools/sample_training80_busy.py",
        "6e4a964b9fe4af69974a73dc1f5d3338d1c7cacc7d68426d0890c58cd158e324"),
    "tuning_queue.py":("tools/tuning_queue.py","cdfb2d5c6b117c65e98e7661fa30054f7a970d32c1f541609ff7185a803a6bc0"),
    "ledger.json":("coordination/training/training80-replacement1194900-node-task-map-v1.json",
        "11d4cc4798f6533dc4bcf31d82dc2c206035aa498ab3842878a7cb56d688f71d"),
}
PRIVATE={
    "pci.json":("local-only/training80-replacement1194900-pci-pins-v1.json",
        "df2b483f320d54383a2ccaf6e80eab3a69e6efbb3f0caf6d7292c56eb96f9b39"),
    "historical-2443.json":("local-only/training80-n03-accounted-probe-0050-998fbfec65b4480f880aa3ebb39b630a.json",
        "5e9403b627d39ed6b0ef08a9945cdf98d2fad7ac508a6dabbe7c8d991107b894"),
}


def encode(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()


def digest(raw):
    return sha256(raw).hexdigest()


def build(request):
    if not re.fullmatch("[0-9a-f]{32}",request) or request=="998fbfec65b4480f880aa3ebb39b630a":
        raise ValueError("fresh distinct singleton request required")
    members={name:(HERE/name).read_bytes() for name in ("bridge.py","controller.py")}
    for name,(relative,wanted) in {**PUBLISHED,**PRIVATE}.items():
        raw=(REPO/relative).read_bytes()
        if digest(raw)!=wanted:
            raise ValueError("exact source drift: "+name)
        members[name]=raw
    for name,raw in members.items():
        if name.endswith(".py"):
            compile(raw,name,"exec")
    members["SOURCE_FILES.json"]=encode(dict(request=request,scope="f13r4n03-original-189-cpu-only",
        files={name:dict(bytes=len(raw),SHA256=digest(raw)) for name,raw in members.items()}))
    return members


def make_owner(request,publication,issued,pci,*,queue_excluded_nodes,original_hold_conflicts):
    value=dict(request=request,publication=publication,parent="1194900",node="f13r4n03",
        predecessor="1194900.189",ports=[43843,43844],queue_excluded_nodes=queue_excluded_nodes,
        original_hold_conflicts=original_hold_conflicts,issued=issued,expires=issued+65,
        historical_SHA256=PRIVATE["historical-2443.json"][1],pci=pci,seconds=10800)
    import controller
    controller.validate_owner(value,request,publication,pci,issued)
    return encode(value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    files=build(args.request)
    if args.output is not None:
        args.output.mkdir(mode=0o700)
        for name,raw in files.items():
            with (args.output/name).open("xb") as stream:
                stream.write(raw)
    print(encode(dict(request=args.request,files=len(files),bytes=sum(map(len,files.values())),
                      manifest_SHA256=digest(files["SOURCE_FILES.json"]),actual_submission=False)).decode(),end="")


if __name__=="__main__":
    main()
