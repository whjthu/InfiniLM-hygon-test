"""Own two bounded concurrent Slurm parent scopes and one conditional expansion."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from audit import verify

ROOT=Path(__file__).resolve().parent
PARENTS=("1087656","1103606")


def save(name,data):
    with (ROOT/name).open("x") as handle:
        json.dump(data,handle,allow_nan=False)


def wave(stage,epoch,request,placement):
    call={}
    out={}
    try:
        for parent in PARENTS:
            stream=(ROOT/f"{stage}-{parent}.log").open("x")
            out[parent]=stream
            call[parent]=subprocess.Popen([sys.executable,"-B",str(ROOT/"dispatch.py"),
                str(epoch),request,stage,parent],stdout=stream,stderr=subprocess.STDOUT)
        save(f"{stage}-submitted.json",dict(stage=stage,request=request,controller_pids={p:c.pid for p,c in call.items()},
            placement=placement,at=time.time()))
        limit=epoch+1082
        for child in call.values():
            child.wait(timeout=max(1,limit-time.time()))
    finally:
        for parent,stream in out.items():
            stream.close()
    terminal={}
    for parent in PARENTS:
        try:
            item=verify(ROOT/f"{stage}-{parent}",placement[parent],request)
            terminal[parent]=dict(controller_rc=call[parent].returncode,result=item)
        except (OSError,ValueError,KeyError,TypeError) as error:
            terminal[parent]=dict(controller_rc=call[parent].returncode,unqualified=type(error).__name__,
                                  code=str(error) if isinstance(error,ValueError) else None)
    save(f"{stage}-review.json",terminal)
    return terminal


def main(epoch,request):
    claim=json.loads((ROOT/"CLAIM.json").read_bytes())
    if request!=claim["request"] or not 0<=time.time()-epoch<10:
        raise ValueError("ROOT_EPOCH")
    plans=json.loads((ROOT/"PLACEMENT.json").read_bytes())
    try:
        gate = claim["qualified_BF16"]
        receipt = (ROOT.parent / gate["path"]).read_bytes()
        if hashlib.sha256(receipt).hexdigest() != gate["SHA256"]:
            raise ValueError("EXACT_BF16_DEVICE_RECEIPT")
        proven = json.loads(receipt)["1103606"]["result"]
        if (proven["complete_pairs"] != 19 or proven["failures"]
                or proven["independent_post"][2:4] != ["COMPLETED","0:0"]
                or any(set(item["shapes"]) != {"6144"} for item in proven["pairs"])):
            raise ValueError("BF16_SHCA_PREREQUISITE_FAILED")
        wide=wave("wide",epoch,request,plans["wide"])
        return int(not all(value["controller_rc"] == 0 and
                           value.get("result", {}).get("complete_pairs") == len(plans["wide"][parent])
                           for parent, value in wide.items()))
    except (OSError,ValueError,subprocess.SubprocessError) as failure:
        save("campaign-exception.json",dict(kind=type(failure).__name__,
            reason=str(failure) if isinstance(failure,ValueError) else None))
        raise


if __name__=="__main__":
    raise SystemExit(main(float(sys.argv[1]),sys.argv[2]))
