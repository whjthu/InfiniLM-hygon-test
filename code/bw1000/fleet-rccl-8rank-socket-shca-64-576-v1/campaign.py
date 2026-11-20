"""Own two bounded concurrent Slurm parent scopes and one conditional expansion."""
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
        limit=epoch+(549 if stage=="sample" else 1082)
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
        sample=wave("sample",epoch,request,plans["sample"])
        ok=all(x["controller_rc"]==0 and x.get("result",{}).get("complete_pairs")==2 and
               x["result"]["controller_failure"] is None and
               all(p["socket_over_shca"]>=2 for p in x["result"]["pairs"]) for x in sample.values())
        save("expansion-decision.json",dict(allowed=ok,required_pairs=4,minimum_per_pair_speedup=2))
        if ok:
            wide=wave("wide",epoch,request,plans["wide"])
            return int(not all(x["controller_rc"]==0 and
                        x.get("result",{}).get("complete_pairs")==len(plans["wide"][parent]) for parent,x in wide.items()))
        return 1
    except (OSError,ValueError,subprocess.SubprocessError) as failure:
        save("campaign-exception.json",dict(kind=type(failure).__name__,
            reason=str(failure) if isinstance(failure,ValueError) else None))
        raise


if __name__=="__main__":
    raise SystemExit(main(float(sys.argv[1]),sys.argv[2]))
