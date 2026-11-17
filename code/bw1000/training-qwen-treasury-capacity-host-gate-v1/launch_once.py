"""Unique three-stage zero-GPU capacity collection under one300second epoch."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from scope import RUN,SRC,LABEL,ARMS,COLS,require,read,pin,budgets
from snapshot_io import separate_logs
from slurm_accounting import wait_terminal
from verify_package import verify


def save(path,value):
    with path.open("x",encoding="utf8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")


def main(publication,claim_sha,request):
    epoch,mono=time.time(),time.monotonic();deadline,mono_end=epoch+300,mono+300
    os.umask(0o077)
    require(re.fullmatch(r"[0-9a-f]{40}",publication) and re.fullmatch(r"[0-9a-f]{64}",claim_sha) and re.fullmatch(r"[0-9a-f]{32}",request),"bad original identity")
    require(Path(__file__).resolve().parent==SRC and not RUN.is_symlink(),"wrong source namespace")
    require(pin(SRC/"CLAIM.json")["SHA256"]==claim_sha,"claim differs");verify(SRC)
    require(not any((RUN/n).exists() for n in ("intent.json","arms","outcome.json")),"original namespace exists; no replay")
    def query(*args,own_end=None):
        remaining=min(8,deadline-time.time(),mono_end-time.monotonic())
        if own_end is not None:remaining=min(remaining,own_end-time.monotonic())
        require(remaining>0,"original observation epoch exhausted")
        return subprocess.check_output(args,text=True,timeout=remaining).strip().splitlines()
    def accounting(parent,label,own_end=None):
        rows=[row for line in query("sacct","-n","-P","-j",parent,"-o",COLS,own_end=own_end) if len(row:=line.split("|"))==7 and row[1]==label]
        require(len(rows)<=1,"duplicate original label");return rows[0] if rows else None
    for arm,spec in ARMS.items():
        require(accounting(spec["parent"],LABEL+"-"+arm) is None,"original CPU step exists")
        jobs=query("squeue","-h","-j",spec["parent"],"-o","%T|%D|%N");require(len(jobs)==1,"parent unavailable")
        state,count,nodes=jobs[0].split("|")
        require((state,count)==("RUNNING","40") and spec["node"] in query("scontrol","show","hostnames",nodes),"parent/node differs")
    original_controller=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-treasury-explicit-dev3-gpu-v3/source/arm_controller.py")
    require(pin(original_controller)["SHA256"]=="53a2d2fc0cf5f1934edfc13e7fb3e77093d02792962429de1fb7a9cb8f0f7ccf","frozen original v3 controller differs")
    save(RUN/"intent.json",dict(request=request,publication=publication,claim_SHA256=claim_sha,epoch=epoch,deadline=deadline,
         arms=ARMS,maximum_CPU_steps=3,serial=True,actual_expected_GPU_cards=0,GPU_GO=False,canary_allowed=False))
    (RUN/"arms").mkdir();results={}
    for arm,spec in ARMS.items():
        folder=RUN/"arms"/arm;folder.mkdir();label=LABEL+"-"+arm
        rc,error=None,None
        try:
            limit=budgets(deadline,mono_end,time.time(),time.monotonic());work_deadline=min(deadline-25,time.time()+50)
            require(accounting(spec["parent"],label) is None,"original CPU label appeared")
            command=["srun","--jobid="+spec["parent"],"--job-name="+label,"--exact","--immediate=5","-N1","-n1","-w",spec["node"],
                     "-c2","--mem=4096M","--gres=none","--time=00:01:00","/usr/bin/timeout","--kill-after=5s","50s",
                     "/usr/local/python3.12/bin/python3","-B",str(SRC/"stage.py"),"--arm",arm,"--deadline",str(work_deadline)]
            require(min(deadline-time.time(),mono_end-time.monotonic())>=90,"CPU setup consumed observation reserve")
            save(folder/"step-intent.json",dict(epoch=epoch,deadline=deadline,work_deadline=work_deadline,limits=limit,command=command))
            with separate_logs(folder/"step.stdout",folder/"step.stderr") as (out,err):
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
                owned=dict(PID=child.pid)
                try:
                    fields=Path("/proc",str(child.pid),"stat").read_text().rsplit(") ",1)[1].split();owned.update(PGID=int(fields[2]),startticks=int(fields[19]))
                except FileNotFoundError:owned["exited_before_identity_observation"]=True
                save(folder/"owned-srun.json",owned)
                try:rc=child.wait(timeout=min(limit["client_seconds"],deadline-time.time()-20,mono_end-time.monotonic()-20))
                except subprocess.TimeoutExpired:pass
        except (OSError,ValueError,subprocess.SubprocessError) as exc:error=type(exc).__name__
        own_end=min(mono_end-5,time.monotonic()+20,time.monotonic()+max(0,deadline-time.time()-5));errors=[]
        def observe():
            try:return accounting(spec["parent"],label,own_end)
            except (OSError,ValueError,subprocess.SubprocessError) as exc:errors.append(type(exc).__name__);return None
        visible=wait_terminal(observe,own_end,interval=2,max_observations=9);row=visible["row"]
        original=None
        if (folder/"stage-outcome.json").is_file():
            try:original=read(folder/"stage-outcome.json")
            except (OSError,ValueError) as exc:error=type(exc).__name__
        passed=bool(rc==0 and error is None and row and row[2:4]==["COMPLETED","0:0"] and row[5]==spec["node"]
            and dict(x.split("=",1) for x in row[6].split(","))=={"cpu":"2","mem":"4G","node":"1"}
            and original and original["passed"] is True and original["step"]==row[0] and original["GPU_GO"] is False)
        item=dict(arm=arm,passed=passed,rc=rc,error_class=error,accounting_columns=COLS,accounting=visible,accounting_errors=errors,
             stage_outcome_pin=pin(folder/"stage-outcome.json") if original else None,actual_allocated_GPU_cards=0 if row and "gres/" not in row[6] else None,
             GPU_GO=False,canary_written=False,replay_allowed=False)
        save(folder/"outcome.json",item);results[arm]=item
    passed=len(results)==3 and all(x["passed"] for x in results.values())
    save(RUN/"outcome.json",dict(request=request,epoch=epoch,deadline=deadline,arms=results,passed=passed,
         GPU_GO=False,actual_GPU_steps=0,automatic_GPU_followon=False,exact_ENOSPC_writer_path_proved=False))
    print(json.dumps(dict(request=request,passed=passed,GPU_GO=False,actual_GPU_steps=0)))
    return 0 if passed else 1


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--publication",required=True);p.add_argument("--claim-sha256",required=True);p.add_argument("--request-id",required=True)
    a=p.parse_args();raise SystemExit(main(a.publication,a.claim_sha256,a.request_id))
