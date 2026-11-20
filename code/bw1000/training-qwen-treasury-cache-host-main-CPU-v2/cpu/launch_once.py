"""Future unique CPU/write scope only; no GPU branch, one900second root epoch."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from scope import RUN,SRC,CPU,LABEL,ARMS,COLS,require,pin,read,remaining,stage_limits,verify_dependencies
from snapshot_io import separate_logs
from slurm_accounting import wait_terminal
from verify_package import verify


def save(path,value):
    with path.open("x",encoding="utf8") as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")


def main(publication,claim_sha,request):
    epoch,mono=time.time(),time.monotonic();deadline,mono_end=epoch+900,mono+900
    os.umask(0o077)
    require(re.fullmatch(r"[0-9a-f]{40}",publication) and re.fullmatch(r"[0-9a-f]{64}",claim_sha) and re.fullmatch(r"[0-9a-f]{32}",request),"bad original launch identity")
    require(Path(__file__).resolve().parent==CPU and not RUN.is_symlink(),"unexpected CPU package namespace")
    require(pin(RUN/"CPU_CLAIM.json")["SHA256"]==claim_sha,"published CPU claim differs");verify(RUN)
    require(not any((RUN/n).exists() for n in ("intent.json","arms","outcome.json")),"existing original namespace requires reconciliation, no replay")
    def query(*args,own_end=None):
        seconds=remaining(deadline,mono_end,time.time(),time.monotonic(),9)
        if own_end is not None:seconds=min(seconds,own_end-time.monotonic())
        require(seconds>0,"own accounting window expired")
        return subprocess.check_output(args,text=True,timeout=seconds).strip().splitlines()
    def accounting(parent,label,own_end=None):
        rows=[row for line in query("sacct","-n","-P","-j",parent,"-o",COLS,own_end=own_end) if len(row:=line.split("|"))==7 and row[1]==label]
        require(len(rows)<=1,"duplicate unique CPU label");return rows[0] if rows else None
    def admit(arm):
        spec=ARMS[arm];rows=query("squeue","-h","-j",spec["parent"],"-o","%T|%D|%N");require(len(rows)==1,"parent unavailable")
        state,count,nodes=rows[0].split("|");require((state,count)==("RUNNING","40") and spec["node"] in query("scontrol","show","hostnames",nodes),"original CPU parent/node differs")
        for line in query("scontrol","show","step",spec["parent"],"--oneliner"):
            fields=dict(word.split("=",1) for word in line.split() if "=" in word);step=fields.get("StepId","")
            if step.startswith(spec["parent"]+".") and step.rsplit(".",1)[1].isdigit():
                require(spec["node"] not in query("scontrol","show","hostnames",fields["NodeList"]),"assigned node holds an original numbered step")
    verify_dependencies(accounting)
    for arm,spec in ARMS.items():
        require(accounting(spec["parent"],LABEL+"-"+arm) is None,"unique CPU label already exists");admit(arm)
    save(RUN/"intent.json",dict(request=request,publication=publication,claim_SHA256=claim_sha,epoch=epoch,deadline=deadline,
         arms=ARMS,maximum_CPU_steps=3,actual_expected_GPU_cards=0,maximum_explicit_canary_bytes_per_arm=40960,
         maximum_explicit_canary_bytes_all_arms=122880,canary_does_not_prove_quota_or_JIT_capacity=True,GPU_GO=False))
    (RUN/"arms").mkdir();results={};stop=None
    for arm,spec in ARMS.items():
        folder=RUN/"arms"/arm;folder.mkdir();label=LABEL+"-"+arm
        if stop:
            result=dict(arm=arm,passed=False,dispatched=False,reason="NOT_DISPATCHED_AFTER_PRIOR_CPU_FAILURE_OR_UNKNOWN",prior=stop,planned_InferRequest_objects=108,planned_canary_bytes=40960,GPU_GO=False,replay_allowed=False)
            save(folder/"outcome.json",result);results[arm]=result;continue
        rc,error=None,None
        try:
            limits=stage_limits(deadline,mono_end,time.time(),time.monotonic());admit(arm)
            require(accounting(spec["parent"],label) is None,"unique CPU label appeared")
            limits=stage_limits(deadline,mono_end,time.time(),time.monotonic())
            work_deadline=min(deadline-40,time.time()+limits["work_seconds"])
            command=["srun","--jobid="+spec["parent"],"--job-name="+label,"--exact","--immediate=5","-N1","-n1","-w",spec["node"],"-c8","--mem=12288M","--gres=none",
                     "--time=00:%02d:00"%(limits["Slurm_seconds"]//60),"/usr/bin/timeout","--kill-after=10s",str(limits["work_seconds"])+"s",
                     "/usr/local/python3.12/bin/python3","-B",str(CPU/"inside_stage.py"),"--arm",arm,"--work-deadline",str(work_deadline)]
            require(limits["Slurm_seconds"]+50<=remaining(deadline,mono_end,time.time(),time.monotonic(),900),"CPU setup consumed original reserve")
            save(folder/"step-intent.json",dict(original_epoch=epoch,deadline=deadline,work_deadline=work_deadline,limits=limits,command=command))
            with separate_logs(folder/"stage.stdout",folder/"Slurm.stderr") as (out,err):
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
                handle=dict(PID=child.pid,limits=limits)
                try:
                    fields=Path("/proc",str(child.pid),"stat").read_text().rsplit(") ",1)[1].split();handle.update(PGID=int(fields[2]),startticks=int(fields[19]))
                except FileNotFoundError:handle["exited_before_identity_observation"]=True
                save(folder/"owned-srun.json",handle)
                try:rc=child.wait(timeout=min(limits["client_seconds"],remaining(deadline,mono_end,time.time(),time.monotonic(),900)-30))
                except subprocess.TimeoutExpired:pass
        except (OSError,ValueError,subprocess.SubprocessError) as exc:error=type(exc).__name__
        own_end=min(mono_end-5,time.monotonic()+30,time.monotonic()+max(0,deadline-time.time()-5));errors=[]
        def observe():
            try:return accounting(spec["parent"],label,own_end)
            except (OSError,ValueError,subprocess.SubprocessError) as exc:errors.append(type(exc).__name__);return None
        visibility=wait_terminal(observe,own_end,interval=2,max_observations=14);row=visibility["row"]
        inside=None
        if (folder/"inside-outcome.json").is_file():
            try:inside=read(folder/"inside-outcome.json")
            except (OSError,ValueError) as exc:error=type(exc).__name__
        passed=bool(rc==0 and error is None and row and row[2:4]==["COMPLETED","0:0"] and row[5]==spec["node"]
            and dict(x.split("=",1) for x in row[6].split(","))=={"cpu":"8","mem":"12G","node":"1"}
            and inside and inside["passed"] is True and inside["step"]==row[0] and inside["allocated_GPU_cards"]==0
            and inside["ownership_pre_post_verified"] is True)
        result=dict(arm=arm,passed=passed,dispatched=True,rc=rc,error_class=error,accounting_columns=COLS,accounting=visibility,
            accounting_errors=errors,inside_outcome_pin=pin(folder/"inside-outcome.json") if inside else None,GPU_GO=False,replay_allowed=False,
            planned_InferRequest_objects=108,planned_canary_bytes=40960)
        save(folder/"outcome.json",result);results[arm]=result
        if not passed:stop=dict(arm=arm,original_accounting=row,error_class=error)
    passed=len(results)==3 and all(x["passed"] for x in results.values())
    result=dict(request=request,epoch=epoch,deadline=deadline,arms=results,passed=passed,GPU_GO=False,actual_GPU_steps=0,
        actual_CPU_steps=sum(x["dispatched"] for x in results.values()),planned_CPU_arms=3,planned_InferRequest_objects=324,
        maximum_explicit_canary_bytes=122880,implicit_library_own_cache_write_bytes="UNKNOWN",user_quota_or_arbitrary_JIT_capacity_qualified=False,
        capacity_producer_only_not_independent_PASS=True,automatic_GPU_followon=False,no_GPUv4=True)
    save(RUN/"outcome.json",result);print(json.dumps(dict(request=request,passed=passed,actual_CPU_steps=result["actual_CPU_steps"],GPU_GO=False)))
    return 0 if passed else 1


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--publication",required=True);p.add_argument("--claim-sha256",required=True);p.add_argument("--request-id",required=True)
    a=p.parse_args();raise SystemExit(main(a.publication,a.claim_sha256,a.request_id))
