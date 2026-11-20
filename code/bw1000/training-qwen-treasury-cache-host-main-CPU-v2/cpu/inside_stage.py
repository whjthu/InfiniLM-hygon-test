"""CPU step: real container/main optin with exact owned RUN, separate logs, and original deadline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from scope import RUN,SRC,ARMS,FLAGS,IMAGE,COLS,require,pin,read,declared_binds,runtime_bind_source
from snapshot_io import separate_logs
from verify_package import verify
from ownership import capture,validate_snapshot,validate_pair,container_exit_identity


def save(path,value):
    with path.open("x",encoding="utf8") as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")


def container_command(arm,work_deadline,seconds):
    args=list(FLAGS)
    for host,target,mode in declared_binds(arm):args.extend(("--bind",runtime_bind_source(host)+":"+target+":"+mode))
    return args+[str(IMAGE),"/usr/bin/timeout","--kill-after=10s",str(seconds)+"s","/bin/bash","--noprofile","--norc",
                 str(SRC/"run_host_main.sh"),arm,str(work_deadline),"--host-main-preflight"]


def validate_receipts(cache,main,arm,cache_sha):
    require(cache["arm"]==arm and cache["all_cache_canaries_verified"] is True and cache["total_canary_bytes"]==40960
            and cache["tested_bytes_only"] is True and cache["user_quota_and_arbitrary_JIT_capacity_qualified"] is False
            and cache["original_home_or_credentials_copied"] is False and cache["GPU_GO"] is False,"ownedcache canary receipt differs")
    require(len(cache["records"])==10 and all(x["bytes_written"]==4096 and x["own_canary_removed"] is True for x in cache["records"].values()),"exact ownedcanary denominator differs")
    require(main["state"]=="TRUE_ARGPARSE_MAIN_REAL_ENGINE_IMPORTS_AND_CPU_PROCESSOR_RETURNED_BEFORE_LOAD_MODEL_TRUE"
            and main["arm"]==arm and main["actual_InferRequest_objects"]==108 and main["actual_AdapterRequest_objects"]==2
            and main["actual_RequestConfig"]=={"max_tokens":64,"temperature":0,"num_beams":1,"n":1,"stream":False}
            and main["ENGINE_PINS_verified"] is True and main["materialize_call_signature_bound"] is True
            and main["Engine_signature_bound_only"] is True and main["load_model_True_calls"]==main["Engine_instances_created"]==main["generation_calls"]==0
            and main["CUDA_success_mocked"] is False and main["explicit_CUDA_probe_calls"]==0
            and main["GPU_GO"] is False and main["user_quota_and_arbitrary_JIT_capacity_qualified"] is False,"true HOST-main receipt differs")
    provider=main["torch_provider"]
    require(provider==read(RUN/"arms"/arm/"host-main/torch-provider.json")
            and provider["file"]==provider["distribution_file"]=="/usr/local/lib/python3.10/dist-packages/torch/__init__.py"
            and provider["version"]=="2.10.0" and provider["HIP_build"]=="6.3.26113"
            and provider["distribution_version"]=="2.10.0+das.opt1.dtk2604.2606161421.g91a580"
            and len(provider["SHA256"])==64,"actual imported and installed Torch receipts differ")
    require(main["cache_readiness_SHA256"]==cache_sha and main["native_main_file"]==str(SRC/"native_generation.py"),"cache/main file binding differs")
    require(main["input_inventory_SHA256"]=="07d906e3ed840d828ee41509cae21b45a754ddef0825d411ca3d8216f606b0d8"
            and main["native_main_SHA256"]==pin(SRC/"native_generation.py")["SHA256"],"actualmain/input identity differs")
    return True


def stage(arm,work_deadline):
    spec=ARMS[arm];folder=RUN/"arms"/arm;intent=read(RUN/"intent.json");start=time.monotonic()
    require(os.uname().nodename.split(".")[0]==spec["node"] and os.environ.get("SLURM_JOB_ID")==spec["parent"],"wrong actual CPU node")
    require(os.environ.get("SLURM_STEP_GPUS") in (None,"","(null)"),"CPU stage was allocated GPU")
    node_source_verification=verify(RUN)
    left=work_deadline-time.time();require(0<left<=150 and work_deadline<=intent["deadline"]-40,"inside CPU deadline differs")
    before=capture(spec["parent"],spec["node"],work_deadline)
    save(folder/"ownership-pre.json",before)
    rc,error=None,None;identity=None;main_deadline=None
    try:
        validate_snapshot(before,spec["parent"],spec["node"])
        # Same original work deadline: pre and post spend it, never restart it.
        left=work_deadline-time.time()
        seconds=int(left-20)  # Reserve10 for owned-child kill grace and10 for one post inventory.
        require(seconds>0,"no original budget after ownership pre and post reserve")
        main_deadline=min(work_deadline-20,time.time()+seconds)
        command=container_command(arm,main_deadline,seconds)
        save(folder/"container-intent.json",dict(original_epoch=intent["epoch"],deadline=intent["deadline"],work_deadline=work_deadline,
            main_deadline=main_deadline,main_timeout_seconds=seconds,post_reserve_seconds=10,flags=FLAGS,binds=declared_binds(arm),
            command=command,explicit_CPU_optin=True,allocated_GPU_cards=0,ownership_pre_pin=pin(folder/"ownership-pre.json")))
        with separate_logs(folder/"canary-hostmain.stdout",folder/"container.stderr") as (out,err):
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            identity=dict(PID=child.pid)
            try:
                fields=Path("/proc",str(child.pid),"stat").read_text().rsplit(") ",1)[1].split();identity.update(PGID=int(fields[2]),startticks=int(fields[19]))
            except FileNotFoundError:identity["exited_before_identity_observation"]=True
            save(folder/"container-owned-process.json",identity)
            try:rc=child.wait(timeout=max(.001,min(seconds+10,work_deadline-time.time()-10)))
            except subprocess.TimeoutExpired:pass
    except (OSError,ValueError,subprocess.SubprocessError) as exc:error=type(exc).__name__
    # Exactly one same-step post even when main/bootstrap/pre fails; expired/denied remains UNKNOWN.
    after=capture(spec["parent"],spec["node"],work_deadline)
    save(folder/"ownership-post.json",after)
    exited=container_exit_identity(identity,work_deadline)
    ownership_ok=False
    try:ownership_ok=validate_pair(before,after,spec["parent"],spec["node"],exited)
    except (ValueError,KeyError,TypeError) as exc:error=error or type(exc).__name__
    save(folder/"ownership-comparison.json",dict(passed=ownership_ok,container_exit=exited,
         original_work_deadline=work_deadline,main_deadline=main_deadline,
         pre_pin=pin(folder/"ownership-pre.json"),post_pin=pin(folder/"ownership-post.json"),
         freshness_is_not_busy=True))
    cache_path=folder/"owned-cache-readiness.json";main_path=folder/"host-main/host-main-receipt.json"
    cache,main=None,None
    try:
        if cache_path.is_file():cache=read(cache_path)
        if main_path.is_file():main=read(main_path)
    except (OSError,ValueError,TypeError) as exc:error=type(exc).__name__
    qualified=False
    if rc==0 and error is None and ownership_ok and cache and main:
        try:qualified=validate_receipts(cache,main,arm,pin(cache_path)["SHA256"])
        except (KeyError,ValueError,TypeError) as exc:error=type(exc).__name__
    pins={}
    for name in ("ownership-pre.json","ownership-post.json","ownership-comparison.json","canary-hostmain.stdout","container.stderr","owned-cache-readiness.json","host-main/host-main-receipt.json","host-main/torch-provider.json"):
        p=folder/name
        if p.is_file():pins[name]=pin(p)
    outcome=dict(arm=arm,passed=bool(qualified),rc=rc,error_class=error,node=spec["node"],step=spec["parent"]+"."+os.environ["SLURM_STEP_ID"],
        node_source_verification=node_source_verification,node_package_MF=pin(RUN/"CPU_PACKAGE_FILES.json"),
        original_epoch=intent["epoch"],deadline=intent["deadline"],work_deadline=work_deadline,main_deadline=main_deadline,wall_seconds=time.monotonic()-start,
        ownership_pre_post_verified=ownership_ok,ownership_freshness_is_not_busy=True,
        original_file_pins=pins,cache_receipt_present=cache is not None,HOST_main_receipt_present=main is not None,
        allocated_GPU_cards=0,canary_explicit_total_bytes=cache["total_canary_bytes"] if cache else None,
        cache_canary_only_tested_bytes=True,implicit_library_cache_write_bytes="UNKNOWN",implicit_library_device_queries="UNKNOWN",
        user_quota_or_arbitrary_JIT_capacity_qualified=False,GPU_GO=False,no_GPUv4=True)
    save(folder/"inside-outcome.json",outcome)
    print(json.dumps(dict(arm=arm,passed=outcome["passed"],GPU_GO=False)))
    return 0 if outcome["passed"] else 1


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--arm",required=True,choices=ARMS);p.add_argument("--work-deadline",required=True,type=float)
    a=p.parse_args();raise SystemExit(stage(a.arm,a.work_deadline))
