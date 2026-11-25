"""One private training process forest inside the existing assigned eight-DCU step."""

import argparse
from hashlib import file_digest
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from evidence import BASE, NATIVE_REQUEST, OBSERVED_UPDATES, WORLD, need, pin, read, summarize
from prepare import INIT_NAMED_SHA, ORIGINAL_SHA, RECIPE_SHA, history, prepare, stage_proof


OWNED_SHA = "63995f2a2a3a6edaa29013fe1942114e0279d9cf48e7a9051515d5158b8552aa"
MODULE_ROOT = Path(__file__).resolve().parent


def save(path,value):
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load(path, expected, name):
    read(path,65536,expected)
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def assigned(raw, parent, node, env, hostname, boot, affinity, monotonic_ns):
    step=env.get("SLURM_STEP_ID","")
    cards=raw["PCI_by_index"]
    lease=raw["leased_snapshot"]
    need(re.fullmatch(r"[0-9]+",parent) and re.fullmatch(r"[0-9]+",step)
         and env.get("SLURM_JOB_ID")==parent and hostname.split(".")[0]==node==raw["node"]==lease["node"]
         and raw["step"]==parent+"."+step and env.get("SLURM_CPUS_PER_TASK")=="64" and len(affinity)>=64
         and len(set(cards))==len(cards)==WORLD and sorted(cards)==lease["pci"] and lease["boot_id"]==boot
         and lease["kfd_holder_count"]==0 and lease["full_fd_inventory_performed"] is False
         and 0<=monotonic_ns-lease["monotonic_ns"]<=45_000_000_000
         and all(re.fullmatch(r"[a-f0-9]{4,8}:[a-f0-9]{2}:[a-f0-9]{2}\.[0-7]",pci) for pci in cards),
         "EXACT_EXISTING_ASSIGNED_EIGHT_DCU_STEP_REQUIRED")
    return raw["step"]


def capture(path):
    ranks, errors, initialization, finished, observed = {}, [], 0, 0, 0
    for index in range(WORLD):
        directory=path/("rank-%02d"%index)
        files={step:directory/("update-%02d.json"%step) for step in range(1,OBSERVED_UPDATES+1)}
        observed+=sum((directory/("batch-%02d-done.json"%n)).is_file() for n in range(8))
        try:
            records={step:json.loads(read(file,131072)) for step,file in files.items() if file.is_file()}
            if (directory/"initial.json").is_file():
                init=json.loads(read(directory/"initial.json",8192))
                need(init["rank"]==index and init["world_size"]==WORLD and
                     init["initial_trainables_SHA256"]==INIT_NAMED_SHA,"EXACT_RANK_INITIALIZATION_RECEIPT")
                initialization+=1
            if (directory/"complete.json").is_file():
                final=json.loads(read(directory/"complete.json",4096))
                need(final["rank"]==index and final["global_step"]==final["actual_optimizer_updates_observed"]==2
                     and final["Model_GO"] is False and set(records)=={1,2}
                     and final["batches"]==sum(len(row["batches"]) for row in records.values()),
                     "EXACT_RANK_COMPLETION_RECEIPT")
                finished+=1
            if records:
                ranks[index]=records
        except (ValueError,OSError,KeyError,TypeError) as error:
            errors.append({"rank":index,"type":type(error).__name__})
    return {**summarize(ranks),"rank_initialization_receipts":initialization,"rank_completion_receipts":finished,
            "rank_batch_done_files":observed,"rank_read_errors":errors,"rank_failure_receipts":
            sum((path/("rank-%02d"%rank)/"failure.json").is_file() for rank in range(WORLD))}


def run_owned(root, launch, owned, deadline, *, start=subprocess.Popen):
    registry={}
    def command_matches(argv, run_root):
        try:
            offset=argv.index(b"--output_dir")
            return argv[offset+1]==os.fsencode(run_root/"train") and b"qwen_street_scene_first_two" in argv
        except (ValueError,IndexError):
            return False
    owned.native_command=command_matches
    result=dict(child_returncode=None,error_type=None,cleanup=None)
    child=None
    try:
        env={**os.environ,**launch["environment_overrides"],"MASTER_ADDR":"127.0.0.1"}
        need(not any(name in os.environ for name in ("RANK","LOCAL_RANK","WORLD_SIZE","LOCAL_WORLD_SIZE","NNODES","NODE_RANK")),
             "EXACT_SINGLE_EIGHT_RANK_LAUNCHER")
        with (root/"native.log").open("xb") as stream:
            child=start([sys.executable,"-B",*launch["command"]],env=env,cwd=launch["cwd"],
                        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            row=owned.snapshot().get(child.pid)
            if row is not None:
                registry[child.pid]={key:row[key] for key in ("PID","PPID","PGID","startticks","state")}
            save(root/"child-owner.json",{"PID":child.pid,"identity":registry.get(child.pid)})
            while child.poll() is None:
                need(time.time()<deadline,"NATIVE_CHILD_DEADLINE")
                owned.refresh(registry,root)
                try:
                    child.wait(timeout=min(1,max(.01,deadline-time.time())))
                except subprocess.TimeoutExpired:
                    pass
            if child.returncode==0:
                until=min(deadline,time.time()+4)
                while owned.refresh(registry,root) and time.time()<until:
                    time.sleep(.1)
    except (ValueError,OSError,RuntimeError,subprocess.SubprocessError) as err:
        result["error_type"]=type(err).__name__
    finally:
        if child is not None:
            live=owned.refresh(registry,root)
            if child.poll() is None and child.pid not in live:
                child.terminate()
            result["cleanup"]=owned.cleanup(registry,root,grace=5)
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
                result["cleanup"]["termination_needed"]=True
            result["child_returncode"]=child.returncode
    if (root/"native.log").is_file():
        with (root/"native.log").open("rb") as stream:
            result["native_log_SHA256"]=file_digest(stream,"sha256").hexdigest()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("framework","output","assignment","original","binding","recipe","owned","public-native","terminal220","assets"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("request","parent","node","assignment-sha"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--port",type=int,required=True)
    for name in ("epoch","deadline","maximum"):
        parser.add_argument("--"+name,type=float,required=True)
    args=parser.parse_args()
    root=args.output
    cache=root/"never-populated-python-cache"
    need(sys.dont_write_bytecode and sys.pycache_prefix==os.environ.get("PYTHONPYCACHEPREFIX")==str(cache)
         and not cache.exists() and not cache.is_symlink(),"EXACT_ISOLATED_TRAINER_PYTHON_CACHE")
    need(sys.platform=="linux" and re.fullmatch(r"[0-9a-f]{32}",args.request)
         and args.request!=NATIVE_REQUEST
         and root.name=="qwen-textvqa-first-two-"+args.request and str(root.parent)==str(BASE)
         and root.resolve(strict=True)==root and root.is_dir()
         and args.epoch<=time.time()<args.deadline<=args.epoch+args.maximum,
         "EXACT_ADMITTED_PRIVATE_OUTPUT_AND_ORIGINAL_EPOCH")
    binding=json.loads(read(args.assignment,32768,args.assignment_sha))
    need(bool(re.fullmatch("[0-9a-f]{64}",args.assignment_sha)),"EXACT_ASSIGNED_RECEIPT_PIN")
    step=assigned(binding,args.parent,args.node,os.environ,socket.gethostname(),
                  Path("/proc/sys/kernel/random/boot_id").read_text().strip(),os.sched_getaffinity(0),time.monotonic_ns())
    sys.path.insert(0,str(args.framework))
    recipe=load(args.recipe,RECIPE_SHA,"exact_original_first_update_recipe").prepare
    owned=load(args.owned,OWNED_SHA,"exact_original_owned_processes")
    model=history(args.public_native,args.terminal220,args.assets,args.binding)
    dataset=stage_proof()
    launch=prepare(recipe,read(args.original,32768,ORIGINAL_SHA),args.binding,args.framework,root,
                   MODULE_ROOT/"street_updates.py",args.port,epoch=args.epoch,deadline=args.deadline,
                   maximum=args.maximum)
    launch["environment_overrides"]["PYTHONPYCACHEPREFIX"]=str(cache)
    launch["environment_overrides"]["PYTHONDONTWRITEBYTECODE"]="1"
    save(root/"native-intent.json",dict(request=args.request,numeric_step=step,epoch=args.epoch,
         deadline=args.deadline,planned_updates=2,nominal_schedule_steps=25,assigned_DCU=8,
         dataset_fingerprints=len(dataset),model_member_count=model["model_member_count"],
         model_manifest_SHA256=model["model_manifest_SHA256"],initial_trainables_SHA256=model["initial_trainables_SHA256"],
         argv_SHA256=pin(launch["command"]),cwd=launch["cwd"],replay_allowed=False))
    try:
        status=run_owned(root,launch,owned,args.deadline)
    except (ValueError,OSError,RuntimeError,subprocess.SubprocessError) as error:
        status=dict(child_returncode=None,error_type=type(error).__name__,cleanup=None)
    try:
        status["observed"]=capture(root/"capture")
    except (ValueError,OSError,KeyError,TypeError) as error:
        status["capture_error_type"]=type(error).__name__
        status["observed"]=dict(planned_optimizer_updates=2,nominal_schedule_steps=25)
    status.update(request=args.request,numeric_step=step,planned_optimizer_updates=2,Model_GO=False,
                  promotion=False,replay_allowed=False)
    save(root/"native-outcome.json",status)
    print(json.dumps(status,sort_keys=True))
    observed=status["observed"]
    clean=status["cleanup"] or {}
    return int(not (status["child_returncode"]==0 and status["error_type"] is None
         and clean.get("owned_processes_remaining")==0 and clean.get("termination_needed") is False
         and observed.get("rank_completion_receipts")==observed.get("rank_initialization_receipts")==WORLD
         and not observed.get("rank_read_errors") and observed.get("native_sampler_verified_all_original_rows")
         and observed.get("qualified_nonzero_update") is not None))


if __name__=="__main__":
    raise SystemExit(main())
