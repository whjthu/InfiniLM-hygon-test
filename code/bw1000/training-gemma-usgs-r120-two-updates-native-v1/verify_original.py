"""Independently reconcile the actual CPU index or two-update training run."""

import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from contract import WHOLE, stable_json
from rendezvous import DIAGNOSTIC, WAIT
from sampler_consensus import VOTE_SCHEMA, UNANIMOUS_SCHEMA
from prepare import LABEL,NEW,ROOT,digest,emit_file,encoded,need,pin,read


OPERATION=ROOT/"training-gemma-usgs-r120-two-updates-native-v1"
REQUEST="0c2fc133102849748fb3e97e4613889c"
SOURCE=Path(__file__).resolve().parent
ACCOUNTING="JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"


def actual_accounting(query=subprocess.check_output):
    from contract import PARENT
    from utc_scope import history_argv
    raw=query(history_argv(PARENT),timeout=10)
    need(len(raw)<=1<<20,"R39_ACCOUNTING_METADATA_BOUND")
    rows=[line.split("|") for line in raw.decode("ascii").splitlines() if line]
    need(all(len(r)==9 for r in rows),"R39_EXACT_INDEPENDENT_NINE_COLUMNS")
    return rows


def sole(rows,label,step,tres):
    matches=[row for row in rows if row[1]==label]
    need(len(matches)==1 and matches[0][0]==step and re.fullmatch(r"1103606\.[0-9]+",step)
         and matches[0][5]=="f13r1n12" and matches[0][6] in tres,"R39_EXACT_ACTUAL_STEP_NODE_AND_TRES")
    return matches[0]


def parsed_utc(value):
    return datetime.strptime(value,"%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def accounting_window(row,epoch,deadline):
    start,end=map(parsed_utc,row[7:])
    need(row[4].isdigit() and epoch-1<=start<=end<=deadline+1 and abs(end-start-int(row[4]))<=1,
         "R39_EXACT_SINGLE_EPOCH_ACCOUNTING_WINDOW")
    return start,end


def audit_ranks(results,natives,index):
    initial_shas=set()
    final_shas=set()
    pairs=[]
    ordinals=[]
    ranks=[]
    for rank,(result,init) in enumerate(zip(results,natives,strict=True)):
        expected=index["rank"][str(rank)]
        model=init["named"]
        need(result["rank"]==init["rank"]==rank and result["world_size"]==8 and result["completed_global_step"]==2
             and result["nominal_scheduler_steps"]==25 and result["actual_per_rank_microbatches"]==8
             and len(result["inputs"])==len(result["outputs"])==8 and len(result["steps"])==2 and not result["saved_checkpoints"]
             and init["mode"]=="TARGET_FRESH_DOMAIN_ONLY" and init["RNG_unchanged"] is init["target_created"] is True
             and init["reference_initial_read"] is init["original_A100_value_equivalence"] is False
             and model["tensors"]==476 and model["elements"]==29802496 and model["finite"] is True,
             "R39_ACTUAL_8_BY_8_NATIVE_TRAINING_AND_FRESH_COHORT")
        initial_shas.add(model["SHA256"])
        final_shas.add(result["final_adapter"]["SHA256"])
        for ordinal,(capture,loss,wanted) in enumerate(zip(result["inputs"],result["outputs"],expected,strict=True),1):
            need(capture["ordinal"]==loss["ordinal"]==ordinal and capture["global_step"]==(ordinal-1)//4
                 and (capture["physical_ordinal"],capture["source_line_SHA256"],capture["native_input_SHA256"],capture["pair"],capture["sign"])==
                     tuple(wanted[field] for field in ("ordinal","line_SHA256","input_SHA256","pair","sign"))
                 and capture["full_input"]["finite"] is True and capture["input_witness_RNG_unchanged"] is True
                 and loss["loss"]["finite"] is True and type(loss["loss_scalar"]) in (int,float) and math.isfinite(loss["loss_scalar"]),
                 "R39_EXACT_ACTUAL_INPUT_IMAGE_AND_FINITE_LOSS")
            ordinals.append(wanted["ordinal"])
            if wanted["pair"] is not None:
                pairs.append((wanted["pair"],wanted["sign"]))
        changes=[]
        for n,step in enumerate(result["steps"],1):
            need(step["global_step"]==n and step["optimizer_slot_count"]==476
                 and all(type(rate) in (int,float) and math.isfinite(rate) and rate>=0 for rate in step["learning_rates"])
                 and step["adapter_changed"] == (step["adapter_before_SHA256"]!=step["adapter_after_SHA256"])
                 and step["zero_learning_rate"]==(not any(step["learning_rates"])),
                 "R39_EXACT_TWO_REAL_OPTIMIZER_BOUNDARIES")
            if n>1:
                need(step["adapter_before_SHA256"]==result["steps"][n-2]["adapter_after_SHA256"],
                     "R39_SAME_PARAMETERS_CONTINUED_BETWEEN_UPDATES")
            if not step["zero_learning_rate"] and step["adapter_changed"]:
                changes.append(n)
        need(changes and result["parameter_update_passed"] is True
             and result["final_adapter"]["SHA256"]==result["steps"][-1]["adapter_after_SHA256"],
             "R39_REAL_NONZERO_LR_PARAMETER_CHANGE_REQUIRED")
        ranks.append({"rank":rank,"forwards":8,"updates":2,"changed_with_nonzero_LR":changes})
    need(len(results)==len(natives)==len(ranks)==8 and len(initial_shas)==len(final_shas)==1 and len(set(ordinals))==len(ordinals)==64
         and len(pairs)==12 and set(pairs)=={(n,s) for n in range(1,7) for s in ("positive","negative")},
         "R39_EXACT_COHORT_64_SLOTS_AND_ALL_6_PAIRS")
    return {"ranks":ranks,"fresh_initial_named_SHA256":next(iter(initial_shas)),
            "adapter_SHA256":next(iter(final_shas)),"actual_forwards":64,"new_exposures":12,"pairs":6,"actual_updates":2}


def adapter_reload(directory,expected):
    # This independent CPU read deserializes only the newly written adapter.
    # It does not initialize the base model or run inference.
    import torch
    from safetensors.torch import load_file
    from vlm.runtime.first_update_audit import fingerprint
    need(not torch.cuda.is_initialized() and {p.name for p in directory.iterdir()}=={"adapter_model.safetensors","adapter_config.json"},
         "R39_CPU_RELOAD_EXACT_ADAPTER_PACKAGE")
    file=directory/"adapter_model.safetensors"
    info=file.lstat()
    need(file.resolve(strict=True)==file and 0<info.st_size<=512<<20,"R39_EXACT_NATIVE_ADAPTER_EXTENT")
    data=load_file(str(file),device="cpu")
    after=file.lstat()
    result=fingerprint(data)
    identity=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    need(identity(info)==identity(after) and result["SHA256"]==expected and result["tensors"]==476 and result["finite"] is True,
         "R39_INDEPENDENT_CPU_ADAPTER_DESERIALIZED_VALUES")
    config=json.loads((directory/"adapter_config.json").read_bytes())
    need(config["peft_type"]=="LORA" and config["r"]==16 and config["lora_alpha"]==32 and config["lora_dropout"]==.05,
         "R39_RELOAD_EXACT_PEFT_PARAMETERS")
    return {"adapter_file_bytes":info.st_size,"named_SHA256":expected,"native_safetensors_reload":True,"base_model_instantiated":False}


def independently_verify(rows):
    intent, _ = stable_json(OPERATION/"intent.json")
    scope, _ = stable_json(OPERATION/"outcome.json")
    need(scope["request_id"] == intent["request_id"] == REQUEST and scope["epoch"] == intent["epoch"]
         and scope["deadline"] == intent["deadline"] == intent["epoch"] + WHOLE,
         "R46_EXACT_OUTCOME_INTENT_EPOCH")
    names = {"pre":{"cpu=2,mem=4G,node=1","cpu=2,mem=4096M,node=1"},
             "gpu":{"cpu=64,gres/dcu=8,mem=224000M,node=1","cpu=64,gres/dcu=8,mem=218.75G,node=1"},
             "post":{"cpu=2,mem=4G,node=1","cpu=2,mem=4096M,node=1"}}
    steps = {}
    physical = False
    physical_error = None
    try:
        for phase, tres in names.items():
            result = scope["results"][phase]
            row = sole(rows, OPERATION.name+"-"+phase, result["terminal"]["row"][0], tres)
            allowed = {"COMPLETED","FAILED","CANCELLED","TIMEOUT","NODE_FAIL","OUT_OF_MEMORY",
                       "PREEMPTED","BOOT_FAIL","DEADLINE","REVOKED"}
            need(row[2] in allowed and (phase=="gpu" or row[2:4]==["COMPLETED","0:0"]),
                 "R46_ACTUAL_TERMINAL_AND_PHYSICAL_STEPS")
            steps[phase] = {"id":row[0],"state":row[2],"exit":row[3],
                            "utc":accounting_window(row,intent["epoch"],intent["deadline"])}
        need(steps["pre"]["utc"][1]<=steps["gpu"]["utc"][0]<=steps["gpu"]["utc"][1]<=steps["post"]["utc"][0],
             "R46_PRE_GPU_POST_CHRONOLOGY")
        before,_ = stable_json(OPERATION/"before.json",scope["before"])
        after,_ = stable_json(OPERATION/"after.json",scope["after"])
        need(before["request_id"]==after["request_id"]==REQUEST and before["step"]==steps["pre"]["id"]
             and after["step"]==steps["post"]["id"] and before["pci"]==after["pci"]
             and len(set(before["pci"]))==8 and before["boot_id"]==after["boot_id"]
             and before["kfd_holder_count"]==after["kfd_holder_count"]==0
             and after["owned_processes_remaining"]==[] and scope["release_permitted"] is True,
             "R46_EXACT_INDEPENDENT_PHYSICAL_RELEASE")
        physical = True
    except (OSError, ValueError, TypeError, KeyError) as exc:
        physical_error = type(exc).__name__
    return scope, steps, physical, physical_error


def read_rank_sampler(member, rank, expected, trace, gpu_step):
    sampler = json.loads(read(member/"native-sampler.json",trace["sampler"]["SHA256"],4096,trace["sampler"]["bytes"]))
    directory = member/"sampler-votes"
    vote,_ = stable_json(directory/DIAGNOSTIC,trace["sampler_vote"],32768)
    unanimous,_ = stable_json(directory/WAIT,trace["sampler_unanimous"],32768)
    need(sampler["rank"]==vote["rank"]==unanimous["rank"]==rank
         and sampler["RNG_unchanged"] is True and sampler["live_sampler_iterated"] is False
         and sampler["real_Swift_shard_isolated"] is sampler["all_eight_rank_agreed"] is True
         and sampler["actual_loader_length"]==100 and sampler["first_eight_physical_ordinals"]==expected
         and vote["schema"]==VOTE_SCHEMA and vote["passed"] is True and vote["optimizer_updates"]==0
         and unanimous["schema"]==UNANIMOUS_SCHEMA and unanimous["all_rank_votes_valid"] is True
         and unanimous["all_rank_barrier_passed"] is True and unanimous["missing"]==unanimous["invalid"]==unanimous["failed"]==[]
         and vote["activation"]["step"]==unanimous["step"]==gpu_step
         and vote["activation"]["activation_SHA256"]==unanimous["activation_SHA256"],
         "R46_INDEPENDENT_ACTUAL_PRODUCTION_SAMPLER_CONSENSUS")
    return vote, unanimous


def audit_training(native, gpu_step):
    path = OPERATION/"continuous/train"
    traces = native["rank_receipts"]
    need(len(traces)==8 and [trace["rank"] for trace in traces]==list(range(8))
         and native["native_two_updates_witnessed"] is True,"R46_NATIVE_RESULTS_NOT_COMPLETE")
    index_result,_ = stable_json(NEW/"INDEX_RESULT.json")
    index_pin=index_result["native_index"]
    index=json.loads(read(NEW/"native64-index-private.json",index_pin["SHA256"],49152,index_pin["bytes"]))
    results, initial, evidence = [], [], []
    for rank, trace in enumerate(traces):
        member = path/"continuous-witness"/f"rank-{rank:02d}"
        for name,key,dest in (("result.json","result",results),("initialization.json","initialization",initial)):
            record=trace[key]
            dest.append(json.loads(read(member/name,record["SHA256"],cap=8<<20,size=record["bytes"])))
        expected = [item["ordinal"] for item in index["rank"][str(rank)]]
        evidence.append(read_rank_sampler(member,rank,expected,trace,gpu_step))
    votes = {str(rank):trace["sampler_vote"]["SHA256"] for rank,trace in enumerate(traces)}
    need(len({vote["activation"]["activation_SHA256"] for vote,_ in evidence})==1
         and len({vote["activation"]["PID"] for vote,_ in evidence})==8
         and all(wait["votes"] == votes for _,wait in evidence),"R46_CROSS_RANK_SAME_ACTIVATION")
    return audit_ranks(results,initial,index)


def gpu_terminal(rows):
    scope,steps,physical,physical_error = independently_verify(rows)
    native = None
    counts = None
    reload = None
    native_error = None
    try:
        if scope["native_outcome"] is not None:
            native,_ = stable_json(OPERATION/"native-outcome.json",scope["native_outcome"])
            need(native["request_id"] == REQUEST and native["epoch"] == scope["epoch"], "R46_EXACT_NATIVE_SCOPE")
            if native["native_two_updates_witnessed"] and "gpu" in steps:
                counts = audit_training(native,steps["gpu"]["id"])
                reload=adapter_reload(OPERATION/"continuous/train/r39-final-adapter",counts["adapter_SHA256"])
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        native_error = type(exc).__name__
    cleanup=bool(native and native["owner_cleanup_passed"])
    engineering=bool(counts and reload and cleanup and physical and steps["gpu"]["state"]=="COMPLETED"
                     and steps["gpu"]["exit"]=="0:0" and scope["engineering_completed"] is True)
    proof = dict(request=REQUEST, independent=True, steps=steps, independent_physical_release=physical,
                 physical_error_class=physical_error, native_error_class=native_error,
                 owner_cleanup_passed=cleanup, engineering_completed=engineering, planned_updates=2,
                 planned_rank_receipts=8, planned_input_signatures=64, counts=counts, reload=reload,
                 supervised_progress=native.get("rank_progress") if native else None,
                 base_model_qualified=False, blind_or_heldout_scoring=False, GPU_replay=False)
    from r120_holder import rank_gate_receipts
    proof["first_batch_gate"] = rank_gate_receipts(OPERATION, steps.get("gpu"), REQUEST)
    emit_file(OPERATION/"INDEPENDENT_R120_TWO_UPDATE_TERMINAL.json",encoded(proof))
    return proof


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("native-training",))
    a=p.parse_args()
    values=actual_accounting()
    proof=gpu_terminal(values)
    print(encoded({"request":REQUEST,"independent":True,"proof":pin(encoded(proof))}).decode(),end="")
