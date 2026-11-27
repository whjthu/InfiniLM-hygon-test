"""Verify all durable native update receipts and checkpoint inventories after child exit."""
import json
import math

import series_contract as contract
from scope import REQUEST, need, pin, write_once


def collect(run,plans,clock):
    progress=[]
    complete_ranks=[]
    errors=[]
    for rank,plan in enumerate(plans):
        path=run/"capture"/f"rank-{rank:02d}"
        partial=dict(rank=rank,forward_slots=0,optimizer_steps=[],save_events=[])
        try:
            for step in contract.STEPS:
                clock()
                for micro in range(1,5):
                    witness=path/f"forward-{step}-{micro}.json"
                    if witness.is_file():
                        need(json.loads(pin(witness,cap=1024))==dict(
                            request=REQUEST,rank=rank,step=step,accumulation=micro),"EXACT_NATIVE_FORWARD_WITNESS")
                        partial["forward_slots"]+=1
                update=path/f"update-{step}.json"
                if update.is_file():
                    item=json.loads(pin(update,cap=8192))
                    start=(step-4)*4
                    need((item["request"],item["rank"],item["step"])==(REQUEST,rank,step)
                         and item["slots"]==plan[start:start+4] and len(item["losses"])==4
                         and all(math.isfinite(loss) for loss in item["losses"])
                         and item["rates"] and all(lr==contract.lr(step-1) and lr>0 for lr in item["rates"]),
                         "EXACT_NATIVE_STEP_AND_704_PLANNED_POSITIONS")
                    partial["optimizer_steps"].append(step)
            hashes={}
            for step in contract.SAVE_STEPS:
                entry=path/f"save-{step}.json"
                if entry.is_file():
                    item=json.loads(pin(entry,cap=1024))
                    need((item["request"],item["rank"],item["step"])==(REQUEST,rank,step),
                         "EXACT_NATIVE_SAVE_WITNESS")
                    hashes[str(step)]=item["parameter_SHA256"]
                    partial["save_events"].append(step)
            terminal=path/"resume25.json"
            if terminal.is_file():
                item=json.loads(pin(terminal,cap=4096))
                need((item["request"],item["rank"],item["final_global_step"],item["updates"],
                      item["forward_slots"],item["native_skip_batches"],item["device_rng_restore_calls"])==
                      (REQUEST,rank,25,22,88,4,1) and item["checkpoint_steps"]==list(contract.SAVE_STEPS)
                      and item["checkpoints"]==hashes and partial["forward_slots"]==88
                      and partial["optimizer_steps"]==list(contract.STEPS)
                      and partial["save_events"]==list(contract.SAVE_STEPS),"EXACT_TERMINAL_RANK_LINEAGE")
                complete_ranks.append(item)
        except (OSError,ValueError,TypeError,KeyError) as error:
            errors.append(dict(rank=rank,type=type(error).__name__))
        progress.append(partial)
    return complete_ranks,progress,errors


def seal(run,reports,seal_checkpoint,clock):
    need(len(reports)==8 and len({row["optimizer_index_to_name_SHA256"] for row in reports})==1 and
         all(len({row["checkpoints"][str(step)] for row in reports})==1 for step in contract.SAVE_STEPS),
         "EIGHT_RANK_PARAMETERS_AND_MAPPING_AGREE")
    inventories={}
    for step in contract.SAVE_STEPS:
        members=seal_checkpoint(run/"train"/f"checkpoint-{step}",step,clock)
        inventories[str(step)]=write_once(run/f"checkpoint-{step}-files.json",members)
    return inventories
