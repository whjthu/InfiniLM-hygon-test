"""Compare both native COCO20 process records without opening tensor bodies."""

import json
from math import isfinite

from boundaries import paired_boundaries


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def compare(root):
    a,b=root/"continuous",root/"resume20"
    left,right=rows(a/"steps.jsonl"),rows(b/"steps.jsonl")
    check([x["global_step"] for x in left]==list(range(1,26)),"Continuous update denominator differs")
    check([x["global_step"] for x in right]==list(range(21,26)),"Resumed update denominator differs")
    for arm in (left,right):
        check(all(x["ordinal"]==(x["global_step"]-1)%20 and isfinite(x["loss"]) for x in arm),
              "Input cursor or step loss differs from the frozen recipe")
    checkpoints,resumed=read(a/"checkpoint-receipts.json"),read(b/"checkpoint-receipts.json")
    check([x["global_step"] for x in checkpoints]==[10,20,25] and len(resumed)==1 and resumed[0]["global_step"]==25,
          "Missing continuous or resumed checkpoints")
    check([x["next_data_cursor"] for x in checkpoints]==[10,0,5] and resumed[0]["next_data_cursor"]==5,
          "Checkpoint next cursor differs")
    restore=read(b/"restore.json")
    check(restore["restored_boundary_exact"] is True and restore["metadata"]["global_step"]==20
          and restore["next"]["global_step"]==21 and restore["next"]["sample_cursor"]==0
          and restore["checkpoint_SHA256"]==checkpoints[1]["checkpoint_SHA256"],
          "Restart did not consume the original step 20 checkpoint")
    boundary = paired_boundaries(left[20:], right)
    c,r=checkpoints[-1],resumed[0]
    matches={name:c[name]==r[name] for name in ("adapter_SHA256","state_SHA256","optimizer_SHA256",
                                                "diffusion_scheduler_SHA256","rng_SHA256","integrity_SHA256")}
    diffs=[abs(x["loss"]-y["loss"]) for x,y in zip(left[20:],right,strict=True)]
    input_match=all(x["operands"]==y["operands"] and x["input_identity"]==y["input_identity"] for x,y in zip(left[20:],right,strict=True))
    return dict(continuous_optimizer_updates=len(left),resumed_optimizer_updates=len(right),
                first_resumed_step=right[0]["global_step"],paired_ordinals=[x["ordinal"] for x in right],
                loss_absolute_differences=diffs,max_loss_absolute_difference=max(diffs),
                paired_input_operands_equal=input_match,state_matches=matches,restored_boundary_exact=True,
                passed=(input_match and max(diffs)<=1e-6 and all(matches.values())
                        and all(step is None for step in boundary["first_difference_by_boundary"].values())),
                tensor_boundary_diagnostic=boundary,
                training_only=True,formal_A100=False,formal_performance=False,heldout_used=False)
