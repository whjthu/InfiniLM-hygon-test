"""Compare completed native 25-update VGGT training with two fresh-process resumes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ARMS = {"continuous":0,"resume20":20}
HASH = re.compile(r"[0-9a-f]{64}")
LOSS_ATOL = 1e-6


def load(path):
    raw = path.read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def compare(inputs):
    if set(inputs) != set(ARMS):
        raise ValueError("Two complete process receipts required")
    processes=[]
    reports={}
    baseline=inputs["continuous"]
    identity_keys=("bundle_SHA256","backend_SHA256","checkpoint_identity","epoch","deadline")
    for arm,first in ARMS.items():
        data=inputs[arm]
        if data["arm"] != arm or data["executed_optimizer_steps"] != 25-first:
            raise ValueError("Arm denominator differs")
        if any(data[key] != baseline[key] for key in identity_keys):
            raise ValueError("Cross-process data, code, epoch or native identity differs")
        process=data["process"]
        token=(process["boot_id"],process["pid"],process["startticks"])
        if token in processes or not all(type(process[k]) is int and process[k]>0 for k in ("pid","startticks")):
            raise ValueError("Expected distinct completed process lifetimes")
        processes.append(token)
        steps=data["steps"]
        if [r["global_optimizer_step"] for r in steps] != list(range(first+1,26)):
            raise ValueError("Missing or duplicate optimizer step")
        if any(r["batch_cursor"] != (r["global_optimizer_step"]-1)%8 for r in steps):
            raise ValueError("Missing canonical eight-pair data cursor")
        if any(not math.isfinite(r[key]) for r in steps
               for key in ("loss","grad_norm","camera_head_grad_norm_before_clip")):
            raise ValueError("Nonfinite native update evidence")
        if data["scheduler"] != dict(type="constant_learning_rate_in_optimizer_param_groups",
                                    mutable_scheduler=False,learning_rate=2e-5) or data["gradient_accumulation"] != 1:
            raise ValueError("LR schedule or accumulation changed")
        expected_checkpoints={str(n) for n in (10,20,25) if n>first}
        if set(data["checkpoints"]) != expected_checkpoints:
            raise ValueError("Checkpoint denominator differs")
        for key,item in data["checkpoints"].items():
            number=int(key)
            if (item["global_step"],item["next_global_step"],item["next_batch_cursor"]) != (number,number+1,number%8):
                raise ValueError("Checkpoint cursor metadata differs")
            for name in ("file_SHA256","model_SHA256","optimizer_SHA256","placements_SHA256","rng_SHA256","full_boundary_SHA256"):
                if not isinstance(item[name],str) or not HASH.fullmatch(item[name]):
                    raise ValueError("Malformed checkpoint state SHA")
        if not first:
            if data["resume_source_SHA256"] is not None:
                raise ValueError("Continuous arm is not a fresh initialization")
            continue
        checkpoint=baseline["checkpoints"][str(first)]
        if data["resume_source_SHA256"] != checkpoint["file_SHA256"]:
            raise ValueError("Resume source is not the original continuous checkpoint")
        reference=baseline["steps"][first:]
        pairs=[dict(global_step=r["global_optimizer_step"],cursor_equal=r["batch_cursor"]==s["batch_cursor"],
                    absolute_loss_error=abs(r["loss"]-s["loss"]),
                    absolute_grad_norm_error=abs(r["grad_norm"]-s["grad_norm"]))
               for r,s in zip(reference,steps,strict=True)]
        comparisons={}
        for number in expected_checkpoints:
            old,new=baseline["checkpoints"][number],data["checkpoints"][number]
            comparisons[number]={key:old[key]==new[key] for key in
                                 ("model_SHA256","optimizer_SHA256","placements_SHA256","rng_SHA256","full_boundary_SHA256")}
        reports[arm]=dict(steps=pairs,checkpoint_component_hashes_equal=comparisons,
                          maximum_loss_error=max(r["absolute_loss_error"] for r in pairs),
                          final_state_equal=data["final_state_SHA256"]==baseline["final_state_SHA256"])
    passed=all(r["maximum_loss_error"]<=LOSS_ATOL and r["final_state_equal"]
               and all(all(flags.values()) for flags in r["checkpoint_component_hashes_equal"].values())
               for r in reports.values())
    return dict(comparison_passed=passed,absolute_loss_tolerance=LOSS_ATOL,arms=reports,
                continuous_updates=25,resume20_updates=5,
                actual_update_invocations=30,planned_training_unique_image_pairs=8,
                formal_CO3D_eligible=False,formal_performance_eligible=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument("--"+arm,type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    loaded={arm:load(getattr(args,arm)) for arm in ARMS}
    result=compare({arm:entry[0] for arm,entry in loaded.items()})
    result["input_receipt_SHA256"]={arm:entry[1] for arm,entry in loaded.items()}
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,sort_keys=True)
        stream.write("\n")
    print(json.dumps(result,sort_keys=True))
    return 0 if result["comparison_passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
