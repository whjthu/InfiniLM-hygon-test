"""Compare a resumed native trajectory with the original uninterrupted trajectory."""

import json
import math
from pathlib import Path

from scope import REQUEST, RESULTS, need, pin


ORIGINAL = RESULTS / "training-internvl-gqa64-continuous25-hip-v1"
STEPS = list(range(21, 26))


def checkpoint_losses(folder):
    state = json.loads(pin(folder / "trainer_state.json", cap=1 << 20))
    need(state["global_step"] == state["max_steps"] == 25, "Step-25 trainer state required")
    series = [(row["step"], row["loss"], row["learning_rate"]) for row in state["log_history"]
              if row.get("step") in STEPS and "loss" in row and "learning_rate" in row]
    need([item[0] for item in series] == STEPS and
         all(math.isfinite(loss) and math.isfinite(lr) for _, loss, lr in series),
         "Missing or non-finite original step-21 to step-25 loss/LR series")
    return series


def gather(run):
    reports = []
    originals = set()
    for rank in range(8):
        path = Path(f"capture/rank-{rank:02d}")
        prior = json.loads(pin(ORIGINAL / path / "run25.json"))
        row = json.loads(pin(run / path / "resume25.json"))
        entry = json.loads(pin(run / path / "restore.json"))
        need(prior["rank"] == row["rank"] == entry["rank"] == rank
             and row["request"] == entry["request"] == REQUEST and prior["global_step"] == 25
             and entry["global_step"] == entry["scheduler_epoch"] == 20
             and entry["optimizer_moments_match"] == 504 and row["rng_restore_calls"] == 1
             and entry["starting_trainables_SHA256"] == row["cp20_trainables_SHA256"]
             and row["checkpoint_steps"] == [25], "Native resume cursor evidence differs")
        steps = row["completed_steps"]
        need([item["step"] for item in steps] == STEPS and
             all(len(item["roster_ordinals"]) == 4 for item in steps), "Missing five native updates or input slots")
        originals.add(prior["final_parameters"])
        reports.append(row)
    before = {r["cp20_trainables_SHA256"] for r in reports}
    after = {r["cp25_trainables_SHA256"] for r in reports}
    need(len(originals) == len(before) == len(after) == 1, "Replicas do not agree on trainable parameters")
    per_step_lr = {tuple(tuple(row["learning_rates"]) for row in r["completed_steps"]) for r in reports}
    need(len(per_step_lr) == 1, "Rank learning rate trajectories differ")
    first = checkpoint_losses(ORIGINAL / "train/checkpoint-25")
    second = checkpoint_losses(run / "train/checkpoint-25")
    params = originals == after
    losses = first == second
    return {"ranks": 8, "native_updates_per_rank": 5, "paired_native_input_count": 0,
            "native_input_count_after_restore": 160, "original_sample_order_directly_observed": False,
            "cp20_restored_trainable_SHA256": before.pop(),
            "uninterrupted_cp25_trainable_SHA256": originals.pop(),
            "resumed_cp25_trainable_SHA256": after.pop(), "exact_final_trainable_match": params,
            "uninterrupted_global_steps_loss_and_lr": first, "resumed_global_steps_loss_and_lr": second,
            "exact_logged_loss_and_lr_match": losses, "rank_trajectories": reports,
            "native_resume_executed": True, "exact_resume_reproduced": params and losses}
