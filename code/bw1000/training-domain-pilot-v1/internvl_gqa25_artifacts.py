"""Validate the full InternVL schedule and the original checkpoint boundary."""
import json
import math
from pathlib import Path
import stat


STEPS = (10, 20, 25)
CHECKPOINT_FILES = (
    "adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
    "optimizer.pt", "scheduler.pt", *(f"rng_state_{rank}.pth" for rank in range(8)),
)
TIMING_BOUNDARY = "before update 6 through completed checkpoint 25; includes checkpoint 10/20/25 I/O"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def json_file(path, cap=256 * 1024):
    path = Path(path)
    need(stat.S_ISREG(path.lstat().st_mode), "Expected an owned regular file")
    with path.open("rb") as stream:
        contents = stream.read(cap + 1)
    need(0 < len(contents) <= cap, "Unexpected evidence file extent")
    return json.loads(contents)


def extent(path):
    st = path.lstat()
    need(stat.S_ISREG(st.st_mode) and st.st_size > 0, "Missing checkpoint payload")
    return st.st_size


def checkpoint_inventory(train):
    inventory = {}
    for step in STEPS:
        folder = train / f"checkpoint-{step}"
        need(stat.S_ISDIR(folder.lstat().st_mode), "Expected an owned checkpoint directory")
        files = {name: extent(folder / name) for name in CHECKPOINT_FILES}
        state = json_file(folder / "trainer_state.json")
        need(state["global_step"] == step and state["max_steps"] == 25, "Checkpoint trainer step differs")
        adapter = json_file(folder / "adapter_config.json")
        need(adapter["r"] == 16 and adapter["lora_alpha"] == 32 and adapter["lora_dropout"] == 0.05,
             "Checkpoint LoRA recipe differs")
        inventory[str(step)] = files
    return inventory


def gather(run, request):
    timing = []
    terminal = []
    for rank in range(8):
        progress = json_file(run / "capture" / f"rank-{rank:02d}" / "run25.json")
        need(progress["rank"] == rank and progress["request"] == request and progress["global_step"] == 25
             and progress["optimizer_callbacks"] == 25 and progress["positive_LR_callbacks"] == 24
             and progress["checkpoint_steps"] == list(STEPS) and progress["final_parameters"] != progress["initial_parameters"],
             "Full 25-step native rank completion differs")
        terminal.append(progress)
        report = json_file(run / "train" / "nvidia-baseline" / f"rank-{rank:02d}.json")
        measurement = report["full_measurement_seconds"]
        need(report["rank"] == rank and report["world_size"] == 8 and report["global_step"] == 25
             and report["warmup_steps"] == 5 and report["measured_steps"] == 20
             and report["effective_batch"] == 32 and report["checkpoint_steps"] == list(STEPS)
             and report["timing_boundary"] == TIMING_BOUNDARY
             and [row["step"] for row in report["step_records"]] == list(range(1, 26))
             and math.isfinite(measurement) and measurement > 0,
             "Original rank measurement boundary differs")
        timing.append(measurement)
    need(len({row["final_parameters"] for row in terminal}) == 1, "Final rank parameters disagree")
    return {"terminal_ranks": terminal, "checkpoint_members_by_step": checkpoint_inventory(run / "train"),
            "rank_measurement_seconds": timing, "slowest_rank_seconds": max(timing),
            "samples_per_second": 640 / max(timing), "timing_boundary": TIMING_BOUNDARY,
            "performance_repetitions": 1, "matches_original_A100_dataset": False,
            "checkpoint_resume_verified": False, "new_process_reload_verified": False}
