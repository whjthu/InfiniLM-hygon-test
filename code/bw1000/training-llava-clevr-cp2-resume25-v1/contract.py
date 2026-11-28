"""Original LLaVA .4983 CP2 and resumed update evidence contract."""

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat


ARCHIVE_SHA = "825ef990fdcb3033e1031ab89c619bef6a56b497b7db010a2d3026e3def9b196"
CP2_RUN = "training-llava-clevr-first2-c59c90a2356d"
REQUEST = "c59c90a2356d4889a6dd57c759f928ee"
TENSORS = "0b6ba7ff7701c165a114bc5460262dff4c053390312d086034e5019d8ecace93"
INPUTS = "d00032b2f0e5eced239d87b2d3f3164c43f369bc3796b0a1180913f3a8950b8b"
TRAIN = "1c88275d408be915584d2b25a5aac4e3e3b7bf97b47cb380cbff70c564932b05"
EPOCHS = tuple(range(1, 13))
STEPS = tuple(range(3, 26))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(raw):
    return sha256(raw).hexdigest()


def checked(path, want, length=None, cap=2 << 20, retain=True):
    need(path.resolve(strict=True) == path, "CANONICAL_RESUME_INPUT")
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap
             and (length is None or before.st_size == length), "PINNED_REGULAR_RESUME_SIZE")
        h = sha256()
        data, count = [], 0
        while chunk := stream.read(4 << 20):
            h.update(chunk)
            count += len(chunk)
            need(count <= before.st_size, "BOUND_RESUME_READ")
            if retain:
                data.append(chunk)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if os.name == "nt":
            fields = fields[:-1]
        stamp = lambda row: tuple(getattr(row, key) for key in fields)
        need(stamp(before) == stamp(os.fstat(stream.fileno())) == stamp(path.stat())
             and h.hexdigest() == want and count == before.st_size, "EXACT_ORIGINAL_RESUME_SHA")
    return b"".join(data) if retain else None


def archive(path):
    evidence = json.loads(checked(path, ARCHIVE_SHA))
    training = evidence["training"]
    steps = {row[0]: row for row in evidence["actual_accounting"]}
    need(evidence["request"] == REQUEST and evidence["terminal_scientific_pass"] is True
         and all(steps[step][2:4] == ["COMPLETED", "0:0"] for step in ("1103606.4983", "1103606.4984"))
         and (training["global_step"], training["schedule_max_steps"], training["actual_forwards"],
              training["rank_RNG_file_count"], training["checkpoint_file_count"]) == (2, 25, 64, 8, 17)
         and len(training["checkpoint_manifest"]) == 17, "EXACT_ARCHIVED_4983_AND_4984")
    return training["checkpoint_manifest"]


def cursor(state, local_batches=8, accumulation=4):
    need(type(state["global_step"]) is int and type(state["max_steps"]) is int
         and (state["global_step"], state["epoch"], state["max_steps"]) == (2, 1.0, 25)
         and (local_batches, accumulation) == (8, 4), "EXACT_CP2_EPOCH_FROM_ORIGINAL_FILE")
    updates = local_batches // accumulation
    epochs, modulo = divmod(state["global_step"], updates)
    need(epochs == state["epoch"] and modulo == 0, "NO_INTRA_EPOCH_SKIP")
    return dict(global_step=2, epoch=epochs, max_steps=25, update_slots_per_epoch=updates, skip_microbatches=0)


def checkpoint(parent, cp, metadata_only=False):
    inventory = archive(parent)
    need(cp.name == "checkpoint-2" and cp.resolve(strict=True) == cp
         and {p.name for p in cp.iterdir()} == set(inventory), "EXACT_ORIGINAL_17_CP2_FILES")
    if not metadata_only:
        for name, row in inventory.items():
            need(Path(name).name == name, "CANONICAL_CP2_MEMBER")
            checked(cp / name, row["SHA256"], length=row["bytes"], cap=384 << 20, retain=False)
    row = inventory["trainer_state.json"]
    saved = json.loads(checked(cp / "trainer_state.json", row["SHA256"], row["bytes"]))
    return cursor(saved)


def assemble(epoch_ranks, expected):
    """Make a plan only from observed ranks and epochs, with exact per-epoch 64-row coverage."""
    need(len(epoch_ranks) == 8 and len(expected) == 64 and set(expected) == set(range(1, 65)),
         "EXACT_64_SOURCE_AND_EIGHT_RANKS")
    updates = {step: [] for step in STEPS}
    for epoch in EPOCHS:
        seen = set()
        for rank, rows in enumerate(epoch_ranks):
            need(len(rows) == 12 and rows[epoch - 1]["epoch"] == epoch
                 and rows[epoch - 1]["seed"] == 42 + epoch, "NATIVE_EPOCH_AND_SEED")
            slots = rows[epoch - 1]["rows"]
            need(len(slots) == 8, "ACTUAL_EIGHT_MICRO_BATCHES_PER_RANK")
            for offset, item in enumerate(slots):
                physical = item["ordinal"]
                need(type(physical) is int and physical in expected and physical not in seen
                     and item == expected[physical], "EXACT_OBSERVED_PHYSICAL_4878_SOURCE")
                seen.add(physical)
                step = 2 * epoch + offset // 4 + 1
                if step <= 25:
                    updates[step].append(dict(rank=rank, accumulation=offset % 4 + 1, offset=offset,
                                              epoch=epoch, **item))
        need(seen == set(expected), "EXACT_FULL_EPOCH_WITHOUT_REPEAT_OR_MISSING")
    for step, rows in updates.items():
        rows.sort(key=lambda row: (row["accumulation"], row["rank"]))
        need(len(rows) == 32 and [(r["accumulation"], r["rank"]) for r in rows]
             == [(m, rank) for m in range(1, 5) for rank in range(8)], "EXACT_STEP_32_FORWARD_SLOTS")
    need(sum(len(value) for value in updates.values()) == 736, "EXACT_23_UPDATES_736_SLOTS")
    return dict(start=cursor(dict(global_step=2, epoch=1.0, max_steps=25)), updates=[
        dict(global_step=step, rows=rows) for step, rows in updates.items()])


def first_resumed_boundary(trace, planned):
    """A later GPU caller supplies actual observations. CPU evidence grants no update."""
    need(trace["global_step"] == planned["global_step"] == 3
         and trace["actual_epoch"] == 1 and trace["actual_skip_microbatches"] == 0
         and trace["rows"] == planned["rows"] and len(trace["rows"]) == 32
         and len(trace["rank_updates"]) == 8, "ACTUAL_RESUMED_32_FIRST_FORWARDS")
    for rank, record in enumerate(trace["rank_updates"]):
        need(record["rank"] == rank and record["actual_adamw_counters_before"] == [2] * 504
             and record["actual_adamw_counters_after"] == [3] * 504 and record["finite_losses"] == 4
             and math.isfinite(record["learning_rate"]) and record["learning_rate"] > 0
             and math.isfinite(record["parameter_delta_l2"]) and record["parameter_delta_l2"] > 0
             and record["pre_SHA256"] != record["post_SHA256"] and record["restored_device_rng"] is True,
             "ACTUAL_STEP3_FINITE_POSITIVE_CHANGE_WITH_RESTORED_STATE")
    return True


def resumed_summary(traces, plan, checkpoint_steps):
    """Reconcile later actual production observations without assuming continuous equivalence."""
    need(len(traces) == len(plan["updates"]) == 23 and checkpoint_steps == [10, 20, 25],
         "ALL_23_UPDATES_AND_THREE_NATIVE_CHECKPOINTS")
    first_resumed_boundary(traces[0], plan["updates"][0])
    for step, (actual, expected) in enumerate(zip(traces, plan["updates"], strict=True), 3):
        need(actual["global_step"] == expected["global_step"] == step
             and actual["rows"] == expected["rows"] and len(actual["rows"]) == 32
             and actual["actual_epoch"] == (step - 1) // 2
             and (actual["finite_losses"], actual["actual_rank_adamw_steps"]) == (32, 8),
             "EXACT_736_REAL_RESUME_FORWARDS_AND_NATIVE_UPDATES")
    return dict(resume_updates=23, real_forwards=736, first_resumed_step=3,
                scientific_recovery=True, uninterrupted_same_recipe_equivalence=None)
