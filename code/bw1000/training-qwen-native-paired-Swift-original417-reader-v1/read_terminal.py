"""Read and compare the actual completed Qwen Swift checkpoint-policy pair."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import subprocess
import sys

PARENT = "1103606"
LABEL = "training-qwen-native-paired-layer-Swift-AB-v3"
COMMIT = "8da1d1d24fd109c9e7269b240cf0a2bd7b9da9f8"
OUTCOME_SHA256 = "23060a92e7049adb814853ae5e8eb4893572af80a1663479b67be931cce44196"
PROVIDER_SHA256 = "40e50bc545325c01bc2a50c324ba161dc1d29aef04d09c58b73f859208f76013"
CHECKPOINTS = (10, 20, 25)
BOUNDARY = "before update 6 through completed checkpoint 25; includes checkpoint 10/20/25 I/O"
HEX = re.compile(r"[0-9a-f]{64}")
CAPTURE_NAMES = {
    "initial.pt",
    *(f"forward-{i:02d}-{kind}.pt" for i in range(1, 9) for kind in ("input", "output")),
    *(f"update-{i:02d}-{kind}.pt" for i in (1, 2)
      for kind in ("preclip", "postclip", "before-optimizer", "after-optimizer")),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def regular(path, cap):
    st = path.lstat()
    require(stat.S_ISREG(st.st_mode) and 0 < st.st_size <= cap, "FILE_BOUNDARY")
    return st


def digest(data):
    return hashlib.sha256(data).hexdigest()


class TerminalReader:
    def __init__(self, root):
        self.root = root
        self.metadata = {}

    def get(self, name):
        path = self.root / name
        st = regular(path, 2 * 1024**2)
        raw = path.read_bytes()
        require(len(raw) == st.st_size and identity(st) == identity(path.lstat()), "JSON_CHANGED")
        self.metadata[name] = digest(raw)
        return json.loads(raw)

    def hash_adapter(self, name):
        path = self.root / name
        before = regular(path, 180 * 1024**2)
        with path.open("rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        require(identity(before) == identity(path.lstat()), "ADAPTER_CHANGED")
        return dict(bytes=before.st_size, sha256=sha)

    def checkpoint(self, arm, step):
        rel = f"{arm}/train/checkpoint-{step}"
        trainer = self.get(f"{rel}/trainer_state.json")
        require(trainer["global_step"] == step and trainer["max_steps"] == 25, "CHECKPOINT_STEP")
        scalars = {row["step"]: {k: row[k] for k in ("loss", "grad_norm", "learning_rate")}
                   for row in trainer["log_history"] if "loss" in row}
        require(sorted(scalars) == list(range(1, step + 1)), "FULL_TRAINER_LOSS_HISTORY")
        require(all(type(x) in (float, int) and math.isfinite(x)
                    for values in scalars.values() for x in values.values()), "NONFINITE_LOSS_HISTORY")
        files = {name: regular(self.root / rel / name, 700 * 1024**2).st_size
                 for name in ("optimizer.pt", "scheduler.pt", *(f"rng_state_{i}.pth" for i in range(8)))}
        return dict(global_step=step, adapter=self.hash_adapter(f"{rel}/adapter_model.safetensors"),
                    state_bytes=files, scalars=scalars)

    def rank(self, arm, rank, epoch, binding_sha):
        p = f"{arm}/capture/rank-{rank:02d}"
        c = self.get(f"{p}/receipt.json")
        require((c["rank"], c["world_size"], c["completed_global_step"], c["optimizer_updates"],
                 c["micro_forwards"], c["clip_calls"]) == (rank, 8, 2, 2, 8, 2), "CAPTURE_COUNTER")
        files = {row["name"]: (row["bytes"], row["SHA256"]) for row in c["files"]}
        require(len(c["files"]) == len(files) == 25 and set(files) == CAPTURE_NAMES, "CAPTURE_COVERAGE")
        for name, (length, sha) in files.items():
            require(HEX.fullmatch(sha) and regular(self.root / p / name, 2*1024**3).st_size == length,
                    "CAPTURE_FILE_SIZE_AND_DIGEST_FORMAT")
        o = self.get(f"{p}/owner.json")
        require((o["rank"], o["local_rank"], o["epoch"], o["binding_SHA256"]) ==
                (rank, rank, epoch, binding_sha), "TRAINING_RANK_BINDING")
        require(self.get(f"{p}/native25-complete.json") == {"rank": rank, "global_step": 25}, "NATIVE25")
        require(self.get(f"{p}/frozen-boundaries.json") ==
                {"model.visual": {"tensors": 351, "parameters": 576388336}}, "FROZEN_VISUAL")
        v = self.get(f"{p}/runtime-providers.json")
        require(v["HIP"] == "6.3.26113" and v["CUDA"] is None and
                {pin["SHA256"] for pin in v["files"].values()} == {PROVIDER_SHA256}, "RUNTIME_PROVIDER")
        policy = self.get(f"{arm}/checkpoint-policy/rank-{rank:02d}-final.json")
        active = arm == "paired"
        require(policy == dict(rank=rank, attempts=100, applied=100 if active else 0, enabled=active),
                "NATIVE_POLICY_COVERAGE")
        f = self.get(f"{arm}/train/nvidia-baseline/rank-{rank:02d}.json")
        require((f["rank"], f["world_size"], f["effective_batch"], f["trainable_parameters"],
                 f["global_step"], f["warmup_steps"], f["measured_steps"]) ==
                (rank, 8, 32, 43646976, 25, 5, 20), "TIMED_RECIPE")
        require(tuple(f["checkpoint_steps"]) == CHECKPOINTS and f["timing_boundary"] == BOUNDARY,
                "TIME_BOUNDARY")
        require([row["step"] for row in f["step_records"]] == list(range(1, 26)) and
                all(math.isfinite(row["seconds"]) and row["seconds"] >= 0 for row in f["step_records"]),
                "TIME_DENOMINATOR")
        sec = f["full_measurement_seconds"]
        require(math.isfinite(sec) and sec > 0 and math.isclose(f["updates_per_second"], 20/sec) and
                math.isclose(f["samples_per_second"], 640/sec), "TIME_ARITHMETIC")
        require(f["initial_trainable_sha256"] == o["initial_trainables_SHA256"] and
                all(HEX.fullmatch(f[k]) for k in ("initial_trainable_sha256", "final_trainable_sha256")),
                "NAMED_TRAINABLE_HASH")
        peak = f["peak_allocated_bytes"]
        require(type(peak) is int and 0 < peak <= f["peak_reserved_bytes"], "MEMORY_COUNTER")
        return dict(seconds=sec, peak=peak, files=files, initial=f["initial_trainable_sha256"],
                    final=f["final_trainable_sha256"])

    def read(self):
        accounting = subprocess.check_output(
            ["sacct", "-n", "-P", "-j", PARENT,
             "--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES%160"],
            text=True, timeout=10,
        )
        steps = {r[0]: r for line in accounting.splitlines()
                 if (r := line.split("|"))[0] in ("1103606.416", "1103606.417", "1103606.419")}
        require(len(steps) == 3, "ORIGINAL_ACCOUNTING_MISSING")
        for step, suffix in ((416, "-pre"), (417, ""), (419, "-post")):
            r = steps[f"{PARENT}.{step}"]
            require(r[1:4] == [LABEL + suffix, "COMPLETED", "0:0"] and r[5] == "f13r1n10",
                    "ORIGINAL_ACCOUNTING_NOT_SUCCESS")
            require(("gres/dcu=8" in r[6].split(",")) == (step == 417), "ORIGINAL_DCU_SCOPE")
        out = self.get("outcome.json")
        require(self.metadata["outcome.json"] == OUTCOME_SHA256 and out["claim_commit"] == COMMIT and
                out["keeper_passed"] and out["error"] is None, "ORIGINAL_KEEPER_IDENTITY")
        before, after = out["before"], out["after"]
        require(before["pci"] == after["pci"] and len(set(after["pci"])) == 8 and
                before["boot_id"] == after["boot_id"] and after["step"] == "1103606.419" and
                after["kfd_holder_count"] == 0 and after["owned_processes_remaining"] == [], "POST_CLEAN")
        binding = self.get("overlay-binding-receipt.json")
        self.get("OVERLAY_INPUT_BINDING.json")
        require(binding == dict(binding_SHA256=self.metadata["OVERLAY_INPUT_BINDING.json"],
                                provider_SHA256=PROVIDER_SHA256), "OVERLAY_BINDING")
        ranks = {arm: [self.rank(arm, i, out["epoch"], binding["binding_SHA256"]) for i in range(8)]
                 for arm in ("stock", "paired")}
        checkpoints = {arm: [self.checkpoint(arm, i) for i in CHECKPOINTS] for arm in ranks}
        stock, paired = ranks["stock"], ranks["paired"]
        equal_files = {n: sum(a["files"][n] == b["files"][n] for a, b in zip(stock, paired, strict=True))
                       for n in sorted(CAPTURE_NAMES)}
        require(all(a["initial"] == b["initial"] == stock[0]["initial"]
                    for a, b in zip(stock, paired, strict=True)), "INITIAL_STATE_DIFFERS")
        stock_final, paired_final = (checkpoints[a][-1]["scalars"] for a in ("stock", "paired"))
        loss = {key: max(abs(stock_final[i][key] - paired_final[i][key]) for i in stock_final)
                for key in ("loss", "grad_norm", "learning_rate")}
        summary = {}
        for arm, rows in ranks.items():
            maximum = max(r["seconds"] for r in rows)
            summary[arm] = dict(rank_seconds=[r["seconds"] for r in rows], slowest_rank_seconds=maximum,
                                updates_per_second=20/maximum, peak_allocated_bytes=max(r["peak"] for r in rows),
                                checkpoints=[{k: v for k, v in ck.items() if k != "scalars"} for ck in checkpoints[arm]])
        json_sha = digest(json.dumps(self.metadata, sort_keys=True, separators=(",", ":")).encode())
        return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), accounting=steps,
                    original_outcome_SHA256=OUTCOME_SHA256, metadata_files=len(self.metadata),
                    metadata_index_SHA256=json_sha, timed_scope=BOUNDARY, by_arm=summary,
                    candidate_over_stock_update_throughput=summary["paired"]["updates_per_second"] /
                    summary["stock"]["updates_per_second"], rounded_trainer_max_absolute_difference=loss,
                    matching_final_named_trainable_digests=sum(a["final"] == b["final"] for a, b in zip(stock, paired)),
                    equal_producer_capture_digests_by_artifact=equal_files,
                    capture_payloads_independently_rehashed=False, checkpoint_adapter_payloads_rehashed=6,
                    true_A100_comparison=False, stage3_passed=False, measured_pairs=1, independent_post_clean=True)


if __name__ == "__main__":
    print(json.dumps(TerminalReader(Path(sys.argv[1])).read(), allow_nan=False))
