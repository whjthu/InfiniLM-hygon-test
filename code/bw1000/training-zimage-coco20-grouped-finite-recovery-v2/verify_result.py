"""Read the exact Z-Image recovery result and accounting without loading Torch."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import subprocess


BASE = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
ORIGINAL = BASE / "training-zimage-coco20-grouped-finite-v1"
OUT = BASE / "training-zimage-coco20-grouped-finite-recovery-v2"
REQUEST = "4fac38ef541d442a883e352fbc13b70b"
STEPNAMES = ("pre", "resume20", "post")
COLUMNS = ("JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES")
FIELDS = ("preclip", "postclip", "LoRA", "step", "exp_avg", "exp_avg_sq")
END_STATE = ("adapter_SHA256", "state_SHA256", "optimizer_SHA256", "diffusion_scheduler_SHA256",
             "rng_SHA256", "integrity_SHA256")


def expect(condition, detail):
    if not condition:
        raise ValueError(detail)


def read(path, maximum=2_000_000):
    expect(path.resolve(strict=True) == path and path.stat().st_size <= maximum, "Evidence path or size differs")
    return path.read_bytes()


def digest_file(path):
    digest = sha256()
    with path.open("rb") as data:
        for block in iter(lambda: data.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    source = read(OUT/"outcome.json")
    outcome = json.loads(source)
    expect(outcome["request"] == REQUEST and outcome["source_step"] == "1103606.1425"
           and outcome["planned_updates"] == list(range(21, 26)), "Original denominator or request differs")
    text = subprocess.check_output(
        ["sacct", "-n", "-P", "-j", "1103606", "--format="+",".join(COLUMNS)], timeout=10, text=True)
    names = {OUT.name+"-"+name: name for name in STEPNAMES}
    accounting = {}
    for line in text.splitlines():
        row = line.split("|")
        expect(len(row) == len(COLUMNS), "Incomplete accounting row")
        if row[1] in names:
            name = names[row[1]]
            expect(name not in accounting and row[5] == "f13r2n07", "Ambiguous phase or node")
            accounting[name] = row
            expect(outcome["phases"][name]["row"] == row, "Controller and Slurm rows differ")
    report = dict(request=REQUEST, source_step=outcome["source_step"], outcome_SHA256=sha256(source).hexdigest(),
                  actual_accounting=accounting, accounting_columns=COLUMNS, controller_passed=outcome["passed"],
                  planned_updates=outcome["planned_updates"], actual_updates=outcome["actual_optimizer_updates"],
                  missing_updates=None, cleanup_verified=False, exact_recovery_verified=False,
                  formal_A100_comparison=False, formal_stage3=False)
    if "error_type" in outcome:
        report["controller_error"] = {"type": outcome["error_type"], "detail": outcome["error"]}
    has_worker = (OUT/"resume20/worker-result.json").is_file()
    if has_worker:
        raw = read(OUT/"resume20/worker-result.json")
        worker = json.loads(raw)
        expect(worker["request"] == REQUEST and worker["step"] == accounting["resume20"][0],
               "Worker belongs to another request or step")
        completed = worker.get("optimizer_returned_steps", [])
        expect(completed == outcome.get("optimizer_returned_steps", []) and len(completed) == worker["actual_optimizer_updates"],
               "Optimizer journals and terminal results disagree")
        report.update(confirmed_optimizer_returns=completed, missing_updates=5-len(completed),
                      worker_SHA256=sha256(raw).hexdigest(), worker_passed=worker["passed"],
                      assigned_PCI=worker.get("physical_PCI"),
                      peak_allocated_bytes=worker.get("peak_allocated_bytes"),
                      peak_reserved_bytes=worker.get("peak_reserved_bytes"))
    if "post" in accounting and outcome.get("post_clean") is True:
        before, after = (json.loads(read(OUT/(name+".json"))) for name in ("pre", "post"))
        report["cleanup_verified"] = (
            all(a["step"] == accounting[name][0] and a["node"] == "f13r2n07"
                and a["kfd_process_count"] == a["own_device_fd_process_count"] == 0
                for name, a in (("pre", before), ("post", after)))
            and len(set(before["PCIs"])) == 8 and before["PCIs"] == after["PCIs"]
            and before["boot_id_SHA256"] == after["boot_id_SHA256"]
            and accounting["post"][2:4] == ["COMPLETED", "0:0"])
    if has_worker and report["missing_updates"] == 0 and worker["passed"]:
        old, new = ([json.loads(row) for row in read(arm/"steps.jsonl").splitlines()]
                    for arm in (ORIGINAL/"continuous", OUT/"resume20"))
        expect([a["global_step"] for a in old] == list(range(1, 26))
               and [a["global_step"] for a in new] == list(range(21, 26)), "Step output coverage differs")
        pairs = list(zip(old[20:], new, strict=True))
        for a, b in pairs:
            expect(a["input_identity"] == b["input_identity"] and a["operands"] == b["operands"]
                   and a["ordinal"] == b["ordinal"] == b["global_step"] - 21,
                   "Wrong data cursor, inputs, or operands")
            expect(set(a["tensor_boundaries"]) == set(b["tensor_boundaries"]) == set(FIELDS)
                   and all(len(a["tensor_boundaries"][k]) == len(b["tensor_boundaries"][k]) == 476 for k in FIELDS),
                   "Tensor boundary denominator differs")
        differences = {field: sum(a["tensor_boundaries"][field] != b["tensor_boundaries"][field] for a, b in pairs)
                       for field in FIELDS}
        losses = [abs(a["loss"]-b["loss"]) for a, b in pairs]
        early, later = (json.loads(read(arm/"checkpoint-receipts.json"))
                        for arm in (ORIGINAL/"continuous", OUT/"resume20"))
        expect([v["global_step"] for v in early] == [10, 20, 25] and [v["global_step"] for v in later] == [25],
               "Checkpoint denominator differs")
        restore = json.loads(read(OUT/"resume20/restore.json"))
        checkpoint = OUT/"resume20/checkpoint-25.private.pt"
        expect(checkpoint.resolve(strict=True) == checkpoint and checkpoint.stat().st_size == later[0]["bytes"]
               and later[0]["bytes"] < 70_000_000 and digest_file(checkpoint) == later[0]["checkpoint_SHA256"],
               "Fresh checkpoint identity differs")
        state = {name: early[-1][name] == later[-1][name] for name in END_STATE}
        report.update(tensor_boundaries_changed_steps=differences, tensors_per_boundary=476,
                      state_matches=state, maximum_loss_difference=max(losses),
                      input_pairs_verified=len(pairs), resumed_checkpoint_SHA256=later[0]["checkpoint_SHA256"])
        report["exact_recovery_verified"] = (
            all(isfinite(v) and v <= 1e-6 for v in losses) and not any(differences.values()) and all(state.values())
            and restore["restored_boundary_exact"] is True and restore["checkpoint_SHA256"] == early[1]["checkpoint_SHA256"]
            and restore["next"]["global_step"] == 21 and restore["next"]["sample_cursor"] == 0
            and worker["physical_PCI"] == "0000:09:00.0"
            and accounting["resume20"][2:4] == ["COMPLETED", "0:0"]
            and "gres/dcu=1" in accounting["resume20"][6].split(","))
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
