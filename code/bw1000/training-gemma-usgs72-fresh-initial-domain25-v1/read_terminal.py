"""Target-local candidate evidence; ROOT independently binds accounting/cleanup.

Only hashes of new checkpoints leave this process. It does not deserialize them.
"""
import argparse
import json
from pathlib import Path
import time

from fresh_contract import clock, cohort, need
from stable_io import pin, read_exact, write_json
from stage import ROOT, TRAINING_REQUEST, verify_package
from worker import CHECKPOINT_FILES


def consume(arm, epoch, deadline):
    clock(epoch, deadline, time.time(), 120)
    def check():
        clock(epoch, deadline, time.time(), 120)
    source = verify_package("READ")
    folder = ROOT / arm / "train"
    snapshots = []
    starts = []
    for rank in range(8):
        check()
        kind = "continuous-witness" if arm == "continuous" else "resume-witness"
        root = folder / kind / ("rank-%02d" % rank)
        result = json.loads(read_exact(root / "result.json", pin(root / "result.json"), check)[0])
        owner = json.loads(read_exact(root / "owner.json", pin(root / "owner.json"), check)[0])
        count = 100 if arm == "continuous" else 20
        start = 0 if arm == "continuous" else 20
        need(result["rank"] == rank and result["world_size"] == 8 and result["completed_global_step"] == 25
             and len(result["inputs"]) == len(result["outputs"]) == count
             and [s["global_step"] for s in result["steps"]] == list(range(start + 1, 26))
             and result["saved_checkpoints"] == ([10, 20, 25] if arm == "continuous" else [25]),
             "rank forwards/updates/checkpoints incomplete")
        need(owner["rank"] == owner["local_rank"] == rank and owner["PID"] > 0 and owner["startticks"] > 0,
             "rank owner metadata incomplete")
        for i, (inp, out) in enumerate(zip(result["inputs"], result["outputs"]), 1):
            need(inp["ordinal"] == out["ordinal"] == i and inp["global_step"] == start + (i - 1) // 4
                 and inp["full_input"]["finite"] is True and out["loss"]["finite"] is True,
                 "input/loss identity or accumulation boundary differs")
        need(any(s["adapter_changed"] for s in result["steps"]), "no actual parameter change")
        snapshots.append(dict(rank=rank, result=pin(root / "result.json"), owner=pin(root / "owner.json"),
                              microbatches=count, updates=25-start))
        if arm == "continuous":
            init = json.loads((root / "initialization.json").read_bytes())
            need(init["mode"] == "TARGET_FRESH_DOMAIN_ONLY" and init["target_created"] is True
                 and init["reference_initial_read"] is False and init["RNG_unchanged"] is True,
                 "fresh target initialization observation missing")
            starts.append(dict(rank=rank, named=init["named"]))
    initial = cohort(starts) if arm == "continuous" else None
    inventories = {}
    for step in ((10, 20, 25) if arm == "continuous" else (25,)):
        p = folder / ("checkpoint-%d" % step)
        need(p.resolve(strict=True) == p, "checkpoint aliases another directory")
        files = {}
        for name in sorted(CHECKPOINT_FILES):
            file = p / name
            need(0 < file.lstat().st_size <= 512 * 1024**2, "native state exceeds file budget")
            expected = pin_stream(file, check)
            files[name] = expected
        need(sum(f["bytes"] for f in files.values()) <= 512 * 1024**2, "checkpoint exceeds total state budget")
        state = json.loads((p / "trainer_state.json").read_bytes())
        args = json.loads((p / "args.json").read_bytes())
        need(state["global_step"] == step and state["max_steps"] == args["max_steps"] == 25
             and args["gradient_accumulation_steps"] == 4 and args["per_device_train_batch_size"] == 1,
             "native checkpoint cursor/accumulation differs")
        inventories[str(step)] = files
    value = dict(state="GEMMA_FRESH_DOMAIN_NATIVE_RECEIPTS_CANDIDATE", request_id=TRAINING_REQUEST,
        arm=arm, passed=True, source_manifest=source, ranks_completed=8, actual_updates=25 if arm == "continuous" else 5,
        microbatches_per_rank=100 if arm == "continuous" else 20, unique_train_QA=72, exposures=800,
        fresh_initial_named_SHA256=initial, ranks=snapshots, checkpoint_files=inventories,
        checkpoint20_files=inventories.get("20"), accounting_independently_bound=False,
        cleanup_independently_bound=False, model_qualified=False, training_ready=False)
    write_json(ROOT / (arm + "-metadata-candidate.json"), value)
    return value


def pin_stream(path, check):
    import hashlib
    import os
    import stat
    from stable_io import identity
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode), "checkpoint member is not a regular file")
    h, total = hashlib.sha256(), 0
    with path.open("rb") as f:
        need(identity(os.fstat(f.fileno())) == identity(before), "checkpoint changed at open")
        while True:
            check()
            raw = f.read(min(1024**2, before.st_size - total + 1))
            if not raw:
                break
            total += len(raw)
            need(total <= before.st_size, "checkpoint grew")
            h.update(raw)
        need(identity(os.fstat(f.fileno())) == identity(before), "checkpoint changed during hash")
    need(total == before.st_size and identity(path.lstat()) == identity(before), "checkpoint unstable")
    return dict(bytes=total, SHA256=h.hexdigest())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm", choices=("continuous", "resumed"))
    p.add_argument("--epoch", type=float, required=True)
    p.add_argument("--deadline", type=float, required=True)
    a = p.parse_args()
    r = consume(a.arm, a.epoch, a.deadline)
    print(json.dumps({k: r[k] for k in ("state", "arm", "passed", "ranks_completed", "actual_updates")}))
