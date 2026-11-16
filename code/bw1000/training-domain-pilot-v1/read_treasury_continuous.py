"""Audit a finished original Treasury continuous arm and pin checkpoint 2 for resume."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess


LABEL = "training-qwen-treasury-continuous-hip-v3"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def require(v, message):
    if not v:
        raise ValueError(message)


def bounded_json(path):
    with path.open("rb") as stream:
        raw = stream.read(262145)
    require(0 < len(raw) <= 262144, f"receipt size differs: {path.name}")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def pin(path, limit):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "checkpoint file shape differs")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.lstat()
    key = lambda p: (p.st_dev, p.st_ino, p.st_size, p.st_mtime_ns, p.st_ctime_ns)
    require(key(before) == key(after), "checkpoint changed during pinning")
    return {"bytes": before.st_size, "SHA256": digest}


def audit(root, request, claim):
    require(re.fullmatch(r"[0-9a-f]{32}", request) and re.fullmatch(r"[0-9a-f]{64}", claim),
            "invalid original request or claim identity")
    launch, launch_hash = bounded_json(root / "launch-receipt.json")
    require(launch["request_id"] == request and launch["claim_SHA256"] == claim
            and launch["node"] == "a01r4n16" and launch["parent"] == "1087656", "wrong continuous request")
    raw = subprocess.check_output(["sacct", "-n", "-P", "-j", launch["parent"], "-o", COLUMNS],
                                  text=True, timeout=14)
    rows = [line.split("|") for line in raw.splitlines() if "|" + LABEL + "-" in line]
    names = [LABEL + "-" + key for key in ("pre", "gpu", "post")]
    require(len(rows) == 3 and {r[1] for r in rows} == set(names), "original step inventory incomplete")
    steps = {r[1].rsplit("-", 1)[-1]: r for r in rows}
    require(all(r[2:4] == ["COMPLETED", "0:0"] and r[5] == launch["node"] for r in rows),
            "original steps lack completed success accounting")
    require("gres/dcu=8" in steps["gpu"][6], "original trained step did not hold eight DCUs")
    result, outcome_hash = bounded_json(root / "controller-outcome.json")
    require(result["passed"] is True and all(result["stages"][key]["row"] == row for key, row in steps.items()),
            "controller disagrees with terminal accounting")
    before, _ = bounded_json(root / "before.json")
    assigned, assigned_hash = bounded_json(root / "continuous/assigned.json")
    post, post_hash = bounded_json(root / "after.json")
    require(before["step"] == steps["pre"][0] and assigned["step"] == steps["gpu"][0]
            and post["step"] == steps["post"][0] and before["boot_id"] == assigned["boot_id"] == post["boot_id"]
            and len(set(before["pci"])) == 8 and sorted(assigned["visible_pci"]) == before["pci"] == post["pci"]
            and before["kfd_holder_count"] == post["kfd_holder_count"] == 0 and post["owned_remaining"] == 0,
            "original GPU identities or post-run cleanup disagree")
    arm = root / "continuous"
    witnesses = []
    for rank in range(8):
        path = arm / "train/treasury-witness" / f"rank-{rank:02d}"
        witness, sha = bounded_json(path / "result.json")
        require(witness["rank"] == rank and witness["completed_global_step"] == 5
                and witness["saved_checkpoints"] == [2, 4, 5]
                and len(witness["inputs"]) == len(witness["outputs"]) == 20
                and [s["global_step"] for s in witness["steps"]] == [1, 2, 3, 4, 5],
                f"continuous witness differs for rank {rank}")
        rng, rng_hash = bounded_json(path / "checkpoint-2-rng.json")
        require(rng["rank"] == rank and rng.get("RNG"), f"missing checkpoint RNG for rank {rank}")
        witnesses.append({"rank": rank, "result_SHA256": sha, "checkpoint_2_rng_witness_SHA256": rng_hash})
    ckpt = arm / "train/checkpoint-2"
    state, state_hash = bounded_json(ckpt / "trainer_state.json")
    require(state["global_step"] == 2 and state["max_steps"] == 5 and abs(state["epoch"] - .4) < 1e-10,
            "checkpoint trainer cursor differs")
    names = {"args.json", "adapter_config.json", "adapter_model.safetensors",
             "trainer_state.json", "optimizer.pt", "scheduler.pt", *(f"rng_state_{i}.pth" for i in range(8))}
    pins = {name: pin(ckpt / name, (512 if name == "optimizer.pt" else 256) * 1024**2) for name in sorted(names)}
    require(sum(info["bytes"] for info in pins.values()) <= 768 * 1024**2
            and pins["trainer_state.json"]["SHA256"] == state_hash, "checkpoint exceeds the native restore bounds")
    return dict(state="ORIGINAL_CONTINUOUS_VERIFIED_CHECKPOINT_2_PINNED", original_request=request,
                accounting_columns=COLUMNS.split(","), accounting=steps, launch_SHA256=launch_hash,
                controller_outcome_SHA256=outcome_hash, assigned_SHA256=assigned_hash, post_SHA256=post_hash,
                actual_optimizer_updates=5, owner_ranks=witnesses, checkpoint_files=pins,
                checkpoint_global_step=2, checkpoint_epoch=state["epoch"], resume_executed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("request")
    p.add_argument("claim")
    p.add_argument("output", type=Path)
    args = p.parse_args()
    result = audit(args.root, args.request, args.claim)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
