"""Bind the original USGS checkpoint and recovery plan to a five-update continuation."""

import hashlib
import json
from pathlib import Path


LABEL = "training-qwen-usgs-resume20-recovery-hip-v3"
ORIGINAL_LABEL = "training-qwen-usgs-exposure25-resume20-hip-v1"
RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
ORIGINAL = RESULTS / ORIGINAL_LABEL
RUN = RESULTS / LABEL
REQUEST = "b0565a6f114209e5e13cf916eb3830e3"
PORT = 29918
WITNESS = "vlm/runtime/resume_witness.py"
CHECKPOINT = "continuous/train/checkpoint-20"
MAX_TOTAL = 512 * 1024**2
ORIGINAL_WITNESS_SHA256 = "4454f738caa07554caa5054234414941e07431416ac8ca3d6b5a3368a14bb02c"
OPTIMIZER = dict(bytes=349601739, SHA256="c45e018140ba43dcec9e2ef5d460a1068914f1b39638ed046206d09108d76f97")
ORIGINAL_PINS = {
    "resumed-plan.json": "73aa45f26c5a656497f017b1a1950286665247827cd40b927914c6828a3b80a2",
    "continuous-checkpoint-20-files.json": "e28ead643b11b4f6a5c5f9851e118839f2c9af0df1b98ef71889cec14a7dfabb",
    "view-receipt.json": "216ae48b1f51d267f257ae9da8e5aa2ab069cfac6a5c7aa6e4412f8aaf8ddaf2",
    "witness-config.json": "cbc1c34d1ba32346a93b40bc92b09c2eb51f0b87146912870d18a31479fda7d4",
    "intent.json": "10ebdb3c507dfac2b203796d2d476f7449db8f57cadd66c2d32a697197c7925e",
    "continuous-checkpoint-25-files.json": "14fa1e4d0d04b61f5c058d5eb47c34f403ac3b538a19c0bb1c87030bbcae841d",
    "after.json": "e447fdfdb923dc2c8a29b89221eb347da3c6ae16e20bdd117e01264e7fd37975",
}
CONTINUOUS_RESULTS = (
    "9a63b865e6355ff851e3497e96f32972d488b967d9dec87a1a57417d3e06c175",
    "661a873017696622ad97fd7842e52137cf0cdd0380cfedaa0121602879513443",
    "c1dcec9b38f6875a48cd2bcda09ec7db9c6ebb5e37327a3a0b733b00ae8b1902",
    "b7af983f7d79e6f44fcd19d04f1b830b9f2af79ad104683913cbecc101a1db6d",
    "5cf28bed8b9683d8af1f597d7f9fca937c9ed2b70d2f7cc1d268f8b842251537",
    "9db23bb46f7e837d4c35db7b0cfad49a30e992427f82e7c5ed046f6093134c1a",
    "ee3572ad8eae3d3490ba5938df9a74a835b493ebff138ccd7eef7d8a92241ac9",
    "b7594fe60494084fc89ff9d7b35b59bf8505a2f00904cd8a26bb2d0d2f0e04f6",
)
CHECKPOINT_NAMES = {
    "trainer_state.json", "args.json", "adapter_model.safetensors",
    "optimizer.pt", "scheduler.pt", "adapter_config.json",
    *(f"rng_state_{rank}.pth" for rank in range(8)),
}


def require(ok, explanation):
    if not ok:
        raise ValueError(explanation)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_inventory(inventory):
    members = inventory["files"]
    require(inventory["arm"] == "continuous" and inventory["global_step"] == 20
            and inventory["member_count"] == len(members) == 14
            and set(members) == CHECKPOINT_NAMES and members["optimizer.pt"] == OPTIMIZER,
            "checkpoint does not identify the original step-20 state")
    for name, pin in members.items():
        maximum = OPTIMIZER["bytes"] if name == "optimizer.pt" else 256 * 1024**2
        sha = pin["SHA256"]
        require(type(pin["bytes"]) is int and 0 < pin["bytes"] <= maximum
                and isinstance(sha, str) and len(sha) == 64
                and all(c in "0123456789abcdef" for c in sha),
                "checkpoint exceeds the frozen member limit")
    total = sum(pin["bytes"] for pin in members.values())
    require(total == inventory["total_bytes"] == 524422373 and total <= MAX_TOTAL,
            "checkpoint exceeds the frozen total")
    return members


def patched_witness(original):
    """Permit the target optimizer pin while preserving every other witness operation."""
    require(digest(original) == ORIGINAL_WITNESS_SHA256, "original shared witness differs")
    anchor = b"        if (type(pin['bytes']) is not int or not 0 < pin['bytes'] <= 256 * 1024**2\n"
    require(original.count(anchor) == 1, "shared checkpoint limit anchor differs")
    candidate = (
        "        if (type(pin['bytes']) is not int or not (0 < pin['bytes'] <= 256 * 1024**2\n"
        f"                or (name == 'optimizer.pt' and pin == {OPTIMIZER!r}))\n"
    ).encode("ascii")
    patched = original.replace(anchor, candidate, 1)
    compile(patched, WITNESS, "exec")
    return patched


def bind_plan(original, source, destination):
    """Replace only the output and installed framework paths in the original plan."""
    replacements = {
        str(source / "framework/vlm/runtime/swift_entry.py"): str(destination / "framework/vlm/runtime/swift_entry.py"),
        str(source / "resumed/train"): str(destination / "resumed/train"),
        str(source / "src/qwen_treasury_witness.py"): str(destination / "src/qwen_treasury_witness.py"),
    }
    plan = json.loads(original)
    require(plan["domain_arm"] == "resumed" and plan["planned_updates"] == 5
            and plan["checkpoint_steps"] == [25], "only the five-update original recovery plan is accepted")
    argv = plan["command"]
    for name, expected in {
        "--resume_from_checkpoint": str(source / CHECKPOINT),
        "--dataset": str(source / "view/training800.jsonl"),
        "--output_dir": str(source / "resumed/train"),
        "--max_steps": "25",
        "--resume_only_model": "false",
        "--ignore_data_skip": "false",
    }.items():
        require(argv.count(name) == 1 and argv[argv.index(name) + 1] == expected,
                f"original resume option differs: {name}")
    require(plan["resume_from_checkpoint"] == str(source / CHECKPOINT)
            and plan["environment_overrides"]["NPROC_PER_NODE"] == "8",
            "original checkpoint or rank topology differs")
    for value in replacements:
        require(argv.count(value) == 1, "original writable or entry path is ambiguous")
    plan["command"] = [replacements.get(value, value) for value in argv]
    plan["required_assets"]["plugins"] = [
        replacements.get(value, value) for value in plan["required_assets"]["plugins"]
    ]
    return plan

def pinned_bytes(path, expected, maximum):
    require(path.is_file() and path.resolve(strict=True) == path, f"aliased input: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum and digest(raw) == expected, f"original identity differs: {path.name}")
    return raw


def original_inputs(source, destination):
    metadata = {
        name: pinned_bytes(source / name, sha, 128 * 1024)
        for name, sha in ORIGINAL_PINS.items()
    }
    checkpoint = verify_inventory(json.loads(metadata["continuous-checkpoint-20-files.json"]))
    for name, pin in checkpoint.items():
        path = source / CHECKPOINT / name
        require(path.is_file() and path.resolve(strict=True) == path and path.stat().st_size == pin["bytes"],
                f"source checkpoint size differs: {name}")
    witness = patched_witness(pinned_bytes(source / "framework" / WITNESS, ORIGINAL_WITNESS_SHA256, 64 * 1024))
    for rank, sha in enumerate(CONTINUOUS_RESULTS):
        item = source / "continuous/train/treasury-witness" / f"rank-{rank:02d}/result.json"
        row = json.loads(pinned_bytes(item, sha, 8 * 1024**2))
        require(row["rank"] == rank and row["world_size"] == 8 and row["completed_global_step"] == 25
                and [step["global_step"] for step in row["steps"]] == list(range(1, 26)),
                "original continuous updates differ")
    plan = bind_plan(metadata["resumed-plan.json"], source, destination)
    view = json.loads(metadata["view-receipt.json"])["view"]
    pinned_bytes(source / "view/training800.jsonl", view["SHA256"], 8 * 1024**2)
    config = json.loads(metadata["witness-config.json"])
    require(config["view_SHA256"] == view["SHA256"] == plan["dataset_SHA256"],
            "original exposure identity differs")
    return plan, config, checkpoint, witness


def verify_continuous_hashes(comparison):
    for rank, sha in enumerate(CONTINUOUS_RESULTS):
        path = f"continuous/train/treasury-witness/rank-{rank:02d}/result.json"
        require(comparison["evidence_SHA256"][path] == sha, "original continuous witness changed during execution")
