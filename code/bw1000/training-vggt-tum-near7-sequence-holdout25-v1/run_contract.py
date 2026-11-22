"""Input, deadline and candidate evidence checks used by the TUM controller."""

import hashlib
import json
import math
from pathlib import Path
import re


LABEL = "training-vggt-tum-directed-translation-v1"
ORIGINAL_CLAIM_SHA256 = "3c4e3ffb19e73628c359a1c8c3d073e4b6ea8adb619a06692b84813e232b8c4f"
BACKEND_SHA256 = "f53220165a356fec024505b65af2ac7f12b1c9272f6c231df22634c060f31ea5"
LOSS_SHA256 = "b08d278311cad3ffc44fed8985e68d3fe969735fdadaf2c16dba03eedccf64ad"
LOSS_PROVIDERS = {
    "pose_enc.py": "845fc3a46cb851bebf0720b8fc2b9f34334ae8e11f1982376ba59f90ff18f22b",
    "rotation.py": "ad3a904e243c6654f0aa5f1d0206ace5f2d61011bc1bf8f492b40e0b76cb2070",
}
ARMS = {"continuous": 0, "resume10": 10, "resume20": 20}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def stage_original_data(claim, original, target):
    """Recheck and copy only the known 17 files from the sealed training scope."""
    raw = claim.read_bytes()
    if sha(raw) != ORIGINAL_CLAIM_SHA256 or original.is_symlink():
        raise ValueError("Original TUM data inventory changed")
    members = json.loads(raw)["data_files"]
    if len(members) != 17 or sum(item["bytes"] for item in members.values()) > 4 * 1024**2:
        raise ValueError("Unexpected original TUM data scope")
    data = {}
    for name, item in members.items():
        path = original / name
        if path.is_symlink() or not path.resolve().is_relative_to(original.resolve()):
            raise ValueError("Original training member escapes the frozen root")
        raw = path.read_bytes()
        if len(raw) != item["bytes"] or sha(raw) != item["sha"]:
            raise ValueError("Original TUM training data differs: " + name)
        data[name] = raw
    target.mkdir()
    for name, raw in data.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(raw)
    return len(data)


def step_budget(phase, now, deadline):
    """Bound the Slurm time limit and reserve time for accounting and post-check."""
    caps = {"pre": 45, "gpu": 1440, "post": 45}
    if phase not in caps or any(not math.isfinite(x) for x in (now, deadline)):
        raise ValueError("Unknown phase or invalid Slurm phase clock")
    remaining = deadline - now
    # Native children validate their own absolute deadline within this envelope.
    limit = min(caps[phase], math.floor(remaining) - (80 if phase == "gpu" else 30))
    if limit <= 5:
        raise ValueError("Original epoch leaves insufficient Slurm phase budget")
    hours, rest = divmod(limit, 3600)
    minutes, seconds = divmod(rest, 60)
    return limit, f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def allocation_matches(row, native):
    """Check an exact Slurm step TRES rather than a substring of the card count."""
    data = {}
    for field in row.split(","):
        key, sep, value = field.partition("=")
        if not sep or not key or not value or key in data:
            return False
        data[key] = value
    accelerators = {key: value for key, value in data.items() if key.startswith("gres/")}
    return data.get("node") == "1" and (
        accelerators == {"gres/dcu": "1"} if native else not accelerators)


def parent_matches(row, members, uid, expected_job, expected_node):
    owner = re.fullmatch(r"[^()]+\(([0-9]+)\)", row.get("UserId", ""))
    accelerators = [item for item in row.get("AllocTRES", "").split(",") if item.startswith("gres/")]
    return (row.get("JobId") == expected_job and row.get("JobState") == "RUNNING"
            and row.get("NumNodes") == "40" and accelerators == ["gres/dcu=320"] and
            len(members) == len(set(members)) == 40 and expected_node in members and
            owner is not None and int(owner[1]) == uid)


def check_directional_receipts(root, receipts):
    """Join the actual auxiliary loss to every native step in all three arms."""
    expected = dict(source_SHA256=LOSS_SHA256, weight=0.1, minimum_target_baseline=0.005,
                    predicted_norm_floor=0.001, pose_source_SHA256=LOSS_PROVIDERS)
    if set(receipts) != set(ARMS):
        raise ValueError("Missing directional training arms")
    costs = {}
    for arm, start in ARMS.items():
        receipt = receipts[arm]["value"]
        raw = (Path(root) / "arms" / arm / "native-result.json").read_bytes()
        if sha(raw) != receipt["result_SHA256"] or receipt["backend_SHA256"] != BACKEND_SHA256:
            raise ValueError("Directional result does not bind its original worker receipt")
        if json.loads(raw)["metrics"]["relative_translation_loss"] != expected:
            raise ValueError("Directional loss source, providers or constants differ")
        rows = receipt["steps"]
        if [row["global_optimizer_step"] for row in rows] != list(range(start + 1, 26)):
            raise ValueError("Incomplete directional optimizer denominator")
        values = [row.get("relative_translation_direction_cost") for row in rows]
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 2
               for value in values):
            raise ValueError("Missing or invalid directional cost")
        costs[arm] = values
    if max(costs["continuous"]) <= 0:
        raise ValueError("Directional term stayed zero across all 25 updates")
    differences = {
        arm: max(abs(a - b) for a, b in zip(costs["continuous"][start:], costs[arm], strict=True))
        for arm, start in ARMS.items() if start
    }
    return dict(comparison_passed=all(value <= 1e-6 for value in differences.values()),
                maximum_resume_direction_cost_errors=differences, loss=expected,
                steps_by_arm={arm: len(values) for arm, values in costs.items()},
                continuous_cost_min=min(costs["continuous"]), continuous_cost_max=max(costs["continuous"]))
