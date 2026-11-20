"""Run two complete native arms sequentially on the same Slurm lease."""
import json
import os
from pathlib import Path
import shutil
import time
import subprocess
import sys

from scope import RUN_ROOT, NATIVE_SECONDS, EpochClock
from snapshot import snapshot


def save(path, value):
    with path.open("x") as destination:
        json.dump(value, destination, allow_nan=False)


def main():
    root = RUN_ROOT
    intent = json.loads((root/"intent.json").read_bytes())
    before = json.loads((root/"before.json").read_bytes())
    clock = EpochClock(intent["epoch"])
    plan = ("stock", "paired")
    arms = {}
    owners = {}
    save(root/"native-intent.json", dict(arms=plan, planned_updates_per_arm=25,
                                         ranks=8, epoch=intent["epoch"]))
    failure = None
    started = time.monotonic()
    for label in plan:
        arm = root/label
        arm.mkdir(mode=0o700)
        shutil.copyfile(root/"intent.json", arm/"intent.json")
        result = None
        try:
            gate = before if label == "stock" else snapshot(True)
            if gate["kfd_holder_count"] or gate["boot_id"] != before["boot_id"] or gate["pci"] != before["pci"]:
                raise RuntimeError("BETWEEN_ARM_PHYSICAL_DEVICE_RECHECK")
            save(arm/"before.json", gate)
            env = {**os.environ, "INFINILM_QWEN_NATIVE_ARM": label}
            with (arm/"arm-supervisor.log").open("x") as log:
                result = subprocess.run([sys.executable, "-B", str(root/"src/native.py")],
                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=clock.left(NATIVE_SECONDS)+25)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            failure = dict(arm=label, type=type(error).__name__)
        outcome = arm/"native-outcome.json"
        value = json.loads(outcome.read_bytes()) if outcome.is_file() else {}
        registry = arm/"owned-registry.json"
        if not value and registry.is_file():
            owners.update(json.loads(registry.read_bytes()))
        owners.update(value.get("owners", {}))
        arms[label] = dict(returncode=result.returncode if result else None, native=value)
        if failure is not None or not value.get("normal_native_exit"):
            break
    good = len(arms) == len(plan) and all(x["native"].get("normal_native_exit") and x["returncode"]==0
                                         for x in arms.values())
    save(root/"native-outcome.json", dict(arms=arms, owners=owners, planned_arms=plan,
        independent_numerical_qualification=False, failure=failure,
        duration_seconds=time.monotonic()-started, normal_native_exit=good and failure is None))
    return 0 if good and failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
