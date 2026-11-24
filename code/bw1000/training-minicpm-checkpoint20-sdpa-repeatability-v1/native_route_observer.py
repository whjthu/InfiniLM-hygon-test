"""Attach a scoped SDPA route to the frozen MiniCPM step-20 native observer."""

from hashlib import sha256
from pathlib import Path
import os
import sys
from types import ModuleType

import torch
from swift.callbacks import callbacks_map


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    from route_control import mode, profile_with_route, require
finally:
    sys.path.pop(0)


KEY = "minicpm_usgs_resume_witness"
PINNED = (
    ("native_observer.py", 3904, "2d875c5b46123482bf69cb3833458a37c3c5dd91ef7b6648ffed92caeeea600c"),
    ("probe.py", 10995, "1d340abb7659f30cb625fd21164f47168b16267fccf86f9cb8da73edcc6bfdc4"),
)


def pinned(root, proof):
    name, length, pin = proof
    path = root / name
    require(path.resolve(strict=True) == path, "aliased observer source")
    with path.open("rb") as stream:
        raw = stream.read(length + 1)
    require(len(raw) == length and sha256(raw).hexdigest() == pin, "original observer source differs")
    return path, raw


def bind():
    arm = os.environ["MINICPM_ROUTE_ARM"]
    policy = mode(arm)
    require(KEY not in callbacks_map, "callback already registered before frozen observer")
    root = Path(os.environ["MINICPM_FIRST_FORWARD_ORIGINAL_OBSERVER"]).resolve(strict=True)
    path, raw = pinned(root, PINNED[0])
    pinned(root, PINNED[1])
    module = ModuleType("minicpm_original_first_forward_route_control")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), vars(module))
    native = callbacks_map[KEY]

    class ScopedRoute(native):
        def on_train_begin(self, args, state, control, **kwargs):
            result = super().on_train_begin(args, state, control, **kwargs)
            original_profile = self.observer.profiler
            original_write = self.witness.write
            self.observer.profiler = lambda: profile_with_route(torch, original_profile, arm)

            def record(name, value):
                if name == "first-forward-state.json":
                    process = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
                    owner = {"arm": arm, "declared_policy": policy, "pid": os.getpid(),
                             "startticks": int(process[19]),
                             "step": os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                             "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
                    value = dict(value, route_owner=owner)
                return original_write(name, value)

            self.witness.write = record
            return result

    callbacks_map[KEY] = ScopedRoute


bind()
