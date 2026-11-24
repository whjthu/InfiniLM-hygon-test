"""Attach the mask observer to the pinned native MiniCPM restore callback."""

from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path
import sys
from types import ModuleType

from swift.callbacks import callbacks_map


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    from tap import MaskTap, require
finally:
    sys.path.pop(0)


ARMS = ("u1", "c1", "c2", "u2")
KEY = "minicpm_usgs_resume_witness"
ORIGINAL = {
    "native_observer.py": (3904, "2d875c5b46123482bf69cb3833458a37c3c5dd91ef7b6648ffed92caeeea600c"),
    "probe.py": (10995, "1d340abb7659f30cb625fd21164f47168b16267fccf86f9cb8da73edcc6bfdc4"),
}


@contextmanager
def profile_scope(original_profiler, model, arm, cap, observations):
    require(arm in ARMS and not observations, "unexpected or repeated source observation")
    if arm in ("c1", "c2"):
        tap = MaskTap(model, max_retained_bytes=cap)
        with original_profiler() as profiler:
            with tap:
                yield profiler
        observations.append(tap.seal())
    else:
        with original_profiler() as profiler:
            yield profiler
        observations.append(None)


def bind():
    arm = os.environ["MINICPM_NATIVE_MASK_ARM"]
    cap = int(os.environ["MINICPM_NATIVE_MASK_STORAGE_LIMIT"])
    require(arm in ARMS and 0 < cap <= 512 * 1024**2, "unbound native mask observation")
    require(KEY not in callbacks_map, "callback registration is already occupied")
    original = Path(os.environ["MINICPM_FIRST_FORWARD_ORIGINAL_OBSERVER"]).resolve(strict=True)
    payload = {}
    for name, (size, pin) in ORIGINAL.items():
        path = original / name
        require(path.resolve(strict=True) == path, "aliased original observer")
        with path.open("rb") as stream:
            raw = stream.read(size + 1)
        require(len(raw) == size and sha256(raw).hexdigest() == pin, "original observer source differs")
        payload[name] = raw
    source = original / "native_observer.py"
    observer = ModuleType("minicpm_frozen_native_mask_observer")
    observer.__file__ = str(source)
    exec(compile(payload["native_observer.py"], str(source), "exec"), vars(observer))
    native = callbacks_map[KEY]

    class MaskObserved(native):
        def on_train_begin(self, args, state, control, **kwargs):
            result = super().on_train_begin(args, state, control, **kwargs)
            original_profile = self.observer.profiler
            original_write = self.witness.write
            observations = []
            self.observer.profiler = lambda: profile_scope(original_profile, self.observer.model, arm, cap, observations)

            def record(name, value):
                if name == "first-forward-state.json":
                    require(len(observations) == 1, "native forward observation did not finish")
                    value = dict(value, native_mask_observation={"arm": arm, "storage_cap": cap,
                                                                 "fingerprints": observations[0]})
                return original_write(name, value)

            self.witness.write = record
            return result

    callbacks_map[KEY] = MaskObserved


bind()
