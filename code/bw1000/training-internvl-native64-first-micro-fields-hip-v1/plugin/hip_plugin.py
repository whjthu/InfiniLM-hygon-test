"""Native Swift plug-in for a new first-micro diagnostic scope."""
import importlib.util
import inspect
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fields import FINGERPRINT_SHA, REQUEST, first_reference, read_small, require
from hip_hook import install


CALLBACK = "internvl_first_micro_fields"
V5_CAPTURE_SHA = "30039768510a0a6cc1664edf710ce58848e91c1d89198ea808a613bd589c8058"
V5_SOURCE = {
    "callback.py": "a22d18937cc795e349c48b17e7d08616ba12ec3d11edf7772d85b4a94a0bf7ee",
    "scope.py": "a0ec6e87a1f8ec510b0f4a7f9e3b1db59f8ec8d0471138c6e2316817f3b62349",
}


def load_frozen():
    base = Path(os.environ["T03_FIELD_FROZEN_V5"])
    require(os.environ["T03_FIELD_REQUEST"] == REQUEST and "scope" not in sys.modules,
            "New InternVL diagnostic scope required")
    for name, digest in V5_SOURCE.items():
        read_small(base / name, expected=digest)
    spec = importlib.util.spec_from_file_location("internvl_frozen_v5_callback", base / "callback.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(sys.modules["scope"].__file__) == base / "scope.py", "Frozen v5 scope provider differs")
    capture_module = sys.modules[module.FirstUpdateCapture.__module__]
    audit_source = Path(inspect.getsourcefile(module.fingerprint))
    capture_source = Path(inspect.getsourcefile(capture_module))
    require(audit_source.parent == capture_source.parent, "Native capture module provider differs")
    read_small(audit_source, expected=FINGERPRINT_SHA, size=6939)
    read_small(capture_source, expected=V5_CAPTURE_SHA)
    return module, capture_module.cpu


base, cpu_projection = load_frozen()


class InternVLFirstMicroFields(base.InternVLFirstTwo):
    def on_train_begin(self, args, state, control, **kwargs):
        root = Path(os.environ["T03_FIELD_OUTPUT"])
        require(root.is_absolute() and root.name == REQUEST
                and root.resolve(strict=True) == root and root.is_dir(),
                "Fresh first-micro metadata directory required")
        scratch_capture = Path(os.environ["T03_FIRST2_CAPTURE"])
        require(scratch_capture == root / "native-gates",
                "Native gate records must use only the new diagnostic namespace")
        control = super().on_train_begin(args, state, control, **kwargs)
        rank = self.capture.rank
        reference = first_reference(sys.modules["scope"].REF_INPUTS)
        require(self.capture.input_fingerprints[0] == reference[rank]["input_fingerprint_SHA256"],
                "Native first slot does not match the original aggregate reference")
        model = kwargs.get("model", self.trainer.model)
        install(self.capture, model, base.torch, base.fingerprint, cpu_projection,
                root / f"rank-{rank:02d}.json")
        return control


require(CALLBACK not in base.callbacks_map, "Repeated first-micro diagnostic callback")
base.callbacks_map[CALLBACK] = InternVLFirstMicroFields
