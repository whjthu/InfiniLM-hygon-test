"""Exercise the isolated twin guard caller against CPU recording."""
from pathlib import Path
import json
import runpy
import sys
import types
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "guarded-numerical-diagnostic-v1"))

import causal
import checkpoint
import twin_guard as module

payloads = causal.inputs(0, 4)
fixture = runpy.run_path(str(root / "numerical-diagnostic-v2/check_diagnostic_CPU.py"))
Recording = fixture["DiagnosticRecording"]
metadata, pins = fixture["metadata"], fixture["pins"]

class Symbol:
    pass

class Recorder(Recording):
    def __init__(self, fault):
        super().__init__()
        self.test_fault = fault
        self.invocations = 0
        self.complete = False
        self.api = {}
        self.library = types.SimpleNamespace()
        if fault != "missing_export":
            setattr(self.library, module.SYMBOL, Symbol())

    def read(self, name, *args):
        result = super().read(name, *args)
        if name == "observe":
            n = len(self.loaded)
            result.update(ready=self.postloaded, past=4 if self.complete else 0,
                          input_guard_bytes_each_side=256, transaction_guard_completed=self.complete,
                          input_audits=2*n+int(self.complete),
                          input_buffers_checked=2*n+4*int(self.complete),
                          input_bytes_checked=(26295808 if self.postloaded else 0)+43072*int(self.complete),
                          input_audit_seconds=(2*n+int(self.complete))/10)
            if self.complete and self.test_fault == "final_counter":
                result["input_bytes_checked"] -= 1
        return result

    def call(self, name, *args):
        if name != "transaction_guard":
            return super().call(name, *args)
        self.invocations += 1
        assert self.postloaded and not self.complete and len(self.api[name].argtypes) == 5
        assert args[0].raw == payloads[0] and args[2].raw == payloads[1]
        assert (args[1], args[3]) == (32768, 8192)
        self.complete = True

outcomes = {}
with patch.object(checkpoint, "headers", return_value=None), patch.object(
    checkpoint, "read_weight", side_effect=lambda _, item: bytes(item["bytes"])
) as read_weight:
    for fault, expected_phase, expected_loads, invocations in (
        (None, None, 7, 1),
        ("missing_export", "before_load", 0, 0),
        ("late_bad_pin", "checkpoint_load", 0, 0),
        ("final_counter", "result", 7, 1),
    ):
        trial = Recorder(fault)
        events = []
        pinset = json.loads(json.dumps(pins))
        if fault == "late_bad_pin":
            pinset["weights"][sorted(metadata["weights"])[-1]]["SHA256"] = "0"*64
        try:
            result = module.run(trial, metadata, pinset, None, lambda _: None, events.append)
        except (AttributeError, ValueError):
            assert expected_phase and events[-1]["event"] == "failed" and events[-1]["phase"] == expected_phase
        else:
            assert fault is None and events[-1]["event"] == "complete"
            assert result["observed"]["input_bytes_checked"] == 26338880 and not result["performance_eligible"]
        assert tuple(events[0]["cases"]) == module.PLANNED
        assert len(trial.loaded) == expected_loads and trial.invocations == invocations
        outcomes[fault or "valid"] = dict(recorded_loads=expected_loads, diagnostic_calls=invocations)
print(json.dumps(outcomes, sort_keys=True))
