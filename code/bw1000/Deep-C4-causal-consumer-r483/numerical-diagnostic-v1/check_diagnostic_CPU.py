"""Exercise the full diagnostic with a CPU recorder and synthetic buffers."""

import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import causal
import checkpoint
from run_diagnostic import run_diagnostic


root = Path(__file__).resolve().parents[1]
sys.path.append(str(root / "causal-input-v1"))
from check_causal_CPU import Recording


class DiagnosticRecording(Recording):
    def __init__(self, corrupt=False):
        super().__init__()
        self.corrupt = corrupt
        self.loaded = []
        self.postloaded = False

    def read(self, name, *args):
        if name == "parameters":
            return [
                dict(name=name, shape=entry["shape"], dtype=metadata["dtype_codes"][entry["dtype"]],
                     bytes=entry["bytes"], contiguous=True, loaded=name in self.loaded)
                for name, entry in metadata["weights"].items()
            ]
        return super().read(name, *args)

    def call(self, name, *args):
        if name == "load":
            key, data, size = args
            assert len(data) == size and hashlib.sha256(data).hexdigest() == pins["weights"][key.decode()]["SHA256"]
            self.loaded.append(key.decode())
            return
        if name == "postload":
            assert len(self.loaded) == 7
            self.postloaded = True
            return
        assert self.postloaded
        return super().call(name, *args)

    def tensor(self, name):
        if name not in ("projected_score", "pooled"):
            return super().tensor(name)
        assert self.pending is not None
        width = 512 if name == "projected_score" else 128
        data = bytearray(self.pending * width * 2)
        if self.corrupt and name == "pooled" and self.past == 8496:
            data[0] = 1
        return dict(shape=[self.pending, width], dtype=9, bytes=len(data), contiguous=True), data


metadata = json.loads((root / "checkpoint-input-v2/CHECKPOINT_METADATA.json").read_bytes())
binding = json.loads((root / "causal-input-v1/INPUT_BINDING.json").read_bytes())
pins = {key: metadata[key] for key in ("layer", "original_config_SHA256", "original_index_SHA256")}
pins["header_SHA256"] = {name: entry["header_SHA256"] for name, entry in metadata["headers"].items()}
pins["weights"] = {
    name: dict(bytes=entry["bytes"], SHA256=hashlib.sha256(bytes(entry["bytes"])).hexdigest())
    for name, entry in metadata["weights"].items()
}


def check():
    events = []
    copies = []
    with patch.object(checkpoint, "headers", return_value=None), patch.object(
        checkpoint, "read_weight", side_effect=lambda _, entry: bytes(entry["bytes"])
    ):
        trial = DiagnosticRecording()
        result = run_diagnostic(
            trial, metadata, pins, None, binding, lambda _: None,
            lambda label, name, m, data: copies.append((label, name, m["bytes"])), events.append,
        )
        diagnostic = result["diagnostic"]
        reports = diagnostic["numeric_audit"]
        assert result["checkpoint"]["loaded"] == sorted(metadata["weights"])
        assert diagnostic["counts"] == dict(prepares=69, commits=68, aborts=1, resets=2)
        assert not diagnostic["performance_eligible"] and len(copies) == 6
        assert len(reports) == 6 and len(events) == 79 and events[-1]["complete"]
        assert sum(report["copy_bytes"] for report in reports) == 1969920
        assert sum(report["projection"]["samples"] for report in reports) == 180
        assert sum(report["pooled"]["boundary_rows"] for report in reports) == 385

        for field in ("header_SHA256", "weights"):
            trial = DiagnosticRecording()
            bad_pins = json.loads(json.dumps(pins))
            if field == "weights":
                bad_pins[field][sorted(metadata["weights"])[-1]]["SHA256"] = "0" * 64
            else:
                bad_pins[field] = {}
            try:
                run_diagnostic(trial, metadata, bad_pins, None, binding, lambda _: None,
                               lambda *args: None, lambda _: None)
            except ValueError:
                assert not trial.loaded and not trial.postloaded
            else:
                raise AssertionError("accepted invalid checkpoint evidence")

        events.clear()
        try:
            run_diagnostic(DiagnosticRecording(corrupt=True), metadata, pins, None, binding,
                           lambda _: None, lambda *args: None, events.append)
        except ValueError as error:
            assert "off-boundary position 8496" in str(error)
            assert len(events[0]["steps"]) == 71 and events[-1]["event"] == "failed"
            assert sum(event["event"] == "step_complete" for event in events) == 18
        else:
            raise AssertionError("accepted nonzero pooled output off boundary")

    print("PASS: six audits, 180 projections, 385 boundary rows, 1969920 copied bytes, three rejected cases")


if __name__ == "__main__":
    check()
