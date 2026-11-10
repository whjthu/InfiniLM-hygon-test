"""Exercise the guarded diagnostic with a CPU recorder."""

import json
from pathlib import Path
import runpy
import struct
from unittest.mock import patch

import checkpoint
from run_diagnostic import run_diagnostic

parent = Path(__file__).resolve().parents[1]
fixtures = runpy.run_path(str(parent / "numerical-diagnostic-v2/check_diagnostic_CPU.py"))
DiagnosticRecording = fixtures["DiagnosticRecording"]
metadata, pins, binding = (fixtures[name] for name in ("metadata", "pins", "binding"))


class ScoreRecording(DiagnosticRecording):
    def call(self, name, *args):
        if name != 'commit':
            return super().call(name, *args)
        first = self.past // 4
        stop = (self.past + self.pending) // 4
        result = super().call(name, *args)
        for slot in range(first, stop):
            page, row = divmod(slot, 64)
            offset = page * 8448 + row * 128
            self.cache[offset:offset + 128] = self.key(slot)
        return result

    @staticmethod
    def key(slot):
        return bytes([(0x38, 0x40, 0x48, 0x50)[slot % 4]]) + bytes(127)

    def tensor(self, name):
        if name == 'new_key_values':
            first, stop = self.past // 4, (self.past + self.pending) // 4
            data = b''.join(self.key(slot) for slot in range(first, stop))
            return dict(shape=[stop - first, 128], dtype=4, bytes=len(data), contiguous=True), data
        if self.pending != 1 or name not in ('query', 'merged_weights', 'scores', 'indices'):
            return super().tensor(name)
        if name == 'query':
            query = bytearray(8192)
            query[0], query[128], query[256] = 0x38, 0xb8, 0x38
            shape, dtype, data = [1, 64, 128], 4, bytes(query)
        elif name == 'merged_weights':
            shape, dtype = [1, 64], 10
            data = struct.pack('<64f', 2, 7, -1, *([0] * 61))
        else:
            width = (self.past + 1) // 4
            values = [(1, 2, 4, 8)[slot % 4] for slot in range(width)]
            if self.corrupt == 'score' and self.past == 32767:
                values[-1] += 2
            reverse_order = self.corrupt == 'topk' and self.past == 32767
            if reverse_order:
                values[0] = 10
            if name == 'scores':
                shape, dtype, data = [1, width], 10, struct.pack(f'<{width}f', *values)
            else:
                ids = sorted(range(width), key=values.__getitem__, reverse=True)[:512]
                if reverse_order:
                    ids.reverse()
                shape, dtype, data = [1, 512], 3, struct.pack('<512q', *ids)
        return dict(shape=shape, dtype=dtype, bytes=len(data), contiguous=True), data


class GuardRecording(ScoreRecording):
    def __init__(self, fault=None):
        super().__init__()
        self.fault = fault
        self.audits = self.buffers = self.nbytes = self.loads = 0

    def call(self, name, *args):
        result = super().call(name, *args)
        if name == "load":
            self.loads += 1
            self.audits += 1
            self.buffers += 1
            self.nbytes += args[2] + 512
        elif name == "prepare":
            self.audits += 1
            self.buffers += 3
            self.nbytes += args[3] + args[5] + args[7] * 8 + 1536
        elif name == "reset" and self.fault == "reset":
            self.nbytes += 1
        return result

    def read(self, name, *args):
        result = super().read(name, *args)
        if name != "observe":
            return result
        result["ready"] = self.postloaded
        result.update(input_guard_bytes_each_side=256, input_audits=self.audits,
                      input_buffers_checked=self.buffers, input_bytes_checked=self.nbytes,
                      input_audit_seconds=self.audits * 0.125)
        if not self.postloaded:
            if self.fault == "missing":
                del result["input_guard_bytes_each_side"]
            elif self.fault == "stale":
                result["input_audits"] = 1
            elif self.fault == "floor":
                result["driver_free"] = 16 * 1024**3 - 1
        if self.fault == "prepare" and self.past == 512 and self.pending:
            result["input_bytes_checked"] -= 512
        return result


def check():
    outcomes = {}
    with patch.object(checkpoint, "headers", return_value=None), patch.object(
        checkpoint, "read_weight", side_effect=lambda _, item: bytes(item["bytes"])
    ) as read_weight:
        for fault, expected_completed in (
            (None, 71), ("missing", 0), ("stale", 0), ("floor", 0), ("reset", 0), ("prepare", 2),
        ):
            trial = GuardRecording(fault)
            events = []
            reads_before = read_weight.call_count
            try:
                result = run_diagnostic(
                    trial, metadata, pins, None, binding, lambda _: None, lambda *args: None, events.append,
                )["diagnostic"]
            except ValueError:
                assert fault
                if fault in ("missing", "stale", "floor"):
                    assert events == [dict(event="checkpoint_failed", planned_steps=71,
                                           completed_steps=0, exception_type="ValueError", complete=False)]
                    assert trial.loads == 0 and read_weight.call_count == reads_before
                else:
                    assert events[-1]["event"] == "failed" and len(events[0]["steps"]) == 71
            else:
                assert fault is None and events[-1]["event"] == "complete"
                assert not result["performance_eligible"] and not result["GPU_input_redzones_qualified"]
                audit = result["input_audit"]
                final = audit["final"]
                assert [final[k] for k in ("input_audits", "input_buffers_checked", "input_bytes_checked")] == [76, 214, 354317576]
                assert sum("prepare_input_audit" in record for record in result["records"]) == 69
                assert events[-1]["input_audit"] == audit and len(events[0]["steps"]) == 71
            completed = sum(event["event"] == "step_complete" for event in events)
            assert completed == expected_completed
            outcomes[fault or "valid"] = dict(planned=71, completed=completed, recorded_loads=trial.loads)
    print(json.dumps(outcomes, sort_keys=True))


if __name__ == "__main__":
    check()
