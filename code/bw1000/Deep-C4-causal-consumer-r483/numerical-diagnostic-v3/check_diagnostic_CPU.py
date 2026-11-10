"""Run the v3 diagnostic against a CPU recorder with synthetic operands."""

import json
from pathlib import Path
import runpy
import struct
from unittest.mock import patch

import checkpoint
from run_diagnostic import run_diagnostic

parent_check = Path(__file__).resolve().parents[1] / "numerical-diagnostic-v2/check_diagnostic_CPU.py"
fixtures = runpy.run_path(str(parent_check))
DiagnosticRecording = fixtures["DiagnosticRecording"]
metadata, pins, binding = (fixtures[name] for name in ("metadata", "pins", "binding"))
import checkpoint
import struct
from unittest.mock import patch

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

results = {}
with patch.object(checkpoint, 'headers', return_value=None), patch.object(
    checkpoint, 'read_weight', side_effect=lambda _, entry: bytes(entry['bytes'])
):
    for fault in (None, 'score', 'topk'):
        events = []
        sinks = []
        try:
            result = run_diagnostic(
                ScoreRecording(corrupt=fault), metadata, pins, None, binding, lambda _: None,
                lambda label, name, m, _: sinks.append((label, name, m['bytes'])), events.append,
            )['diagnostic']
        except ValueError as error:
            assert fault == 'topk' and 'TopK scores or descending order differ' in str(error)
            assert len(events[0]['steps']) == 71 and events[-1]['event'] == 'failed'
            completed = sum(event['event'] == 'step_complete' for event in events)
            assert completed == 68
            results['invalid_topk'] = dict(planned=71, completed=completed, error=str(error))
            continue
        assert fault != 'topk' and len(sinks) == 6 and not result['performance_eligible']
        assert result['counts'] == dict(prepares=69, commits=68, aborts=1, resets=2)
        audits = [row['report'] for row in result['cache_audit']]
        reports = [row['score'] for row in audits if row['score'] is not None]
        assert len(audits) == 6 and sum(row['copy_bytes'] for row in audits) == 19590116
        assert [row['visible_slots'] for row in reports] == [2124, 2124, 8192]
        assert sum(len(row['slots']) for row in reports) == 40 and reports[-1]['slots'][-1] == 8191
        assert sum(row['copy_bytes'] for row in reports) == 75104
        assert [row['max_abs_error'] for row in reports] == [0.0, 0.0, 2.0 if fault else 0.0]
        results[fault or 'valid'] = dict(samples=40, extra_copy_bytes=75104, max_abs_error=reports[-1]['max_abs_error'])

print(json.dumps(results, sort_keys=True))
