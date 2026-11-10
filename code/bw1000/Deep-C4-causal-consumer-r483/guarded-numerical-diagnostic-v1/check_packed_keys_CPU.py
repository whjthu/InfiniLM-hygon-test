"""Verify the pending C4 key check through the existing cache capture."""
import json
import struct

from cache_audit import CACHE_BYTES, CacheCapture
from causal import schedule
from numeric import SAMPLE_ROWS

raw_cache = bytes(CACHE_BYTES)
cases = [step for step in schedule() if step["action"] == "prepare" and
         tuple(step[key] for key in ("label", "past", "tokens", "variant")) in SAMPLE_ROWS]


def capture(step, key_byte=None, scale=None):
    count = (step["past"] + step["tokens"]) // 4 - step["past"] // 4
    values = bytearray(bytes([0x7e, 0xfe, 0x00, 0x80]) * (count * 32))
    scales = bytearray(struct.pack("<f", 1.0) * count)
    if key_byte is not None:
        values[-1] = key_byte
    if scale is not None:
        scales[-4:] = struct.pack("<f", scale)
    calls = []

    def copy_tensor(_, name):
        calls.append(name)
        shape, dtype, payload = {
            "raw_cache": ([128, 8448], 4, raw_cache),
            "new_key_values": ([count, 128], 4, bytes(values)),
            "new_key_scales": ([count, 1], 10, bytes(scales)),
        }[name]
        return {"shape": shape, "dtype": dtype, "bytes": len(payload)}, payload

    trial = CacheCapture(None, step, copy_tensor, lambda _: None)
    return trial, calls, count


multirow = [step for step in cases if step["tokens"] > 1]
for step in multirow:
    trial, calls, count = capture(step)
    trial.pending()
    assert len(trial.packed[0]) == count * 128 and len(trial.packed[1]) == count * 4
    assert calls == ["raw_cache", "raw_cache", "new_key_values", "new_key_scales"]

step = multirow[0]
rejected = 0
for key_byte, scale in ((0x7f, None), (0xff, None), (None, float("nan")),
                        (None, float("inf")), (None, float("-inf")),
                        (None, 0.0), (None, -0.0), (None, -1.0)):
    trial, calls, count = capture(step, key_byte, scale)
    try:
        trial.pending()
    except ValueError as error:
        last_slot = step["past"] // 4 + count - 1
        assert f"slot {last_slot}" in str(error)
        assert trial.packed is None
        assert calls == ["raw_cache", "raw_cache", "new_key_values", "new_key_scales"]
        rejected += 1
    else:
        raise AssertionError("invalid pending key encoding passed")

coverage = sum((step["past"] + step["tokens"]) // 4 - step["past"] // 4 for step in cases)
assert len(cases) == 6 and coverage == 385
# Run only the new success-path integration. Existing negative cases are separate.
from unittest.mock import patch
import checkpoint
from check_diagnostic_CPU import GuardRecording, binding, metadata, pins, run_diagnostic

events = []
with patch.object(checkpoint, "headers", return_value=None), patch.object(
    checkpoint, "read_weight", side_effect=lambda _, item: bytes(item["bytes"])
):
    report = run_diagnostic(GuardRecording(), metadata, pins, None, binding, lambda _: None,
                            lambda *args: None, events.append)["diagnostic"]
rows = [entry["report"]["finite_packed_key_rows"] for entry in report["cache_audit"]]
assert rows == [128, 128, 0, 0, 1, 128]
assert sum(event["event"] == "step_complete" for event in events) == 71
print(json.dumps(dict(covered_schedule_steps=len(cases), recorded_pending_keys=sum(rows),
                      packed_rows_per_sample=rows, valid_multirow_fixtures=len(multirow),
                      invalid_last_rows_rejected=rejected, completed_schedule_steps=71,
                      added_device_copy_bytes=0, actual_device=False)))
