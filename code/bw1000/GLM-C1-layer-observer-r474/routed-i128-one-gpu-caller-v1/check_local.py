"""Local fixture and orchestration checks without a HIP driver or target objects."""
from contextlib import ExitStack
import ctypes as C
import json
import math
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import caller
import fixture as F
import hip_runtime as H


def decode(bits):
    sign = -1 if bits >> 7 else 1
    exp, frac = (bits >> 3) & 15, bits & 7
    return sign * (math.ldexp(1 + frac / 8, exp - 7) if exp else math.ldexp(frac, -9))


def at(data, index):
    return struct.unpack_from("<f", data, index * 4)[0]


def float_samples(index, role, x, xs, w, ws):
    _, offset, experts = F.CASES[index]
    shard = offset // 128
    down = role == 4
    size = 6144 if down else 128
    result = F.reference(index, "down" if down else "gate")[int(role == 3)]
    for route, row in ((0, 0), (3, size // 2 - 1), (7, size - 1)):
        e = experts[route]
        total = 0.
        for block in range(1 if down else 48):
            x0 = route * 128 if down else block * 128
            w0 = (e * 6144 + row) * 128 if down else (e * 128 + row) * 6144 + block * 128
            dot = sum(decode(x[x0 + k]) * decode(w[w0 + k]) for k in range(128))
            slot = e * 768 + (row // 128) * 16 + shard if down else e * 768 + shard * 48 + block
            total += dot * at(xs, route if down else block) * at(ws, slot)
        assert struct.unpack_from("<H", result, (route * size + row) * 2)[0] == F.bf16(total)


def check_fixtures():
    expected = caller.fixture_binding()
    outputs, source_bytes, samples = 0, 0, 0
    for index, (name, offset, ids) in enumerate(F.CASES):
        spec = expected[name]
        actual = F.inputs(index)
        assert spec["offset"] == offset and spec["routes"] == list(ids)
        for role in (2, 3, 4):
            source = F.weights(index, role)
            (wn, w), (sn, s) = source
            for key, data in ((wn, w), (sn, s)):
                assert spec["inputs"][key] == dict(bytes=len(data), sha256=F.sha(data))
                source_bytes += len(data)
            prefix = "down" if role == 4 else "gate"
            float_samples(index, role, actual[prefix + "_x"], actual[prefix + "_xs"], w, s)
            samples += 3
            del w, data
        for key, data in actual.items():
            assert spec["inputs"][key] == dict(bytes=len(data), sha256=F.sha(data))
            source_bytes += len(data)
        for op in ("gate", "down"):
            v = F.reference(index, op)
            assert [dict(bytes=len(buf), sha256=F.sha(buf)) for buf in v] == spec["references"][op]
            outputs += sum(len(buf) // 2 for buf in v)
            assert F.passing(F.comparisons(v, v))
    good = struct.pack("<4H", 0x0000, 0x3f80, 0x4000, 0x4040)
    bad = struct.pack("<4H", 0x8000, 0x3f80, 0x4080, 0x7f81)
    row, = F.comparisons((bad,), (good,))
    assert row["bit_differences"] == 3 and row["numeric_differences"] == 2
    assert row["nonfinite"] == 1 and row["max_abs_error"] == 2
    assert F.bf16(1 + 1 / 256) == 0x3f80
    return dict(frozen_input_bytes=source_bytes, independent_float_outputs=samples, oracle_values=outputs)


class RecordedHip:
    def __init__(self):
        self.trace = []
        self.memory = {}

    def call(self, op, *args):
        self.trace.append(op)
        if op == "Malloc":
            size = args[1]
            buf = C.create_string_buffer(size)
            addr = C.addressof(buf)
            C.cast(args[0], C.POINTER(C.c_void_p)).contents.value = addr
            self.memory[addr] = buf
        elif op == "Free":
            self.memory.pop(args[0].value)
        elif op == "Memcpy":
            C.memmove(*args[:3])
        elif op == "MemsetAsync":
            C.memset(*args[:3])
        elif op == "EventElapsedTime":
            C.cast(args[0], C.POINTER(C.c_float)).contents.value = 10.


def check_guard():
    hip = RecordedHip()
    with ExitStack() as cleanup:
        region = H.Guarded(hip, cleanup, 7)
        region.upload(b"input07")
        assert region.readback() == b"input07"
        region.input_audit(lambda: None)
        C.memset(region.ptr, 0, 1)
        try:
            region.input_audit(lambda: None)
        except RuntimeError as error:
            assert "payload" in str(error)
        else:
            raise AssertionError("input mutation missed")
        C.memset(region.suffix, 0, 1)
        try:
            region.guards()
        except RuntimeError as error:
            assert "guard" in str(error)
        else:
            raise AssertionError("canary mutation missed")
    assert not hip.memory


def check_timing():
    hip = RecordedHip()
    owner, start, end = (H.V(1), H.V(2), H.V(3))
    plan = tuple(SimpleNamespace(ptr=H.V(i)) for i in range(11, 13))
    inputs = {key: SimpleNamespace(ptr=H.V(i)) for i, key in enumerate(
        ("gate_x", "gate_xs", "ids", "gate_w", "gate_ws", "up_w", "up_ws"), 20)}
    reference = (b"\x80\x3f", b"\x00\x40")
    native = []

    def gate(*args):
        native.append(args)
        hip.trace.append("gate")
        return 0

    for region, value in zip(plan, reference):
        region.readback = lambda value=value: value
        region.sentinel = lambda stream: hip.trace.append("sentinel")
    result = {}
    caller.timed_arm(hip, SimpleNamespace(gate=gate), "gate", inputs, plan, 1920,
                     (owner, start, end), reference, lambda: None, result)
    assert len(native) == 60 and all(row[-3:] == (8, 1920, owner) for row in native)
    assert all(tuple(v.value for v in row[:9]) == (11, 12, 20, 21, 22, 23, 24, 25, 26) for row in native)
    assert hip.trace == (["sentinel"] * 2 + ["gate"] * 20 + ["sentinel"] * 2 + ["StreamSynchronize", "EventRecord"]
                         + ["gate"] * 40 + ["EventRecord", "EventSynchronize", "EventElapsedTime"])
    assert result["event_ms_per_call"] == .25 and result["warm_launches_enqueued"] == 20
    assert result["timed_launches_enqueued"] == 40


def check_refusal(path):
    cfg = path / "binding.json"
    result = path / "report.json"
    assert not result.exists()
    rows = {}
    for name in (*F.PROVIDERS, "hip"):
        p = path / (name + ".so")
        p.write_bytes(name.encode() if name != "D4" else b"")
        rows[name] = dict(path=str(p.resolve()), sha256=F.sha(p.read_bytes()))
    cfg.write_text(json.dumps(dict(schema=caller.SCHEMA, providers={name: rows[name] for name in F.PROVIDERS},
                                   hip=rows["hip"], expected_pci="0000:af:00.0", epoch=0, deadline=10**16)))
    with patch.object(H, "own_symbols", side_effect=AssertionError("premature HIP or DSO loading")) as own:
        assert caller.main(["--binding", str(cfg), "--out", str(result)]) == 1
        own.assert_not_called()
    row = json.loads(result.read_bytes())
    assert "absent, empty" in row["error"]["summary"]
    assert row["completed"] and not row["passed"] and row["timed_launches_enqueued"] == 0
    assert (row["numeric_counts"]["MISSING"], row["pair_counts"]["MISSING"],
            row["arm_counts"]["MISSING"]) == (24, 48, 96)
    Path(rows["D4"]["path"]).write_bytes(b"new file, old digest")
    second = path / "wrong-sha.json"
    with patch.object(H, "own_symbols", side_effect=AssertionError("premature HIP or DSO loading")) as own:
        assert caller.main(["--binding", str(cfg), "--out", str(second)]) == 1
        own.assert_not_called()
    assert "SHA256 mismatch" in json.loads(second.read_bytes())["error"]["summary"]


def check_failure_denominators(path):
    report = caller.Report(path / "partial.json")
    case = next(iter(report.cases.values()))
    chosen = case["pairs"]["G1.1"]
    chosen["arms"]["ORIG"].update(state="PASS", timed_launches_enqueued=F.TIMED, warm_launches_enqueued=F.WARM)
    chosen["arms"]["G1"].update(state="FAILED", timed_launches_enqueued=2, warm_launches_enqueued=F.WARM)
    chosen["state"] = "FAILED"
    case["numeric"]["ORIG.gate"]["state"] = "PASS"
    case["numeric"]["D2.down"]["state"] = "FAILED"
    report.finish()
    landed = json.loads(report.destination.read_bytes())
    assert landed["numeric_counts"] == dict(PASS=1, FAILED=1, MISSING=22, BLOCKED_NUMERIC=0)
    assert landed["pair_counts"] == dict(PASS=0, FAILED=1, MISSING=47, BLOCKED_NUMERIC=0)
    assert landed["arm_counts"] == dict(PASS=1, FAILED=1, MISSING=94, BLOCKED_NUMERIC=0)
    assert landed["timed_launches_enqueued"] == 42 and landed["passed_timed_launches"] == 40


def main():
    result = check_fixtures()
    assert H.pci_identity(b"AF:00.1") == "0000:af:00.1"
    assert H.pci_identity(b"0001:af:00.1") == "0001:af:00.1"
    check_guard()
    check_timing()
    with TemporaryDirectory() as temporary:
        check_refusal(Path(temporary))
        check_failure_denominators(Path(temporary))
    result.update(host_guard_rejections=2, recorded_timing_arms=1, recorded_timing_launches=60,
                  artifact_refusals_before_dlopen=2, partial_denominator_verified=True,
                  pci_formats_verified=2, actual_hip_calls=0, actual_GPU=0)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
