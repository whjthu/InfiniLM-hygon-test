"""One-card GLM TP16 I128 leaf comparison using admitted standalone providers."""
import argparse
from contextlib import ExitStack
import ctypes as C
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import struct
import sys
import tempfile
import time

# The frozen sibling modules must remain usable with Python -I -S -B.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixture as F
import hip_runtime as H


SCHEMA = "glm_i128_one_gpu_v1"
LEAVES = (("ORIG", "gate"), ("ORIG", "down"), *F.ARMS.items())
INPUT_PIN = "19a5fcd6e25ae80ecfbe538d9f83bac2edb9d009d128b4d1a7e15191e2cf645e"


def fixture_binding():
    raw = Path(__file__).with_name("INPUT_BINDING.json").read_bytes()
    if F.sha(raw) != INPUT_PIN:
        raise ValueError("synthetic fixture binding SHA mismatch")
    return json.loads(raw)["cases"]


def binding(raw):
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError("missing admitted binding schema")
    providers = raw.get("providers")
    if not isinstance(providers, dict) or set(providers) != set(F.PROVIDERS):
        raise ValueError("exact five artifact providers required")
    pins = {role: H.file_pin(providers[role]) for role in F.PROVIDERS}
    hip_pin = H.file_pin(raw.get("hip"))
    all_pins = [*pins.values(), hip_pin]
    if len({path for path, _ in all_pins}) != 6 or len({sha for _, sha in all_pins}) != 6:
        raise ValueError("HIP and candidate artifacts must have distinct identities")
    identity = raw.get("expected_pci")
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[01][0-9a-f]\.[0-7]", identity):
        raise ValueError("lower-case physical GPU PCI required")
    epoch, expires = raw.get("epoch"), raw.get("deadline")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in (epoch, expires)):
        raise ValueError("single finite launcher epoch/deadline required")
    wall_now, mono_now = time.time(), time.monotonic()
    if not epoch <= wall_now < expires:
        raise ValueError("admitted launcher budget expired or not active")
    monotonic_expiry = mono_now + (expires - wall_now)

    def remaining():
        if time.time() >= expires or time.monotonic() >= monotonic_expiry:
            raise TimeoutError("original launcher budget exceeded")

    return pins, hip_pin, identity, remaining


class Report:
    def __init__(self, destination):
        self.destination = destination
        self.cases = {}
        for name, offset, routes in F.CASES:
            checks = {f"{role}.{op}": {"state": "PLANNED"} for role, op in LEAVES}
            pairs = {}
            for candidate, op in F.ARMS.items():
                for index, reversed_pair in enumerate(F.PAIR_REVERSE, 1):
                    order = [candidate, "ORIG"] if reversed_pair else ["ORIG", candidate]
                    pairs[f"{candidate}.{index}"] = dict(
                        operation=op, order=order, state="PLANNED",
                        arms={arm: {"state": "PLANNED"} for arm in order})
            self.cases[name] = dict(offset=offset, route_ids=routes, state="PLANNED",
                                    numeric=checks, pairs=pairs)
        self.data = dict(schema=SCHEMA, fixture_kind="SYNTHETIC_PACKED_E4M3_COMPONENT",
                         fixture_binding_sha256=INPUT_PIN,
                         tokens=1, routes=F.ROUTES, hidden=F.HIDDEN, local_width=F.LOCAL,
                         experts=F.EXPERTS, warm_per_arm=F.WARM, timed_per_arm=F.TIMED,
                         expected_numeric_checks=24, expected_pair_windows=48,
                         expected_arm_windows=96, expected_timed_launches=3840,
                         expected_warm_launches=1920, completed=False, passed=False, cases=self.cases)
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(self.data, handle, allow_nan=False)
            handle.write("\n")

    def save(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.destination.parent,
                                         prefix=".i128-", delete=False) as handle:
            temp = Path(handle.name)
            try:
                json.dump(self.data, handle, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                handle.close()
                temp.unlink()
                raise
        try:
            os.replace(temp, self.destination)
        finally:
            temp.unlink(missing_ok=True)

    def finish(self):
        numeric = []
        pairs = []
        arms = []
        for case in self.cases.values():
            if case["state"] == "PLANNED":
                case["state"] = "MISSING"
            numeric.extend(case["numeric"].values())
            for pair in case["pairs"].values():
                pairs.append(pair)
                arms.extend(pair["arms"].values())
        for row in (*numeric, *pairs, *arms):
            if row["state"] in ("PLANNED", "RUNNING"):
                row["state"] = "MISSING"

        def state_counts(rows):
            return {state: sum(row["state"] == state for row in rows)
                    for state in ("PASS", "FAILED", "MISSING", "BLOCKED_NUMERIC")}

        complete = (all(row["state"] == "PASS" for row in self.cases.values())
                    and all(row["state"] == "PASS" for row in (*numeric, *pairs, *arms))
                    and "error" not in self.data)
        self.data.update(completed=True, passed=complete, numeric_counts=state_counts(numeric),
                         pair_counts=state_counts(pairs), arm_counts=state_counts(arms),
                         timed_launches_enqueued=sum(row.get("timed_launches_enqueued", 0) for row in arms),
                         warm_launches_enqueued=sum(row.get("warm_launches_enqueued", 0) for row in arms),
                         passed_timed_launches=F.TIMED * sum(row["state"] == "PASS" for row in arms))
        self.save()


def err(error):
    return dict(type=type(error).__name__, summary=str(error)[:300])


def resources(hip, cleanup, index, expected_inputs, remaining):
    inputs = {}
    # Generate one full host projection bank at a time.
    def values():
        yield from F.inputs(index).items()
        for role in (2, 3, 4):
            yield from F.weights(index, role)

    for name, content in values():
        remaining()
        region = H.Guarded(hip, cleanup, len(content))
        region.upload(content)
        if expected_inputs.get(name) != dict(bytes=region.size, sha256=region.pin):
            raise ValueError(f"synthetic input fixture mismatch: {name}")
        inputs[name] = region
    if inputs.keys() != expected_inputs.keys():
        raise ValueError("synthetic input set mismatch")
    outputs = {kind: tuple(H.Guarded(hip, cleanup, values * 2) for values in sizes)
               for kind, sizes in F.BF16_COUNTS.items()}
    return inputs, outputs


def invoke(api, operation, inputs, outputs, offset, stream):
    selected = ["gate_x", "gate_xs", "ids", "gate_w", "gate_ws", "up_w", "up_ws"] if operation == "gate" else [
        "down_x", "down_xs", "ids", "down_w", "down_ws"]
    args = [*(r.ptr for r in outputs), *(inputs[key].ptr for key in selected), F.ROUTES, offset, stream]
    H.checked(getattr(api, operation)(*args), "routed I128 " + operation)


def validate_routes(hip, api, ids, status, stream, expected):
    status.sentinel(stream, 0)
    H.checked(api.validate(ids.ptr, F.ROUTES, F.EXPERTS, status.ptr, stream), "routed I128 validate")
    hip.call("StreamSynchronize", stream)
    observed, = struct.unpack("<i", status.readback())
    if observed != expected:
        raise RuntimeError("routed I128 validate status mismatch")


def numeric(hip, provider, operation, input_map, outputs, case, stream, reference, baseline):
    for item in outputs:
        item.sentinel(stream)
    invoke(provider, operation, input_map, outputs, case["offset"], stream)
    hip.call("StreamSynchronize", stream)
    actual = tuple(output.readback() for output in outputs)
    oracle = F.comparisons(actual, reference)
    compare = F.comparisons(actual, baseline) if baseline is not None else None
    return actual, dict(reference=oracle, baseline=compare,
                        passed=F.passing(oracle) and (compare is None or F.passing(compare)))


def timed_arm(hip, provider, operation, input_map, outputs, offset, events, reference, remaining, row):
    stream, start, end = events
    for item in outputs:
        item.sentinel(stream)
    for _ in range(F.WARM):
        remaining()
        invoke(provider, operation, input_map, outputs, offset, stream)
        row["warm_launches_enqueued"] = row.get("warm_launches_enqueued", 0) + 1
    # Reset after warm-up so an unwritten timed output cannot reuse warm data.
    for item in outputs:
        item.sentinel(stream)
    hip.call("StreamSynchronize", stream)
    remaining()
    wall = time.perf_counter()
    hip.call("EventRecord", start, stream)
    for _ in range(F.TIMED):
        remaining()
        invoke(provider, operation, input_map, outputs, offset, stream)
        row["timed_launches_enqueued"] = row.get("timed_launches_enqueued", 0) + 1
    hip.call("EventRecord", end, stream)
    hip.call("EventSynchronize", end)
    elapsed_wall_ms = 1000 * (time.perf_counter() - wall)
    ms = C.c_float()
    hip.call("EventElapsedTime", C.byref(ms), start, end)
    if not math.isfinite(ms.value) or ms.value <= 0:
        raise RuntimeError("HIP stream event interval unavailable")
    row.update(event_window_ms=ms.value, event_ms_per_call=ms.value / F.TIMED,
               host_window_ms=elapsed_wall_ms)
    remaining()
    checks = F.comparisons(tuple(output.readback() for output in outputs), reference)
    row.update(post_timing=checks)
    if not F.passing(checks):
        raise RuntimeError("post-timing BF16 output differs")


def run_case(report, index, hip, providers, events, remaining, frozen):
    name, offset, experts = F.CASES[index]
    case = report.cases[name]
    stream = events[0]
    case["state"] = "RUNNING"
    report.save()
    with ExitStack() as cleanup:
        try:
            if offset != frozen["offset"] or list(experts) != frozen["routes"]:
                raise ValueError("synthetic control case mismatch")
            reference = {op: F.reference(index, op) for op in F.BF16_COUNTS}
            if frozen["references"] != {op: [dict(bytes=len(v), sha256=F.sha(v)) for v in rows]
                                         for op, rows in reference.items()}:
                raise ValueError("synthetic arithmetic oracle mismatch")
            case["memory_before"] = hip.mem()
            input_map, outputs = resources(hip, cleanup, index, frozen["inputs"], remaining)
            status = H.Guarded(hip, cleanup, 4)
            wrong_ids = H.Guarded(hip, cleanup, F.ROUTES * 8)
            case["memory_resident"] = hip.mem()
            case["input_sha256"] = {key: guard.pin for key, guard in input_map.items()}
            case["input_payload_bytes"] = sum(guard.size for guard in input_map.values())
            case["validated_routes"] = {}
            for role, api in providers.items():
                remaining()
                validate_routes(hip, api, input_map["ids"], status, stream, 0)
                for invalid in (-1, F.EXPERTS):
                    wrong_ids.upload(struct.pack("<8q", *case["route_ids"][:-1], invalid))
                    validate_routes(hip, api, wrong_ids, status, stream, 1)
                    wrong_ids.input_audit(remaining)
                case["validated_routes"][role] = dict(good=1, invalid_rejected=2)
            baseline = {}
            for role, operation in LEAVES:
                remaining()
                row = case["numeric"][f"{role}.{operation}"]
                try:
                    actual, results = numeric(hip, providers[role], operation, input_map,
                                              outputs[operation], case, stream, reference[operation],
                                              baseline.get(operation))
                    row.update(results)
                    if not row["passed"]:
                        raise RuntimeError("independent synthetic BF16 oracle or baseline mismatch")
                    if role == "ORIG":
                        baseline[operation] = actual
                    elif operation not in baseline:
                        raise RuntimeError("original comparison unavailable")
                    row["state"] = "PASS"
                except (OSError, ValueError, RuntimeError) as error:
                    row.update(state="FAILED", error=err(error))
                    # Drain the owned stream before submitting any further leaf.
                    hip.call("StreamSynchronize", stream)
                report.save()
            for ident, pair in case["pairs"].items():
                remaining()
                candidate = ident.split(".")[0]
                operation = pair["operation"]
                if any(case["numeric"][f"{role}.{operation}"]["state"] != "PASS" for role in ("ORIG", candidate)):
                    pair["state"] = "BLOCKED_NUMERIC"
                    for row in pair["arms"].values():
                        row["state"] = "BLOCKED_NUMERIC"
                    report.save()
                    continue
                for role in pair["order"]:
                    row = pair["arms"][role]
                    try:
                        timed_arm(hip, providers[role], operation, input_map, outputs[operation],
                                  case["offset"], events, reference[operation], remaining, row)
                        row["state"] = "PASS"
                    except (OSError, ValueError, RuntimeError) as error:
                        row.update(state="FAILED", error=err(error))
                        hip.call("StreamSynchronize", stream)
                        if isinstance(error, TimeoutError):
                            raise
                    report.save()
                passed = all(row["state"] == "PASS" for row in pair["arms"].values())
                pair["state"] = "PASS" if passed else "FAILED"
                if passed:
                    pair["event_candidate_over_original"] = (
                        pair["arms"][candidate]["event_ms_per_call"] / pair["arms"]["ORIG"]["event_ms_per_call"]
                    )
                report.save()
            for region in input_map.values():
                region.input_audit(remaining)
            case["input_readback_verified"] = True
            case["state"] = "PASS" if all(row["state"] == "PASS" for row in
                                          (*case["numeric"].values(), *case["pairs"].values())) else "FAILED"
            case["paired_median"] = {}
            for candidate in F.ARMS:
                original_pairs = [case["pairs"][f"{candidate}.{i}"] for i in range(1, 4)]
                if all(row["state"] == "PASS" for row in original_pairs):
                    case["paired_median"][candidate] = median(row["event_candidate_over_original"]
                                                              for row in original_pairs)
        except (OSError, ValueError, RuntimeError) as error:
            case.update(state="FAILED", error=err(error))
            if isinstance(error, TimeoutError):
                raise
        finally:
            hip.call("StreamSynchronize", stream)
            report.save()
    case["memory_after_free"] = hip.mem()
    report.save()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    config = parser.parse_args(args)
    report = Report(config.out)
    try:
        with config.binding.open("rb") as handle:
            raw = handle.read(65537)
        if len(raw) > 65536:
            raise ValueError("binding exceeds 64 KiB")
        loaded = json.loads(raw)
        pins, hip_pin, identity, remaining = binding(loaded)
        frozen = fixture_binding()
        report.data.update(binding_sha256=F.sha(raw), deadline=loaded["deadline"], epoch=loaded["epoch"],
                           providers={role: dict(path=str(path), sha256=sha) for role, (path, sha) in pins.items()},
                           hip=dict(path=str(hip_pin[0]), sha256=hip_pin[1]))
        remaining()
        hip, providers = H.own_symbols(pins, hip_pin)
        report.data["dladdr_verified_symbols_per_provider"] = 3
        report.data["device"] = hip.device(identity)
        with hip.stream() as events:
            for index, (name, _, _) in enumerate(F.CASES):
                remaining()
                run_case(report, index, hip, providers, events, remaining, frozen[name])
        report.data["normal_stream_and_event_destroy"] = True
    except (OSError, ValueError, RuntimeError, KeyError, AttributeError, C.ArgumentError) as error:
        report.data["error"] = err(error)
    finally:
        report.finish()
    return 0 if report.data["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
