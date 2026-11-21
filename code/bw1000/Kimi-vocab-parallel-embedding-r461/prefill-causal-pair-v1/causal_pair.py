"""Diagnostic for two prefill widths using one resident Kimi CAPI context per TP node."""

import ctypes as ct
from hashlib import sha256
import json
import math
import re
import sys
import time

import numpy as np


VOCAB = 163840
EOS = {163585, 163586}
LEGACY_FLOOR_BYTES = 4 << 30
WIDTHS = (256, 512)
SYMBOL = "modern_kimi_engine_forward_ar_storage_r456"


def digest(raw):
    return sha256(raw).hexdigest()


def compact(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def input_digest(prompt, suffix):
    return digest(compact(dict(prompt=prompt, teacher_forced_suffix=suffix)))


def require(condition, error):
    if not condition:
        raise ValueError(error)


def validate_inputs(prompt, forced_suffix, expected_input_sha):
    require(isinstance(prompt, list) and 0 < len(prompt) <= 32768 - 4096 and isinstance(forced_suffix, list)
            and len(forced_suffix) == 2, "Invalid full prompt or forced two-token continuation")
    require(all(type(t) is int and 0 <= t < VOCAB for t in prompt + forced_suffix)
            and not EOS.intersection(forced_suffix), "Invalid diagnostic token identity")
    require(input_digest(prompt, forced_suffix) == expected_input_sha, "Diagnostic input identity differs")


def matched_peers(peers, packet):
    require(len(peers) == 4 and all(type(row.get("node")) is int for row in peers)
            and {row["node"] for row in peers} == set(range(4)), "Incomplete TP32 node participation")
    keys = ("kind", "past", "width", "total", "ids_SHA256", "positions_SHA256")
    require(all(all(peer.get(key) == packet[key] for key in keys) for peer in peers),
            "TP32 continuation metadata differs")
    by_node = sorted(peers, key=lambda p: p["node"])
    require(by_node[packet["node"]] == packet, "TP32 local result does not match the broadcast")
    valid_token = by_node[0].get("selected")
    require(type(valid_token) is int and 0 <= valid_token < VOCAB
            and all(p.get("selected") is None for p in by_node[1:]), "Invalid selected-token owner")
    require(all(isinstance(p.get("logits_SHA256"), str) and re.fullmatch("[0-9a-f]{64}", p["logits_SHA256"])
                and type(p.get("capi_elapsed_s")) is float and math.isfinite(p["capi_elapsed_s"])
                and p["capi_elapsed_s"] >= 0 for p in by_node),
            "Invalid TP logits/clock evidence")
    return by_node


def reset_agreement(control, label, node):
    peers = control.collect(label, dict(node=node))
    require(len(peers) == 4 and all(type(p.get("node")) is int for p in peers)
            and {p["node"] for p in peers} == set(range(4)), "Kimi diagnostic reset agreement failed")


def schedule(prompt, suffix, width):
    past = 0
    for start in range(0, len(prompt), width):
        ids = prompt[start:start + width]
        past = start + len(ids)
        # The two arm boundaries coincide after a multiple of 512 and at prompt end.
        checkpoint = ("prompt-end" if past == len(prompt) else "prefix512" if past == 512 else None)
        yield 0, start, ids, checkpoint
    for i, token in enumerate(suffix, 1):
        yield 1, past, [token], f"continuation-{i}"
        past += 1


def run(lib, ctx, node, control, prompt, forced_suffix, expected_input_sha, scope,
        check_deadline, audit_floor, save_logits, record, *, after_first_prefill=None):
    """Execute a diagnostic, using the same forced continuation in both arms.

    Callers supply their admitted runtime, the original floor and monotonic deadline,
    a private byte sink, and durable progress. This caller consumes a context that
    has already loaded weights. It never constructs, closes, or serves a model.
    """
    require(sys.byteorder == "little" and type(node) is int and 0 <= node < 4
            and isinstance(scope, str) and re.fullmatch("[0-9a-f]{32}", scope),
            "Expected a unique TP32 diagnostic scope")
    validate_inputs(prompt, forced_suffix, expected_input_sha)
    forward = getattr(lib, SYMBOL)
    forward.argtypes = [ct.c_void_p, ct.c_int, ct.c_size_t, ct.c_size_t, ct.c_int] + [ct.c_void_p] * 9
    forward.restype = ct.c_int
    reset = lib.modern_kimi_engine_reset_r2475
    reset.argtypes = [ct.c_void_p]
    reset.restype = ct.c_int

    arms = []
    for width in WIDTHS:
        work = list(schedule(prompt, forced_suffix, width))
        prefix = f"{scope}:{width}"
        check_deadline()
        record("arm-start", dict(width=width, scheduled_calls=len(work), input_SHA256=expected_input_sha))
        calls = []
        initial_reset_succeeded = False
        try:
            require(reset(ctx) == 0, "Kimi diagnostic initial reset failed")
            initial_reset_succeeded = True
            reset_agreement(control, prefix + ":reset-before", node)
            for ordinal, (kind, past, ids, landmark) in enumerate(work):
                check_deadline()
                total = past + len(ids)
                buffers = [np.array([1], dtype=np.int32), np.array([past], dtype=np.int32),
                           np.array([total], dtype=np.int32), np.array([0, len(ids)], dtype=np.int32),
                           np.asarray(ids, dtype=np.int64), np.arange(past, total, dtype=np.int64),
                           np.full((1, VOCAB), 0x7fc0, dtype=np.uint16), np.array([-1], dtype=np.int64),
                           np.array([-1], dtype=np.int32)]
                before = [digest(b.tobytes()) for b in buffers[:6]]
                record("call-start", dict(arm=width, ordinal=ordinal, kind=kind, past=past, total=total))
                started = time.perf_counter()
                rc = forward(ctx, kind, len(ids), 1, 3, *[ct.c_void_p(b.ctypes.data) for b in buffers])
                elapsed = time.perf_counter() - started
                if rc:
                    raise RuntimeError(f"Kimi diagnostic CAPI failure with status {rc}")
                require(before == [digest(b.tobytes()) for b in buffers[:6]], "CAPI modified host causal inputs")
                logits, selected, flags = buffers[-3:]
                require(flags[0] == (3 if node == 0 else 1)
                        and not np.any((logits & 0x7f80) == 0x7f80),
                        "Invalid flags or incomplete/nonfinite native BF16 logits")
                packet = dict(node=node, kind=kind, past=past, total=total, width=len(ids),
                              ids_SHA256=before[4], positions_SHA256=before[5],
                              logits_SHA256=digest(logits.tobytes()), capi_elapsed_s=elapsed,
                              selected=int(selected[0]) if node == 0 else None)
                peers = matched_peers(control.collect(f"{prefix}:call:{ordinal}", packet), packet)
                if after_first_prefill is not None and width == WIDTHS[0] and ordinal == 0:
                    after_first_prefill()
                floor_min = audit_floor(f"{prefix}:floor:{ordinal}")
                require(type(floor_min) is int and floor_min > LEGACY_FLOOR_BYTES, "Original sampled 4 GiB floor violated")
                report = dict(ordinal=ordinal, landmark=landmark, width=len(ids), past=past, total=total,
                              kind=kind, memory_minimum_free_bytes=floor_min, peers=peers)
                if landmark and node == 0:
                    raw = logits.astype("<u2", copy=False).tobytes()
                    pin = save_logits(f"{prefix}:{landmark}", raw)
                    require(pin.get("SHA256") == digest(raw) and pin.get("bytes") == VOCAB * 2,
                            "Private BF16 capture identity differs")
                    report["private_logits"] = pin
                calls.append(report)
                record("call", dict(arm=width, **report))
        except BaseException as primary:
            try:
                record("arm-error", dict(width=width, planned_calls=len(work), completed_calls=len(calls)))
            except Exception as recording_error:
                if hasattr(primary, "add_note"):
                    primary.add_note(f"Kimi diagnostic progress failure: {type(recording_error).__name__}")
            raise
        finally:
            primary = sys.exc_info()[1]
            if initial_reset_succeeded:
                try:
                    require(reset(ctx) == 0, "Kimi diagnostic terminal reset failed")
                    reset_agreement(control, prefix + ":reset-after", node)
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    if hasattr(primary, "add_note"):
                        primary.add_note(f"Kimi diagnostic cleanup: {type(cleanup).__name__}")
        arms.append(dict(width=width, calls=calls, input_SHA256=expected_input_sha, planned_calls=len(work)))
        record("arm-end", dict(width=width, completed_calls=len(calls), normal_reset=True))
    return dict(node=node, scope=scope, arms=arms, prefill_tokens=len(prompt), generation_cap=4096,
                teacher_forced_diagnostic=True, service_performance_eligible=False)


def compare_logits(reference, candidate):
    """Compare two bounded little-endian BF16 vectors, preserving exact and FP error."""
    require(len(reference) == len(candidate) == VOCAB * 2, "Expected full-vocab BF16")
    a = np.frombuffer(reference, dtype="<u2")
    b = np.frombuffer(candidate, dtype="<u2")
    require(not np.any((a & 0x7f80) == 0x7f80) and not np.any((b & 0x7f80) == 0x7f80),
            "Nonfinite logits cannot establish numerical equivalence")
    a_float = (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    b_float = (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    delta = np.abs(a_float - b_float)
    return dict(elements=VOCAB, changed_BF16_bits=int(np.count_nonzero(a != b)),
                max_abs_delta=float(delta.max()), mean_abs_delta=float(delta.mean()),
                same_argmax=bool(np.argmax(a_float) == np.argmax(b_float)))


def compare_run(report, prompt, forced_suffix, expected_input_sha, read_private):
    """Recheck every call and compare only matching semantic output positions."""
    validate_inputs(prompt, forced_suffix, expected_input_sha)
    require(report.get("node") == 0 and report.get("teacher_forced_diagnostic") is True
            and report.get("service_performance_eligible") is False
            and report.get("prefill_tokens") == len(prompt) and report.get("generation_cap") == 4096,
            "Invalid diagnostic result scope")
    arms = report["arms"]
    require([arm["width"] for arm in arms] == list(WIDTHS), "Missing or repeated diagnostic arm")
    comparable = {}
    floors = {}
    for arm in arms:
        width = arm["width"]
        work = list(schedule(prompt, forced_suffix, width))
        require(arm["input_SHA256"] == expected_input_sha and len(arm["calls"]) == arm["planned_calls"] == len(work),
                "Incomplete or mismatched diagnostic arm")
        selected = {}
        floor_values = []
        for i, (call, (kind, past, ids, landmark)) in enumerate(zip(arm["calls"], work, strict=True)):
            total = past + len(ids)
            expected = dict(kind=kind, past=past, total=total, width=len(ids))
            require(call["ordinal"] == i and call["landmark"] == landmark
                    and all(call[key] == value for key, value in expected.items()), "Causal call order differs")
            expected.update(ids_SHA256=digest(np.asarray(ids, dtype="<i8").tobytes()),
                            positions_SHA256=digest(np.arange(past, total, dtype="<i8").tobytes()))
            peers = call["peers"]
            require(peers and peers[0].get("node") == 0, "Expected canonical TP leader report")
            leader = matched_peers(peers, dict(peers[0], **expected))[0]
            floor = call["memory_minimum_free_bytes"]
            require(type(floor) is int and floor > LEGACY_FLOOR_BYTES, "Original sampled floor was not met")
            floor_values.append(floor)
            if landmark is not None:
                pin = call["private_logits"]
                raw = read_private(pin)
                require(len(raw) == pin["bytes"] == VOCAB * 2
                        and digest(raw) == pin["SHA256"] == leader["logits_SHA256"], "Bound logits changed")
                selected[landmark] = (kind, total, raw, leader["selected"])
            else:
                require("private_logits" not in call, "Unregistered raw-logit capture")
        comparable[width] = selected
        floors[width] = min(floor_values)
    reference, candidate = comparable[512], comparable[256]
    require(reference.keys() == candidate.keys(), "Common semantic positions differ")
    summary = {}
    for point, (kind, pos, a, a_token) in reference.items():
        b_kind, b_pos, b, b_token = candidate[point]
        require((kind, pos) == (b_kind, b_pos), "Output context differs at the shared position")
        summary[point] = dict(kind=kind, context_tokens=pos, selected_token_matches=a_token == b_token,
                              **compare_logits(a, b))
    return dict(planned_arms=2, compared=summary, sampled_minimum_free_bytes=floors,
                actual_KV_tensor_checked=False, serving_performance_eligible=False)
