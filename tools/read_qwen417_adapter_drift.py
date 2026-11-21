"""Recompute original Qwen .417 adapter differences at their archived location."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import mmap
import os
from pathlib import Path
import signal
import stat
import struct
import time

import numpy as np


REPO = Path(__file__).resolve().parents[1]
RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
ARCHIVE = RESULTS / "training-qwen-native-paired-layer-Swift-AB-v3"
OUTPUT = RESULTS / "training-qwen-native-paired-swift-original417-adapter-drift-v1"
EVIDENCE = REPO / "results/bw1000/2026-11-20/Qwen-native-paired-Swift-original417-terminal-and-checkpoints.json"
EVIDENCE_SHA = "7418d3f843f18ef93dfa9bf5d7c35366b486b1180173d5c1c1dad104c5ff5421"
OUTCOME_SHA = "23060a92e7049adb814853ae5e8eb4893572af80a1663479b67be931cce44196"
STEPS = (10, 20, 25)
ARMS = ("stock", "paired")
DTYPE = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}
MAX_HEADER_BYTES = 8 * 1024**2
CHUNK_BYTES = 1024**2
DEADLINE_SECONDS = 180


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def uniq(pairs):
    result = {}
    for name, value in pairs:
        need(name not in result, "duplicate safetensors key")
        result[name] = value
    return result


def tensor_header(mapped):
    need(len(mapped) > 8, "invalid safetensors header size")
    length, = struct.unpack_from("<Q", mapped)
    body = 8 + length
    need(0 < length <= MAX_HEADER_BYTES and body < len(mapped), "invalid safetensors header size")
    document = json.loads(mapped[8:body], object_pairs_hook=uniq)
    need(isinstance(document, dict), "invalid safetensors header")
    tensors = {}
    for name, item in document.items():
        if name == "__metadata__":
            need(isinstance(item, dict) and all(isinstance(value, str) for value in item.values()),
                 "invalid safetensors user metadata")
            continue
        need(name and isinstance(item, dict) and item.keys() == {"dtype", "shape", "data_offsets"},
             "invalid safetensors tensor entry")
        dtype, shape, offsets = item["dtype"], item["shape"], item["data_offsets"]
        need(isinstance(dtype, str) and dtype in DTYPE
             and isinstance(shape, list) and all(type(dim) is int and dim > 0 for dim in shape)
             and isinstance(offsets, list) and len(offsets) == 2 and all(type(pos) is int for pos in offsets),
             "invalid adapter dtype, shape or offsets")
        start, end = offsets
        need(0 <= start < end <= len(mapped) - body and end - start == math.prod(shape) * np.dtype(DTYPE[dtype]).itemsize,
             "tensor offsets do not match the adapter shape")
        tensors[name] = (dtype, tuple(shape), body + start, body + end)
    next_offset = body
    for _, _, start, end in sorted(tensors.values(), key=lambda row: row[2]):
        need(start == next_offset, "overlap or gap in safetensors adapter payload")
        next_offset = end
    need(tensors and next_offset == len(mapped), "adapter tensor inventory does not cover all payload bytes")
    return tensors


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@contextmanager
def adapter(path, pin, check):
    before = path.lstat()
    need(path.resolve(strict=True) == path and stat.S_ISREG(before.st_mode)
         and 8 < before.st_size == pin["bytes"], "wrong archived adapter path or size")
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        opened = os.fstat(handle.fileno())
        need(stat.S_ISREG(opened.st_mode) and opened.st_size == before.st_size
             and identity(before) == identity(path.lstat())
             and (os.name == "nt" or identity(before) == identity(opened)), "archived adapter changed during open")
        digest = sha256()
        for first in range(0, len(mapped), CHUNK_BYTES):
            check()
            digest.update(mapped[first:first + CHUNK_BYTES])
        need(digest.hexdigest() == pin["sha256"], "archived adapter SHA-256 differs")
        yield mapped, tensor_header(mapped)
        need(identity(opened) == identity(os.fstat(handle.fileno())) and identity(before) == identity(path.lstat()),
             "archived adapter changed during comparison")


def values(raw, dtype):
    stored = np.frombuffer(raw, dtype=DTYPE[dtype])
    if dtype == "BF16":
        stored = (stored.astype("<u4") << 16).view("<f4")
    out = stored.astype(np.float64)
    need(np.isfinite(out).all(), "nonfinite value in adapter")
    return out


def compare_tensor(left, right, stock, paired, check):
    dtype, shape, first, end = stock
    paired_dtype, paired_shape, right_first, right_end = paired
    need((dtype, shape, end-first) == (paired_dtype, paired_shape, right_end-right_first),
         "paired tensor dtype or shape differs")
    count = math.prod(shape)
    differing, max_abs = 0, 0.
    stock_squares, delta_squares = [], []
    payload_equal = True
    for offset in range(0, end-first, CHUNK_BYTES):
        check()
        size = min(CHUNK_BYTES, end-first-offset)
        a = left[first+offset:first+offset+size]
        b = right[right_first+offset:right_first+offset+size]
        payload_equal &= a == b
        stock_values, paired_values = values(a, dtype), values(b, dtype)
        delta = paired_values - stock_values
        differing += int(np.count_nonzero(delta))
        max_abs = max(max_abs, float(np.max(np.abs(delta))))
        stock_squares.append(float(np.sum(stock_values * stock_values)))
        delta_squares.append(float(np.sum(delta * delta)))
    return dict(dtype=dtype, elements=count, differing_values=differing, tensor_body_bitwise_equal=payload_equal,
                max_abs=max_abs, stock_squared=math.fsum(stock_squares), delta_squared=math.fsum(delta_squares))


def summarize(rows):
    stock = math.fsum(row["stock_squared"] for row in rows)
    delta = math.fsum(row["delta_squared"] for row in rows)
    count = sum(row["elements"] for row in rows)
    return dict(tensors=len(rows), elements=count, differing_values=sum(row["differing_values"] for row in rows),
                differing_tensor_bodies=sum(not row["tensor_body_bitwise_equal"] for row in rows),
                max_abs=max(row["max_abs"] for row in rows), rms_abs=math.sqrt(delta/count),
                relative_l2_to_stock_tensor_values=math.sqrt(delta/stock) if stock else None)


def compare(stock, paired, pins, check=lambda: None):
    with adapter(stock, pins["stock"], check) as (left, a), adapter(paired, pins["paired"], check) as (right, b):
        need(a.keys() == b.keys(), "paired adapter tensor inventory differs")
        rows = {name: compare_tensor(left, right, a[name], b[name], check) for name in sorted(a)}
    families = {}
    for name, row in rows.items():
        family = next((part for part in name.split(".") if part in ("lora_A", "lora_B")), "other")
        families.setdefault(family, []).append(row)
    return dict(all=summarize(list(rows.values())),
                by_family={name: summarize(items) for name, items in families.items()}), rows


def checkpoint_plan():
    source = EVIDENCE.read_bytes()
    need(sha256(source).hexdigest() == EVIDENCE_SHA, "original .417 terminal evidence differs")
    evidence = json.loads(source)
    need(evidence["original_outcome_SHA256"] == OUTCOME_SHA and evidence["independent_post_clean"] is True
         and evidence["checkpoint_adapter_payloads_rehashed"] == 6, "original .417 archived binding differs")
    by_arm = {arm: evidence["by_arm"][arm]["checkpoints"] for arm in ARMS}
    need(all(tuple(item["global_step"] for item in rows) == STEPS for rows in by_arm.values()),
         "archived paired steps differ")
    return [dict(step=step, files={arm: dict(
        path=f"{arm}/train/checkpoint-{step}/adapter_model.safetensors", **by_arm[arm][i]["adapter"]) for arm in ARMS})
        for i, step in enumerate(STEPS)]


def measure(check, save_step):
    plan = checkpoint_plan()
    outcome = ARCHIVE / "outcome.json"
    need(outcome.resolve(strict=True) == outcome and outcome.is_file(), "original run namespace differs")
    with outcome.open("rb") as handle:
        raw = handle.read(2 * 1024**2 + 1)
    need(len(raw) <= 2 * 1024**2 and sha256(raw).hexdigest() == OUTCOME_SHA, "original controller outcome differs")
    for entry in plan:
        pins = entry["files"]
        paths = [ARCHIVE / pins[arm]["path"] for arm in ARMS]
        summary, rows = compare(*paths, pins, check)
        save_step(dict(step=entry["step"], verified_files=pins, **summary), rows)


def save(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="verify and enumerate the pinned paths without opening adapters")
    if parser.parse_args().plan:
        print(json.dumps(dict(checkpoints=checkpoint_plan(), actual_adapter_files_opened=0), allow_nan=False))
        return 0
    need(os.name == "posix" and OUTPUT.parent.resolve(strict=True) == RESULTS, "original target results root is required")
    OUTPUT.mkdir(mode=0o700)
    started = time.monotonic()
    result = []

    def expired(*_):
        raise TimeoutError("adapter audit deadline exceeded")

    def check():
        if time.monotonic() - started >= DEADLINE_SECONDS:
            expired()

    def save_step(summary, rows):
        step = summary["step"]
        pin = save(OUTPUT / f"checkpoint-{step}-private.json", dict(summary=summary, tensors=rows))
        result.append(dict(summary, private_report_sha256=pin))

    error = None
    save(OUTPUT / "intent.json", dict(original_step="1103606.417", checkpoints=STEPS,
                                     source_summary_SHA256=EVIDENCE_SHA, wall_seconds=DEADLINE_SECONDS))
    signal.signal(signal.SIGALRM, expired)
    try:
        check()
        signal.setitimer(signal.ITIMER_REAL, max(.001, DEADLINE_SECONDS - (time.monotonic() - started)))
        measure(check, save_step)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        error = type(exc).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    pending = list(STEPS[len(result):])
    changed = next((row["step"] for row in result if row["all"]["differing_values"]), None)
    report = dict(original_step="1103606.417", checkpoint_pairs=result, unverified_checkpoints=pending,
                  passed=not pending and error is None, error=error,
                  earliest_observed_changed_checkpoint=changed, seconds=time.monotonic()-started,
                  actual_Slurm_submissions=0, actual_GPU_steps=0, actual_optimizer_updates=0, training_READY_added=0)
    pin = save(OUTPUT / "result.json", report)
    print(json.dumps(dict(report, result_sha256=pin), allow_nan=False))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
