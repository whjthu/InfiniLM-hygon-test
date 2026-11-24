"""Check and stage the entire selected CSA layer before the first Native write."""

import ctypes
from hashlib import sha256
import os
from pathlib import Path
import struct
import time

from bind import BASE, METADATA_SHA, bind, digest, module, need, pinned


CONTRACT = "csa-identity-swa-checkpoint-v1/WEIGHTS.json"
CONTRACT_SHA = "f15fc11e1799a53c39f4cab5be92e8f784b9bb822636bed12c418b818edf8cc0"
BLOCK = 1024 * 1024


def signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def stage_payloads(root, metadata, expected_pins, deadline):
    """Use one file identity, keep typed bytes pinned, and defer every Native call."""
    need(len(metadata["headers"]) == 1, "one original CSA shard required")
    name, expected_header = next(iter(metadata["headers"].items()))
    need(Path(name).name == name and name.endswith(".safetensors"), "invalid frozen shard name")
    path = root / name
    deadline("checkpoint.stage.header")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    buffers = {}
    with os.fdopen(fd, "rb", buffering=0) as source:
        before = signature(os.fstat(fd))
        prefix = source.read(8)
        need(len(prefix) == 8, "missing shard header length")
        size, = struct.unpack("<Q", prefix)
        need(before[2] == expected_header["file_bytes"] and size == expected_header["header_bytes"]
             and 0 < size <= 1024**2, "original shard extent changed")
        header = source.read(size)
        need(len(header) == size and digest(header) == expected_header["header_SHA256"], "original shard header changed")
        for parameter, row in sorted(metadata["weights"].items()):
            deadline("checkpoint.stage." + parameter)
            pin = expected_pins[parameter]
            need(row["shard"] == name and type(row["bytes"]) is int and row["bytes"] == pin["bytes"]
                 and size + 8 <= row["absolute_offset"] < row["absolute_offset"] + row["bytes"] <= before[2],
                 "pinned range bounds differ")
            source.seek(row["absolute_offset"])
            buffer = ctypes.create_string_buffer(row["bytes"])
            h = sha256()
            with memoryview(buffer).cast("B") as data:
                offset = 0
                while offset < len(data):
                    deadline("checkpoint.read." + parameter)
                    with data[offset:offset + BLOCK] as window:
                        extent = source.readinto(window)
                        need(extent is not None and extent > 0, "pinned range was truncated")
                        h.update(window[:extent])
                    offset += extent
            need(h.hexdigest() == pin["SHA256"], "original payload digest differs: " + parameter)
            buffers[parameter] = buffer
        need(signature(os.fstat(fd)) == before and signature(path.stat()) == before,
             "original shard changed or was replaced while staging")
    return buffers


def load_buffers(trial, buffers, deadline):
    """Consume already checked buffers with a synchronous Native ABI."""
    loaded, size = [], 0
    started = False
    try:
        for parameter in tuple(buffers):
            deadline("checkpoint.load." + parameter)
            buffer = buffers.pop(parameter)
            started = True
            trial.call("load", parameter.encode("utf-8"), buffer, len(buffer))
            loaded.append(parameter)
            size += len(buffer)
            del buffer
        deadline("checkpoint.postload")
        trial.call("postload")
    except BaseException:
        if started:
            trial.close()
        raise
    return loaded, size


def load_checkpoint(trial, *, metadata_raw, non_indexer_raw, original_indexer, checkpoint_root,
                    runtime_config_sha, deadline):
    """Load once into a fresh TP1 CsaAttention trial after all 23 pins pass."""
    started = time.monotonic()
    coverage, merged = bind(metadata_raw, non_indexer_raw, original_indexer)
    need(merged is not None and coverage["full_payload_pins_complete"] is True,
         "seven original indexer ranges required before any checkpoint access")
    original = module("csa-identity-swa-checkpoint-v1/inspect_checkpoint.py")
    contract = original.parse(pinned(BASE / CONTRACT, 65536, CONTRACT_SHA))
    metadata = original.parse(metadata_raw)
    need(runtime_config_sha == metadata["runtime_config_SHA256"], "worker runtime config differs")
    original.verify_native_parameters(trial.read("parameters"), contract)
    pins = original.bind_range_pins(metadata_raw, METADATA_SHA, merged, coverage["merged_receipt_SHA256"])
    root = Path(checkpoint_root).resolve(strict=True)
    deadline("checkpoint.discover")
    need(original.discover(root, contract) == metadata, "checkpoint metadata changed before staging")
    buffers = stage_payloads(root, metadata, pins, deadline)
    need(len(buffers) == 23 and sum(len(buffer) for buffer in buffers.values()) == 170605056,
         "staged payload extent differs before Native load")
    clean_stage_seconds = time.monotonic() - started
    loaded, size = load_buffers(trial, buffers, deadline)
    return dict(layer=2, TP=1, checkpoint_scope="CSA layer-2 component", loaded_parameters=loaded,
                payload_bytes=size, merged_receipt_SHA256=coverage["merged_receipt_SHA256"],
                verified_before_first_Native_call=True, staging_wall_seconds=clean_stage_seconds,
                total_load_wall_seconds=time.monotonic() - started)
