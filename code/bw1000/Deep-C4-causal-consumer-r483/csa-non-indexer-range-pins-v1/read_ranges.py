"""Pin the disjoint sixteen non-indexer payload ranges for Deep CSA layer 2."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import struct
import time


PARENT = "1194900"
NODE = "f12r3n17"
SHARD = "model-00004-of-00046.safetensors"
META_SHA = "39e2af65e0f30502d4a0d1f97df54fc27f0023f2ba1861810a634b1065a7e4f7"
EXPECTED_BYTES = 157460736
SOURCE_PAYLOAD = 170605056


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(fd):
    info = os.fstat(fd)
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def selection(meta):
    all_ranges = meta["weights"]
    indexer = {k for k in all_ranges if k.startswith("indexer.")}
    require(len(all_ranges) == 23 and len(indexer) == 7 and meta["selected_payload_bytes"] == SOURCE_PAYLOAD
            and set(meta["headers"]) == {SHARD} and meta["parameter_values_read"] == 0,
            "Unexpected original metadata or indexer count")
    upper = meta["headers"][SHARD]
    header_end = 8 + upper["header_bytes"]
    rows = []
    for name, row in all_ranges.items():
        start, end = row["data_offsets"]
        require(row["shard"] == SHARD and row["checkpoint_key"] == "layers.2.attn." + name
                and 0 <= start < end and end - start == row["bytes"]
                and row["absolute_offset"] == start + header_end and end + header_end <= upper["file_bytes"],
                "Unexpected byte range")
        rows.append((start + header_end, end + header_end, name))
    rows.sort()
    require(all(a[1] <= b[0] for a, b in zip(rows, rows[1:])), "Selected ranges overlap")
    included = [row for row in rows if row[2] not in indexer]
    require(len(included) == 16 and sum(end - start for start, end, _ in included) == EXPECTED_BYTES,
            "Non-indexer scope changed")
    return included


def hash_selected(path, meta, deadline):
    rows = selection(meta)
    target = meta["headers"][SHARD]
    results = {}
    fd = os.open(path / SHARD, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = identity(fd)
        require(before[2] == target["file_bytes"], "Checkpoint file size changed")
        length, = struct.unpack("<Q", os.pread(fd, 8, 0))
        require(length == target["header_bytes"] and length <= 1024**2, "Checkpoint header extent changed")
        header = os.pread(fd, length, 8)
        require(len(header) == length and sha256(header).hexdigest() == target["header_SHA256"],
                "Checkpoint header digest changed")
        for start, end, name in rows:
            offset = start
            digest = sha256()
            while offset < end:
                require(time.monotonic() < deadline, "Range scan deadline expired")
                block = os.pread(fd, min(1024**2, end - offset), offset)
                require(block, "Checkpoint range truncated")
                digest.update(block)
                offset += len(block)
            results[name] = {"bytes": end - start, "SHA256": digest.hexdigest()}
        current = (path / SHARD).stat()
        require(identity(fd) == before and (current.st_dev, current.st_ino, current.st_size,
                 current.st_mtime_ns, current.st_ctime_ns) == before, "Checkpoint file was replaced or changed")
        return results
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job, ordinal = os.environ["SLURM_JOB_ID"], os.environ["SLURM_STEP_ID"]
    require(job == PARENT and ordinal.isdecimal() and socket.gethostname().split(".")[0] == NODE
            and os.environ.get("SLURM_STEP_GPUS") in {None, "", "(null)"}
            and os.environ.get("SLURM_CPUS_PER_TASK") == "1" and len(os.sched_getaffinity(0)) == 1,
            "Unexpected zero-GRES observer")
    start = time.monotonic()
    with args.metadata.open("rb") as source:
        raw = source.read(65537)
    require(len(raw) == 6071 and sha256(raw).hexdigest() == META_SHA, "Original layer-2 metadata changed")
    meta = json.loads(raw)
    ranges = hash_selected(args.checkpoint.resolve(strict=True), meta, start + 45)
    report = {"schema": "Deep-CSA-layer2-non-indexer-range-pins-v1", "step": job + "." + ordinal, "node": NODE,
              "metadata_SHA256": META_SHA, "original_config_SHA256": meta["original_config_SHA256"],
              "original_index_SHA256": meta["original_index_SHA256"],
              "header_SHA256": {SHARD: meta["headers"][SHARD]["header_SHA256"]}, "weights": ranges,
              "actual_value_bytes_read": sum(row["bytes"] for row in ranges.values()), "indexer_payload_bytes_read": 0,
              "actual_GPU_calls": 0, "full_checkpoint_bound": False, "elapsed_monotonic_s": time.monotonic() - start}
    profile = (json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with args.output.open("xb") as output:
        output.write(profile)
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps({key: report[key] for key in ("step", "actual_value_bytes_read", "indexer_payload_bytes_read",
                                                   "actual_GPU_calls", "elapsed_monotonic_s")}
                     | {"range_receipt_SHA256": sha256(profile).hexdigest(), "range_count": len(ranges)}))


if __name__ == "__main__":
    main()
