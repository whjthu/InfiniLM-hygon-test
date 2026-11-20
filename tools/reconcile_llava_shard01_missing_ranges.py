"""Read the original LLaVA shard 01 relay receipts without opening weight files."""

import argparse
import hashlib
import json
from pathlib import Path
import stat


REQUEST = "2cb64431972e4c0ea0db1bf69d2ff3cf"
FILENAME = "model-00001-of-00004.safetensors"
WHOLE_BYTES = 4995008464
WHOLE_SHA = "cfb2091e22729bd1e9e2241ce872137a122150229c647ae9e8b3295fc659d69e"
INTENT_SHA = "dcd5dde39a710d0ba4ff495f458601820c0e5c4041733425916325548f38d074"
RECEIVER_INTENT_SHA = "42e81121ea28557f94fdbae104eee12506b92f5c096eda2668e8965793488e6e"
LOCAL_OUTCOME_SHA = "86f2fd4035153414f48c9b4382f992071411ebf56937d2aa3ce96ce34ec013f1"
TARGET = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/"
              "training-T04-public-official-shard01-missing-ranges-a100-relay-v1")


def read_small(path, *, optional=False):
    try:
        info = path.lstat()
    except FileNotFoundError:
        if optional:
            return None, None
        raise
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 16384:
        raise ValueError("expected a bounded original receipt")
    with path.open("rb") as source:
        raw = source.read(16385)
    if len(raw) != info.st_size:
        raise ValueError("receipt changed while reading")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def file_size(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("original shard path is not a regular file")
    return info.st_size


def exact_identity(binding):
    if (binding["request"], binding["file"], binding["bytes"], binding["SHA256"]) != (
            REQUEST, FILENAME, WHOLE_BYTES, WHOLE_SHA):
        raise ValueError("original request or shard identity differs")


def bound(row, binding):
    if any(row.get(key) != value for key, value in binding.items()):
        raise ValueError("receipt belongs to a different request")


def local_outcome(root):
    binding, intent_sha = read_small(root / "intent.json")
    row, terminal_sha = read_small(root / "outcome.json")
    exact_identity(binding)
    bound(row, binding)
    if (intent_sha != INTENT_SHA or terminal_sha != LOCAL_OUTCOME_SHA or row["passed"] is not False
            or row["failure_code"] != "SOURCE_TERMINAL" or row["local_weight_bytes_written"] != 0
            or row["transport_rcs"] != [1, 255]):
        raise ValueError("original bounded local relay terminal differs")
    frame_bytes = len((json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n").encode("utf8"))
    planned = binding["wire_weight_bytes"] + frame_bytes
    wire = row["wire_bytes"]
    if not frame_bytes < wire < planned or file_size(root / "receiver.stdout.private") != 0:
        raise ValueError("original partial pipe or empty receiver receipt differs")
    return dict(state="LOCAL_RELAY_FAILED_TARGET_UNVERIFIED", request=REQUEST,
                local_intent_SHA256=intent_sha, local_terminal_SHA256=terminal_sha,
                frame_bytes=frame_bytes, planned_pipe_bytes=planned, actual_pipe_bytes=wire,
                missing_pipe_bytes=planned-wire, weight_bytes_in_pipe_upper_bound=wire-frame_bytes,
                receiver_exit=1, sender_exit=255, sent_commit=False, failure_code=row["failure_code"],
                target_durable_bytes=None, independent_target_whole_SHA_verified=False,
                actual_model_weight_file_reads=0)


def target_snapshot(root):
    if root.resolve() != root.absolute():
        raise ValueError("original namespace aliases another path")
    binding, intent_sha = read_small(root / "intent.json", optional=True)
    payload = dict(request=REQUEST, target_durable_bytes=None, independent_target_whole_SHA_verified=False,
                   actual_model_weight_file_reads=0, original_receiver_intent_SHA256=intent_sha)
    if binding is None:
        return dict(payload, state="TARGET_INTENT_NOT_OBSERVED")
    exact_identity(binding)
    if intent_sha != RECEIVER_INTENT_SHA:
        raise ValueError("original receiver intent has changed")
    terminal, terminal_sha = read_small(root / "terminal.json", optional=True)
    partial = file_size(root / "shard.incomplete")
    final = file_size(root / FILENAME)
    payload.update(receiver_terminal_SHA256=terminal_sha, incomplete_file_stat_bytes=partial, final_file_stat_bytes=final)
    if terminal is None:
        return dict(payload, state="TARGET_TERMINAL_NOT_OBSERVED")
    bound(terminal, binding)
    durable, written = terminal["durable_bytes"], terminal["written_bytes"]
    if (terminal["role"] != "receiver" or type(durable) is not int or type(written) is not int
            or not 0 <= durable <= written <= WHOLE_BYTES or type(terminal["passed"]) is not bool):
        raise ValueError("invalid receiver terminal counters")
    if terminal["passed"] or final is not None:
        raise ValueError("receiver publication conflicts with the original unsent commit")
    retained, prefix_sha = read_small(root / "retained-prefix.json", optional=True)
    if retained is None and durable > 0:
        retained, prefix_sha = read_small(root / f"prefix-{durable:010d}.json", optional=True)
    if retained is not None:
        bound(retained, binding)
        prefix_bytes = retained["durable_bytes"]
        reported_sha = retained["prefix_SHA256"]
        if (type(prefix_bytes) is not int or not 0 <= prefix_bytes <= durable or partial is None
                or not prefix_bytes <= partial <= WHOLE_BYTES or not isinstance(reported_sha, str)
                or len(reported_sha) != 64 or any(c not in "0123456789abcdef" for c in reported_sha)):
            raise ValueError("original retained-prefix receipt differs from the terminal and partial file")
        payload.update(retained_prefix_receipt_SHA256=prefix_sha, reported_prefix_SHA256=reported_sha,
                       reported_prefix_durable_bytes=prefix_bytes)
    return dict(payload, state="TARGET_RECEIVER_FAILED", target_durable_bytes=durable, target_written_bytes=written)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    local = modes.add_parser("local")
    local.add_argument("original_local_root", type=Path)
    modes.add_parser("target")
    args = parser.parse_args()
    result = local_outcome(args.original_local_root) if args.mode == "local" else target_snapshot(TARGET)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
