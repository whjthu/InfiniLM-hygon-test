"""Accept one checksum-bound, train-only VizWiz pilot packet."""

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import sys
import time


TARGET = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/"
              "training-gemma-vizwiz-reviewed-train-native-v1")
FILE_PATTERN = re.compile(r"VizWiz_train_[0-9]{8}\.jpg")
SHA_PATTERN = re.compile(r"[0-9a-f]{64}")
MAX_ROWS = 39
MAX_IMAGE = 512 * 1024
MAX_CONTROL = 128 * 1024
MAX_PACKET = 16 * 1024 * 1024
BUDGET = 240


def identity(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
            + "\n").encode("utf-8")


def save(path, raw):
    with path.open("xb") as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())


def check_pin(pin, maximum):
    if (set(pin) != {"bytes", "SHA256"} or type(pin["bytes"]) is not int
            or not 0 < pin["bytes"] <= maximum or not SHA_PATTERN.fullmatch(pin["SHA256"])):
        raise ValueError("invalid file pin")


def checked_member(stream, pin):
    raw = stream.read(pin["bytes"])
    if identity(raw) != pin:
        raise ValueError("partial packet or changed member")
    return raw


def check_claim(claim):
    if (claim["target"] != TARGET.as_posix() or claim["slurm"] != 0 or claim["gpu"] != 0
            or claim["budget_seconds"] != BUDGET or claim["replay_allowed"] is not False
            or not re.fullmatch(r"[0-9a-f]{32}", claim["request"])
            or not SHA_PATTERN.fullmatch(claim["quality_receipt_SHA256"])
            or type(claim["train_rows"]) is not int or not 0 < claim["train_rows"] <= MAX_ROWS
            or type(claim["payload_bytes"]) is not int or not 0 < claim["payload_bytes"] <= MAX_PACKET):
        raise ValueError("unexpected training-only staging claim")
    for role in ("binding", "rows"):
        check_pin(claim[role], MAX_CONTROL)


def validate_scope(claim, binding, rows_raw):
    data_root = PurePosixPath(claim["target"]) / "data"
    data = json.loads(binding)
    rows = [json.loads(line) for line in rows_raw.splitlines()]
    media = data["media"]
    if (data["role"] != "train" or data["target_root"] != data_root.as_posix()
            or data["quality_receipt_SHA256"] != claim["quality_receipt_SHA256"]
            or len(rows) != claim["train_rows"] or len(media) != len(rows)
            or len(set(pin["SHA256"] for pin in media.values())) != len(rows)):
        raise ValueError("training-only packet scope differs")
    for name, pin in media.items():
        if not FILE_PATTERN.fullmatch(name):
            raise ValueError("unexpected training image basename")
        check_pin(pin, MAX_IMAGE)
    expected_paths = {(data_root / "images" / name).as_posix() for name in media}
    paths = set()
    for row in rows:
        if (set(row) != {"group", "id", "images", "messages", "source", "split"}
                or row["split"] != "train" or row["source"] != "VizWiz-VQA"
                or len(row["images"]) != 1 or not SHA_PATTERN.fullmatch(row["group"])):
            raise ValueError("unexpected training row or admitted role")
        path, = row["images"]
        name = PurePosixPath(path).name
        messages = row["messages"]
        if (path not in expected_paths or path in paths or len(messages) != 2
                or row["group"] != sha256(name.encode()).hexdigest()
                or any(set(message) != {"role", "content"} for message in messages)
                or [message["role"] for message in messages] != ["user", "assistant"]
                or not messages[0]["content"].startswith("<image>\n") or not messages[1]["content"].strip()):
            raise ValueError("image binding or train supervision differs")
        paths.add(path)
    if paths != expected_paths or binding_size(claim, media) != claim["payload_bytes"]:
        raise ValueError("unexpected role, row or byte denominator")
    return media


def binding_size(claim, media):
    return sum(claim[field]["bytes"] for field in ("binding", "rows")) + sum(m["bytes"] for m in media.values())


def install(root, claim, stream, *, clock=time.monotonic):
    check_claim(claim)
    deadline = clock() + BUDGET
    binding, rows = (checked_member(stream, claim[k]) for k in ("binding", "rows"))
    media = validate_scope(claim, binding, rows)
    if not root.parent.is_dir() or root.parent.resolve(strict=True) != root.parent:
        raise ValueError("the permitted stage parent is missing or aliased")
    root.mkdir(mode=0o700)
    save(root / "stage-intent.json", encode(dict(request=claim["request"], replay_allowed=False,
                                               quality_receipt_SHA256=claim["quality_receipt_SHA256"])))
    pending = root / ".data-incoming"
    pending.mkdir(mode=0o700)
    (pending / "images").mkdir(mode=0o700)
    completed = 0
    try:
        save(pending / "train-swift.jsonl", rows)
        save(pending / "data-binding-private.json", binding)
        completed = 2
        for name, pin in sorted(media.items()):
            if clock() >= deadline:
                raise TimeoutError("original staging budget expired")
            save(pending / "images" / name, checked_member(stream, pin))
            completed += 1
        if stream.read(1) or clock() >= deadline:
            raise ValueError("unexpected packet ending")
        pending.rename(root / "data")
    except (OSError, ValueError) as error:
        save(root / "stage-outcome.json", encode(dict(request=claim["request"], state="FAILED_RETAINED",
                                                    completed=completed, error=type(error).__name__)))
        raise
    evidence = dict(request=claim["request"], state="DATA_STAGED", role="train",
                    completed=completed, train_rows=claim["train_rows"], payload_bytes=claim["payload_bytes"],
                    slurm=0, gpu=0, optimizer_updates=0, training_READY=False)
    save(root / "stage-outcome.json", encode(evidence))
    return evidence


def expired(*_):
    raise TimeoutError("the original receiver budget expired")


def receive(claim):
    os.umask(0o077)
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(BUDGET)
    print(json.dumps(install(TARGET, claim, sys.stdin.buffer), sort_keys=True), flush=True)


if __name__ == "__main__":
    receive(CLAIM)
