"""Receive a single, checksum-bound official CLIP-L snapshot on the project host."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys


REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
MEMBERS = ("config.json", "merges.txt", "special_tokens_map.json", "tokenizer.json",
           "tokenizer_config.json", "vocab.json", "model.safetensors")
MODEL_PIN = {"bytes": 1710540580,
             "SHA256": "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb"}
CHUNK_BYTES = 512 * 1024
PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LEAF = "training-hymotion-official-clip-L-32bd642-v1"
BUDGET_SECONDS = 1500


def validate_manifest(claim):
    if (claim["revision"] != REVISION or claim["destination"] != (PROJECT / "candidates" / LEAF).as_posix()
            or claim["budget_seconds"] != BUDGET_SECONDS or claim["sequence"] != list(MEMBERS)
            or set(claim["files"]) != set(MEMBERS) or claim["files"]["model.safetensors"] != MODEL_PIN
            or not re.fullmatch(r"[0-9a-f]{32}", claim["request"])):
        raise ValueError("unexpected CLIP-L scope")
    for name in MEMBERS:
        pin = claim["files"][name]
        maximum = MODEL_PIN["bytes"] if name == "model.safetensors" else 3_000_000
        if (type(pin["bytes"]) is not int or not 0 < pin["bytes"] <= maximum
                or not re.fullmatch(r"[a-f0-9]{64}", pin["SHA256"])):
            raise ValueError("invalid member length or digest")


def save_json(path, body):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def receive_exact(source, target, expected):
    digest = sha256()
    remaining = expected["bytes"]
    while remaining:
        block = source.read(min(CHUNK_BYTES, remaining))
        if not block:
            raise EOFError("incomplete official member")
        target.write(block)
        digest.update(block)
        remaining -= len(block)
    if digest.hexdigest() != expected["SHA256"]:
        raise ValueError("official member digest mismatch")
    return {"bytes": expected["bytes"], "SHA256": digest.hexdigest()}


def emit(event, request, **details):
    print(json.dumps(dict(event=event, request=request, **details), sort_keys=True), flush=True)


def completion_marker(request):
    return ("CLIP_L_READY " + request + "\n").encode("ascii")


def install(claim, claim_pin, source):
    validate_manifest(claim)
    parent = (PROJECT / "candidates").resolve(strict=True)
    destination, request = parent / LEAF, claim["request"]
    stage = parent / ("." + LEAF + ".incoming-" + request)
    lock = parent / ("." + LEAF + ".exclusive")
    need = sum(pin["bytes"] for pin in claim["files"].values())
    if destination.exists() or stage.exists() or lock.exists():
        raise FileExistsError("original installation namespace is already reserved")
    if shutil.disk_usage(parent).free < need + 128 * 1024 * 1024:
        raise OSError("project storage has insufficient free capacity")
    lock.mkdir(mode=0o700)
    stage.mkdir(mode=0o700)
    save_json(stage / "intent.json", dict(request=request, revision=REVISION, claim_SHA256=claim_pin,
                                          planned_bytes=need, budget_seconds=BUDGET_SECONDS))
    emit("READY", request, require_bytes=need)
    observed = {}
    try:
        for name in MEMBERS:
            with (stage / name).open("xb") as target:
                observed[name] = receive_exact(source, target, claim["files"][name])
                target.flush()
                os.fsync(target.fileno())
        marker = completion_marker(request)
        if source.read(len(marker)) != marker:
            raise ValueError("sender did not confirm verification")
        if source.read(1):
            raise ValueError("unexpected trailing stream bytes")
        receipt = dict(request=request, revision=REVISION, claim_SHA256=claim_pin, files=observed,
                       installed_bytes=need, adopted_encoder_rows=0)
        save_json(stage / "receipt.json", receipt)
        if destination.exists():
            raise FileExistsError("installation destination became occupied")
        os.rename(stage, destination)
    except (OSError, TimeoutError, ValueError, EOFError) as error:
        if not stage.is_dir():
            raise
        save_json(stage / "failure.json", dict(
            request=request, error=type(error).__name__, verified_members=observed))
        emit("FAILED_RETAINED", request, error=type(error).__name__, verified_members=list(observed))
        return 1
    emit("COMPLETE", request, receipt_SHA256=sha256((destination / "receipt.json").read_bytes()).hexdigest(),
         installed_bytes=need)
    return 0


def expired(*_):
    raise TimeoutError("original receive budget expired")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(BUDGET_SECONDS)
    sys.exit(install(CLAIM, CLAIM_SHA256, sys.stdin.buffer))
