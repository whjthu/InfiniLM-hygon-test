"""Read and verify the retained CLIP-L text ranges, then build one new text-only artifact."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import sys
import time

import assemble_hymotion_clip_text_safetensors as projector


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
OLD_LEAF = "training-hymotion-official-clip-L-32bd642-v1"
OLD_REQUEST = "98ab7aeb09cf4ba6a12071cbb6b941e7"
NEW_LEAF = "training-hymotion-clip-text-only-32bd642-v1"
OLD_INTENT = "bce538e4253c516ca62e726a93e92cf78ba6f8fd01c5f980cd28f322691f2026"
OLD_FAILURE = "73cc7229e83f61ce00ba4cdd43040896c39368cea806fd3889e2a084351378b8"
REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
RETAINED_BYTES = 472383488
HEADER_BYTES = 71856
TOTAL_TEXT_BYTES = 492242536
BUDGET = 230
METADATA = ("config.json", "merges.txt", "special_tokens_map.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json")


def sig(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def save(path, raw):
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())


def emit(event, request, **fields):
    print(json.dumps(dict(event=event, request=request, **fields), sort_keys=True), flush=True)


def check_claim(claim):
    if (claim["revision"] != REVISION or claim["original_request"] != OLD_REQUEST
            or claim["source_failure_SHA256"] != OLD_FAILURE or claim["source_intent_SHA256"] != OLD_INTENT
            or claim["destination"] != (PROJECT / "candidates" / NEW_LEAF).as_posix()
            or claim["receiver_seconds"] != BUDGET or claim["retained_end"] != RETAINED_BYTES
            or claim["official_bytes"] != 1710540580 or claim["text_tensor_count"] != 197
            or claim["official_intervals"] != [[71856,72472],[74532,492316452]]
            or not re.fullmatch("[0-9a-f]{32}", claim["request"])
            or set(claim["metadata"]) != set(METADATA) or claim["replay_allowed"] is not False
            or claim["gpu"] != 0 or claim["slurm"] != 0 or claim["original_scope_write_allowed"] is not False
            or claim["preflight_required"] is not True or claim["whole_official_checkpoint_verified"] is not False):
        raise ValueError("unexpected official text-only recovery scope")
    expected = [[71856,72472],[74532,RETAINED_BYTES],[RETAINED_BYTES,492316452]]
    ranges = claim["source_ranges"]
    if ([u["interval"] for u in ranges] != expected
            or any(u["bytes"] != u["interval"][1] - u["interval"][0] for u in ranges)
            or not all(re.fullmatch("[0-9a-f]{64}", u["SHA256"]) for u in ranges)):
        raise ValueError("unexpected official text ranges")


def assert_directory(path):
    entry = path.lstat()
    if (not stat.S_ISDIR(entry.st_mode) or entry.st_uid != os.getuid()
            or path.resolve(strict=True) != path):
        raise ValueError("expected the exact project owned directory")


def exact_file(path, size, limit=None):
    entry = path.lstat()
    if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.getuid() or path.resolve(strict=True) != path
            or entry.st_size != size or (limit is not None and size > limit)):
        raise ValueError("unexpected project retained file")
    return entry


def signature(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def read_small(path, pin):
    entry = exact_file(path, pin["bytes"], 4 * 1024 * 1024)
    with path.open("rb") as stream:
        value = stream.read(pin["bytes"] + 1)
        if signature(os.fstat(stream.fileno())) != signature(entry):
            raise ValueError("retained metadata changed while it was read")
    if sig(value) != pin:
        raise ValueError("original retained metadata has a different digest")
    return value


def origin(claim):
    base = PROJECT / "candidates"
    if base.resolve(strict=True) != base or not base.is_dir():
        raise ValueError("expected the authorized candidate storage root")
    old = base / ("." + OLD_LEAF + ".incoming-" + OLD_REQUEST)
    lock = base / ("." + OLD_LEAF + ".exclusive")
    assert_directory(old)
    assert_directory(lock)
    if any(p.exists() or p.is_symlink() for p in (base / OLD_LEAF, old / "receipt.json")):
        raise ValueError("the original timeout has changed state")
    intent = json.loads(read_small(old / "intent.json", dict(bytes=241, SHA256=OLD_INTENT)))
    failure = json.loads(read_small(old / "failure.json", dict(bytes=783, SHA256=OLD_FAILURE)))
    if (intent["request"] != OLD_REQUEST or failure["request"] != OLD_REQUEST or failure["error"] != "TimeoutError"
            or failure["verified_members"] != claim["metadata"]):
        raise ValueError("unexpected prior transfer failure or verified metadata")
    return base, old


class DigestSink:
    def write(self, block):
        return len(block)


class BoundWriter:
    def __init__(self, file):
        self.file = file
        self.digest = sha256()
        self.bytes = 0

    def write(self, block):
        n = self.file.write(block)
        if n:
            self.digest.update(memoryview(block)[:n])
            self.bytes += n
        return n


def prefix_read(claim, where, deadline):
    original = where / "model.safetensors"
    known = exact_file(original, RETAINED_BYTES)
    with original.open("rb") as source:
        if signature(known) != signature(os.fstat(source.fileno())):
            raise ValueError("retained weight file changed before verification")
        for member in claim["source_ranges"][:2]:
            projector.copy_checked(DigestSink(), source, member["interval"][0],
                                   member["bytes"], member["SHA256"], deadline)
        if (signature(known) != signature(os.fstat(source.fileno()))
                or signature(known) != signature(original.lstat())):
            raise ValueError("retained weight file changed during verification")
    return {"verified_prefix_intervals": claim["source_ranges"][:2], "retained_file_bytes": known.st_size,
            "descriptor_identity": list(signature(known))}


def preflight(claim):
    check_claim(claim)
    before = time.monotonic()
    _, path = origin(claim)
    retained = prefix_read(claim, path, before + 60)
    metadata = {name: sig(read_small(path / name, pin)) for name, pin in claim["metadata"].items()}
    emit("PREFLIGHT_COMPLETE", claim["request"], metadata=metadata, **retained,
         elapsed_seconds=time.monotonic() - before, writes=0, gpu=0, slurm=0)


def install(claim, claim_sha, client_preflight, stream):
    check_claim(claim)
    os.umask(0o077)
    start = time.monotonic()
    base, old = origin(claim)
    if (client_preflight["request"] != claim["request"] or client_preflight["metadata"] != claim["metadata"]
            or client_preflight["verified_prefix_intervals"] != claim["source_ranges"][:2]
            or client_preflight["retained_file_bytes"] != RETAINED_BYTES):
        raise ValueError("independent retained text prefix was not qualified")
    target = base / NEW_LEAF
    pending = base / ("." + NEW_LEAF + ".incoming-" + claim["request"])
    if any(p.exists() or p.is_symlink() for p in (target, pending)):
        raise FileExistsError("original text-only derivative already exists")
    needed = TOTAL_TEXT_BYTES + sum(p["bytes"] for p in claim["metadata"].values()) + 1024 * 1024
    if shutil.disk_usage(base).free < needed:
        raise OSError("insufficient authorized project storage")
    path = old / "model.safetensors"
    known = exact_file(path, RETAINED_BYTES)
    if list(signature(known)) != client_preflight["descriptor_identity"]:
        raise ValueError("retained descriptor changed after independent preflight")
    pending.mkdir(mode=0o700)
    intent = dict(request=claim["request"], claim_SHA256=claim_sha, derivative="CLIP_L_TEXT_ONLY",
                  old_scope_preserved=True, replay_allowed=False)
    save(pending / "intent.json", json_bytes(intent))
    emit("READY", claim["request"], header_bytes=HEADER_BYTES, tail_bytes=claim["source_ranges"][-1]["bytes"])
    try:
        header = stream.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES or sha256(header).hexdigest() != claim["official_header_SHA256"]:
            raise ValueError("transferred official header differs")
        with path.open("rb") as original, (pending / "model.safetensors").open("xb") as out:
            if signature(os.fstat(original.fileno())) != signature(known):
                raise ValueError("retained source changed before assembly")
            output = BoundWriter(out)
            observed = projector.assemble(output, original, stream, header, claim, start + BUDGET)
            if (signature(os.fstat(original.fileno())) != signature(known)
                    or signature(path.lstat()) != signature(known)
                    or output.bytes != observed["file_bytes"]):
                raise ValueError("retained source or derivative extent changed")
            out.flush()
            os.fsync(out.fileno())
        for name, pin in claim["metadata"].items():
            save(pending / name, read_small(old / name, pin))
        receipt = dict(intent, **observed, official_source_ranges=claim["source_ranges"],
                       output_model={"bytes": output.bytes, "SHA256": output.digest.hexdigest()},
                       metadata=claim["metadata"], gpu=0, slurm=0, encoder_rows=0, training_READY=False)
        raw = json_bytes(receipt)
        save(pending / "receipt.json", raw)
        if target.exists() or target.is_symlink():
            raise FileExistsError("the text-only target became occupied")
        pending.rename(target)
    except (OSError, ValueError, EOFError) as error:
        save(pending / "failure.json", json_bytes(dict(intent, error=type(error).__name__)))
        emit("FAILED_RETAINED", claim["request"], error=type(error).__name__)
        return 1
    emit("COMPLETE", claim["request"], derivative_bytes=output.bytes, receipt_SHA256=sha256(raw).hexdigest())
    return 0


def expired(*_):
    raise TimeoutError("the original bounded CLIP text derivative exceeded its deadline")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(65 if MODE == "preflight" else BUDGET)
    if MODE == "preflight":
        preflight(CLAIM)
    else:
        sys.exit(install(CLAIM, CLAIM_SHA256, PREFLIGHT, sys.stdin.buffer))
