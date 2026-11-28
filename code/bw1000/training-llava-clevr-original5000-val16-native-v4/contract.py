"""Exact answerless CLEVR16 input and a 32-slot durable prediction ledger."""

from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import stat


ARMS = ("base", "cp25")
SLOTS = tuple(range(16))
CAP = 128
PROMPT = ("<image>List the visible objects by color and shape with their counts. "
          "Only include objects that are present.")
OUTCOMES = {"eos", "cap", "stop", "error", "missing"}
HEX = re.compile(r"[0-9a-f]{64}")


def need(condition, code):
    if not condition:
        raise ValueError(code)


def sha(raw):
    return sha256(raw).hexdigest()


def pairs(items):
    result = dict(items)
    need(len(result) == len(items), "DUPLICATE_JSON_KEY")
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=pairs)


def wire(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def put(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def identity(info):
    return (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read(path, limit, expected=None, size=None):
    need(path.is_absolute() and path.resolve(strict=True) == path, "NONCANONICAL_FILE")
    prior = path.lstat()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        old = os.fstat(stream.fileno())
        need(stat.S_ISREG(prior.st_mode) and stat.S_ISREG(old.st_mode) and
             (prior.st_dev, prior.st_ino, prior.st_size) == (old.st_dev, old.st_ino, old.st_size)
             and 0 < old.st_size <= limit and (size is None or old.st_size == size), "INVALID_MEMBER_EXTENT")
        raw = stream.read(limit + 1)
        need(len(raw) == old.st_size and (old.st_dev, identity(old)) ==
             (os.fstat(stream.fileno()).st_dev, identity(os.fstat(stream.fileno())))
             and (prior.st_dev, identity(prior)) == (path.lstat().st_dev, identity(path.lstat())),
             "MUTATED_MEMBER")
    need(expected is None or HEX.fullmatch(expected) and sha(raw) == expected, "MEMBER_HASH_DIFFERS")
    return raw


def inputs(root, request_sha, image_manifest_sha):
    from PIL import Image

    need(root.resolve(strict=True) == root and {item.name for item in root.iterdir()} ==
         {"requests.jsonl", "IMAGES.json", "images"}, "EXACT_ANSWERLESS_CLOSURE")
    image_root = root / "images"
    need(image_root.resolve(strict=True) == image_root and {item.name for item in image_root.iterdir()} ==
         {f"{slot:02d}.png" for slot in SLOTS}, "EXACT_IMAGE_DIRECTORY")
    requests = [parse(line) for line in read(root / "requests.jsonl", 48 << 10, request_sha).splitlines()]
    pins = parse(read(root / "IMAGES.json", 16 << 10, image_manifest_sha))
    need(type(pins) is list and len(requests) == len(pins) == len(SLOTS), "EXACT_16_SLOTS")
    seen = set()
    for slot, row, pin in zip(SLOTS, requests, pins, strict=True):
        name = f"images/{slot:02d}.png"
        need(type(row) is dict and type(row.get("slot")) is int and row ==
             dict(slot=slot, images=[name], messages=[dict(role="user", content=PROMPT)]), "NATIVE_CLEVR_PROMPT_OR_SLOT")
        need(type(pin) is dict and set(pin) == {"slot", "name", "bytes", "SHA256"}
             and type(pin["slot"]) is int and pin["slot"] == slot
             and pin["name"] in (name, f"{slot:02d}.png")
             and type(pin["bytes"]) is int and 0 < pin["bytes"] <= 2 << 20
             and type(pin["SHA256"]) is str and HEX.fullmatch(pin["SHA256"])
             and pin["SHA256"] not in seen, "IMAGE_PIN_SLOT_OR_DUPLICATE")
        seen.add(pin["SHA256"])
        blob = read(root / name, 2 << 20, pin["SHA256"], pin["bytes"])
        with Image.open(io.BytesIO(blob)) as image:
            need(image.format == "PNG" and image.size == (480, 320) and image.mode in ("RGB", "RGBA"),
                 "EXACT_ORIGINAL_CLEVR_IMAGE")
            image.load()
    return requests


def generation(ids, eos, pad, finish, echoed):
    need(type(ids) is list and 0 < len(ids) <= CAP and all(type(item) is int and item >= 0 for item in ids)
         and bool(eos) and all(type(item) is int and item >= 0 for item in eos), "NATIVE_CONTINUATION_IDS")
    real = [token for token in ids if token != pad or token in eos]
    need(type(echoed) is list and (echoed == real or echoed == [item for item in ids if item != pad]),
         "DECODED_TOKEN_IDS_DIFFER")
    for index, token in enumerate(ids):
        if token in eos:
            need(finish in ("stop", "eos") and all(x == pad for x in ids[index+1:]), "NATIVE_EOS_BOUNDARY")
            return "eos"
    if len(ids) == CAP and finish == "length":
        return "cap"
    need(len(ids) < CAP and finish == "stop", "NATIVE_STOP_OR_CAP_UNPROVEN")
    return "stop"


def record(root, row):
    arm, slot, outcome, text = row["arm"], row["slot"], row["outcome"], row["text"]
    need(set(row) == {"arm", "slot", "outcome", "text"} and arm in ARMS and type(slot) is int and slot in SLOTS
         and outcome in OUTCOMES - {"missing"} and ((type(text) is str and len(text) <= 8192)
             if outcome in {"eos", "cap", "stop"} else text is None), "NATIVE_LEDGER_ROW")
    put(root / f"{arm}-{slot:02d}.private.json", wire(row))


def seal(root, request, facts, failure):
    rows = []
    for arm in ARMS:
        for slot in SLOTS:
            member = root / f"{arm}-{slot:02d}.private.json"
            row = dict(arm=arm, slot=slot, outcome="missing", text=None)
            if member.exists():
                row = parse(read(member, 64 << 10))
                need(row.get("arm") == arm and row.get("slot") == slot
                     and row.get("outcome") in OUTCOMES - {"missing"}, "LEDGER_READBACK_IDENTITY")
            rows.append(row)
    encoded = b"".join(wire(row) for row in rows)
    put(root / "predictions.jsonl", encoded)
    counts = {arm: {state: sum(row["arm"] == arm and row["outcome"] == state for row in rows)
                    for state in sorted(OUTCOMES)} for arm in ARMS}
    passed = (failure is None and facts["model_loads"] == 1 and facts["actual_generate_calls"] == 32
              and facts["adapter_tensors_verified"] == 504 and facts["paired_equal_native_inputs"] == 16
              and all(not arm_counts["error"] and not arm_counts["missing"] for arm_counts in counts.values()))
    result = dict(request=request, planned=32, rows_per_arm=16, counts=counts, passed=passed,
                  predictions_SHA256=sha(encoded), predictions_bytes=len(encoded), failure=failure,
                  facts=facts, scoring_performed=False, gold_opened=False, formal_domain_increment=0)
    put(root / "result.json", wire(result))
    return result
