"""Install only the five frozen VizWiz audit inputs under the project result root."""

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import sys
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/"
            "training-gemma-vizwiz-five-native-input-audit-v1")
BINDING = {"bytes": 2743, "SHA256": "389c4d3d0284b153f25800ae66e78401b26b4c209ce3571f5cba2999c5fdee85"}
ROWS = {"bytes": 2164, "SHA256": "be03f5443f6158f1b8358410632100b88cc15e0bba6649ecebd54648cbb88717"}
MEDIA_BYTES = 1405423
BUDGET = 90


def identity(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def save(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def as_json(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def take(stream, pin):
    data = stream.read(pin["bytes"])
    if identity(data) != pin:
        raise ValueError("packet member checksum or length differs")
    return data


def manifest(binding_raw, rows_raw, data_root):
    if identity(binding_raw) != BINDING or identity(rows_raw) != ROWS:
        raise ValueError("frozen data control file differs")
    binding = json.loads(binding_raw)
    rows = [json.loads(line) for line in rows_raw.splitlines()]
    media = binding["media"]
    if (binding["target_root"] != data_root.as_posix()
            or binding["target_rows"] != ROWS or [binding["planned_rows"], binding["bound_rows"]] != [6, 5]
            or len(binding["omitted"]) != 1 or len(media) != 5 or len(rows) != 5):
        raise ValueError("frozen pilot denominator or scope differs")
    for name, info in media.items():
        if (not re.fullmatch(r"VizWiz_train_\d{8}\.jpg", name)
                or set(info) != {"bytes", "SHA256", "source_member", "target_path"}
                or not re.fullmatch(r"[0-9a-f]{64}", info["SHA256"])
                or type(info["bytes"]) is not int or not 0 < info["bytes"] <= 384 * 1024
                or info["source_member"] != "images/" + name
                or info["target_path"] != (data_root / "images" / name).as_posix()):
            raise ValueError("frozen media path or bound differs")
    if (sum(x["bytes"] for x in media.values()) != MEDIA_BYTES
            or {r["images"][0] for r in rows if r["split"] == "train" and len(r["images"]) == 1}
            != {x["target_path"] for x in media.values()}):
        raise ValueError("frozen five train media members differ")
    return media


def check_claim(claim):
    if (claim["target"] != ROOT.as_posix() or claim["max_seconds"] != BUDGET or claim["files"] != 7
            or claim["media_bytes"] != MEDIA_BYTES or claim["gpu"] != 0 or claim["slurm"] != 0
            or claim["binding"] != BINDING or claim["rows"] != ROWS
            or not re.fullmatch(r"[0-9a-f]{32}", claim["request"])):
        raise ValueError("unexpected bounded staging claim")


def install(root, request, stream, scope_data_root, *, clock=time.monotonic):
    if not root.parent.is_dir() or root.parent.resolve(strict=True) != root.parent:
        raise ValueError("result parent is missing or aliased")
    deadline = clock() + BUDGET
    binding = take(stream, BINDING)
    rows = take(stream, ROWS)
    media = manifest(binding, rows, scope_data_root)
    root.mkdir(mode=0o700)
    scratch = root / (".data-incoming-" + request)
    scratch.mkdir(mode=0o700)
    (scratch / "images").mkdir(mode=0o700)
    save(root / "stage-intent.json", as_json(dict(request=request, replay_allowed=False)))
    completed = 0
    try:
        save(scratch / "data-binding-private.json", binding)
        save(scratch / "train-swift.jsonl", rows)
        completed = 2
        for name, wanted in sorted(media.items()):
            if clock() >= deadline:
                raise TimeoutError("staging file budget expired")
            content = take(stream, {key: wanted[key] for key in ("bytes", "SHA256")})
            save(scratch / "images" / name, content)
            completed += 1
        if stream.read(1) or clock() >= deadline:
            raise ValueError("unexpected packet ending")
        os.rename(scratch, root / "data")
    except (OSError, ValueError) as error:
        save(root / "stage-outcome.json", as_json(dict(
            state="FAILED_RETAINED", request=request, completed=completed, error=type(error).__name__)))
        raise
    report = dict(request=request, state="DATA_STAGED", completed=completed,
                  payload_bytes=MEDIA_BYTES + BINDING["bytes"] + ROWS["bytes"], slurm=0,
                  gpu=0, optimizer_updates=0, training_READY=False)
    save(root / "stage-outcome.json", as_json(report))
    return report


def expired(*_):
    raise TimeoutError("original staging budget expired")


def receive(claim, stream):
    check_claim(claim)
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(BUDGET)
    os.umask(0o077)
    report = install(ROOT, claim["request"], stream, PurePosixPath(ROOT.as_posix()) / "data")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    receive(CLAIM, sys.stdin.buffer)
