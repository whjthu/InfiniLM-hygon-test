"""Bind the 18 original OneVision files into one isolated model directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import time

PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
METADATA = PROJECT / "training-vlm-T03-T04-metadata-v2/llava/model"
OUTPUT = PROJECT / "training-llava-original-model18-v1"
HERE = Path(__file__).resolve().parent
SHARDS = {f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)}
BLOCK = 4 << 20


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def file_identity(s):
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def exact_regular(path):
    s = path.lstat()
    need(stat.S_ISREG(s.st_mode), "SOURCE_NOT_REGULAR")
    return s


def checked_path(root, path):
    need(path.is_absolute() and path.is_relative_to(root) and path == path.resolve(strict=True),
         "SOURCE_PATH_OUTSIDE_OR_NONCANONICAL")
    return path


def verify(path, pin, check_deadline):
    initial = exact_regular(path)
    need(initial.st_size == pin["bytes"], "ORIGINAL_FILE_SIZE")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        opened = os.fstat(f.fileno())
        need(file_identity(opened) == file_identity(initial), "SOURCE_REPLACED_BEFORE_OPEN")
        for chunk in iter(lambda: f.read(BLOCK), b""):
            check_deadline()
            digest.update(chunk)
        need(file_identity(os.fstat(f.fileno())) == file_identity(opened), "SOURCE_CHANGED_DURING_HASH")
    need(file_identity(exact_regular(path)) == file_identity(initial), "SOURCE_CHANGED_AFTER_HASH")
    need(digest.hexdigest() == pin["SHA256"], "ORIGINAL_FILE_SHA256")
    return file_identity(initial)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def input_mapping(plan, files, root, metadata):
    need(set(plan) == {"case", "weights"} and plan["case"] == "T04.sft"
         and set(plan["weights"]) == SHARDS and len(files) == 18 and SHARDS.issubset(files),
         "MISSING_ORIGINAL_FILE_MEMBERSHIP")
    sources = {}
    for name, pin in files.items():
        need(Path(name).name == name and type(pin["bytes"]) is int and pin["bytes"] > 0
             and len(pin["SHA256"]) == 64 and all(c in "0123456789abcdef" for c in pin["SHA256"]),
             "INVALID_OFFICIAL_FILE_PIN")
        if name in SHARDS:
            path = Path(plan["weights"][name])
            need(path.name == name, "WEIGHT_FINAL_NAME_REQUIRED")
        else:
            path = metadata / name
        sources[name] = checked_path(root, path)
    need(len(set(sources.values())) == 18, "DUPLICATE_SOURCE")
    return sources


def pinned_json(path, expected_sha256, limit):
    size = exact_regular(path).st_size
    need(1 <= size <= limit, "MANIFEST_SIZE")
    payload = path.read_bytes()
    need(len(payload) == size and hashlib.sha256(payload).hexdigest() == expected_sha256, "MANIFEST_HASH")
    return json.loads(payload)


def assemble(plan_path, plan_sha256, manifest_sha256, seconds):
    need(0 < seconds <= 600, "BOUND_ASSEMBLER_SECONDS")
    started = time.monotonic()

    def check_deadline():
        if time.monotonic() >= started+seconds:
            raise TimeoutError("MODEL_DIRECTORY_ASSEMBLER_DEADLINE")

    signal.signal(signal.SIGALRM, lambda *_: check_deadline())
    signal.setitimer(signal.ITIMER_REAL, seconds)
    plan = pinned_json(plan_path, plan_sha256, 65536)
    manifest = pinned_json(HERE / "MODEL_FILES.json", manifest_sha256, 65536)
    files = manifest["files"]
    need(manifest["case"] == "T04.sft", "EXACT_MODEL_CASE")
    sources = input_mapping(plan, files, PROJECT, METADATA)
    need(not os.path.lexists(OUTPUT), "ORIGINAL_OUTPUT_ALREADY_EXISTS")
    ids = {name: verify(path, files[name], check_deadline) for name, path in sources.items()}
    index = pinned_json(sources["model.safetensors.index.json"], files["model.safetensors.index.json"]["SHA256"], 1 << 20)
    need(len(index["weight_map"]) == 698 and set(index["weight_map"].values()) == SHARDS,
         "CHECKPOINT_INDEX_SHARD_MEMBERSHIP")
    check_deadline()
    OUTPUT.mkdir(mode=0o700)
    stage = OUTPUT / "model.incomplete"
    stage.mkdir(mode=0o700)
    for name, source in sources.items():
        check_deadline()
        need(file_identity(exact_regular(source)) == ids[name], "VERIFIED_SOURCE_CHANGED")
        target = stage / name
        os.link(source, target, follow_symlinks=False)
        need(file_identity(exact_regular(target)) == ids[name], "LINK_IDENTITY")
    sync_dir(stage)
    check_deadline()
    model = OUTPUT / "model"
    stage.rename(model)
    sync_dir(OUTPUT)
    need({p.name for p in model.iterdir()} == set(files) and
         all(file_identity(exact_regular(model/name)) == ids[name] for name in files), "FINAL_MODEL_FILE_SET")
    receipt = dict(case="T04.sft", verified_original_files=18, verified_complete_shards=4,
                   hash_bytes=sum(f["bytes"] for f in files.values()), copied_weight_bytes=0,
                   plan_SHA256=plan_sha256, manifest_SHA256=manifest_sha256,
                   hardlinked_model_directory=True, actual_GPU_steps=0, model_loaded=False,
                   initial_trainables_qualified=False, runtime_loader_qualified=False,
                   wall_seconds=time.monotonic()-started)

    receipt["files"] = {name: dict(files[name], device=ids[name][0], inode=ids[name][1],
                                 mtime_ns=str(ids[name][3])) for name in sorted(files)}
    with (OUTPUT / "receipt.json").open("x", encoding="utf8") as stream:
        json.dump(receipt, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(OUTPUT)
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(receipt, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--seconds", type=int, required=True)
    args = parser.parse_args()
    assemble(args.plan, args.plan_sha256, args.manifest_sha256, args.seconds)
