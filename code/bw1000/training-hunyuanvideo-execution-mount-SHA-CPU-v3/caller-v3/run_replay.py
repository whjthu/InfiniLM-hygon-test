"""Bind T09.sft to the audited private inputs immediately before the native replay."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from check_hunyuan_sft_replay_inputs import CACHE, MODEL, REPLAY, inspect

LAUNCH_SHA256 = "3fdd29e7ea1a09fc54872c044699199cd5ef16bd73af6e7d522fe2ff1b32662f"
ENTRY_SHA256 = "8236aeb91210b0ec971b7470b78775519657b0b6049d890f2146451a6b9dc0c3"
NATIVE_SOURCE = Path("/t09-native/hunyuanvideo-1.5")


def read_pinned(path, maximum, digest):
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Pinned source differs: {path.name}")
    return raw


def remaining(deadline):
    seconds = deadline - time.time()
    if not math.isfinite(seconds) or not 0 < seconds <= 3600:
        raise ValueError("Native replay deadline is expired or outside its epoch")
    return seconds


def prepare(args):
    remaining(args.deadline)
    if not re.fullmatch("[0-9]+", args.card):
        raise ValueError("Expected one numeric Slurm-visible card index")
    if not args.workspace.is_dir() or args.workspace.is_symlink():
        raise ValueError("Expected an existing private run directory")
    replay = args.results / REPLAY
    weights = args.results / MODEL / "hunyuanvideo-1.5"
    cache = args.results / CACHE / "sample.pt"
    source = NATIVE_SOURCE
    if not source.is_dir() or source.is_symlink() or not (source / "hyvideo/__init__.py").is_file():
        raise ValueError("Canonical HunyuanVideo source is missing")
    entry = args.framework / "tools/hygon_hunyuanvideo_first_update.py"
    original = json.loads(read_pinned(args.launch, 16384, LAUNCH_SHA256))
    read_pinned(entry, 32768, ENTRY_SHA256)
    report = inspect(args.results, args.audit, args.audit_sha256, execution_deadline=args.deadline,
                     execution_authority=args.execution_authority, execution_authority_sha=args.execution_authority_sha256)
    if not report["input_identity_ready"]:
        states = {name: row["state"] for name, row in report["files"].items()}
        states["transformer"] = report["transformer"]["state"]
        raise ValueError("Replay admission rejected: " + json.dumps(states, sort_keys=True))
    args.weight_admission = report["transformer"]
    available = remaining(args.deadline)
    asset_map = args.workspace / "assets.json"
    assets = {
        original["weights"]: str(weights),
        original["source"]: str(source),
        original["config"]["training_cache"]: str(cache),
    }
    with asset_map.open("x", encoding="utf-8") as stream:
        json.dump(assets, stream, indent=2)
        stream.write("\n")
    return [
        sys.executable, "-B", str(entry),
        "--capture-root", str(args.workspace / "capture"),
        "--capture-timeout", str(available),
        "--a100-launch", str(args.launch),
        "--capture-accelerator", "hip",
        "--replay-root", str(replay),
        "--run",
        "--asset-map", str(asset_map),
        "--cards", args.card,
        "--source-root", str(source.parent),
        "--output-root", str(args.workspace / "outputs"),
        "--run-name", "fixed-a100-replay",
        "--lock-root", str(args.lock_root),
        "--lock-host", args.lock_host,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "workspace", "framework", "launch", "audit", "lock-root", "execution-authority"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("audit-sha256", "card", "lock-host", "execution-authority-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    argv = prepare(args)
    argv[argv.index("--capture-timeout") + 1] = str(remaining(args.deadline))
    os.execv(argv[0], argv)


if __name__ == "__main__":
    main()
