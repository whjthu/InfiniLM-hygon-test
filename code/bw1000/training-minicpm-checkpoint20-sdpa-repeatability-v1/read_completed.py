"""Read fixed original MiniCPM references and the four new native forward receipts."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from route_control import ARMS, comparison, require


ORIGINAL_READER = (
    ("compare_first.py", 6593, "e11c8b7b0955be3ea4b3e35dc8d747cb7edfbf51f62a0416ffad765b406200dd"),
    ("probe.py", 10995, "1d340abb7659f30cb625fd21164f47168b16267fccf86f9cb8da73edcc6bfdc4"),
)


def read(path, cap, expected=None):
    require(path.resolve(strict=True) == path, "aliased evidence file")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    pin = sha256(raw).hexdigest()
    require(0 < len(raw) <= cap and (expected is None or pin == expected), "evidence extent or pin differs")
    return raw, pin


def read_arms(root, step):
    records, sources, failed, contexts = {}, {}, [], []
    for arm in ARMS:
        for rank in range(8):
            relative = Path(arm) / "train/minicpm-usgs-witness" / f"rank-{rank:02d}"
            try:
                values = {}
                for suffix, limit in (("state", 2 * 1024**2), ("ready", 4096), ("stop", 4096)):
                    name = relative / f"first-forward-{suffix}.json"
                    raw, pin = read(root / name, limit)
                    values[suffix] = json.loads(raw)
                    sources[name.as_posix()] = pin
                ready, stop = values["ready"], values["stop"]
                require(ready["run"]["actual_step"] == step and ready["rank"] == rank and ready["world_size"] == 8
                        and ready["checkpoint_step"] == 20 and ready["state"] == "FORWARD_RETURNED",
                        "cohort ready identity changed")
                require(stop == {"rank": rank, "state": "ALL_RANKS_STOPPED", "global_step": 20,
                                 "backward_calls": 0, "optimizer_updates": 0}, "cohort stop identity changed")
                records[(rank, arm)] = values["state"]
                contexts.append(ready["run"])
            except (OSError, KeyError, TypeError, ValueError) as error:
                failed.append({"arm": arm, "rank": rank, "error": type(error).__name__})
    require(not contexts or all(row == contexts[0] for row in contexts), "shared run or checkpoint configuration changed")
    return records, sources, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "original-root", "original-public", "original-observer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--actual-gpu-step", required=True)
    args = parser.parse_args()
    observer = args.original_observer.resolve(strict=True)
    for name, cap, pin in ORIGINAL_READER:
        read(observer / name, cap, pin)
    sys.path.insert(0, str(observer))
    from compare_first import original_rows

    history = original_rows(args.original_root.resolve(strict=True), args.original_public.resolve(strict=True))
    records, sources, failed = read_arms(args.root.resolve(strict=True), args.actual_gpu_step)
    result = comparison(records, args.actual_gpu_step, history)
    result.update(native_state_receipts_SHA256=sources, missing_or_invalid_receipt_files=failed,
                  external_accounting_and_post_required=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in
                      ("actual_step", "planned_forwards", "verified_forwards", "comparable_ranks", "complete")}))
    return int(not result["complete"])


if __name__ == "__main__":
    raise SystemExit(main())
