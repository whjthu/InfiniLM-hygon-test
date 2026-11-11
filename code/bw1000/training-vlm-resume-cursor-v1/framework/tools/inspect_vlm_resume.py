"""Inspect pinned Trainer JSON and plan an epoch-boundary VLM resume on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vlm" / "runtime"))
from resume_cursor import plan_epoch_boundary_resume, read_pinned_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trainer-state", "saved-args"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    for name in ("checkpoint-step", "max-steps", "world-size", "per-device-batch",
                 "gradient-accumulation-steps", "dataloader-batches-per-rank"):
        parser.add_argument(f"--{name}", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        state = read_pinned_json(args.trainer_state, args.trainer_state_sha256,
                                 maximum_bytes=65536)
        saved = read_pinned_json(args.saved_args, args.saved_args_sha256,
                                 maximum_bytes=262144)
        result = plan_epoch_boundary_resume(
            state, saved, checkpoint_step=args.checkpoint_step, max_steps=args.max_steps,
            world_size=args.world_size, per_device_batch=args.per_device_batch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            dataloader_batches_per_rank=args.dataloader_batches_per_rank)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    result["source_identities"] = {
        "trainer_state_SHA256": args.trainer_state_sha256,
        "saved_args_SHA256": args.saved_args_sha256,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
