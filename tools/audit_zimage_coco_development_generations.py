"""Bind all 256 COCO development pairs to native render outcomes before scoring."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import re

from render_zimage_coco_development_pairs import (
    ADAPTER_SHA256,
    ARMS,
    CHECKPOINT_SHA256,
    DEVELOPMENT_COUNT,
    NOISE_SHA256,
    PROMPTS_SHA256,
    ROSTER_SHA256,
    read_inputs,
    require,
    shard_plan,
)


def digest(raw):
    return sha256(raw).hexdigest()


def read_bounded(path, limit):
    require(not path.is_symlink() and path.is_file(), "missing or aliased generation artifact")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "generation artifact exceeds its size bound")
    return raw


def plan_for(directory):
    raw = read_bounded(directory / "plan.jsonl", 4096)
    require(raw.endswith(b"\n") and len(raw.splitlines()) == 1, "expected one durable shard plan")
    plan = json.loads(raw)
    require(plan == shard_plan(plan["first"], plan["stop"]), "shard plan differs from native caller")
    return plan, digest(raw)


def verify_image(directory, record, expected):
    from PIL import Image

    name = f'{record["ordinal"]:03d}-{record["arm"]}.private.png'
    require(record["image_relative"] == name and record["initial_noise_SHA256"] == expected,
            "generated image path or actual noise differs")
    raw = read_bounded(directory / name, 1024 * 1024)
    require(len(raw) == record["image_bytes"] and digest(raw) == record["image_SHA256"],
            "generated image bytes differ")
    with Image.open(io.BytesIO(raw)) as image:
        require(image.format == "PNG" and image.mode == "RGB" and image.size == (256, 256)
                and digest(image.tobytes()) == record["RGB_SHA256"], "generated image pixels differ")
    return {"image": str(directory / name), "image_SHA256": record["image_SHA256"],
            "RGB_SHA256": record["RGB_SHA256"], "noise_SHA256": expected}


def read_shard(directory, plan, rows, replay):
    path = directory / "generations.jsonl"
    raw = read_bounded(path, 1024 * 1024) if path.exists() or path.is_symlink() else b""
    # A crash can leave an uncommitted suffix. The caller fsyncs each full record.
    complete = raw.split(b"\n")
    trailing = complete.pop()
    expected_order = [(index, arm) for index in range(plan["first"], plan["stop"])
                      for arm in (ARMS if index % 2 == 0 else ARMS[::-1])]
    require(len(complete) < len(expected_order) or len(complete) == len(expected_order) and not trailing,
            "extra generation events")
    outcomes = {}
    for line, identity in zip(complete, expected_order):
        record = json.loads(line)
        ordinal, arm = identity
        row = rows[ordinal]
        require((record["ordinal"], record["arm"]) == identity and record["image_id"] == row["image_id"]
                and record["prompt_SHA256"] == row["prompt_SHA256"], "generation event identity or order differs")
        require(type(record["completed"]) is bool, "invalid generation event state")
        if record["completed"]:
            require(record["seed"] == row["seed"] and record["inference_steps"] == 8
                    and record["guidance"] == 1.0 and record["height"] == record["width"] == 256,
                    "generation event recipe differs")
            _, expected = replay.sample(row["image_id"], row["prompt_SHA256"], row["seed"])
            outcomes[identity] = {"state": "COMPLETED", **verify_image(directory, record, expected)}
        else:
            require(re.fullmatch("[A-Za-z_][A-Za-z_0-9]*", record["error_type"]) is not None,
                    "invalid generation failure record")
            outcomes[identity] = {"state": "FAILED", "error_type": record["error_type"]}
    return outcomes, {"generations_SHA256": digest(raw), "uncertain_trailing_bytes": len(trailing)}


def audit(directories, rows, replay):
    require(len(rows) == DEVELOPMENT_COUNT, "invalid frozen development cohort")
    reserved, outcomes, sources = set(), {}, []
    for supplied in directories:
        require(not supplied.is_symlink(), "aliased shard root")
        directory = supplied.resolve(strict=True)
        plan, plan_sha = plan_for(directory)
        indices = set(range(plan["first"], plan["stop"]))
        require(not indices.intersection(reserved), "overlapping generation shards")
        reserved.update(indices)
        records, log = read_shard(directory, plan, rows, replay)
        outcomes.update(records)
        sources.append({"first": plan["first"], "stop": plan["stop"], "plan_SHA256": plan_sha, **log})

    pairs = []
    counts = {arm: dict.fromkeys(("COMPLETED", "FAILED", "MISSING"), 0) for arm in ARMS}
    for index, entry in enumerate(rows):
        pair = {"ordinal": index, "image_id": entry["image_id"], "prompt_SHA256": entry["prompt_SHA256"]}
        for arm in ARMS:
            record = outcomes.get((index, arm), {"state": "MISSING"})
            counts[arm][record["state"]] += 1
            pair[arm] = record
        pairs.append(pair)
    complete_pairs = sum(all(pair[arm]["state"] == "COMPLETED" for arm in ARMS) for pair in pairs)
    return {"schema": "training.zimage.coco.development_generation_inventory.v1",
            "scope": "COCO development only",
            "sources": sorted(sources, key=lambda entry: entry["first"]),
            "prompts_SHA256": PROMPTS_SHA256, "roster_SHA256": ROSTER_SHA256, "noise_manifest_SHA256": NOISE_SHA256,
            "step25_checkpoint_SHA256": CHECKPOINT_SHA256, "step25_adapter_SHA256": ADAPTER_SHA256,
            "planned_pairs": DEVELOPMENT_COUNT, "planned_outputs": DEVELOPMENT_COUNT * 2,
            "per_arm": counts, "complete_pairs": complete_pairs,
            "complete_development_inventory": complete_pairs == DEVELOPMENT_COUNT,
            "scored_images": 0, "formal_quality_accepted": False, "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prompts", "noise-manifest", "roster", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--shard", action="append", required=True, type=Path)
    args = parser.parse_args()
    rows, replay = read_inputs(args.prompts, args.noise_manifest, args.roster)
    result = audit(args.shard, rows, replay)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("planned_pairs", "planned_outputs", "per_arm",
                                                  "complete_pairs", "complete_development_inventory")},
                     sort_keys=True))
    return 0 if result["complete_development_inventory"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
