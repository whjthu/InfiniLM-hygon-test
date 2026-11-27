"""Freeze a 96-image secondary GQA relation evaluation without answer-based selection."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from build_gqa_relation_train import deflate_parts, load_pinned, question_rows
from materialize_gqa_relation_pilot import (
    HOLDOUT, OLD, ZipIndex, document, image_identity, near_duplicate, read_jpeg, select_manifest, write,
)


QUOTAS = {"verify": 48, "query": 48}
SEED = "internvl-gqa-relation-val96-20261127-v1"
VAL_PIN = "ba38c43ae905821145f318942d9f272da92c77262b811ecf6912cc56ed7aeea2"
TRAIN_MEDIA_PIN = "8f692dab37c976bb819fd56c405721882b20258b008138430ae38c6f34d7b5b7"
TRAIN_METADATA_PIN = "fca123196d7ae669b959cae6deceeba8123998f5290d61c6d33796a987df9f1e"


def ordered(purpose, key):
    return sha256(f"{SEED}:{purpose}:{key}".encode()).digest()


def protected_inputs(train):
    media = train / "actual-gqa-relation-64-frozen-input-v2"
    used = load_pinned(media / "media-receipt.json", TRAIN_MEDIA_PIN)["admitted"]
    candidates = load_pinned(train / "coco-vg-candidates-original-v1.json", TRAIN_METADATA_PIN)["candidates"]
    dims = {str(item["gqa_image_id"]): item["visual_genome_dimensions"] for item in candidates}
    inventory = load_pinned(train / OLD / "receipt.json",
                           "a023f6dd5e794cdd63bfcc3348a54e5c97d7db48f8d3597bee245a0856cb42ff")
    held = load_pinned(train / HOLDOUT, "22fe71b7205e3b19ebe087d4ff9a9f5829834e551971a5d1a49bef1e6de1db57")
    identities = []
    for key, item in used.items():
        h = image_identity((media / "media" / f"{key}.jpg").read_bytes(), dims[key])
        if h[0] != item["SHA256"]:
            raise ValueError("Training image identity changed")
        identities.append(h)
    for item in inventory["images"]:
        key = str(item["gqa_image_id"])
        if key in held["image_ids"]:
            h = image_identity((train / OLD / f"{key}.jpg").read_bytes(), item["JPEG_dimensions"])
            if h[0] not in held["image_SHA256"]:
                raise ValueError("Engineering holdout identity changed")
            identities.append(h)
    if len(identities) != 71:
        raise ValueError("Protected photo denominator differs")
    return set(used) | set(held["image_ids"]), {v["source_group"] for v in used.values()} | set(held["source_groups"]), identities


def pick_question_pool(validation, metadata, eligible):
    parts = deflate_parts(validation / "val_balanced_questions.json.deflate",
                          metadata["member_unpacked_SHA256"], metadata["member_packed_SHA256"],
                          metadata["member_unpacked_bytes"])
    pool = defaultdict(dict)
    for ident, example in question_rows(parts):
        kind = example["types"].get("structural")
        key = example["imageId"]
        if kind not in QUOTAS or key not in eligible or example["types"].get("semantic") not in {"rel", "relation"}:
            continue
        if key not in pool[kind] or ordered(kind, ident) < ordered(kind, pool[kind][key][0]):
            pool[kind][key] = ident, example
    if any(len(pool[kind]) < minimum * 2 for kind, minimum in QUOTAS.items()):
        raise ValueError("Per-type publisher pool is insufficient")
    return pool


def main(train, validation, output):
    origin = load_pinned(validation / "candidate-identities.json", VAL_PIN)
    excluded_ids, excluded_groups, protected = protected_inputs(train)
    allowed = {str(item["gqa_image_id"]): item for item in origin["candidates"]}
    if (len(allowed) != 645 or excluded_ids & set(allowed)
            or excluded_groups & {v["source_group"] for v in allowed.values()}):
        raise ValueError("Prior photo or source group present in evaluation pool")
    pool = pick_question_pool(validation, origin["summary"], allowed)
    selections = {kind: 0 for kind in QUOTAS}
    plans = {kind: sorted(pool[kind], key=lambda key: ordered("image-" + kind, key))[:80] for kind in QUOTAS}
    zip_index = ZipIndex()
    with ZipFile(zip_index) as archive:
        members = select_manifest(archive, set().union(*map(set, plans.values())))
    if sum(v.compress_size + 8522 for v in members.values()) > 45 * 1024**2:
        raise ValueError("GQA validation image range budget exceeded")
    output.mkdir()
    media = output / "media"
    media.mkdir()
    prompts, gold, lineage, failures = [], [], [], []
    used = set()
    transferred = 0
    for kind, plan in plans.items():
        for offset in range(0, len(plan), 4):
            ids = [key for key in plan[offset:offset + 4] if key not in used]
            with ThreadPoolExecutor(max_workers=2) as pool_io:
                jobs = {key: pool_io.submit(read_jpeg, members[key]) for key in ids}
                for key in ids:
                    try:
                        data, size = jobs[key].result()
                        transferred += size
                        info = allowed[key]
                        photo = image_identity(data, info["visual_genome_dimensions"])
                    except (OSError, ValueError) as error:
                        failures.append({"key": key, "class": type(error).__name__})
                        continue
                    if any(near_duplicate(photo, earlier) for earlier in protected):
                        failures.append({"key": key, "class": "near_duplicate"})
                        continue
                    if selections[kind] == QUOTAS[kind]:
                        continue
                    ident, row = pool[kind][key]
                    ordinal = len(prompts) + 1
                    path = f"media/{key}.jpg"
                    write(output / path, data)
                    prompts.append({"slot": ordinal, "type": kind, "image": path, "question": row["question"]})
                    gold.append({"slot": ordinal, "answer": row["answer"]})
                    lineage.append({"slot": ordinal, "question_id": ident, "source_group": info["source_group"],
                                    "license": info["coco_license_url"], "source_url": info["source_url"], "image_SHA256": photo[0]})
                    used.add(key)
                    protected.append(photo)
                    selections[kind] += 1
            if selections[kind] == QUOTAS[kind]:
                break
    if selections != QUOTAS or len({r["source_group"] for r in lineage}) != 96:
        write(output / "incomplete.json", document({"selected": selections, "failures": failures}))
        raise ValueError("Unable to freeze 96 distinct licensed images")
    for name, data in (("prompts.json", prompts), ("gold.private.json", gold), ("lineage.private.json", lineage)):
        write(output / name, document(data))
    result = {"seed": SEED, "heldout_manifest_SHA256": VAL_PIN, "kind_counts": selections, "protected_image_identities": 71,
              "prompts_SHA256": sha256((output / "prompts.json").read_bytes()).hexdigest(),
              "gold_SHA256": sha256((output / "gold.private.json").read_bytes()).hexdigest(),
              "lineage_SHA256": sha256((output / "lineage.private.json").read_bytes()).hexdigest(),
              "media_transferred_range_bytes": transferred, "excluded_bad_image_candidates": len(failures),
              "accuracy_min_gain_percentage_points": 3, "paired_image_bootstrap_lower_bound_must_exceed_zero": True,
              "source_split": "GQA balanced validation relationship subset", "testdev_official_metric": False,
              "scored_checkpoint_outputs": 0}
    write(output / "freeze.json", document(result))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.training, args.validation, args.output)
