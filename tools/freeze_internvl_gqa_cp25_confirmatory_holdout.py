"""Build the source-disjoint InternVL GQA confirmation set before model generation."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from PIL import Image


REPO = Path(__file__).resolve().parents[1]
DOMAIN = REPO / "code/bw1000/training-domain-pilot-v1"
sys.path.insert(0, str(DOMAIN))
from build_gqa_relation_train import deflate_parts, load_pinned, question_rows
from freeze_internvl_gqa_val96 import VAL_PIN, protected_inputs
from internvl_gqa_val96 import encode, fields, normalize, prompt_rows, read, write
from materialize_gqa_relation_pilot import ZipIndex, image_identity, near_duplicate, read_jpeg, select_manifest


SEED = "internvl-gqa56-cp25-confirmatory-20261128-v1"
QUOTA = {"query": 48, "verify": 48}
MAX_CANDIDATES = 112
MAX_RANGE_BYTES = 64 << 20
OLD_FREEZE = "b1e56ef230399827d144ee76adfdecc64a0909c8a8372f3ff54e524d127a246c"
OLD_PROMPTS = "2c783210c2285a5d8a6d968698adccf94bd4079fe3f8fbba8a2ecb137337f37e"
OLD_LINEAGE = "dab20f681c439e1284fd531d9802541532c5f3622a38f272dc66bd295be0adc1"
CP25 = "13ef509956869b3addf62dfa9fb404d4276198f237251928292fdf589b7d8c29"


def order(label, value):
    return sha256(f"{SEED}:{label}:{value}".encode()).digest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def questions(root, meta):
    return question_rows(deflate_parts(root / "val_balanced_questions.json.deflate",
                         meta["member_unpacked_SHA256"], meta["member_packed_SHA256"], meta["member_unpacked_bytes"]))


def old_groups(root):
    frozen = json.loads(read(root / "freeze.json", 4096, OLD_FREEZE))
    prompts = prompt_rows(read(root / "prompts.json", 1 << 18, OLD_PROMPTS))
    lineage = json.loads(read(root / "lineage.private.json", 1 << 18, OLD_LINEAGE))
    require(frozen["prompts_SHA256"] == OLD_PROMPTS and frozen["lineage_SHA256"] == OLD_LINEAGE,
            "Original evaluation identity changed")
    require(len(lineage) == 96 and len({row["source_group"] for row in lineage}) == 96,
            "Original evaluation group coverage differs")
    ids, groups, photos = set(), set(), []
    for prompt, origin in zip(prompts, lineage, strict=True):
        key = Path(prompt["image"]).stem
        require(prompt["slot"] == origin["slot"] and key not in ids, "Original evaluation slots differ")
        raw = read(root / prompt["image"], 2 << 20, origin["image_SHA256"])
        with Image.open(BytesIO(raw)) as img:
            dims = list(img.size)
        ids.add(key)
        groups.add(origin["source_group"])
        photos.append(image_identity(raw, dims))
    return ids, groups, photos


def pool_by_stratum(rows, allowed):
    pool = defaultdict(dict)
    for key, row in rows:
        kind, image = row["types"].get("structural"), row["imageId"]
        if kind not in QUOTA or image not in allowed or row["types"].get("semantic") not in {"rel", "relation"}:
            continue
        text = row["question"]
        require(isinstance(text, str) and 0 < len(text) <= 2048, "Malformed question")
        if image not in pool[kind] or order(kind, key) < order(kind, pool[kind][image][0]):
            pool[kind][image] = key, text
    require(all(len(pool[kind]) >= 2 * count for kind, count in QUOTA.items()), "Independent question pool too small")
    return pool


def freeze(local, output):
    train = local / "internvl-gqa-stanford-train-compressed-member-v1"
    val = local / "internvl-gqa-val-source-disjoint-secondary-v1"
    old = local / "internvl-gqa-val96-blind-frozen-v1"
    train_ids, train_groups, train_photos = protected_inputs(train)
    old_ids, old_source_groups, old_photos = old_groups(old)
    require(not train_ids & old_ids and not train_groups & old_source_groups, "Historical train and eval overlap")
    source = load_pinned(val / "candidate-identities.json", VAL_PIN)
    excluded_ids = train_ids | old_ids
    excluded_groups = train_groups | old_source_groups
    candidates = [row for row in source["candidates"] if str(row["gqa_image_id"]) not in excluded_ids
                  and row["source_group"] not in excluded_groups]
    eligible = {str(row["gqa_image_id"]): row for row in candidates}
    require(len(source["candidates"]) == 645 and len(eligible) == len(candidates), "Publisher candidate set differs")
    pool = pool_by_stratum(questions(val, source["summary"]), eligible)
    ordered = {kind: sorted(pool[kind], key=lambda key: order("image-" + kind, key))[:MAX_CANDIDATES]
               for kind in QUOTA}
    with ZipFile(ZipIndex()) as archive:
        members = select_manifest(archive, set().union(*map(set, ordered.values())))
    require(sum(row.compress_size + 8522 for row in members.values()) < MAX_RANGE_BYTES, "Candidate range budget exceeded")

    output.mkdir(mode=0o700)
    media = output / "media"
    media.mkdir(mode=0o700)
    protected = train_photos + old_photos
    used_images, used_groups = set(), set()
    prompts, lineage = [], []
    counts, failures = Counter(), Counter()
    transferred = 0
    with ThreadPoolExecutor(max_workers=2) as network:
        for kind, planned in ordered.items():
            for offset in range(0, len(planned), 4):
                keys = [key for key in planned[offset:offset + 4] if key not in used_images
                        and eligible[key]["source_group"] not in used_groups]
                futures = {key: network.submit(read_jpeg, members[key]) for key in keys}
                for key, future in futures.items():
                    info = eligible[key]
                    try:
                        data, extent = future.result()
                        transferred += extent
                        photo = image_identity(data, info["visual_genome_dimensions"])
                    except (OSError, ValueError) as error:
                        failures[type(error).__name__] += 1
                        continue
                    if counts[kind] == QUOTA[kind] or info["source_group"] in used_groups:
                        continue
                    if any(near_duplicate(photo, prior) for prior in protected):
                        failures["near_duplicate"] += 1
                        continue
                    qid, question = pool[kind][key]
                    slot = len(prompts) + 1
                    name = f"media/{key}.jpg"
                    write(output / name, data)
                    prompts.append(dict(slot=slot, type=kind, image=name, question=question))
                    lineage.append(dict(slot=slot, question_id=qid, source_group=info["source_group"],
                                        image_id=key, image_SHA256=photo[0], image_bytes=len(data),
                                        license=info["coco_license_url"], source_url=info["source_url"]))
                    used_images.add(key)
                    used_groups.add(info["source_group"])
                    protected.append(photo)
                    counts[kind] += 1
                if counts[kind] == QUOTA[kind]:
                    break
    if counts != QUOTA:
        write(output / "incomplete.json", encode(dict(eligible=len(eligible), counts=counts, failures=failures)))
        raise ValueError("Unable to freeze 96 independent licensed image groups")
    require(len(used_images) == len(used_groups) == 96 and not used_images & excluded_ids
            and not used_groups & excluded_groups and transferred < MAX_RANGE_BYTES, "Final independence differs")
    prompts_bytes = encode(prompts)
    lineage_bytes = encode(lineage)
    prompt_rows(prompts_bytes)
    media_pins = {row["image"]: dict(bytes=witness["image_bytes"], SHA256=witness["image_SHA256"])
                  for row, witness in zip(prompts, lineage, strict=True)}
    manifest = encode(dict(schema="INTERNVL_GQA_VAL96_INPUT_V1", images=96, gold_included=False,
                           members={"prompts.json": fields(prompts_bytes), **media_pins}))
    write(output / "prompts.json", prompts_bytes)
    write(output / "lineage.private.json", lineage_bytes)
    write(output / "INPUT.json", manifest)
    freeze_doc = dict(schema="INTERNVL_GQA56_CP25_CONFIRMATORY_VAL96_V1", seed=SEED, quotas=QUOTA,
                      adapter_parameter_SHA256=CP25, source_candidate_SHA256=VAL_PIN,
                      original_eval_freeze_SHA256=OLD_FREEZE, input=fields(manifest), lineage=fields(lineage_bytes),
                      tested_arms=["base", "domain25"], planned_slots_per_arm=96,
                      metric="normalized exact answer, missing and capped predictions are incorrect",
                      min_gain_percentage_points=3, resampling_unit="unique source image", bootstrap_resamples=20000,
                      min_paired_bootstrap_95_percent_lower_bound_percentage_points_exclusive=0,
                      excluded_train_and_engineering_image_count=len(train_photos), excluded_original_eval_image_count=len(old_photos),
                      disjoint_image_ids=True, disjoint_source_groups=True, no_detected_near_duplicate=True,
                      selected_candidate_pool_images=len(eligible), actual_source_range_bytes=transferred,
                      image_range_rejections=failures, selection_uses_answers=False, generated_predictions=0)
    frozen_bytes = encode(freeze_doc)
    write(output / "frozen-before-gold.json", frozen_bytes)
    selected = {row["question_id"]: row["slot"] for row in lineage}
    labels = {}
    for qid, row in questions(val, source["summary"]):
        if qid in selected:
            answer = row["answer"]
            require(isinstance(answer, str) and bool(normalize(answer)), "Invalid official answer")
            labels[selected[qid]] = answer
    require(len(labels) == 96, "Official answer denominator differs")
    gold = encode([dict(slot=slot, answer=labels[slot]) for slot in range(1, 97)])
    write(output / "gold.private.json", gold)
    summary = dict(**freeze_doc, freeze=fields(frozen_bytes), local_gold=fields(gold), actual_GPU_steps=0)
    write(output / "receipt.json", encode(summary))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    print(json.dumps(freeze(REPO / "local-only", args.output), sort_keys=True, allow_nan=False))
