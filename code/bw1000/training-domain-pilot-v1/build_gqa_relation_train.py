"""Build an InternVL relation-QA Swift slice from admitted GQA balanced training assets."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re

from build_textvqa_street_scene import item


TRAIN_SOURCE_NAME = "train_balanced_questions.json"


def digest(data):
    return sha256(data).hexdigest()


def pinned(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("invalid SHA256 pin")
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError("input SHA256 differs")
    return raw


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON member")
    return result


def load_pinned(path, expected):
    return json.loads(pinned(path, expected), object_pairs_hook=unique_object)


def media_path(root, relative):
    if not isinstance(relative, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", relative):
        raise ValueError("invalid relative image path")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != relative:
        raise ValueError("invalid relative image path")
    path = root.joinpath(*parsed.parts)
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("image must be a regular file inside the media root")
    return path


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def build(questions, admitted, holdout, root, count, seed):
    if not isinstance(questions, dict) or not isinstance(admitted, dict):
        raise ValueError("expected question and admitted-image maps")
    if not isinstance(holdout, dict) or set(holdout) != {"image_ids", "source_groups", "image_SHA256"}:
        raise ValueError("expected identities-only holdout exclusion")
    if any(not isinstance(values, list) or not values or not all(map(nonempty, values))
           for values in holdout.values()):
        raise ValueError("holdout exclusion needs all three identity sets")
    excluded_ids, excluded_groups, excluded_hashes = (
        set(holdout[key]) for key in ("image_ids", "source_groups", "image_SHA256")
    )
    if not all(re.fullmatch(r"[0-9a-f]{64}", pin) for pin in excluded_hashes):
        raise ValueError("invalid holdout image SHA256")
    for image_id, entry in admitted.items():
        if (not isinstance(entry, dict)
                or set(entry) != {"path", "source_group", "license", "source_url", "SHA256"}
                or not all(map(nonempty, entry.values()))
                or not entry["source_url"].startswith("https://")
                or not re.fullmatch(r"[0-9a-f]{64}", entry["SHA256"])):
            raise ValueError(f"incomplete image provenance for {image_id}")

    candidates = []
    rejected = Counter()
    for question_id, row in questions.items():
        if row["isBalanced"] is not True or row["types"]["semantic"] != "relation":
            rejected["other_question_type"] += 1
            continue
        image_id = row["imageId"]
        if image_id not in admitted:
            rejected["image_not_admitted"] += 1
            continue
        image = admitted[image_id]
        if (image_id in excluded_ids or image["source_group"] in excluded_groups
                or image["SHA256"] in excluded_hashes):
            rejected["overlap_with_holdout"] += 1
            continue
        row_key = digest(f"{seed}\0{question_id}".encode())
        candidates.append((row_key, question_id, image_id, row))

    if not 1 <= count <= 512:
        raise ValueError("slice count must be in [1, 512]")
    rows, lineage = [], []
    groups, image_hashes = set(), set()
    for _, question_id, image_id, row in sorted(candidates):
        image = admitted[image_id]
        if image["source_group"] in groups or image["SHA256"] in image_hashes:
            rejected["same_training_group_or_image"] += 1
            continue
        path = media_path(root, image["path"])
        pinned(path, image["SHA256"])
        rows.append(item(str(path), row["question"], row["answer"]))
        lineage.append({"question_id": question_id, "image_id": image_id, **image})
        groups.add(image["source_group"])
        image_hashes.add(image["SHA256"])
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(f"only {len(rows)} distinct admitted relation examples available")
    proof = {"source_questions": len(questions), "relation_candidates": len(candidates),
             "selection_rejections": dict(rejected), "selected": len(rows), "seed": seed, "lineage": lineage,
             "native_processor_rows": 0, "optimizer_updates": 0, "training_ready": False}
    return rows, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("questions", "admitted", "holdout"):
        parser.add_argument(f"--{name}", required=True, type=Path)
        parser.add_argument(f"--{name}-sha", required=True)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", default="internvl-gqa-relation-train-v1")
    args = parser.parse_args()
    if args.questions.name != TRAIN_SOURCE_NAME:
        parser.error("use the pinned balanced training member")
    pins = {name: getattr(args, f"{name}_sha") for name in ("questions", "admitted", "holdout")}
    questions, admitted, holdout = (load_pinned(getattr(args, name), pin)
                                     for name, pin in pins.items())
    rows, proof = build(questions, admitted, holdout, args.media_root.resolve(strict=True),
                        args.count, args.seed)
    train = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in rows).encode()
    proof.update(train_SHA256=digest(train), source_SHA256=pins, train_bytes=len(train))
    receipt = (json.dumps(proof, ensure_ascii=False, sort_keys=True) + "\n").encode()
    args.output_dir.mkdir(exist_ok=False)
    (args.output_dir / "train.jsonl").write_bytes(train)
    (args.output_dir / "receipt.json").write_bytes(receipt)
    print(json.dumps({"selected": len(rows), "train_SHA256": digest(train),
                      "receipt_SHA256": digest(receipt), "training_ready": False}))


if __name__ == "__main__":
    main()
