"""Build an InternVL relation-QA Swift slice from admitted GQA balanced training assets."""
import argparse
import codecs
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import zlib

from build_textvqa_street_scene import item


TRAIN_SOURCE_NAME = "train_balanced_questions.json"
BLOCK = 1024 * 1024


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


def deflate_parts(path, expected_sha, expected_packed_sha, expected_size):
    plain_sha, packed_sha = sha256(), sha256()
    plain_size = 0
    inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    utf8 = codecs.getincrementaldecoder("utf-8")()
    with path.open("rb") as source:
        while packed := source.read(65536):
            packed_sha.update(packed)
            while packed:
                raw = inflater.decompress(packed, BLOCK)
                packed = inflater.unconsumed_tail
                plain_size += len(raw)
                if plain_size > expected_size or inflater.unused_data:
                    raise ValueError("unexpected compressed question member extent")
                plain_sha.update(raw)
                if raw:
                    yield utf8.decode(raw)
    tail = utf8.decode(b"", final=True)
    if tail:
        yield tail
    if (not inflater.eof or inflater.flush() or plain_size != expected_size
            or plain_sha.hexdigest() != expected_sha or packed_sha.hexdigest() != expected_packed_sha):
        raise ValueError("question member extent or SHA256 differs")


def question_rows(parts):
    """Read a single top-level object without materializing all question text."""
    decoder = json.JSONDecoder(object_pairs_hook=unique_object)
    chunks = iter(parts)
    text, pos = "", 0

    def refill():
        nonlocal text, pos
        for chunk in chunks:
            if chunk:
                text, pos = text[pos:] + chunk, 0
                return True
        return False

    def current():
        nonlocal pos
        while True:
            while pos < len(text) and text[pos] in " \t\r\n":
                pos += 1
            if pos < len(text):
                return text[pos]
            if not refill():
                return None

    def value():
        nonlocal pos
        current()
        while True:
            try:
                obj, pos = decoder.raw_decode(text, pos)
                return obj
            except json.JSONDecodeError:
                if len(text) - pos > BLOCK or not refill():
                    raise

    if current() != "{":
        raise ValueError("expected a question map")
    pos += 1
    seen = set()
    if current() != "}":
        while True:
            if current() != '"':
                raise ValueError("expected question ID string")
            question_id = value()
            if question_id in seen or current() != ":":
                raise ValueError("duplicate question ID or missing colon")
            seen.add(question_id)
            pos += 1
            if current() != "{":
                raise ValueError("expected a question record")
            yield question_id, value()
            if current() != ",":
                break
            pos += 1
    if current() != "}":
        raise ValueError("unterminated question map")
    pos += 1
    if current() is not None:
        raise ValueError("unexpected trailing question data")


def is_balanced_relation(row):
    return row["isBalanced"] is True and row["types"]["semantic"] in {"rel", "relation"}


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
    if not isinstance(admitted, dict):
        raise ValueError("expected an admitted-image map")
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
    total = 0
    if isinstance(questions, dict):
        questions = questions.items()
    for question_id, row in questions:
        total += 1
        if not is_balanced_relation(row):
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
    proof = {"source_questions": total, "relation_candidates": len(candidates),
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
    parser.add_argument("--questions-compressed-sha")
    parser.add_argument("--questions-uncompressed-bytes", type=int)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", default="internvl-gqa-relation-train-v1")
    args = parser.parse_args()
    pins = {name: getattr(args, f"{name}_sha") for name in ("questions", "admitted", "holdout")}
    if args.questions.name == TRAIN_SOURCE_NAME + ".deflate":
        if (not args.questions_compressed_sha or not args.questions_uncompressed_bytes
                or args.questions_uncompressed_bytes < 0):
            parser.error("compressed member needs its SHA256 and expanded byte count")
        pins["questions_compressed"] = args.questions_compressed_sha
        questions = question_rows(deflate_parts(args.questions, pins["questions"],
                                               pins["questions_compressed"], args.questions_uncompressed_bytes))
    elif args.questions.name == TRAIN_SOURCE_NAME:
        questions = load_pinned(args.questions, pins["questions"])
    else:
        parser.error("use the pinned balanced training member")
    admitted = load_pinned(args.admitted, pins["admitted"])
    holdout = load_pinned(args.holdout, pins["holdout"])
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
