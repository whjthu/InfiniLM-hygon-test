"""Build and independently score the frozen InternVL GQA validation comparison."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import unicodedata


PROMPTS_SHA = "2c783210c2285a5d8a6d968698adccf94bd4079fe3f8fbba8a2ecb137337f37e"
GOLD_SHA = "4277e03d034a3f8bd05a20a81265adf59b631757197e738ddea600f0b95844d1"
LINEAGE_SHA = "dab20f681c439e1284fd531d9802541532c5f3622a38f272dc66bd295be0adc1"
ARMS = ("base", "domain25")
COUNT = 96
CAP = 64
IMAGE = re.compile(r"media/[0-9]+\.jpg")
HEX = re.compile(r"[0-9a-f]{64}")


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def encode(document):
    return (json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()


def fields(data):
    return {"bytes": len(data), "SHA256": digest(data)}


def read(path, limit, expected=None):
    need(path.is_file() and not path.is_symlink(), f"Expected a regular {path.name}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    need(0 < len(data) <= limit, f"Invalid byte extent for {path.name}")
    need(expected is None or digest(data) == expected, f"Hash differs for {path.name}")
    return data


def write(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def prompt_rows(data):
    rows = json.loads(data)
    need(isinstance(rows, list) and len(rows) == COUNT, "Frozen prompt denominator differs")
    for slot, row in enumerate(rows, 1):
        need(set(row) == {"slot", "type", "image", "question"} and type(row["slot"]) is int
             and row["slot"] == slot and IMAGE.fullmatch(row["image"])
             and isinstance(row["question"], str) and 0 < len(row["question"]) <= 2048,
             "Invalid user-only prompt row")
    need(Counter(row["type"] for row in rows) == {"query": 48, "verify": 48}
         and len({row["image"] for row in rows}) == COUNT, "Frozen prompt strata or image identity differs")
    return rows


def frozen_members(frozen):
    """Read only pinned user prompts, the matching publisher JPEGs and their lineage."""
    prompts = read(frozen / "prompts.json", 1 << 18, PROMPTS_SHA)
    rows = prompt_rows(prompts)
    lineage = json.loads(read(frozen / "lineage.private.json", 1 << 18, LINEAGE_SHA))
    need(len(lineage) == COUNT and len({row["source_group"] for row in lineage}) == COUNT,
         "Held-out source groups differ")
    images = {}
    for row, witness in zip(rows, lineage):
        need(row["slot"] == witness["slot"], "Held-out lineage order differs")
        images[row["image"]] = read(frozen / row["image"], 2 << 20, witness["image_SHA256"])
    need(len({digest(data) for data in images.values()}) == COUNT, "Duplicate held-out image bytes")
    assets = {"prompts.json": prompts, **images}
    need(sum(map(len, assets.values())) < 20 << 20, "Evaluation transfer exceeds fixed byte budget")
    manifest = {"schema": "INTERNVL_GQA_VAL96_INPUT_V1", "images": COUNT, "gold_included": False,
                "members": {name: fields(data) for name, data in assets.items()}}
    return {"INPUT.json": encode(manifest), **assets}


def prepare(frozen, output):
    assets = frozen_members(frozen)
    output.mkdir(mode=0o700)
    (output / "media").mkdir(mode=0o700)
    for name, data in assets.items():
        write(output / name, data)
    return fields(assets["INPUT.json"])


def inputs(root, input_sha):
    manifest = json.loads(read(root / "INPUT.json", 1 << 17, input_sha))
    need(manifest["schema"] == "INTERNVL_GQA_VAL96_INPUT_V1" and manifest["images"] == COUNT
         and manifest["gold_included"] is False, "Wrong user-only input manifest")
    members = manifest["members"]
    prompts = read(root / "prompts.json", 1 << 18, PROMPTS_SHA)
    rows = prompt_rows(prompts)
    need(set(members) == {"prompts.json", *(row["image"] for row in rows)}
         and fields(prompts) == members["prompts.json"], "Input manifest extent differs")
    need({p.name for p in root.iterdir()} == {"INPUT.json", "prompts.json", "media"}
         and not (root / "media").is_symlink()
         and {p.name for p in (root / "media").iterdir()} == {Path(row["image"]).name for row in rows},
         "Input directory contains unregistered or missing files")
    for row in rows:
        name = row["image"]
        need(fields(read(root / name, 2 << 20)) == members[name], "Frozen JPEG differs")
    return rows


def normalize(answer):
    # Match the whole answer with a fixed case, space and terminal punctuation rule.
    text = " ".join(unicodedata.normalize("NFKC", answer).casefold().split())
    return text.rstrip(".?!").rstrip()


def prediction_rows(raw, arm, input_sha):
    data = json.loads(raw)
    need(set(data) == {"schema", "arm", "input_SHA256", "rows"}
         and (data["schema"], data["arm"], data["input_SHA256"]) == (
             "INTERNVL_GQA_VAL96_PREDICTION_V1", arm, input_sha), "Unexpected prediction binding")
    rows = data["rows"]
    need(isinstance(rows, list) and len(rows) == COUNT, "Every held-out slot must be recorded")
    for slot, row in enumerate(rows, 1):
        need(set(row) == {"slot", "status", "text", "processed_input_SHA256", "generated_tokens"}
             and type(row["slot"]) is int and row["slot"] == slot, "Missing, duplicate or misordered prediction slot")
        status, body, token_count, pin = (
            row["status"], row["text"], row["generated_tokens"], row["processed_input_SHA256"]
        )
        need(status in {"eos", "cap", "error", "missing"}, "Unknown generation status")
        if status in {"eos", "cap"}:
            need(isinstance(body, str) and len(body) <= 16384 and type(token_count) is int
                 and 1 <= token_count <= CAP and (status != "cap" or token_count == CAP)
                 and isinstance(pin, str) and HEX.fullmatch(pin), "Invalid generated output")
        else:
            need(body is None and token_count is None and (pin is None or isinstance(pin, str) and HEX.fullmatch(pin)),
                 "Failure must preserve the slot without a generated answer")
    return rows


def bootstrap_interval(deltas):
    rng = random.Random(271196)
    counts = sorted(sum(sample) for sample in (
        rng.choices(deltas, k=COUNT) for _ in range(20_000)
    ))
    return 100 * counts[499] / COUNT, 100 * counts[19499] / COUNT


def score(frozen, predictions, seal_sha, input_sha, output):
    """Open the frozen answers only after both private prediction arms validate."""
    marks = json.loads(read(predictions / "SEALED.json", 1 << 16, seal_sha))
    need(set(marks) == {"schema", "input_SHA256", "arms"}
         and marks["schema"] == "INTERNVL_GQA_VAL96_SEALED_V1" and marks["input_SHA256"] == input_sha
         and set(marks["arms"]) == set(ARMS), "A complete sealed comparison is required")
    arms = {}
    for arm in ARMS:
        raw = read(predictions / f"{arm}.json", 4 << 20)
        need(fields(raw) == marks["arms"][arm], "Sealed predictions changed")
        arms[arm] = prediction_rows(raw, arm, input_sha)
    for left, right in zip(*arms.values()):
        first, second = left["processed_input_SHA256"], right["processed_input_SHA256"]
        need(first is None or second is None or first == second, "Native inputs differ across arms")
    prompts = prompt_rows(read(frozen / "prompts.json", 1 << 18, PROMPTS_SHA))
    gold = json.loads(read(frozen / "gold.private.json", 1 << 18, GOLD_SHA))
    need(len(gold) == COUNT, "Gold denominator differs")
    results = []
    for prompt, truth in zip(prompts, gold):
        slot, kind = prompt["slot"], prompt["type"]
        need(set(truth) == {"slot", "answer"} and truth["slot"] == slot
             and isinstance(truth["answer"], str) and normalize(truth["answer"]), "Invalid gold row")
        keyed = {"slot": slot, "type": kind}
        for name, rows in arms.items():
            pred = rows[slot - 1]
            status = pred["status"]
            keyed[name] = {"status": status, "correct": status == "eos"
                           and normalize(pred["text"]) == normalize(truth["answer"])}
        results.append(keyed)
    by_arm = {}
    for name in ARMS:
        by_arm[name] = {"correct": sum(r[name]["correct"] for r in results), "total": COUNT,
                        "status": dict(Counter(r[name]["status"] for r in results)),
                        "by_type": {kind: {"correct": sum(r[name]["correct"] for r in results if r["type"] == kind),
                                           "total": 48} for kind in ("query", "verify")}}
    deltas = [int(r["domain25"]["correct"]) - int(r["base"]["correct"]) for r in results]
    interval = bootstrap_interval(deltas)
    report = {"input_SHA256": input_sha, "seal_SHA256": seal_sha, "gold_SHA256": GOLD_SHA,
              "split": "frozen GQA balanced validation relation subset", "official_testdev": False,
              "by_arm": by_arm, "paired_image_groups": COUNT, "gain_pp": 100 * sum(deltas) / COUNT,
              "CI95_pp": interval, "blind_domain_effect_gate": 100 * sum(deltas) >= 3 * COUNT and interval[0] > 0,
              "formal_domain_delivery": False, "rows": results}
    write(output, encode(report))
    return {key: value for key, value in report.items() if key != "rows"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("prepare")
    pack.add_argument("frozen", type=Path)
    pack.add_argument("output", type=Path)
    verify = sub.add_parser("check-input")
    verify.add_argument("input", type=Path)
    verify.add_argument("input_sha")
    scoring = sub.add_parser("score")
    scoring.add_argument("frozen", type=Path)
    scoring.add_argument("predictions", type=Path)
    scoring.add_argument("seal_sha")
    scoring.add_argument("input_sha")
    scoring.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.frozen, args.output)
    elif args.command == "check-input":
        result = {"rows": len(inputs(args.input, args.input_sha)), "input_SHA256": args.input_sha}
    else:
        result = score(args.frozen, args.predictions, args.seal_sha, args.input_sha, args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
