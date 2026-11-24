"""Bind an official train-only VizWiz pilot to the shared VLM JSONL row format."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re


SOURCE = "VizWiz-VQA"
IMAGE_NAME = re.compile(r"VizWiz_train_(\d{8})\.jpg")
MEDIA_MARKERS = ("<image>", "<video>", "<audio>", "<start_of_image>", "<end_of_image>")
LABEL = "unanswerable"


def pin(blob):
    return {"bytes": len(blob), "SHA256": sha256(blob).hexdigest()}


def serialize(row):
    return (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def valid_message(text):
    return isinstance(text, str) and bool(text.strip()) and not any(marker in text for marker in MEDIA_MARKERS)


def answer(row):
    answers = row["answers"]
    if len(answers) != 10 or any(not valid_message(item.get("answer")) for item in answers):
        raise ValueError("invalid annotated answer set")
    if row["answerable"] == 0:
        return LABEL
    # Resolve the easy supervised cases first. A tie or an answer with too few
    # confident votes stays in the reported denominator for further review.
    confident = [item["answer"].strip() for item in answers if item["answer_confidence"] == "yes"]
    votes = Counter(text.casefold() for text in confident if text.casefold() != LABEL)
    candidates = votes.most_common(2)
    if not candidates or candidates[0][1] < 3 or (len(candidates) > 1 and candidates[0][1] == candidates[1][1]):
        return None
    return next(text for text in confident if text.casefold() == candidates[0][0])


def prepare(source, output, receipt_sha, source_rows_sha):
    raw_receipt = (source / "pilot-receipt.json").read_bytes()
    raw_rows = (source / "training-pilot-private.json").read_bytes()
    if pin(raw_receipt)["SHA256"] != receipt_sha or pin(raw_rows)["SHA256"] != source_rows_sha:
        raise ValueError("frozen public-pilot source identity differs")
    receipt, original = json.loads(raw_receipt), json.loads(raw_rows)
    if (receipt["mode"] != "TRAIN_ONLY_PILOT" or receipt["pilot_rows"] != 6 or len(original) != 6
            or len(receipt["assets"]) != 6 or receipt["evaluation_annotation_members_decoded"] != 0):
        raise ValueError("unexpected train-only pilot scope")
    entities, images = set(), set()
    rows, omitted = [], []
    for item, row in zip(receipt["assets"], original, strict=True):
        key = row["image"]
        matched = IMAGE_NAME.fullmatch(key)
        if (not matched or 23431 <= int(matched[1]) <= 23958 or key != item["group"] or key in entities
                or type(row["answerable"]) is not int or row["answerable"] not in (0, 1)
                or row["answerable"] != item["answerable"] or not valid_message(row["question"])):
            raise ValueError("unexpected pilot image, split or supervision")
        path = source / "images" / key
        if path.is_symlink() or pin(path.read_bytes()) != item["image"]:
            raise ValueError("frozen image payload differs")
        image_sha = item["image"]["SHA256"]
        if image_sha in images:
            raise ValueError("duplicate image bytes")
        entities.add(key)
        images.add(image_sha)
        group = sha256(key.encode()).hexdigest()
        supervised = answer(row)
        if supervised is None:
            omitted.append(dict(group_SHA256=group, reason="AMBIGUOUS_ANSWERABLE_SUPERVISION"))
            continue
        rows.append(dict(
            id=f"vizwiz-{matched[1]}", source=SOURCE, split="train", group=group,
            images=[str(path.resolve())],
            messages=[dict(role="user", content="<image>\n" + row["question"]),
                      dict(role="assistant", content=supervised)],
        ))
    output.mkdir()
    data = b"".join(serialize(row) for row in rows)
    (output / "train-swift.jsonl").write_bytes(data)
    summary = dict(planned_rows=len(original), emitted_rows=len(rows), omitted=omitted,
                   emitted_by_answerability=dict(Counter(
                       "unanswerable" if row["messages"][1]["content"] == LABEL else "answerable" for row in rows)),
                   source_receipt=pin(raw_receipt), source_rows=pin(raw_rows), swift_rows=pin(data),
                   native_processor_rows=0, optimizer_updates=0, training_READY=False)
    (output / "row-contract-receipt.json").write_bytes(serialize(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen_pilot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--rows-sha256", required=True)
    args = parser.parse_args()
    report = prepare(args.frozen_pilot, args.output, args.receipt_sha256, args.rows_sha256)
    print(json.dumps({key: value for key, value in report.items() if key != "omitted"}, sort_keys=True))


if __name__ == "__main__":
    main()
