"""Project the frozen TextVQA validation annotation into a label-free identity file."""

import argparse
import hashlib
import json
from pathlib import Path
import re


ANNOTATION_SHA256 = "4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d"
ANNOTATION_BYTES = 3_116_162
IMAGE_ID = re.compile(r"[0-9a-f]{16}")


def project(source):
    with source.open("rb") as stream:
        raw = stream.read(ANNOTATION_BYTES + 1)
    if len(raw) != ANNOTATION_BYTES or hashlib.sha256(raw).hexdigest() != ANNOTATION_SHA256:
        raise ValueError("TextVQA v0.5.1 frozen validation annotation differs")

    def identity_fields(record):
        if "question_id" in record:
            return {key: record[key] for key in ("question_id", "image_id", "set_name")}
        return record

    document = json.loads(raw, object_hook=identity_fields)
    if (document["dataset_type"], document["dataset_version"]) != ("val", "0.5.1"):
        raise ValueError("Unexpected annotation version or split")
    rows = document["data"]
    if any(type(row["question_id"]) is not int or row["question_id"] < 0
           or not isinstance(row["image_id"], str) or not IMAGE_ID.fullmatch(row["image_id"])
           or row["set_name"] != "val" for row in rows):
        raise ValueError("Invalid validation identity")
    if len(rows) != 5000 or len({row["question_id"] for row in rows}) != 5000:
        raise ValueError("Unexpected validation question denominator")
    if len({row["image_id"] for row in rows}) != 3166:
        raise ValueError("Unexpected validation image denominator")
    return dict(schema="TEXTVQA_V051_VAL_IDENTITY_ONLY_V1", annotation_SHA256=ANNOTATION_SHA256,
                split="val", version="0.5.1",
                rows=[dict(question_id=row["question_id"], image_id=row["image_id"]) for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("identity_output", type=Path)
    args = parser.parse_args()
    projection = project(args.annotation)
    raw = (json.dumps(projection, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with args.identity_output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(questions=5000, image_ids=3166, bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())))


if __name__ == "__main__":
    main()
