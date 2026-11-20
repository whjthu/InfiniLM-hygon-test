"""Prepare private table-reading records from the frozen public USGS catalog."""
import argparse
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re

from PIL import Image, ImageDraw, ImageFont


CREDIT = "Source: U.S. Geological Survey, ComCat"
FIELDS = ("UTC date", "Latitude", "Longitude", "Depth (km)", "Magnitude")
CASES = {"T01.sft", "T02.sft", "T03.sft", "T04.sft", "T05.sft"}
VIEW = ("train", "dev")


def digest(raw):
    return sha256(raw).hexdigest()


def verify(raw, expected):
    if digest(raw) != expected:
        raise ValueError("USGS data identity differs")
    return raw


def decimal(value, places):
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError("nonfinite USGS data")
    return format(result.quantize(Decimal(places), rounding=ROUND_HALF_UP), "f")


def values(raw, record):
    data = []
    for line in raw.splitlines():
        event = json.loads(line)
        start, end = record["window_half_open"]
        if not re.fullmatch("us[a-z0-9]+", event["id"]) or not start <= event["time"][:10] < end:
            raise ValueError("USGS event provenance or date mismatch")
        data.append((event["id"], event["time"][:10], decimal(event["latitude"], "0.01"),
                     decimal(event["longitude"], "0.01"), decimal(event["depth"], "0.1"),
                     decimal(event["mag"], "0.1")))
    if len(data) != record["accepted"] or len(data) < 2 or len({row[0] for row in data}) != len(data):
        raise ValueError("USGS original event denominator differs")
    return data


def groups(records):
    output = [records[start:start + 6] for start in range(0, len(records), 6)]
    if len(output[-1]) == 1:
        output[-2].extend(output.pop())
    return output


def render(path, records):
    font = ImageFont.load_default(size=19)
    title = ImageFont.load_default(size=24)
    row_height = 39
    x = (28, 91, 243, 366, 497, 633)
    image = Image.new("RGB", (777, 126 + row_height * len(records)), "white")
    draw = ImageDraw.Draw(image)
    draw.text((28, 16), "Earthquake observations", font=title, fill="#0b283a")
    draw.rectangle((20, 52, 756, 88), fill="#123449")
    for col, text in zip(x, ("Row", *FIELDS)):
        draw.text((col, 61), text, font=font, fill="white")
    for index, row in enumerate(records, 1):
        y = 89 + (index - 1) * row_height
        draw.rectangle((20, y, 756, y + row_height - 1), fill="#edf4f7" if index % 2 else "#ffffff")
        for col, text in zip(x, (str(index), *row[1:])):
            draw.text((col, y + 8), text, font=font, fill="#0b283a")
    draw.text((28, image.height - 26), CREDIT, font=font, fill="#3e5c6f")
    image.save(path, format="PNG")
    return digest(path.read_bytes())


def questions(records):
    last = len(records) - 1
    return [
        (f"What is the magnitude in row {last + 1}? Return one decimal place.", records[last][5]),
        ("What is the depth in kilometers in row 1? Return one decimal place.", records[0][4]),
        ("What is the date in row 2? Return YYYY-MM-DD.", records[1][1]),
        ("Which row has the deepest event? If tied, give the lowest row number.",
         str(1 + max(range(len(records)), key=lambda i: Decimal(records[i][4])))),
    ]


def jsonlines(path, rows):
    raw = b"".join((json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
                   for row in rows)
    with path.open("xb") as f:
        f.write(raw)
    return {"rows": len(rows), "bytes": len(raw), "SHA256": digest(raw)}


def build(source, source_sha, output, target, case):
    raw = verify((source / "source-provenance.json").read_bytes(), source_sha)
    provenance = json.loads(raw)
    if not target.is_absolute() or ".." in target.parts or case not in CASES:
        raise ValueError("exact native VLM case and staging root required")
    material = {split: values(verify((source / (split + "-public-events.jsonl")).read_bytes(),
                                     provenance["split"][split]["qualified_SHA256"]),
                              provenance["split"][split]) for split in VIEW}
    if {row[0] for row in material["train"]} & {row[0] for row in material["dev"]}:
        raise ValueError("develop and training event identities overlap")
    output.mkdir(exist_ok=False)
    evidence = {}
    for split in VIEW:
        images = output / split / "media"
        images.mkdir(parents=True)
        records = []
        gold = []
        media = {}
        for index, board in enumerate(groups(material[split]), 1):
            board_name = f"{split}/media/table-{index:03d}.png"
            media[board_name] = render(output / board_name, board)
            for ordinal, (question, answer) in enumerate(questions(board), 1):
                ident = f"{split}-{index:03d}-{ordinal}"
                messages = [{"role": "user", "content": "<image>\n" + question}]
                if split == "train":
                    messages.append({"role": "assistant", "content": "FINAL: " + answer})
                records.append({"id": ident, "messages": messages, "images": [(target / board_name).as_posix()]})
                if split == "dev":
                    gold.append({"id": ident, "answer": answer})
        evidence[split] = {"events": len(material[split]), "media": media,
                           "inputs": jsonlines(output / split / "swift.jsonl", records)}
        if gold:
            evidence[split]["private_scoring"] = jsonlines(output / split / "labels-private.jsonl", gold)
    result = {"schema": "USGS_TABLE_READ_TRAIN_DEV_V1", "source_SHA256": digest(raw), "native_case": case,
              "data": evidence, "blind_source_read_for_rendering": False, "target_verified": False,
              "training_GO": False}
    raw_manifest = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    with (output / "binding-private.json").open("xb") as f:
        f.write(raw_manifest)
    return {"events": {name: result["data"][name]["events"] for name in VIEW},
            "questions": {name: result["data"][name]["inputs"]["rows"] for name in VIEW},
            "manifest_SHA256": digest(raw_manifest)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-data", type=PurePosixPath, required=True)
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.source_sha, args.output, args.target_data, args.case)))
