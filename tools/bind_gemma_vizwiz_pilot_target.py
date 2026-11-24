"""Freeze five bounded VizWiz rows and their exact future target media paths."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re


DESTINATION = PurePosixPath(
    "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/"
    "training-gemma-vizwiz-five-native-input-audit-v1/data"
)
PILOT_RECEIPT_SHA256 = "9464368d900442afc06055e601b296a35a550a5d96f72682aaf2387d92df50fa"
ROW_RECEIPT_SHA256 = "3382c10a1feaf6e7864726e7917358afb4607262f8d042f8ddf37391a1c9c97d"
ROW_SOURCE_SHA256 = "a226804fa23cfba8546806fd440587419a8da584f5d1e45802dbd98e109a70e7"
NAME = re.compile(r"VizWiz_train_([0-9]{8})\.jpg")
MAX_IMAGE_BYTES = 384 * 1024


def pin(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def exact(path, digest, maximum):
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("source must be a bounded regular file")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum or pin(raw)["SHA256"] != digest:
        raise ValueError("source bytes differ")
    return raw


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def bind(source_root, rows_root, output):
    official = json.loads(exact(source_root / "pilot-receipt.json", PILOT_RECEIPT_SHA256, 8192))
    transform = json.loads(exact(rows_root / "row-contract-receipt.json", ROW_RECEIPT_SHA256, 8192))
    source = exact(rows_root / "train-swift.jsonl", ROW_SOURCE_SHA256, 8192)
    rows = [json.loads(line) for line in source.splitlines()]
    by_image = {asset["group"]: asset for asset in official["assets"]}
    if (len(rows) != 5 or transform["planned_rows"] != 6 or transform["emitted_rows"] != 5
            or len(by_image) != 6 or len(transform["omitted"]) != 1):
        raise ValueError("pilot denominator changed")
    images_dir = (source_root / "images").resolve(strict=True)
    media, groups, identities = {}, set(), []
    for row in rows:
        if row["split"] != "train" or row["source"] != "VizWiz-VQA" or len(row["images"]) != 1:
            raise ValueError("expected one train-only image and supervision")
        local = Path(row["images"][0])
        name = local.name
        matched = NAME.fullmatch(name)
        if (not matched or local != images_dir / name or name in media or row["id"] != "vizwiz-" + matched[1]
                or row["group"] != sha256(name.encode()).hexdigest() or row["group"] in groups):
            raise ValueError("wrong or duplicate original image identity")
        messages = row["messages"]
        if ([entry["role"] for entry in messages] != ["user", "assistant"]
                or not messages[0]["content"].startswith("<image>\n")
                or (messages[1]["content"] != "unanswerable") != bool(by_image[name]["answerable"])):
            raise ValueError("row supervision differs from the pilot")
        expected = by_image[name]["image"]
        blob = exact(local, expected["SHA256"], MAX_IMAGE_BYTES)
        if pin(blob) != expected:
            raise ValueError("media byte count differs")
        groups.add(row["group"])
        target = DESTINATION / "images" / name
        row["images"] = [target.as_posix()]
        media[name] = dict(expected, source_member=f"images/{name}", target_path=target.as_posix())
        identities.append(row["group"])
    target_rows = b"".join(canonical(row) for row in rows)
    record = dict(
        planned_rows=6, bound_rows=5, omitted=transform["omitted"], target_root=DESTINATION.as_posix(),
        source_rows=pin(source), target_rows=pin(target_rows), media=media,
        group_SHA256_order=identities, rows_by_answerability=dict(Counter(
            "unanswerable" if row["messages"][1]["content"] == "unanswerable" else "answerable" for row in rows)),
        native_rows=0, optimizer_updates=0, training_READY=False, local_image_copies_written=0,
    )
    output.mkdir()
    (output / "train-swift.jsonl").write_bytes(target_rows)
    (output / "data-binding-private.json").write_bytes(canonical(record))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("rows_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = bind(args.source_root, args.rows_root, args.output)
    print(json.dumps({key: receipt[key] for key in ("planned_rows", "bound_rows", "target_rows",
                                                   "native_rows", "training_READY")}, sort_keys=True))


if __name__ == "__main__":
    main()
