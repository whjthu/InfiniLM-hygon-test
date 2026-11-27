"""Private OFF train-only inputs. No dataset discovery."""
import hashlib
import json
from pathlib import Path
import re
import stat

MEDIA = {f"off-image-{i:02d}.jpg" for i in range(1, 5)}


def need(ok, code):
    if not ok:
        raise ValueError(code)


def read_file(path, extent):
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_size == extent, "OFF_INPUT_TYPE_OR_EXTENT")
    body = path.read_bytes()
    after = path.lstat()
    key = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(key(before) == key(after), "OFF_INPUT_CHANGED")
    return body


def check_row(row, ordinal):
    need(type(row) is dict and set(row) == {"id", "images", "messages"} and row["id"] == f"off-train-{ordinal:02d}", "OFF_ROW_IDENTITY")
    images, messages = row["images"], row["messages"]
    need(type(images) is list and len(images) == 1 and images[0] in MEDIA, "OFF_MEDIA_NAME")
    need(type(messages) is list and len(messages) == 2
         and all(type(x) is dict and set(x) == {"role", "content"} and type(x["content"]) is str for x in messages)
         and [x["role"] for x in messages] == ["user", "assistant"], "OFF_MESSAGES")
    need(messages[0]["content"].count("<image>") == 1, "OFF_IMAGE_PLACEHOLDER")
    answer = messages[1]["content"]
    if answer == "not readable":
        return "not_readable"
    need(re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)? g", answer), "OFF_ANSWER_UNIT")
    return "numeric_g"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_inputs(root, expected):
    root = Path(root)
    need(root.resolve(strict=True) == root and root.is_dir(), "OFF_INPUT_ROOT")
    manifest_path = root / "INPUT_FILES.json"
    size = manifest_path.lstat().st_size
    need(0 < size <= 8192, "OFF_INPUT_MANIFEST_EXTENT")
    manifest_raw = read_file(manifest_path, size)
    need(digest(manifest_raw) == expected, "OFF_INPUT_MANIFEST_SHA")
    manifest = json.loads(manifest_raw)
    files = manifest["files"]
    allowed = MEDIA | {"train.jsonl"}
    need(manifest["role"] == "OFF_TRAIN_ONLY" and manifest["planned"] == 6 and manifest["unique_media"] == 4
         and set(files) == allowed and {p.name for p in root.iterdir()} == allowed | {"INPUT_FILES.json"}, "OFF_TRAIN_SCOPE")
    for name, pin in files.items():
        cap = 8192 if name == "train.jsonl" else 2 * 1024**2
        need(0 < pin["bytes"] <= cap, "OFF_INPUT_MEMBER_EXTENT")
        raw = read_file(root / name, pin["bytes"])
        need(digest(raw) == pin["SHA256"], "OFF_INPUT_MEMBER_SHA")
        if name == "train.jsonl":
            lines = raw.splitlines()
    need(len(lines) == len(manifest["rows"]) == 6
         and len({files[name]["SHA256"] for name in MEDIA}) == 4, "OFF_TRAIN_COUNTS")
    rows, used, counts = [], set(), {"not_readable": 0, "numeric_g": 0}
    for ordinal, (line, provenance) in enumerate(zip(lines, manifest["rows"]), 1):
        row = json.loads(line)
        classification = check_row(row, ordinal)
        image = row["images"][0]
        need(provenance["ordinal"] == ordinal and provenance["staged_row_SHA256"] == digest(line)
             and provenance["image"] == image and provenance["answer_class"] == classification
             and re.fullmatch(r"[0-9a-f]{64}", provenance["original_row_SHA256"]), "OFF_ROW_PROVENANCE")
        rows.append(dict(row, images=[str(root / image)]))
        used.add(image)
        counts[classification] += 1
    need(used == MEDIA and counts == {"not_readable": 2, "numeric_g": 4}, "OFF_CONTENT_COUNTS")
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input-sha", required=True)
    args = parser.parse_args()
    rows = read_inputs(args.root, args.input_sha)
    print(json.dumps(dict(status="OFF_TRAIN_STAGE_EXACT", input_manifest_SHA256=args.input_sha,
                          train_rows=len(rows), media_files=len({r["images"][0] for r in rows})),
                     sort_keys=True), flush=True)
