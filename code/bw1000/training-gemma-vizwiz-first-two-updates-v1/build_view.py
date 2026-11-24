"""Build a training-only Gemma view from the two pinned VizWiz streams."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re


SCOPE_SHA256 = "7b494f529c5d202b522fb35cd5cf89af67bb97775534968518aee109668bd9d0"
TARGET_RESULTS = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
TRAIN_IMAGE = re.compile(r"VizWiz_train_[0-9]{8}\.jpg")


def read_exact(path, size, digest):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file for {path.name}")
    with path.open("rb") as source:
        raw = source.read(size + 1)
    if len(raw) != size or sha256(raw).hexdigest() != digest:
        raise ValueError(f"File identity differs for {path.name}")
    return raw


def load_scope(path):
    size = path.stat().st_size
    if not 0 < size <= 8192:
        raise ValueError("Invalid training scope length")
    return json.loads(read_exact(path, size, SCOPE_SHA256))


def build_view(scope, streams):
    declarations = scope["existing_frozen_rows"]
    if scope["role"] != "train" or tuple(streams) != tuple(row["name"] for row in declarations):
        raise ValueError("Training input sources or order differ")

    source_lines = []
    image_names = set()
    classes = Counter()
    for declaration in declarations:
        stream = streams[declaration["name"]]
        pin = declaration["source_stream"]
        if len(stream) != pin["bytes"] or sha256(stream).hexdigest() != pin["SHA256"]:
            raise ValueError("Frozen training stream identity differs")
        lines = stream.splitlines(keepends=True)
        if [sha256(line).hexdigest() for line in lines] != declaration["ordered_line_SHA256"]:
            raise ValueError("Frozen training line identity or order differs")
        image_root = TARGET_RESULTS / declaration["original_target_stage"] / "data" / "images"
        for line in lines:
            if not line.endswith(b"\n"):
                raise ValueError("Incomplete training line")
            row = json.loads(line)
            messages = row["messages"]
            if [message["role"] for message in messages] != ["user", "assistant"] or len(row["images"]) != 1:
                raise ValueError("Expected one supervised image and conversation")
            image = PurePosixPath(row["images"][0])
            if image.parent != image_root or not TRAIN_IMAGE.fullmatch(image.name) or image.name in image_names:
                raise ValueError("Image is duplicated or outside the original train-only mount")
            answer = messages[1]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Missing SFT supervision")
            image_names.add(image.name)
            classes["unanswerable" if answer == "unanswerable" else "answerable"] += 1
        source_lines.extend(lines)

    if (len(source_lines) != scope["source_row_count"] or len(image_names) != scope["source_image_count"]
            or dict(classes) != scope["class_counts"]):
        raise ValueError("Training source totals differ from the frozen scope")

    view_rows = scope["view_rows"]
    order = [index % len(source_lines) for index in range(view_rows)]
    counts = Counter(order)
    view = b"".join(source_lines[index] for index in order)
    receipt = {
        "schema": "training.gemma.vizwiz.train_only_view.receipt.v1",
        "scope_SHA256": SCOPE_SHA256,
        "source_rows": len(source_lines),
        "source_images": len(image_names),
        "label_class_counts": dict(classes),
        "view_rows": view_rows,
        "view_bytes": len(view),
        "view_SHA256": sha256(view).hexdigest(),
        "exposure_count_by_source_index": [counts[index] for index in range(len(source_lines))],
        "native_rank_input_order_observed": False,
        "optimizer_updates": 0,
    }
    return view, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--five", type=Path, required=True)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scope = load_scope(args.scope)
    streams = {}
    for declaration in scope["existing_frozen_rows"]:
        name = declaration["name"]
        pin = declaration["source_stream"]
        streams[name] = read_exact(getattr(args, name), pin["bytes"], pin["SHA256"])
    view, receipt = build_view(scope, streams)
    args.output.mkdir(mode=0o700)
    with (args.output / "train800-swift.jsonl").open("xb") as output:
        output.write(view)
    with (args.output / "view-receipt.json").open("x", encoding="utf-8", newline="\n") as output:
        json.dump(receipt, output, sort_keys=True)
        output.write("\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
