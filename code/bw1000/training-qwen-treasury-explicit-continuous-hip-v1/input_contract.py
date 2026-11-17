"""Verify the independently pinned train-only instruction revision."""
import hashlib
import json
from pathlib import PurePosixPath

OLD_SHA = "e0219347307a94259b826b929a88f3fe991b688026677c86c25075c6941bc19b"
NEW_SHA = "5a6a8f02dbfc77aa46487a0180000275771aefd56e3cc933ffdcf7c97496162e"
IMAGE_ROOT = PurePosixPath("/refactor/results/training-qwen-treasury-pilot-data-v1/data/train")
FORMAT_INSTRUCTION = (
    " Output only the amount as a decimal with exactly two digits after the decimal point. "
    "Use a minus sign for negative amounts and no currency symbol, commas, or explanation."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_revision(original, revised):
    require(hashlib.sha256(original).hexdigest() == OLD_SHA, "original dataset changed")
    require(hashlib.sha256(revised).hexdigest() == NEW_SHA, "revised dataset changed")
    old = [json.loads(line) for line in original.splitlines()]
    new = [json.loads(line) for line in revised.splitlines()]
    require(len(old) == len(new) == 160, "training denominator changed")
    images = set()
    changed = 0
    for before, after in zip(old, new):
        require(set(before) == set(after) == {"images", "messages"}, "row fields changed")
        require(before["images"] == after["images"] and len(after["images"]) == 1,
                "image identity or order changed")
        image = PurePosixPath(after["images"][0])
        require(image.parent == IMAGE_ROOT and image.suffix == ".png", "image outside train split")
        images.add(image.name)
        require(len(before["messages"]) == len(after["messages"]) == 2, "message count changed")
        require([m["role"] for m in after["messages"]] == ["user", "assistant"], "message roles changed")
        require(before["messages"][1] == after["messages"][1], "training answer changed")
        left, right = before["messages"][0], after["messages"][0]
        require(set(left) == set(right) == {"role", "content"}, "user message schema changed")
        if left != right:
            require(right["content"] == left["content"] + FORMAT_INSTRUCTION
                    and "What is the signed change in " in left["content"], "unexpected instruction edit")
            changed += 1
    require(changed == 53 and len(images) == 72, "instruction or image denominator changed")
    return dict(rows=160, images=72, changed_user_instructions=53, labels_changed=0, row_order_changed=False)
