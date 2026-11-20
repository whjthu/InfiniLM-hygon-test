"""Bind the fixed 80 development prompts, before making any model requests."""
import hashlib
import json
from pathlib import Path
import re


def pinned(path, expected):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError("development input identity differs")
    return raw


def bound_inputs(root, inventory):
    binding=json.loads(inventory.read_bytes())
    rows=[json.loads(line) for line in pinned(root/"swift.jsonl", binding["dataset_SHA256"]).splitlines()]
    if len(rows)!=80 or len({row["id"] for row in rows})!=80:
        raise ValueError("development prompt denominator")
    found=set()
    for row in rows:
        if not re.fullmatch(r"dev-[0-9]{3}-[1-4]",row["id"]) or len(row["messages"])!=1 or row["messages"][0]["role"]!="user":
            raise ValueError("unexpected development prompt or labels")
        [image]=row["images"]
        path=Path(image)
        if path.parent != root/"media" or not re.fullmatch("table-[0-9]{3}[.]png",path.name):
            raise ValueError("development media scope")
        if path.name not in found:
            pinned(path,binding["media"][path.name])
            found.add(path.name)
    if found!=set(binding["media"]):
        raise ValueError("development media denominator")
    return rows,found
