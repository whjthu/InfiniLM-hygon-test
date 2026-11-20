"""Local stdlib checks; no device, network or data-body access."""
import ast
import hashlib
import json
from pathlib import Path


def verify(directory, *, adopted=False):
    directory=Path(directory)
    inventory=json.loads((directory/"SOURCE_FILES.json").read_bytes())
    for name,pin in inventory["files"].items():
        path=directory/name
        if path.is_symlink() or path.parent!=directory:
            raise ValueError("Unexpected source member")
        raw=path.read_bytes()
        if len(raw)!=pin["bytes"] or hashlib.sha256(raw).hexdigest()!=pin["SHA256"]:
            raise ValueError("Source identity mismatch: "+name)
        if name.endswith(".py"):
            ast.parse(raw)
    claim=json.loads((directory/"CLAIM.json").read_bytes())
    expected=dict(continuous=25,resume20=5)
    if (claim["label"]!=directory.name or claim["updates_by_phase"]!=expected
            or claim["request"]!="ab8a312813524d949608defa7124c3bf"
            or claim["planned_updates"]!=30 or claim["ordered_training_views"]!=4
            or claim["physical_training_pairs"]!=2 or claim["whole_seconds"]!=1800
            or claim["GPU_GO"] is not adopted or claim["root_GO"] is not adopted
            or claim["training_GO"] is not adopted or claim["Model_GO"] is not False
            or claim["formal17_GO"] is not False or claim["effect_GO"] is not False):
        raise ValueError("Task scope or adoption flags differ")
    repo=directory.parents[2]
    for name,pin in claim["sources"].items():
        path=repo/pin["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=pin["sha"]:
            raise ValueError("Dependency identity differs: "+name)
    return dict(passed=True,source_members=len(inventory["files"]),adopted=adopted,
                actual_GPU=0,actual_training_updates=0)


if __name__=="__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent),sort_keys=True))
