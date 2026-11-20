"""Verify this narrow source-only CPU scope; never import native libraries."""
import ast
import hashlib
import json
from pathlib import Path


def verify(directory,adopted=False):
    directory=Path(directory)
    value=json.loads((directory/"SOURCE_FILES.json").read_bytes())
    for name,pin in value["files"].items():
        p=directory/name
        raw=p.read_bytes()
        if p.is_symlink() or p.parent!=directory or len(raw)!=pin["bytes"] or hashlib.sha256(raw).hexdigest()!=pin["SHA256"]:
            raise ValueError("Native CPU source identity differs")
        if name.endswith(".py"):
            ast.parse(raw,feature_version=(3,10))
    claim=json.loads((directory/"CLAIM.json").read_bytes())
    if (claim["request"]!="7f9a0e2809bf4f53a7294fdc3dc74225" or claim["label"]!=directory.name
            or claim["planned_rows"]!=88 or claim["planned_encode_calls"]!=176
            or claim["planned_collator_calls"]!=88 or claim["GRES"]!=0
            or claim["whole_seconds"]!=300 or claim["Slurm_seconds"]!=240 or claim["worker_seconds"]!=210
            or any(claim[k] is not adopted for k in ("root_GO","CPU_GO"))
            or any(claim[k] is not False for k in ("GPU_GO","training_GO","Model_GO"))):
        raise ValueError("CPU-only request, bound, or adoption differs")
    assets=json.loads((directory/"ASSETS.json").read_bytes())
    if (len(assets["metadata"])!=9 or "model.safetensors" in assets["metadata"] or len(assets["media"])!=22
            or assets["initial"]["SHA256"]!="d3824dc5a788a0238861ae86dec0374dac7b5bf7e6e945e163c32877d6b86d83"
            or assets["initial"]["named_SHA256"]!="9ac0f645b35f75f38aa29770c26f9298e765a4ee4190c86c9481a34bea6bac16"):
        raise ValueError("Existing model metadata/initial/train input scope differs")
    root=directory.parents[2]
    for pin in claim["public_prerequisites"]:
        raw=(root/pin["path"]).read_bytes()
        if len(raw)!=pin["bytes"] or hashlib.sha256(raw).hexdigest()!=pin["SHA256"]:
            raise ValueError("Published prerequisite differs")
    return dict(passed=True,source_members=len(value["files"]),Python310_AST=True,adopted=adopted,
                actual_native_calls=0,actual_GPU=0,actual_optimizer_updates=0)
