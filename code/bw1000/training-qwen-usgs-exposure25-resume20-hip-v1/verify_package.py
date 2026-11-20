"""Source/dependency/GO checks only; never import model libraries or read data."""
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
            raise ValueError("Task source identity differs: "+name)
        if name.endswith(".py"):
            ast.parse(raw)
    c=json.loads((directory/"CLAIM.json").read_bytes())
    if (c["request"]!="f572fda6784140cbbdcae7bb91a916b9" or c["label"]!=directory.name
            or c["optimizer_updates"]!=30 or c["updates_by_arm"]!={"continuous":25,"resumed":5}
            or c["train_source_rows"]!=88 or c["training_exposure_rows"]!=800
            or c["whole_seconds"]!=2400 or c["gpu_seconds"]!=2100 or c["native_seconds"]!=1980
            or any(c[k] is not adopted for k in ("root_GO","GPU_GO","training_GO"))
            or any(c[k] is not False for k in ("Model_GO","formal_training_GO","effect_GO","performance_GO"))):
        raise ValueError("Task scope/adoption differs")
    root=directory.parents[2]
    for item in c["public_prerequisites"]:
        raw=(root/item["path"]).read_bytes()
        if len(raw)!=item["bytes"] or hashlib.sha256(raw).hexdigest()!=item["SHA256"]:
            raise ValueError("Published prerequisite differs")
    return dict(passed=True,members=len(value["files"]),adopted=adopted,
                actual_GPU=0,actual_new_native_processor=0,actual_optimizer_updates=0)
