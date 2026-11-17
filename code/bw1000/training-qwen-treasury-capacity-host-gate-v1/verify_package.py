"""Validate source, original runtime bindings, and zero-GPU scope without execution."""
import ast
import hashlib
import json
from pathlib import Path
import runpy


def verify(source):
    raw=(source/"CLAIM.json").read_bytes();claim=json.loads(raw);expected=dict(claim["source_SHA256"]);expected["CLAIM.json"]=hashlib.sha256(raw).hexdigest()
    mf=json.loads((source/"SOURCE_FILES.json").read_bytes())["files"]
    if set(mf)!=set(expected) or {p.name for p in source.iterdir()}!=set(expected)|{"SOURCE_FILES.json"}:raise ValueError("source denominator differs")
    for name,digest in expected.items():
        p=source/name
        if p.name!=name or p.is_symlink() or not p.is_file():raise ValueError("unsafe source")
        b=p.read_bytes()
        if hashlib.sha256(b).hexdigest()!=digest or mf[name]!={"bytes":len(b),"SHA256":digest}:raise ValueError("source pin differs")
        if name.endswith(".py"):ast.parse(b)
    scope=runpy.run_path(str(source/"scope.py"));frozen=json.loads((source/"FROZEN_RUNTIME.json").read_bytes())
    if scope["FLAGS"]!=frozen["flags"] or str(scope["IMAGE"])!=frozen["image"]:raise ValueError("original container runtime differs")
    if claim["source_install"] != str(scope["RESULTS"] / scope["LABEL"] / "source") or "\\" in claim["source_install"]:
        raise ValueError("canonical POSIX source install differs")
    for arm in scope["ARMS"]:
        if scope["original_binds"](arm)!=frozen["declared_binds"][arm]:raise ValueError("original container binding differs")
    if claim["epoch_seconds"]!=300 or claim["maximum_CPU_steps"]!=3 or claim["GPU_GO"] is not False or claim["canary_allowed"] is not False:raise ValueError("CPU-only bounded scope differs")
    return dict(passed=True,source_files=len(mf)+1,actual_remote_calls=0,actual_GPU_calls=0)


if __name__=="__main__":print(json.dumps(verify(Path(__file__).resolve().parent)))
