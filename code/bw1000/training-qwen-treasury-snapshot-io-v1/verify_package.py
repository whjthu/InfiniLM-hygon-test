"""Verify the bounded zero-GPU transport scope without platform/SDK imports."""
import ast
import hashlib
import json
from pathlib import Path


def verify(source):
    raw=(source/"CLAIM.json").read_bytes()
    claim=json.loads(raw)
    expected=dict(claim["source_SHA256"])
    expected["CLAIM.json"]=hashlib.sha256(raw).hexdigest()
    manifest=json.loads((source/"SOURCE_FILES.json").read_bytes())["files"]
    if set(manifest)!=set(expected) or {p.name for p in source.iterdir()}!=set(expected)|{"SOURCE_FILES.json"}:
        raise ValueError("source denominator differs")
    for name,digest in expected.items():
        p=source/name
        if p.name!=name or p.is_symlink() or not p.is_file():
            raise ValueError("unsafe source")
        raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest or manifest[name]!={"bytes":len(raw),"SHA256":digest}:
            raise ValueError("source identity differs")
        if name.endswith(".py"):
            ast.parse(raw)
    if claim["GPU_GO"] is not False or claim["actual_expected_GPU_cards"]!=0 or claim["maximum_CPU_steps"]!=3 or claim["epoch_seconds"]!=600:
        raise ValueError("HOST-only scope differs")
    return dict(passed=True,source_files=len(manifest)+1,actual_GPU_calls=0)


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent)))
