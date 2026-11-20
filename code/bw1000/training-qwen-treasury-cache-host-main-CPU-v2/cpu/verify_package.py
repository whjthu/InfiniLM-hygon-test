"""Check exact nested12 candidate and bounded CPU package without any native import."""
import ast
import hashlib
import json
from pathlib import Path,PurePosixPath


def verify(root):
    raw=(root/"CPU_CLAIM.json").read_bytes();claim=json.loads(raw);expected=dict(claim["source_SHA256"]);expected["CPU_CLAIM.json"]=hashlib.sha256(raw).hexdigest()
    manifest=json.loads((root/"CPU_PACKAGE_FILES.json").read_bytes())["files"]
    actual={"CPU_CLAIM.json"}
    for directory in ("source","cpu"):
        for p in (root/directory).iterdir():
            if not p.is_file() or p.is_symlink():raise ValueError("unsafe package directory member")
            actual.add(directory+"/"+p.name)
    if actual!=set(expected) or set(manifest)!=set(expected):raise ValueError("CPU package denominator differs")
    for name,digest in expected.items():
        posix=PurePosixPath(name)
        if posix.is_absolute() or ".." in posix.parts or "\\" in name:raise ValueError("unsafe source path")
        p=root/name;b=p.read_bytes()
        if p.is_symlink() or len(b)>2*1024**2 or hashlib.sha256(b).hexdigest()!=digest or manifest[name]!={"bytes":len(b),"SHA256":digest}:raise ValueError("CPU source pin differs")
        if name.endswith(".py"):ast.parse(b)
    raw=(root/"source/SOURCE_FILES.json").read_bytes()
    if hashlib.sha256(raw).hexdigest()!="6119c09c1a0b1cd854f330287f7099577140682cc204f4007534141b8dd24002":raise ValueError("revised12 candidate differs")
    candidate=json.loads(raw)["files"]
    if len(candidate)+1!=12 or {p.name for p in (root/"source").iterdir()}!=set(candidate)|{"SOURCE_FILES.json"}:raise ValueError("candidate12 denominator differs")
    for name,info in candidate.items():
        b=(root/"source"/name).read_bytes()
        if info!={"bytes":len(b),"SHA256":hashlib.sha256(b).hexdigest()}:raise ValueError("candidate source member differs")
    if claim["epoch_seconds"]!=900 or claim["maximum_CPU_steps"]!=3 or claim["GPU_GO"] is not False or claim["maximum_explicit_canary_bytes"]!=122880:raise ValueError("CPU/write budget differs")
    return dict(passed=True,package_files=len(manifest)+1,candidate_exact_files=12,actual_target_calls=0)


if __name__=="__main__":print(json.dumps(verify(Path(__file__).resolve().parent.parent)))
