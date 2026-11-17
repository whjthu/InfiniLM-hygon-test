"""Verify the exact CPU-only three-node entry-gate package without importing native modules."""
import ast
import hashlib
import json
from pathlib import Path


def verify(source):
    raw=(source/"CLAIM.json").read_bytes();claim=json.loads(raw)
    expected=dict(claim["source_SHA256"]);expected["CLAIM.json"]=hashlib.sha256(raw).hexdigest()
    manifest=json.loads((source/"SOURCE_FILES.json").read_bytes())["files"]
    if set(manifest)!=set(expected) or {x.name for x in source.iterdir()}!=set(expected)|{"SOURCE_FILES.json"}:
        raise ValueError("CPU gate source denominator differs")
    for name,digest in expected.items():
        path=source/name
        if Path(name).name!=name or path.is_symlink() or not path.is_file() or path.stat().st_size>1024**2:
            raise ValueError("unsafe CPU gate source")
        data=path.read_bytes()
        if len(data)!=manifest[name]["bytes"] or hashlib.sha256(data).hexdigest()!=digest or manifest[name]["SHA256"]!=digest:
            raise ValueError("CPU gate source changed")
        if name.endswith(".py"):ast.parse(data)
    expected_arms={"base":{"parent":"1087656","node":"a01r4n14"},"old_domain":{"parent":"1087656","node":"a01r4n16"},"domain":{"parent":"1103606","node":"f13r1n10"}}
    if claim["arms"]!=expected_arms or (claim["GPU_cards"],claim["whole_epoch_seconds"],claim["inside_seconds_per_arm"],claim["Slurm_seconds_per_arm"],claim["GPU_GO"])!=(0,600,90,120,False):
        raise ValueError("CPU gate resources or qualification changed")
    frozen=json.loads((source/"FROZEN_NATIVE.json").read_bytes())
    if set(frozen)!={"native_generation.py","check_explicit_dev_inputs.py","check_qwen_native_inputs.py"}:
        raise ValueError("native entry source denominator differs")
    for name,digest in frozen.items():
        if name not in ("native_generation.py","check_explicit_dev_inputs.py","check_qwen_native_inputs.py") or hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("frozen native bound-input entry differs")
    return dict(passed=True,package_files=len(manifest)+1,AST_checked=True,actual_GPU_invocations=0,
                actual_container_or_native_entry_executed=False,GPU_GO=False)


if __name__=="__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent)))
