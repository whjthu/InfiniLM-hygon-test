"""Verify exact package bytes and three independent one-card placements, without native imports."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath


EXPECTED_ADAPTERS = {
    "base": None,
    "old_domain": "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-treasury-sft-v3/continuous/train/checkpoint-5",
    "domain": "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-treasury-explicit-sft-v1/continuous/train/checkpoint-5",
}


def verify_adapter_paths(arms, scope_arms):
    if set(arms) != set(EXPECTED_ADAPTERS) or set(scope_arms) != set(EXPECTED_ADAPTERS):
        raise ValueError("adapter arm denominator differs")
    for arm, expected in EXPECTED_ADAPTERS.items():
        actual = arms[arm]["adapter"]
        if actual != expected or scope_arms[arm]["adapter"] != expected:
            raise ValueError("claim/scope adapter differs from fixed POSIX path")
        if actual is not None:
            path = PurePosixPath(actual)
            if not path.is_absolute() or path.as_posix() != actual or ".." in path.parts or "\\" in actual:
                raise ValueError("adapter is not canonical absolute POSIX")
    return True


def verify(source):
    raw=(source/"CLAIM.json").read_bytes(); claim=json.loads(raw)
    expected=dict(claim["source_SHA256"]); expected["CLAIM.json"]=hashlib.sha256(raw).hexdigest()
    manifest=json.loads((source/"SOURCE_FILES.json").read_bytes())["files"]
    if set(manifest)!=set(expected) or {x.name for x in source.iterdir()}!=set(expected)|{"SOURCE_FILES.json"}:
        raise ValueError("source member denominator differs")
    for name,digest in expected.items():
        p=source/name
        if Path(name).name!=name or p.is_symlink() or not p.is_file() or p.stat().st_size>1024**2:
            raise ValueError("unsafe source member")
        value=p.read_bytes()
        if len(value)!=manifest[name]["bytes"] or hashlib.sha256(value).hexdigest()!=digest or manifest[name]["SHA256"]!=digest:
            raise ValueError("source member differs")
        if name.endswith(".py"):ast.parse(value)
    locations={arm:(v["parent"],v["node"],v["DCU"],v["CPU_cores"],v["memory_MiB"]) for arm,v in claim["arms"].items()}
    if locations!={"base":("1087656","a01r4n14",1,8,65536),"old_domain":("1087656","a01r4n16",1,8,65536),"domain":("1103606","f13r1n10",1,8,65536)}:
        raise ValueError("three independent placements differ")
    # Scope contains only stdlib definitions. Its as_posix paths are platform-independent.
    import runpy
    scope = runpy.run_path(str(source / "scope.py"))
    scope_arms = scope["ARMS"]
    if claim["label"] != scope["LABEL"] or claim["label"] != "training-qwen-treasury-explicit-dev3-gpu-v2":
        raise ValueError("new namespace differs")
    binding_bytes = (source / "CPU_ENTRY_GATE_BINDING.json").read_bytes()
    if hashlib.sha256(binding_bytes).hexdigest() != "382091ad2cb140044b859ae3588661a96f78c64cf8c434ee5a493d01baf71979":
        raise ValueError("actual independent CPU entry authority differs")
    binding = json.loads(binding_bytes)
    if binding["originals_passed"] is not True or set(binding["arms"]) != set(scope_arms):
        raise ValueError("incomplete CPU entry authority")
    for arm in scope_arms:
        for name, digest in binding["arms"][arm]["native_source_SHA256"].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise ValueError("frozen actual native input entry differs")
    verify_adapter_paths(claim["arms"], scope_arms)
    if (claim["whole_epoch_seconds"],claim["maximum_load_generation_seconds_per_arm"],claim["maximum_GPU_Slurm_seconds_per_arm"],claim["planned_total_rows"])!=(1800,900,960,324):
        raise ValueError("frozen budget or complete denominator differs")
    return dict(passed=True,package_files=len(manifest)+1,AST_checked=True,actual_GPU_invocations=0)


if __name__=="__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent)))
