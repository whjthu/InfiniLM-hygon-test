"""Verify frozen package bytes and CPU-only execution contract, without native imports."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(source):
    claim_bytes = (source / "CLAIM.json").read_bytes()
    claim = json.loads(claim_bytes)
    manifest = json.loads((source / "SOURCE_FILES.json").read_bytes())
    expected = dict(claim["source_SHA256"])
    expected["CLAIM.json"] = hashlib.sha256(claim_bytes).hexdigest()
    if set(manifest["files"]) != set(expected):
        raise ValueError("package inventory changed")
    if {path.name for path in source.iterdir()} != set(expected) | {"SOURCE_FILES.json"}:
        raise ValueError("installed package has extra or missing entries")
    for name, digest in expected.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or relative.as_posix() != name or len(relative.parts) != 1:
            raise ValueError("unsafe package path")
        path = source / name
        pin = manifest["files"][name]
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 * 1024:
            raise ValueError("unsafe package file")
        raw = path.read_bytes()
        if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != digest or digest != pin["SHA256"]:
            raise ValueError("published package bytes differ")
        if name.endswith(".py"):
            ast.parse(raw)
    if (claim["parent"], claim["node"], claim["label"], claim["whole_epoch_seconds"], claim["GPU_cards"],
        claim["CPU_cores"], claim["memory_MiB"], claim["maximum_CPU_invocations"]) != (
        "1087656", "a01r4n14", "training-qwen-treasury-explicit-native-CPU-v1", 600, 0, 8, 12288, 1):
        raise ValueError("CPU execution contract changed")
    checker = ast.parse((source / "check_explicit_native_inputs.py").read_bytes())
    dataset_pins = [node.value.value for node in checker.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "DATASET_SHA256"
                            for target in node.targets)]
    if dataset_pins != [claim["dataset_SHA256"]]:
        raise ValueError("native checker dataset pin changed")
    return dict(passed=True, package_files=len(expected) + 1, AST_checked=True,
                actual_native_preprocessing=False, actual_Slurm_steps=0, GPU_cards=0)


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), sort_keys=True))
