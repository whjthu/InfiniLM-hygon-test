"""Verify the exact five-update package without importing training providers."""
import ast
import hashlib
import json
from pathlib import Path


def verify(source):
    claim_raw = (source / "CLAIM.json").read_bytes()
    claim = json.loads(claim_raw)
    manifest = json.loads((source / "SOURCE_FILES.json").read_bytes())["files"]
    expected = dict(claim["files"])
    expected["CLAIM.json"] = hashlib.sha256(claim_raw).hexdigest()
    if set(expected) != set(manifest) or {p.name for p in source.iterdir()} != set(expected) | {"SOURCE_FILES.json"}:
        raise ValueError("package member denominator differs")
    for name, digest in expected.items():
        path = source / name
        if Path(name).name != name or path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
            raise ValueError("unsafe source member")
        raw = path.read_bytes()
        if (len(raw) != manifest[name]["bytes"] or hashlib.sha256(raw).hexdigest() != digest
                or manifest[name]["SHA256"] != digest):
            raise ValueError("published source member changed")
        if name.endswith(".py"):
            ast.parse(raw)
    if (claim["label_prefix"], claim["run_directory"], claim["parent"], claim["node"],
        claim["DCU"], claim["cpus"], claim["memory_MiB"], claim["whole_epoch_seconds"],
        claim["max_GPU_invocations"], claim["max_native_optimizer_steps"], claim["required_checkpoint_steps"]) != (
        "training-qwen-treasury-explicit-continuous-hip-v1", "training-qwen-treasury-explicit-sft-v1",
        "1087656", "a01r4n16", 8, 64, 224000, 1620, 1, 5, [2, 4, 5]):
        raise ValueError("frozen independent training scope changed")
    return dict(passed=True, package_files=len(manifest) + 1, AST_checked=True,
                actual_GPU_invocations=0, native_training_validated=False)


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent)))
