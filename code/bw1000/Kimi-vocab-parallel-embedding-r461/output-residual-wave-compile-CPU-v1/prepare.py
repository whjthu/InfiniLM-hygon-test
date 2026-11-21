"""Materialize exactly the two existing Kimi output-residual score variants."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARMS = {"full-tree": "output-residual-fused-scores-v1", "wave32": "output-residual-wave-reduction-v1"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(output):
    if output.is_symlink():
        raise FileExistsError(output)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((HERE / "BUILD_PLAN.json").read_bytes())
    for path, expected in plan["repo_source_SHA256"].items():
        if sha((ROOT / path).read_bytes()) != expected:
            raise ValueError(f"frozen source drift: {path}")
    with TemporaryDirectory(prefix="kimi-score-compare-", dir=output.parent) as temp:
        stage = Path(temp) / "source"
        stage.mkdir(mode=0o700)
        inventory = {}
        for arm, source in ARMS.items():
            runpy.run_path(str(HERE.parent / source / "render.py"))["materialize"](stage / arm)
            inventory[arm] = {p.name: sha(p.read_bytes()) for p in (stage / arm).iterdir()}
        if inventory != plan["rendered_files_SHA256"]:
            raise ValueError("rendered score variant drift")
        before, after = (inventory[arm] for arm in ARMS)
        if set(after) - set(before) != {"reduction.hpp"} or before.keys() - after.keys() or {
                k for k in before if before[k] != after[k]} != {"fused_scores.cu"}:
            raise ValueError("unexpected score variant changes")
        for name in ("compile_once.py", "BUILD_PLAN.json"):
            (stage / name).write_bytes((HERE / name).read_bytes())
        pins = {p.relative_to(stage).as_posix(): sha(p.read_bytes()) for p in sorted(stage.rglob("*")) if p.is_file()}
        manifest = json.dumps({"files_SHA256": pins}, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        (stage / "SOURCE_FILES.json").write_bytes(manifest)
        stage.rename(output)
    return {"manifest_SHA256": sha(manifest), "files": len(pins), "target_compiles": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_output", type=Path)
    print(json.dumps(prepare(parser.parse_args().new_output), indent=2, allow_nan=False))
