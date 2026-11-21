"""Replace only the materialized fused kernel with the same-tree shuffle variant."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "output-residual-fused-scores-v1"
BASELINE_KERNEL = "0f77263b5c773e9d3e22d00f01c81962c8fc8d20625513a453b5924e9debc528"


def materialize(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if sha256((PARENT / "fused_scores.cu").read_bytes()).hexdigest() != BASELINE_KERNEL:
        raise ValueError("baseline fused kernel SHA-256 differs")
    baseline = runpy.run_path(str(PARENT / "render.py"))["materialize"]
    with TemporaryDirectory(prefix="kimi-wave-reduction-", dir=output.parent) as folder:
        staged = Path(folder) / "component"
        original = baseline(staged)
        for name in ("fused_scores.cu", "reduction.hpp"):
            (staged / name).write_bytes((HERE / name).read_bytes())
        pins = {path.name: sha256(path.read_bytes()).hexdigest() for path in staged.iterdir()}
        if pins == original or {name for name in original if original[name] != pins[name]} != {"fused_scores.cu"}:
            raise ValueError("rendered component changed outside the fused kernel")
        staged.rename(output)
    return dict(baseline_kernel_SHA256=BASELINE_KERNEL, files_SHA256=pins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(materialize(parser.parse_args().output), indent=2))
