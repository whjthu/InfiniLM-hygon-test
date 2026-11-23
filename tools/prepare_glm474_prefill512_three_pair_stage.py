"""Assemble the frozen GLM474 TP16 three-pair launcher from public repo sources."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import tempfile


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "code/bw1000/GLM-C1-layer-observer-r474"
LAUNCHER = ROOT / "fresh-kv-prefill512-resident-three-pairs-slurm-v1"
SOURCE_SHA = "1cdd9d696a5526e712be0c1d1098a46a462aeecace2788d096f2d8a22b24dd33"
SUPPORT = ROOT / "full-model-attention-detail-v1"
RENDERER = ROOT / "fresh-kv-prefill512-caller-source-v1"
RENDERER_MANIFEST_SHA = "151db4f53f0932b05cf8fb17abe76b3d1cd851a7e62346ad30d778faeb5c2d55"
NATIVE_PROOF = REPO / "results/bw1000/2026-11-22/GLM474-t512-native-original1071-host-verified.json"


def checked(path, digest):
    with path.open("rb") as handle:
        raw = handle.read(2**20 + 1)
    if not 0 < len(raw) <= 2**20 or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Frozen file differs: {path}")
    return raw


def assemble(stage):
    stage.mkdir()
    launcher_manifest = checked(LAUNCHER / "SOURCE_FILES.json", SOURCE_SHA)
    launcher_files = json.loads(launcher_manifest)["files"]
    contract_path = stage / "contract.py"
    contract_path.write_bytes(checked(LAUNCHER / "contract.py", launcher_files["contract.py"]))
    contract = runpy.run_path(str(contract_path))
    (stage / "SOURCE_FILES.json").write_bytes(launcher_manifest)
    for name, digest in launcher_files.items():
        if name != "contract.py":
            (stage / name).write_bytes(checked(LAUNCHER / name, digest))

    render_manifest = json.loads(checked(RENDERER / "SOURCE_FILES.json", RENDERER_MANIFEST_SHA))
    for member in render_manifest["files"]:
        checked(REPO / member["path"], member["SHA256"])
    binding = runpy.run_path(str(RENDERER / "render.py"))["sources"]()["typed_binding.py"]
    for name, digest in contract["STAGED"].items():
        if name == "typed_binding.py":
            if hashlib.sha256(binding).hexdigest() != digest:
                raise ValueError("Generated typed binding differs")
            raw = binding
        else:
            raw = checked(NATIVE_PROOF if name == "1071.json" else SUPPORT / name, digest)
        (stage / name).write_bytes(raw)

    for source, target, manifest_sha in (
        (ROOT / "fresh-kv-prefill512-resident-three-pairs-source-v1", stage / "pair", contract["PAIR_MANIFEST"]),
        (ROOT / "fresh-kv-prefill512-resident-pair-source-v2", stage / "pair/base", contract["BASE_MANIFEST"]),
    ):
        target.mkdir()
        manifest = checked(source / "SOURCE_FILES.json", manifest_sha)
        (target / "SOURCE_FILES.json").write_bytes(manifest)
        for name, record in json.loads(manifest)["files"].items():
            (target / name).write_bytes(checked(source / name, record["SHA256"]))
    contract["verify_stage"](stage, SOURCE_SHA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for the verified 28-file stage")
    args = parser.parse_args()
    # Validate the entire closure before publishing a new local staging directory.
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        stage = Path(temporary) / "stage"
        assemble(stage)
        shutil.copytree(stage, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "source_manifest_SHA256": SOURCE_SHA,
                      "launcher_files_verified": 28, "stage_verified": True,
                      "cluster_admission_checked": False, "slurm_submissions": 0}))


if __name__ == "__main__":
    main()
