"""Compose the shared Swift resume opt-in against the frozen a68952c framework."""

import argparse
import hashlib
import json
from pathlib import Path


SOURCES = {
    "tools/hygon_vlm_recipe.py": (
        "code/bw1000/training-qwen-target-native-CPU-v1/deltas/tools/hygon_vlm_recipe.py",
        "71417958a659380bfeea4edace29e9948b2ce549ef3e1dd20e5b9dc17fc402c2",
    ),
    "vlm/runtime/resume_cursor.py": (
        "code/bw1000/training-vlm-resume-cursor-v1/framework/vlm/runtime/resume_cursor.py",
        "ab49439782518c0abd91bb3515d0f102fe056087ccd250effd681eadb9e640d0",
    ),
    "vlm/runtime/resume_witness.py": (
        "code/bw1000/training-qwen-target-unified-framework-a68952c-v1/deltas/vlm/runtime/resume_witness.py",
        "8dcdf503002ecfbb7c0b933196639ac86044abe955924643b5ecc1ba6168af15",
    ),
}
MID_EPOCH_HELPER = (
    "code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/mid_epoch_recovery.py",
    "c40071d9cf48dc128d0b779de14a3e2360147c55730bb6be5c42fa59d3f3a283",
)
GEMMA_WITNESS = (
    "code/bw1000/training-gemma-usgs-shared-resume-v1/gemma_domain_resume.py",
    "188e07812f9b3c9f5916da520f05785e6e7c6c34b95225b75440e78ce777fa01",
)

ORIGINAL_GATE = b'''        if (case != "T05.sft" or len(observers) != 1
                or observers[0]["callback"] != "native_resume_witness"
                or argv.count("nvidia_baseline") != 1):
            raise ValueError("MiniCPM full resume requires its single dedicated native witness")'''
SHARED_GATE = b'''        resume_callback = {
            "T01.sft": "qwen_treasury_resume_witness",
            "T02.sft": "gemma_domain_resume_witness",
            "T05.sft": "native_resume_witness",
        }.get(case)
        if (resume_callback is None or len(observers) != 1
                or observers[0]["callback"] != resume_callback
                or argv.count("nvidia_baseline") != 1):
            raise ValueError("full resume requires the case-specific native witness")'''


def pinned(repo, path, expected):
    raw = (repo / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"frozen input differs: {path}")
    return raw


def build_sources(repo, framework=None):
    source = {path: pinned(repo, published, digest) if framework is None else pinned(framework, path, digest)
              for path, (published, digest) in SOURCES.items()}
    helper = {"__name__": "frozen_mid_epoch_recovery"}
    exec(compile(pinned(repo, *MID_EPOCH_HELPER), MID_EPOCH_HELPER[0], "exec"), helper)
    translator = "tools/hygon_vlm_recipe.py"
    if source[translator].count(ORIGINAL_GATE) != 1:
        raise ValueError("frozen Swift recovery gate differs")
    source[translator] = source[translator].replace(ORIGINAL_GATE, SHARED_GATE, 1)
    source[helper["CURSOR"]] = helper["patch_cursor"](source[helper["CURSOR"]])
    source[helper["WITNESS"]] = helper["patch_witness"](source[helper["WITNESS"]])
    source["vlm/plugins/gemma_domain_resume.py"] = pinned(repo, *GEMMA_WITNESS)
    for path, raw in source.items():
        compile(raw, path, "exec")
    return source


def materialize(repo, output, framework=None):
    if framework is not None and output.resolve().is_relative_to(framework.resolve()):
        raise ValueError("output must be separate from the original framework")
    files = build_sources(repo, framework)
    output.mkdir(mode=0o700)
    for path, raw in files.items():
        destination = output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    return {path: hashlib.sha256(raw).hexdigest() for path, raw in files.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--framework", type=Path, help="optional exact target source tree")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.repo, args.output, args.framework), sort_keys=True))


if __name__ == "__main__":
    main()
