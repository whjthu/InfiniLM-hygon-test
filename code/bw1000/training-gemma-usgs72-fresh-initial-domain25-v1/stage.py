"""Zero-GRES target-local verification/copy of software and train-only metadata.

No Torch, model deserializer, network, Slurm, or subprocess is used here.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import time

from fresh_contract import clock, need
from stable_io import read_exact, write_json, pin

D = Path(__file__).resolve().parent
R = "/public/home/aclsylqidf/hygon-inference-refactor-20261014"
ROOT = Path(R + "/results/training-gemma-usgs72-fresh-initial-domain25-v1")
OLD = Path(R + "/results/training-gemma-usgs-render72-exposure25-resume20-v1")
DATA = Path(R + "/results/training-gemma-usgs-render72-native-CPU-v1/data")
STAGE_REQUEST = "7b01c69d091a4230ad18b068a5dbf740"
TRAINING_REQUEST = "247fa3682a114542ac3640fbc726aaae"
PINS = {
    "manifest": dict(bytes=92454, SHA256="7abc7f3183ea45fb7e3681a58b6a16c190bab2cc6cd79c11f40bbf13b55648ff"),
    "view": dict(bytes=602794, SHA256="74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc"),
    "ledger": dict(bytes=8196, SHA256="f2b661c7375b2a418b3aad8c901848807bd4cfe301496aade021b0242ce83a44"),
    "cpu": dict(bytes=344, SHA256="3832af1aef87b88149d955417ad39f665c50fe957a5f91caaf42f44bbc992e80"),
    "train": dict(bytes=54246, SHA256="a897e7d9e7823d01bd2b503104d138f4a50975cabc06a2fefcda1dfb36ff05fd"),
    "binding": dict(bytes=10123, SHA256="7182f239841f2b09ebc6645c2ee9a6d3c2dd1416840d1124e6da88b8c6ee921d"),
}


def safe_relative(value):
    p = PurePosixPath(value)
    need(not p.is_absolute() and ".." not in p.parts and p.parts, "unsafe relative member")
    return p


def verify_package(kind):
    files = json.loads((D / "SOURCE_FILES.json").read_bytes())["files"]
    for name, expected in files.items():
        safe_relative(name)
        need(pin(D / name) == expected, "package member changed")
    claim = json.loads((D / "CLAIM.json").read_bytes())
    need(claim["root_GO"] is True and claim[kind + "_GO"] is True,
         "operative package permission is false")
    if kind == "GPU":
        need(claim["training_GO"] is True, "training permission is false")
    return pin(D / "SOURCE_FILES.json")


def stage(epoch, deadline):
    mono_end = time.monotonic() + clock(epoch, deadline, time.time(), 300)
    def check():
        clock(epoch, deadline, time.time(), 300)
        need(time.monotonic() < mono_end, "staging monotonic deadline exhausted")
    source = verify_package("CPU")
    need(all(os.environ.get(k) == "-1" for k in
             ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")),
         "zero-GRES stage must explicitly hide all GPUs")
    need(Path(__file__).resolve() != ROOT / "stage.py", "source package must remain separate")
    ROOT.mkdir(mode=0o700)  # exclusive; an attempt never overwrites another
    write_json(ROOT / "stage-intent.json", dict(request_id=STAGE_REQUEST, epoch=epoch,
        deadline=deadline, source_manifest=source, actual_updates=0, GPU_steps=0))
    reads = {}
    try:
        def read(path, expected, collect=True):
            raw, actual = read_exact(path, expected, check, collect)
            reads[str(path)] = actual
            return raw
        model = json.loads((D / "MODEL_BINDING.json").read_bytes())
        need(len(model["files"]) == 16 and model["actual_full_target_weight_SHA_verified"] is True,
             "historical model qualification missing")
        for name, expected in model["files"].items():
            need(PurePosixPath(name).name == name, "model leaf is not flat")
            read(Path(model["target_model_directory"]) / name, expected, False)
        mp = OLD / "candidate-framework-private.json"
        manifest_raw = read(mp, PINS["manifest"])
        manifest = json.loads(manifest_raw)
        need(len(manifest["files"]) == 396, "expected existing isolated framework396")
        final = {}
        for name, expected in manifest["files"].items():
            safe_relative(name)
            raw = read(OLD / "framework" / name, expected)
            overlay = D / "overlay" / name
            if overlay.is_file():
                raw = overlay.read_bytes()
            target = ROOT / "framework" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as f:
                f.write(raw)
            target.chmod(expected.get("mode", 0o644))
            final[name] = pin(target)
        overlay_names = {p.relative_to(D / "overlay").as_posix() for p in (D / "overlay").rglob("*") if p.is_file()}
        need(overlay_names <= set(final), "overlay unexpectedly adds an unbound framework member")
        view = read(OLD / "view/train800-swift.jsonl", PINS["view"])
        ledger = read(OLD / "view/exposure-ledger-private.json", PINS["ledger"])
        cpu_raw = read(OLD / "native-input-cpu-result.json", PINS["cpu"])
        cpu = json.loads(cpu_raw)
        need(cpu["passed"] is True and cpu["unique_rows"] == 72 and cpu["encode_calls"] == 144
             and cpu["collator_calls"] == 72 and cpu["optimizer_updates"] == 0
             and cpu["model_loaded"] is False, "native72 receipt incomplete")
        lines = read(DATA / "train-swift.jsonl", PINS["train"]).splitlines(keepends=True)
        need(len(lines) == 72 and view == b"".join(lines[i % 72] for i in range(800)),
             "800-view differs from exact72 source lines")
        binding = json.loads(read(DATA / "data-binding-private.json", PINS["binding"]))
        media = {p["target_path"]: p for n, p in binding["media"].items() if n.startswith("train/media/")}
        paths = {json.loads(line)["images"][0] for line in lines}
        need(len(paths) == 12 and paths == media.keys(), "train-only image mapping differs")
        for path, expected in media.items():
            need(Path(path).is_relative_to(DATA / "train/media"), "training image escapes original train split")
            read(Path(path), expected, False)
        (ROOT / "view").mkdir()
        for name, raw in (("train800-swift.jsonl", view), ("exposure-ledger-private.json", ledger)):
            (ROOT / "view" / name).write_bytes(raw)
        (ROOT / "native-input-cpu-result.json").write_bytes(cpu_raw)
        (ROOT / "structural-binding.json").write_bytes((D / "STRUCTURAL_BINDING.json").read_bytes())
        for arm in ("continuous", "resumed"):
            (ROOT / (arm + "-plan.json")).write_bytes((D / (arm + "-plan.json")).read_bytes())
        write_json(ROOT / "fresh-framework-files.json", dict(files=final))
        versions = {n: importlib.metadata.version(n) for n in
                    ("torch", "transformers", "ms-swift", "tokenizers", "Pillow")}
        need(versions == {"torch": "2.10.0+das.opt1.dtk2604.2606161421.g91a580",
            "transformers": "5.12.1", "ms-swift": "4.4.2", "tokenizers": "0.22.2", "Pillow": "11.3.0"},
            "target metadata versions differ from native72")
        trainer = Path(importlib.metadata.distribution("transformers").locate_file("transformers/trainer.py")).resolve(strict=True)
        trainer_size = trainer.stat().st_size
        need(0 < trainer_size <= 2 * 1024**2, "Trainer source size invalid")
        trainer_raw = trainer.read_bytes()
        trainer_pin = dict(bytes=trainer_size, SHA256=hashlib.sha256(trainer_raw).hexdigest())
        read(trainer, trainer_pin, False)
        write_json(ROOT / "runtime-binding.json", dict(versions=versions,
            trainer=dict(path=str(trainer), **trainer_pin), imports_Torch=False))
        result = dict(state="GEMMA_FRESH_DOMAIN_ASSETS_STAGED", passed=True,
            request_id=STAGE_REQUEST, training_request_id=TRAINING_REQUEST, epoch=epoch, deadline=deadline,
            source_manifest=source, framework_files=396, model_files=16, model_full_SHA_fresh=True,
            unique_train_QA=72, train_media=12, exposures=800, native72_receipt=PINS["cpu"],
            fresh_framework_manifest=pin(ROOT / "fresh-framework-files.json"),
            runtime_binding=pin(ROOT / "runtime-binding.json"),
            initial_reference_read=False, model_deserialized=False, actual_updates=0, GPU_steps=0,
            implicit_Torch_GPU_initialization="NOT_IMPORTED", dev_or_blind_gold_read=False,
            reads=reads, training_ready=False)
        check()
        write_json(ROOT / "stage-complete.json", result)
        return result
    except BaseException as e:
        write_json(ROOT / "stage-failure.json", dict(passed=False, error_type=type(e).__name__,
            reason=str(e)[:512], reads=reads, actual_updates=0, planned_unique_QA=72,
            planned_exposures=800, GPU_steps=0))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoch", required=True, type=float)
    p.add_argument("--deadline", required=True, type=float)
    a = p.parse_args()
    r = stage(a.epoch, a.deadline)
    print(json.dumps({k: r[k] for k in ("state", "passed", "framework_files", "model_files", "actual_updates")}))
