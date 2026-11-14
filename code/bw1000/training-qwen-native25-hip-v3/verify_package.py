"""Verify real target sources and assets before reserving a training step."""
import hashlib
import json
from pathlib import Path
import shutil
import stat
from scope import ROOT

R = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
BASE = R/"training-qwen-a68952c-swift-initial-CPU-v1"
STAGE = BASE/"source"
FRAMEWORK = BASE/"framework"
MODEL = R/"training-qwen-original-public-model-v1/model"
INITIAL = R/"training-qwen-initial-trainables-v1/initial.pt"
ASSETS = R/"training-vlm-original-assets-v1"
SWIFT = R/"training-vlm-swift-offline-runtime-v1"
VISION = R/"training-qwen-vision014-CPU-v2/vision"
PROCESSOR = R/"training-minicpm-hygon-processor-r2/cpu/runtime-site"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def exact(path, digest, size=None):
    before = path.lstat()
    assert stat.S_ISREG(before.st_mode) and (size is None or before.st_size == size), str(path)
    got = sha(path)
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    assert identity(before) == identity(after) and got == digest, str(path)


def verify():
    src = ROOT/"src"
    intent = json.loads((ROOT/"intent.json").read_bytes())
    exact(src/"SOURCE_FILES.json", intent["source_manifest_SHA256"])
    for name, pin in json.loads((src/"SOURCE_FILES.json").read_bytes())["files"].items():
        exact(src/name, pin["SHA256"], pin["bytes"])
    exact(R/"training-qwen-a68952c-swift-initial-CPU-v4/outcome.json",
          "f396253981fd63f9f997f7c05383d2dd37bc9943079a0a172db6e1b747cfacbb")
    mf = STAGE/"FRAMEWORK_FILES.json"
    exact(mf, "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41")
    files = json.loads(mf.read_bytes())["files"]
    assert len(files) == 393 and {p.relative_to(FRAMEWORK).as_posix()
            for p in FRAMEWORK.rglob("*") if p.is_file()} == set(files)
    for name, pin in files.items():
        p = FRAMEWORK/name
        exact(p, pin["SHA256"], pin["bytes"])
        assert stat.S_IMODE(p.stat().st_mode) == pin["mode"], name
    pin = json.loads((src/"ASSETS.json").read_bytes())
    assert {p.name for p in MODEL.iterdir()} == set(pin["model"]) and len(pin["model"]) == 14
    for name, file in pin["model"].items():
        exact(MODEL/name, file["SHA256"], file["bytes"])
    for name, file in pin["images"].items():
        exact(ASSETS/name, file["SHA256"], file["bytes"])
    exact(ASSETS/"vlm_report_coco64.hip.jsonl", "1c9aaddad5b681b50a6ddfa79c08e30e4820ee6ffcdc81bd473d0c2e84b35a0f", 26817)
    exact(ASSETS/"plugins/swift_callback_nvidia_baseline_v2.py", "d731f9889c22381514c2181c27f99ac86051930ab2e7792958db5b34e34f27e8", 4692)
    exact(INITIAL,"2a0df4d5fdd4d793d47859f7e6ee9102cd3324199f90d0ed34760d3c7198612f",174821899)
    for name, digest in pin["staged_source"].items():
        exact(STAGE/name,digest)
    exact(SWIFT/"site/swift/cli/main.py", "d84a44f9911c01c9af83db19baf2068ad39a8ec926f6e2c58f6f69779bfdad68")
    exact(SWIFT/"site/swift/cli/sft.py", "97162d10eb5b82cdfadf1ec30a0eac50cc0bc9646140b8342806df9e8e6e07f0")
    exact(VISION.parent/"outcome.json", "235851389a1a6a39c1cf5762c89af2ce77802ae7ed45b45365f5863524a983f3")
    vision = json.loads((src/"VISION_FILES.json").read_bytes())
    assert len(vision)==7 and {p.relative_to(VISION).as_posix()
        for p in VISION.rglob("*") if p.is_file()} == set(vision)
    for name, file in vision.items():
        exact(VISION/name, file["SHA256"], file["bytes"])
    assert shutil.disk_usage(ROOT).free >= 80*1024**3
    return dict(passed=True, framework_files=393, model_files=14,
                model_bytes_rehashed=sum(p["bytes"] for p in pin["model"].values()),
                image_files=len(pin["images"]), GPU_calls=0)
