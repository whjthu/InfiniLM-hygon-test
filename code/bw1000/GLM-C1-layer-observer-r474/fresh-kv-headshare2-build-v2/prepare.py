"""Stage the one-TU headshare revision with one exact optional DTK dependency."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import ModuleType


HERE = Path(__file__).resolve().parent
OLD = HERE.with_name("fresh-kv-headshare2-candidate-v1")
LABEL = "GLM474-fresh-kv-headshare2-gfx936-CPU-v2"
REQUEST = "2591f935e17d4a02be02d534af889236"
PREVIOUS_LABEL = "GLM474-fresh-kv-headshare2-gfx936-CPU-v1"
HEADER = "/opt/dtk/include/hip/hip_version.h"
HEADER_BYTES = 377
HEADER_SHA = "5ed79dd7d6ab3e6cfffc9dee3bf34a61df16083e9e30a1490a165013b1d5b9d9"
HISTORICAL = HERE.with_name("depth-stratified-observer-CPU-v1") / "PLAN.json"
HISTORICAL_SHA = "48188514c7137d61d553232eb08f050f6f65d1c41a0be020aeceefb2275fe6c6"
PREPARE_SHA = "132a24ff612022f0769e022df4d5937c42ead6c6edc9183c0e9732d4c22ffc98"
PREVIOUS_STAGE = {
    "fresh_kv_headshare.cu": "9314af2453ed8e798a688b621444cb7af525b5f4cea45cf2ce4c468fe35c7841",
    "compile_once.py": "875f822c05f6fa3b1618ea07c02806da2dfa49e896e7ae1e145a182172f3f890",
    "PLAN.json": "aec9ca6a6e6413b63ba66c73bad99941da421a8b24eb3f60ba35c5dee1880687",
    "SOURCE_FILES.json": "e588d6b8b2d0ba0e14c6dd89921ac1258f293f48fe6da90bc1b99b579652a59c",
}


def digest(raw):
    return sha256(raw).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("expected one frozen compiler boundary")
    return source.replace(old, new)


def frozen_stage():
    historical = HISTORICAL.read_bytes()
    if digest(historical) != HISTORICAL_SHA or json.loads(historical)["system_pins"].get(HEADER) != HEADER_SHA:
        raise ValueError("historical HIP version header pin changed")
    path = OLD / "prepare.py"
    raw = path.read_bytes()
    if digest(raw) != PREPARE_SHA:
        raise ValueError("previous renderer changed")
    previous = ModuleType("headshare_v1")
    previous.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), vars(previous))
    staged = previous.build_inputs()
    if {name: digest(raw) for name, raw in staged.items()} != PREVIOUS_STAGE:
        raise ValueError("previous stage bytes changed")
    return staged


def render(previous):
    if digest(previous) != PREVIOUS_STAGE["compile_once.py"]:
        raise ValueError("previous compiler bytes changed")
    source = replace_once(previous, PREVIOUS_LABEL.encode(), LABEL.encode())
    source = replace_once(source, b"import socket\n", b"import socket\nimport stat\n")
    boundary = b"def main():\n"
    helper = f'''HIP_VERSION = {HEADER!r}
HIP_VERSION_BYTES = {HEADER_BYTES}
HIP_VERSION_SHA = {HEADER_SHA!r}


def admit_hip_version(fresh, allowed):
    if HIP_VERSION not in fresh:
        return
    header = Path(HIP_VERSION)
    info = header.lstat()
    require(stat.S_ISREG(info.st_mode) and str(header.resolve(strict=True)) == HIP_VERSION
            and info.st_size == HIP_VERSION_BYTES and pin(header) == HIP_VERSION_SHA,
            "DTK_HIP_VERSION_HEADER_IDENTITY")
    require(allowed.setdefault(HIP_VERSION, HIP_VERSION_SHA) == HIP_VERSION_SHA,
            "DTK_HIP_VERSION_HEADER_CONFLICT")


'''.encode()
    source = replace_once(source, boundary, helper + boundary)
    boundary = b"        fresh = dependencies(before, obj)\n"
    return replace_once(source, boundary, boundary + b"        admit_hip_version(fresh, allowed)\n")


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def build_inputs():
    staged = frozen_stage()
    staged["compile_once.py"] = render(staged["compile_once.py"])
    plan = json.loads(staged["PLAN.json"])
    plan.update(label=LABEL, compile_entry_SHA256=digest(staged["compile_once.py"]),
                optional_exact_hip_version_header=dict(path=HEADER, bytes=HEADER_BYTES, SHA256=HEADER_SHA))
    staged["PLAN.json"] = json_bytes(plan)
    staged.pop("SOURCE_FILES.json")
    staged["SOURCE_FILES.json"] = json_bytes(
        {"files_SHA256": {name: digest(raw) for name, raw in staged.items()}})
    return staged


def stage(output):
    staged = build_inputs()
    claim = json.loads((HERE / "CLAIM.json").read_bytes())
    manifest = digest(staged["SOURCE_FILES.json"])
    if (claim["label"] != LABEL or claim["stage_manifest_SHA256"] != manifest
            or claim["request"] != REQUEST or not re.fullmatch("[0-9a-f]{32}", REQUEST)):
        raise ValueError("build claim does not bind the rendered source")
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with TemporaryDirectory(prefix="glm-headshare-build-v2-", dir=output.parent) as tmp:
        tree = Path(tmp) / "source"
        tree.mkdir(mode=0o700)
        for name, raw in staged.items():
            (tree / name).write_bytes(raw)
        tree.rename(output)
    return dict(label=LABEL, request=claim["request"], compile_manifest_SHA256=manifest,
                actual_target_compiles=0, actual_GPU_steps=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(stage(parser.parse_args().output), sort_keys=True))
