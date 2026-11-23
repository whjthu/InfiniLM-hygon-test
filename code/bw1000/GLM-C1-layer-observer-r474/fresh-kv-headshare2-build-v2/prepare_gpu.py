"""Render the original single pilot against the adopted v2 CPU artifact."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import LABEL, OLD, REQUEST, digest, json_bytes, replace_once


PILOT_REQUEST = "0f8bc49a2a514f4187487a07dd9f0055"
V1_REQUEST = "f564ba1c74c94b5aa6bd890874595d24"
REPO = HERE.parents[3]
INPUTS = {
    "device.py": (OLD / "device.py", "06c1179bd53d4fda3c2283e9e7f9a13ff4e4dc86e1b827897fa06cf2e8f51677"),
    "prepare.py": (OLD / "prepare.py", "132a24ff612022f0769e022df4d5937c42ead6c6edc9183c0e9732d4c22ffc98"),
    "launch_plan.py": (OLD / "launch_plan.py", "f6267e3cbc4ae5228f3f5050619c67f081683a8850ee052767e397ec4e8e5d5e"),
    "read_terminal.py": (OLD / "read_terminal.py", "2bc1afc240184671a5f9d37b23d242ec83e7e4204daeda07e4fc4c0f198ea0aa"),
    "hip_runtime.py": (HERE.parent / "routed-i128-one-gpu-caller-v1" / "hip_runtime.py",
                       "74a50bc161afa681043f498977d7c0e79e9de1defb3352905013645bcda16c89"),
    "timing.py": (HERE.parent / "fresh-kv-pack-event-paired-v1" / "timing.py",
                  "6962475a0135702b647a2237eec7fc328676e5e8b277f05bcf6c0c9514f260bf"),
    "node_task_map.json": (REPO / "coordination/training/training80-active-node-task-map-v1.json",
                           "36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9"),
}
BUILD_MANIFEST = "99a4dda85f2485a0d4fa61c82b6699e115d1b5dedb260e36960ab71c962bab75"


def build_inputs():
    staged = {}
    for name, (path, expected) in INPUTS.items():
        raw = path.read_bytes()
        if digest(raw) != expected:
            raise ValueError("frozen GPU source changed: " + name)
        staged[name] = raw
    device = replace_once(
        staged["device.py"], b", LABEL as BUILD_LABEL, digest, pinned\n",
        f", digest, pinned\nBUILD_LABEL = {LABEL!r}\nBUILD_MANIFEST = {BUILD_MANIFEST!r}\n".encode())
    staged["device.py"] = replace_once(
        device, b"build['manifest_SHA256']==scope['build_source_manifest_SHA256']",
        b"build['manifest_SHA256']==scope['build_source_manifest_SHA256']==BUILD_MANIFEST")
    launch = replace_once(staged["launch_plan.py"], f"BUILD_REQUEST = {V1_REQUEST!r}\n".encode(),
                          f"BUILD_REQUESTS = ({V1_REQUEST!r}, {REQUEST!r})\n".encode())
    launch = replace_once(launch, b"request not in (BUILD_REQUEST,PILOT_REQUEST)",
                          b"request not in (*BUILD_REQUESTS,PILOT_REQUEST)")
    staged["launch_plan.py"] = replace_once(
        launch, b"Path(__file__).resolve().parents[4]/'coordination/training/training80-active-node-task-map-v1.json'",
        b"Path(__file__).resolve().with_name('node_task_map.json')")
    if f"PILOT_REQUEST = {PILOT_REQUEST!r}".encode() not in staged["launch_plan.py"]:
        raise ValueError("original pilot request changed")
    staged["SOURCE_FILES.json"] = json_bytes(
        {"files_SHA256": {name: digest(raw) for name, raw in staged.items()},
         "pilot_request": PILOT_REQUEST, "adopted_build_manifest_SHA256": BUILD_MANIFEST})
    return staged


def stage(output):
    sources = build_inputs()
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with TemporaryDirectory(prefix="glm-headshare-gpu-adoption-", dir=output.parent) as tmp:
        tree = Path(tmp) / "source"
        tree.mkdir(mode=0o700)
        for name, raw in sources.items():
            (tree / name).write_bytes(raw)
        tree.rename(output)
    return dict(pilot_request=PILOT_REQUEST, build_label=LABEL, build_manifest_SHA256=BUILD_MANIFEST,
                staged_manifest_SHA256=digest(sources["SOURCE_FILES.json"]), target_GPU_steps=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(stage(parser.parse_args().output), sort_keys=True))
