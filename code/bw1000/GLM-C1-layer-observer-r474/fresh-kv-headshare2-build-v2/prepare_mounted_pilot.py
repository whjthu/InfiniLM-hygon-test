"""Build a new pilot with a runtime-importable shallow container package."""

import json
from pathlib import Path

from prepare_pilot import REQUEST as PRIOR, build_inputs as previous_inputs, digest


REQUEST = "b9313d611fce4c30a27f14fd769ea8cb"
FAILED_SHA256 = "c3e328abf3e7e5b22592594819a7f79136aaeafc39031c508bafc8993b7f7ddf"


def replace(raw, before, after):
    if raw.count(before) != 1:
        raise ValueError("Original pilot source anchor differs")
    return raw.replace(before, after, 1)


def build_inputs(repo):
    failure = (repo / "results/bw1000/2026-11-23/GLM-headshare-pilot-original1450-target-import-terminal.json").read_bytes()
    prior = json.loads(failure)
    if (digest(failure) != FAILED_SHA256 or prior["request"] != PRIOR or not prior["post_clean"]
            or prior["device_calls_confirmed"] != 0 or prior["actual_accounting"][0][0] != "1103606.1450"):
        raise ValueError("Previous pilot terminal and independent post differ")
    files = previous_inputs(repo)
    files.pop("SOURCE_FILES.json")
    files["launch_plan.py"] = replace(files["launch_plan.py"],
        f"PILOT_REQUEST = {PRIOR!r}\n".encode(), f"PILOT_REQUEST = {REQUEST!r}\n".encode())
    ready = files["prepare.py"]
    ready = replace(ready, b"REPO = HERE.parents[3]\n", b"")
    ready = replace(ready, b"def build_inputs():\n", b"def build_inputs():\n    repo = HERE.parents[3]\n")
    if ready.count(b"pinned(REPO/") != 2:
        raise ValueError("Original build-only path callers differ")
    files["prepare.py"] = ready.replace(b"pinned(REPO/", b"pinned(repo/")
    namespace = {"__file__": str(Path(Path.cwd().anchor) / "headshare" / "prepare.py"), "__name__": "installed"}
    exec(compile(files["prepare.py"], "prepare.py", "exec"), namespace)
    if len(namespace["EXPORTS"]) != 2 or not callable(namespace["pinned"]):
        raise ValueError("Shallow installed package lost runtime constants")
    files["SOURCE_FILES.json"] = (json.dumps({
        "files_SHA256": {name: digest(raw) for name, raw in files.items()},
        "pilot_request": REQUEST,
        "previous_terminal_SHA256": FAILED_SHA256,
    }, sort_keys=True, indent=2) + "\n").encode()
    return files
