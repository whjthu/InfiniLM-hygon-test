"""Create a new pilot scope after the original GPU prelaunch rejection."""

from hashlib import sha256
import json
from pathlib import Path

from prepare_gpu import PILOT_REQUEST as ORIGINAL_REQUEST, build_inputs as original_inputs


REQUEST = "30f1580d7463474c8e3cc0df482302cf"
ORIGINAL_PREFLIGHT_EVIDENCE = "17e9e2e09b2caeb458a3d7cbbb678c634f98fb6dd784864e4c6ed212e692f171"
ORIGINAL_MANIFEST = "48da961f6812b82b81ca6614af48eeadddadeace6e48b22e82316ae5e776a0f7"


def digest(raw):
    return sha256(raw).hexdigest()


def build_inputs(repo):
    prior = repo / "results/bw1000/2026-11-23/GLM-headshare-pilot-original1447-pre-GPU-declined.json"
    raw = prior.read_bytes()
    if digest(raw) != ORIGINAL_PREFLIGHT_EVIDENCE:
        raise ValueError("Original admission terminal evidence differs")
    evidence = json.loads(raw)
    if (evidence["request"] != ORIGINAL_REQUEST or evidence["actual_GPU_steps"] != 0
            or evidence["post_clean"] is not True):
        raise ValueError("Original GPU scope has not been reconciled")
    files = original_inputs()
    manifest = files.pop("SOURCE_FILES.json")
    if digest(manifest) != ORIGINAL_MANIFEST:
        raise ValueError("Original GPU package differs")
    old = f"PILOT_REQUEST = {ORIGINAL_REQUEST!r}\n".encode()
    launch = files["launch_plan.py"]
    if launch.count(old) != 1:
        raise ValueError("Original pilot admission anchor differs")
    files["launch_plan.py"] = launch.replace(old, f"PILOT_REQUEST = {REQUEST!r}\n".encode(), 1)
    files["launch_bound.py"] = Path(__file__).with_name("launch_bound.py").read_bytes()
    accounting = (repo / "code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/slurm_accounting.py").read_bytes()
    if digest(accounting) != "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b":
        raise ValueError("Bounded original Slurm accounting helper differs")
    files["slurm_accounting.py"] = accounting
    compile(files["launch_bound.py"], "launch_bound.py", "exec")
    files["SOURCE_FILES.json"] = (json.dumps({
        "files_SHA256": {name: digest(data) for name, data in files.items()},
        "pilot_request": REQUEST,
        "original_prelaunch_evidence_SHA256": ORIGINAL_PREFLIGHT_EVIDENCE,
    }, sort_keys=True, indent=2) + "\n").encode()
    return files
