"""Bind the same checkout at the absolute paths used by frozen dependency symlinks."""

from hashlib import sha256
import json

from prepare_target import LABEL as ORIGINAL_LABEL, REQUEST as ORIGINAL_REQUEST, materials as original_materials
from render import checked_replace


LABEL = "deep-rope-route-tu15-cpu-v2"
REQUEST = "deep-rope-route-tu15-build-20261123-v2"


def materials():
    files = original_materials()
    claim = json.loads(files["CLAIM.json"])
    launcher = files["launch_once.py"].decode()
    for old, new in ((ORIGINAL_LABEL, LABEL), (ORIGINAL_REQUEST, REQUEST)):
        launcher = checked_replace(launcher, old, new)
    launcher = checked_replace(
        launcher,
        '        "--bind", f"{PROJECT}:/refactor:ro", "--bind", f"{STAGE}:/audit:ro",',
        '        "--bind", f"{PROJECT}:/refactor:ro", "--bind", f"{PROJECT}:{PROJECT}:ro",\n'
        '        "--bind", f"{STAGE}:/audit:ro",',
    )
    compile(launcher, "launch_once.py", "exec")
    files["launch_once.py"] = launcher.encode()
    claim.update(label=LABEL, request=REQUEST, supersedes_failed_step="1103606.1427")
    claim["materials"]["launch_once.py"] = sha256(files["launch_once.py"]).hexdigest()
    files["CLAIM.json"] = (json.dumps(claim, sort_keys=True, allow_nan=False) + "\n").encode()
    return files
