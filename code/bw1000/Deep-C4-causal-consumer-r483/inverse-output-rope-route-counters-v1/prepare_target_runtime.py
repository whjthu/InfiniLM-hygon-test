"""Restore the original container's pinned libnuma injection for the CPU-only build."""

from hashlib import sha256
import json

from prepare_target_absolute import LABEL as PREVIOUS_LABEL, REQUEST as PREVIOUS_REQUEST
from prepare_target_absolute import materials as previous_materials
from render import checked_replace


LABEL = "deep-rope-route-tu15-cpu-v3"
REQUEST = "deep-rope-route-tu15-build-20261123-v3"


def materials():
    files = previous_materials()
    claim = json.loads(files["CLAIM.json"])
    launcher = files["launch_once.py"].decode()
    for old, new in ((PREVIOUS_LABEL, LABEL), (PREVIOUS_REQUEST, REQUEST)):
        launcher = checked_replace(launcher, old, new)
    launcher = checked_replace(launcher, '"exec", "--contain", "--no-home",',
                               '"exec", "--rocm", "--contain", "--no-home",')
    compile(launcher, "launch_once.py", "exec")
    files["launch_once.py"] = launcher.encode()
    claim.update(label=LABEL, request=REQUEST, supersedes_failed_step="1103606.1428")
    claim["materials"]["launch_once.py"] = sha256(files["launch_once.py"]).hexdigest()
    files["CLAIM.json"] = (json.dumps(claim, sort_keys=True, allow_nan=False) + "\n").encode()
    return files
