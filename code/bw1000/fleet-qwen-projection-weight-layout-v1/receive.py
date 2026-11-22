"""Receive the published narrow runtime and start its owned controller once."""

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
REQUEST = "cde5c3eabf0943a8b303a3f260963588"
FILES = {"CLAIM.json", "dispatch.py", "worker.py", "layout_scope.py", "admission.py",
         "ledger.json", "node_sample.py", "qualified_sampler.py"}


def main(payload):
    os.umask(0o077)
    plan = payload["plan"]
    if plan["request"] != REQUEST or not 0 <= time.time() - plan["epoch"] < 40 \
            or not re.fullmatch("[0-9a-f]{40}", payload["publication"]) or set(payload["files"]) != FILES:
        raise ValueError("invalid original backfill envelope")
    runtime = {}
    for name, (digest, raw) in payload["files"].items():
        raw = base64.b64decode(raw, validate=True)
        if not 0 < len(raw) < 32768 or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("source envelope changed")
        runtime[name] = raw
    claim = json.loads(runtime["CLAIM.json"])
    if claim["request"] != REQUEST or claim["max_invocations"] != 1 or claim["operator_GO"] is not True:
        raise ValueError("backfill claim changed")
    target = ROOT / "incoming" / ("fleet-qwen-projection-weight-layout-" + REQUEST)
    target.mkdir(mode=0o700, exist_ok=False)
    for name, raw in runtime.items():
        with (target / name).open("xb") as out:
            out.write(raw)
    p = target / "plan.json"
    with p.open("x", encoding="utf-8") as out:
        json.dump(dict(plan, publication=payload["publication"]), out, allow_nan=False)
    return subprocess.run([sys.executable, "-B", "-u", str(target / "dispatch.py"), str(p)],
                          stdin=subprocess.DEVNULL, timeout=max(0.1, plan["epoch"] + 245 - time.time())).returncode


if __name__ == "__main__":
    raise SystemExit(main(PAYLOAD))
