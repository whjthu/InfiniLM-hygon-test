"""Run the exact grouped-finite training implementation against its saved checkpoint."""

import hashlib
import json
from pathlib import Path
import sys


def worker_source(raw):
    replacements = {
        'OUT = ROOT / mode': 'OUT = Path("/recovery") / mode',
        'intent = json.loads((ROOT / "intent.json").read_bytes())':
            'intent = json.loads(Path("/recovery/intent.json").read_bytes())',
        'deadline = intent["epoch"] + claim["worker_seconds"]':
            'deadline = intent["gpu_deadline"]',
        'intent["request"] == claim["request"]':
            'intent["source_request"] == claim["request"]',
        'result = dict(request=claim["request"], step=step, passed=False,':
            'result = dict(request=intent["request"], source_request=claim["request"], step=step, passed=False,',
    }
    for anchor, replacement in replacements.items():
        key = anchor.encode("ascii")
        if raw.count(key) != 1:
            raise ValueError("Pinned worker anchor differs: " + anchor)
        raw = raw.replace(key, replacement.encode("ascii"), 1)
    return raw


def main():
    original = Path("/fusion-run")
    recovery = Path("/recovery")
    claim = json.loads((recovery / "CLAIM.json").read_bytes())
    pins = claim["source_work"]["small_receipts"]
    manifest = (original / "SOURCE_FILES.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != pins["SOURCE_FILES.json"]:
        raise ValueError("Original checkpoint contract source does not match")
    expected = json.loads(manifest)["files"]["worker.py"]
    entry = original / "worker.py"
    worker = entry.read_bytes()
    if len(worker) != expected["bytes"] or hashlib.sha256(worker).hexdigest() != expected["SHA256"]:
        raise ValueError("Original worker does not match")
    sys.path.insert(0, str(original))
    sys.argv[:] = [str(entry), "resume20"]
    exec(compile(worker_source(worker), str(entry), "exec"), {"__name__": "__main__", "__file__": str(entry)})


if __name__ == "__main__":
    main()
