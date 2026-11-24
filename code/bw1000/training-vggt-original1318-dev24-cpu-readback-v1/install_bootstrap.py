"""Execute the pinned CPU host installer from its in-memory source-only offer."""

from base64 import b64decode
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType


LIMIT = 300000
NAMES = ("worker.py", "host.py", "verify_terminal.py")
PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")


def load_install(raw, packet_sha, pins, parent, request):
    if not 0 < len(raw) <= LIMIT or sha256(raw).hexdigest() != packet_sha:
        raise ValueError("EXACT_SOURCE_ONLY_OFFER")
    embedded = json.loads(raw)["cpu_sources"]
    if set(embedded) != set(pins) or set(pins) != set(NAMES):
        raise ValueError("EXACT_BOOT_SOURCE_SET")
    payload = {name: b64decode(embedded[name], validate=True) for name in NAMES}
    if any({"bytes": len(data), "SHA256": sha256(data).hexdigest()} != pins[name] for name, data in payload.items()):
        raise ValueError("EXACT_BOOT_SOURCE_BYTES")
    existing = sys.modules.pop("worker", None)
    try:
        modules = {}
        for name in ("worker", "host"):
            module = ModuleType(name)
            module.__file__ = "<exact offered " + name + ".py>"
            exec(compile(payload[name + ".py"], module.__file__, "exec"), module.__dict__)
            modules[name] = module
            if name == "worker":
                sys.modules[name] = module
        return modules["host"].install(parent, request, raw, packet_sha, pins)
    finally:
        sys.modules.pop("worker", None)
        if existing is not None:
            sys.modules["worker"] = existing


def main():
    if not (sys.platform == "linux" and sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise ValueError("ISOLATED_SHARED_HOST_INSTALL_ONLY")
    os.umask(0o077)
    request, sha, stamp = sys.argv[1:]
    result = load_install(sys.stdin.buffer.read(LIMIT + 1), sha, json.loads(stamp), PROJECT / "incoming", request)
    print(json.dumps({"install": result, "request": request, "source_files": 21, "Slurm": 0, "GPU": 0}))


if __name__ == "__main__":
    main()
