"""Emit the exact controller invocation only. Transport and dispatch belong to root."""
import argparse
import json
import math
from pathlib import Path
from contract import (Budget, LABEL, NODE, PARENT, REQUEST, RUN, WHOLE, load,
                      need, validate_preflight, verify_package)


def prepare(epoch, manifest_sha, preflight_sha):
    Budget(epoch)
    verify_package(Path(RUN) / "src", manifest_sha)
    p, pin = load(RUN / "preflight.json")
    need(pin["SHA256"] == preflight_sha, "Exact original preflight binding differs")
    validate_preflight(p, epoch)
    return {"request_id": REQUEST, "parent": PARENT, "node": NODE,
            "epoch": epoch, "deadline": epoch + WHOLE,
            "argv": ["/usr/local/python3.12/bin/python3", "-B", str(RUN / "src/controller.py"),
                     "--workspace", str(RUN), "--epoch", str(epoch),
                     "--source-manifest-sha256", manifest_sha, "--preflight-sha256", preflight_sha],
            "transport": "ROOT_ONLY_NO_DEFAULT_SSH", "implicit_loader_GPU_use": "UNKNOWN"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoch", type=float, required=True)
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--preflight-sha256", required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.epoch, a.source_manifest_sha256, a.preflight_sha256)))


if __name__ == "__main__":
    main()
