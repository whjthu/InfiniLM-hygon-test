"""Consume the real sealed v3 package; resealed false flags must fail ingress."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from unittest.mock import patch

import contract


def digest(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "SHA256": hashlib.sha256(raw).hexdigest()}


def windows_stat_shim(stack):
    """Test-only FD metadata injection; all real file bytes/SHA still pass the production reader."""
    opened = {}
    original_open = Path.open
    original_fstat = os.fstat

    def tracked(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        opened[stream.fileno()] = path
        return stream

    def metadata(fd):
        return opened[fd].lstat() if fd in opened else original_fstat(fd)

    stack.enter_context(patch.object(Path, "open", new=tracked))
    stack.enter_context(patch("contract.os.fstat", side_effect=metadata))


def seal_fixture(folder):
    files = {}
    for path in sorted(folder.iterdir()):
        if path.name != "SOURCE_FILES.json":
            raw = path.read_bytes()
            files[path.name] = {"bytes": len(raw), "SHA256": hashlib.sha256(raw).hexdigest()}
    raw = (json.dumps({"files": files}, indent=2) + "\n").encode()
    (folder / "SOURCE_FILES.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def check(output):
    output.mkdir(exist_ok=False)
    package = Path(__file__).resolve().parent
    actual_manifest = digest(package / "SOURCE_FILES.json")
    started = time.monotonic()
    result = {"passed": False, "request_id": contract.REQUEST,
              "actual_v2_source_manifest": actual_manifest, "operative_claim": digest(package / "CLAIM.json"),
              "GPU_calls": 0, "actual_optimizer_updates": 0, "target_filesystem_not_qualified": True}
    with ExitStack() as stack:
        if os.name == "nt":
            try:
                contract.verify_package(package, actual_manifest["SHA256"])
                result["initial_real_Windows_stat_ingress"] = "PASS"
            except ValueError as error:
                if str(error) not in ("Metadata open identity differs", "Metadata changed during read",
                                      "Metadata path changed"):
                    raise
                result["initial_real_Windows_stat_ingress"] = "REJECTED_RETAINED"
                result["initial_metadata_rejection"] = str(error)
            windows_stat_shim(stack)
            result["Windows_test_FD_stat_injection"] = True
        else:
            result["Windows_test_FD_stat_injection"] = False
        # Exact real v3 package and its real final manifest; no synthetic positive claim.
        contract.verify_package(package, actual_manifest["SHA256"])
        result["actual_package_ingress_passed"] = True
        rejected = []
        with tempfile.TemporaryDirectory(prefix="t09-v3-operative-flags-") as folder:
            for name, expected in contract.EXPECTED_CLAIM_FLAGS.items():
                target = Path(folder) / name
                shutil.copytree(package, target)
                claim = json.loads((target / "CLAIM.json").read_bytes())
                claim[name] = not expected
                (target / "CLAIM.json").write_text(json.dumps(claim, indent=2) + "\n", encoding="utf-8")
                # Rehash the negative fixture so rejection proves operative flag admission,
                # rather than merely noticing an unsealed byte change.
                manifest_sha = seal_fixture(target)
                try:
                    contract.verify_package(target, manifest_sha)
                except ValueError as error:
                    if str(error) != "Operative root claim flag mismatch":
                        raise
                    rejected.append(name)
                else:
                    raise AssertionError("False/mismatched operative claim accepted: " + name)
        result["rejected_resealed_mismatched_flags"] = rejected
        result["negative_count"] = len(rejected)
        result["passed"] = len(rejected) == len(contract.EXPECTED_CLAIM_FLAGS)
    result["wall_seconds"] = time.monotonic() - started
    result["receipt_schema"] = "T09_V3_REAL_PACKAGE_OPERATIVE_CLAIM_INGRESS_CPU"
    raw = (json.dumps(result, indent=2) + "\n").encode()
    with (output / "receipt.json").open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"passed": result["passed"], "negative_count": result["negative_count"],
                      "source_manifest_SHA256": actual_manifest["SHA256"]}))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    result = check(p.parse_args().output)
    raise SystemExit(0 if result["passed"] else 1)
