"""Bind the DTK Torch distribution and the module it actually imported."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

DIST = "2.10.0+das.opt1.dtk2604.2606161421.g91a580"
MODULE = "2.10.0"
HIP = "6.3.26113"
MODULE_FILE = "/usr/local/lib/python3.10/dist-packages/torch/__init__.py"


def capture(torch):
    distribution = importlib.metadata.distribution("torch")
    origin = Path(torch.__file__).resolve(strict=True)
    declared = Path(distribution.locate_file("torch/__init__.py")).resolve(strict=True)
    with origin.open("rb") as f:
        raw = f.read(2 * 1024**2 + 1)
    if not 0 < len(raw) <= 2 * 1024**2:
        raise ValueError("unexpected Torch provider source size")
    return dict(file=str(origin), distribution_file=str(declared), SHA256=hashlib.sha256(raw).hexdigest(),
                version=str(torch.__version__), distribution_version=distribution.version,
                HIP_build=torch.version.hip)


def validate(identity):
    expected = dict(file=MODULE_FILE, distribution_file=MODULE_FILE,
                    version=MODULE, distribution_version=DIST, HIP_build=HIP)
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("DTK Torch module/distribution/HIP binding differs")
    digest = identity.get("SHA256")
    if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid imported Torch source digest")


def write_and_validate(torch, output):
    identity = capture(torch)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(identity, stream, indent=2, allow_nan=False)
        stream.write("\n")
    validate(identity)
    return identity
