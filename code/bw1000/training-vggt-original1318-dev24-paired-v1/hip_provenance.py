"""Record the actual libamdhip64 shared object loaded by the single-device scorer."""

from hashlib import sha256
from pathlib import Path
import re


def hip_dso():
    paths = {Path(parts[5]) for line in Path("/proc/self/maps").read_text().splitlines()
             if len(parts := line.split(maxsplit=5)) == 6 and "/libamdhip64.so" in parts[5]}
    if not paths or any(not path.is_absolute() or str(path).endswith(" (deleted)") for path in paths):
        raise ValueError("expected the actual named HIP shared object")
    selected = {path.resolve(strict=True) for path in paths}
    if len(selected) != 1:
        raise ValueError("expected exactly one actual HIP shared object")
    path, = selected
    if not re.fullmatch(r"libamdhip64\.so(?:\.[0-9]+)*", path.name):
        raise ValueError("HIP shared object path differs")
    signature = sha256()
    size = path.stat().st_size
    if not 0 < size <= 256 * 1024**2:
        raise ValueError("HIP shared object exceeds read boundary")
    with path.open("rb") as stream:
        for raw in iter(lambda: stream.read(1024**2), b""):
            signature.update(raw)
    return {"path": path.as_posix(), "bytes": size, "SHA256": signature.hexdigest()}
