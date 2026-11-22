"""Run the sealed nine-pair preflight with explicit zero-byte upstream pins."""

from hashlib import sha256
import importlib.util
from pathlib import Path
import sys


V1_SOURCE_SHA256 = "269aea94adc3d5a90558252028933c4a750c0cdccd081b071da1bba2558563b3"


def allow_explicit_empty(original):
    def read(path, expected, cap, count=None):
        if type(count) is not int or count != 0:
            return original(path, expected, cap, count)
        if (not path.is_absolute() or path.is_symlink()
                or path.resolve(strict=True) != path or not path.is_file()):
            raise ValueError("expected an unaliased original file")
        with path.open("rb") as stream:
            data = stream.read(1)
        if data or expected != sha256(b"").hexdigest():
            raise ValueError(f"frozen zero-byte file differs: {path.name}")
        return data
    return read


def load_v1(path):
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != V1_SOURCE_SHA256:
        raise ValueError("sealed v1 preflight source differs")
    spec = importlib.util.spec_from_file_location("sealed_vggt_nine_pair_v1", path)
    if spec is None:
        raise ValueError("sealed v1 preflight module missing")
    module = importlib.util.module_from_spec(spec)
    # Compile the exact verified bytes so neither cache nor another read changes the source.
    exec(compile(raw, str(path), "exec"), module.__dict__)
    module.read = allow_explicit_empty(module.read)
    return module


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("expected sealed v1 source followed by install or native arguments")
    load_v1(Path(sys.argv.pop(1))).main()
