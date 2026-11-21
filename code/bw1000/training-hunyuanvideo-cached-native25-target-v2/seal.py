"""Refresh only this package's small source manifest after explicit root adoption."""
import hashlib
import json
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    files = {}
    for path in sorted(root.iterdir()):
        if path.name == "SOURCE_FILES.json":
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError("Only delivered regular source members are permitted")
        raw = path.read_bytes()
        files[path.name] = {"bytes": len(raw), "SHA256": hashlib.sha256(raw).hexdigest()}
    raw = (json.dumps({"files": files}, indent=2) + "\n").encode()
    (root / "SOURCE_FILES.json").write_bytes(raw)
    print(json.dumps({"files": len(files) + 1, "bytes": len(raw), "SHA256": hashlib.sha256(raw).hexdigest()}))
