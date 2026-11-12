"""Identify executed package contents independently of user-supplied labels."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def package_digest(package: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".yaml"}:
            continue
        name = path.relative_to(package).as_posix().encode("utf-8")
        data = path.read_bytes().replace(b"\r\n", b"\n")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def source_identity(package: Path | None = None) -> dict[str, Any]:
    package = Path(__file__).resolve().parent if package is None else package.resolve()
    identity: dict[str, Any] = {
        "package_sha256": package_digest(package),
        "commit": None,
        "tree": None,
        "dirty": None,
        "origin": "package-content",
    }
    # Do not attribute an installed package to an unrelated enclosing checkout.
    root = package.parents[2]
    if (root / "generation" / "src" / "gencuda").resolve() == package and (root / ".git").exists():
        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(root), *args], check=True, capture_output=True,
                text=True, timeout=10,
            ).stdout.strip()

        try:
            commit = git("rev-parse", "HEAD")
            tree = git("rev-parse", "HEAD^{tree}")
            dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
        except (OSError, subprocess.SubprocessError):
            pass
        else:
            identity.update(commit=commit, tree=tree, dirty=dirty, origin="git-checkout")
            return identity
    build_file = package / "_build_identity.json"
    if build_file.is_file():
        build = json.loads(build_file.read_text(encoding="utf-8"))
        if not isinstance(build, dict) or build.get("package_sha256") != identity["package_sha256"]:
            raise ValueError("installed package does not match its build identity")
        identity.update(
            commit=build.get("commit"), tree=build.get("tree"),
            dirty=build.get("dirty"), origin="package-build",
        )
    return identity


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
