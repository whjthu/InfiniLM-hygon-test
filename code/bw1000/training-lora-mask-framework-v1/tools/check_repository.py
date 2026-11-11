"""Reject generated artifacts, private paths and review residue in source trees."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORIES = {".git", "__pycache__", ".pytest_cache", ".venv", "build", "dist"}
FORBIDDEN_DIRECTORIES = {"reports", "evidence", "diagnostics", "outputs", "checkpoints"}
FORBIDDEN_SUFFIXES = {
    ".log", ".ndjson", ".safetensors", ".ckpt", ".pt", ".pth", ".bin",
    ".mp4", ".glb", ".docx", ".pdf", ".tar", ".zip", ".pyc",
}
# Keep provider names out of the checker source while checking generated residue.
RESIDUE = re.compile(
    r"(?:co" r"dex|chat" r"gpt|co-authored-by|as an ai|chain.of.thought|<analysis>|<tool_call>)",
    re.IGNORECASE,
)
PRIVATE_PATH = re.compile(r"(?:/mnt/afs/user_home/|/home/whj/|C:[/\\]Users[/\\])")
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
ALLOWED_FILES = {
    ".gitattributes", ".gitignore", ".githooks/pre-commit", "README.md", "UPSTREAM.toml",
    "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md",
    "generation/README.md", "generation/pyproject.toml", "generation/setup.py",
    "generation/sources.lock.json", "generation/.dockerignore",
    "generation/hygon-a100-backend-origin.json",
    "vlm/README.md", "vlm/Dockerfile", "vlm/.dockerignore", "vlm/requirements-mlu.txt",
    "generation/docs/host-controller.md",
    "vlm/runtime/MMF-LICENSE.txt", "vlm/runtime/TEXTVQA-NOTICE.md",
}
SOURCE_DIRECTORIES = {
    "tools/": {".py"}, "tests/": {".py"},
    "generation/src/genmlu/": {".py", ".yaml"},
    "generation/src/gencuda/": {".py", ".yaml"},
    "generation/configs/": {".yaml"}, "generation/patches/": {".patch"},
    "generation/scripts/": {".py", ".sh"}, "generation/tests/": {".py", ".sh"},
    "vlm/runtime/": {".py", ".sh"}, "vlm/plugins/": {".py"},
    "vlm/scripts/": {".py", ".sh"}, "vlm/tests/": {".py", ".sh"},
}


def allowed_source_path(path: PurePosixPath) -> bool:
    name = path.as_posix()
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        return False
    if name in ALLOWED_FILES:
        return True
    if path.parent.as_posix() == "generation/docker" and path.name.startswith("Dockerfile."):
        return bool(re.fullmatch(r"Dockerfile\.[A-Za-z0-9_-]+", path.name))
    return any(name.startswith(directory) and path.suffix in suffixes
               for directory, suffixes in SOURCE_DIRECTORIES.items())


def check_file(name: str, data: bytes) -> list[str]:
    path = PurePosixPath(name)
    errors = []
    if any(part == ".env" or part.startswith(".env.") for part in path.parts):
        errors.append("environment file")
    if not allowed_source_path(path):
        errors.append("path is outside the source/configuration allowlist")
    if any(part in FORBIDDEN_DIRECTORIES for part in path.parts):
        errors.append("generated or review-only directory")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        errors.append("generated or binary artifact")
    if len(data) > 2_000_000:
        errors.append("file exceeds the source-file size limit")
    if b"\0" in data:
        errors.append("binary content")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return errors + ["source is not UTF-8"]
    # The checker defines examples of rejected input; do not flag its own rules.
    if name not in {"tools/check_repository.py", "tests/test_repository_policy.py"}:
        if RESIDUE.search(text):
            errors.append("review or generated-message residue")
        if PRIVATE_PATH.search(text):
            errors.append("developer-specific absolute path")
        if PRIVATE_KEY.search(text):
            errors.append("private key material")
    return errors


def index_files(root: Path):
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--stage", "-z"],
        capture_output=True, check=True,
    )
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        metadata, raw_name = entry.split(b"\t", 1)
        mode, object_id, stage = metadata.decode().split()
        name = raw_name.decode("utf-8")
        if stage != "0" or mode not in {"100644", "100755"}:
            yield name, None
            continue
        blob = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob", object_id],
            capture_output=True, check=True,
        )
        yield name, blob.stdout


def worktree_files(root: Path):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRECTORIES or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            yield relative.as_posix(), None
        elif path.is_file():
            yield relative.as_posix(), path.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--index", action="store_true")
    scope.add_argument("--worktree", action="store_true")
    args = parser.parse_args()
    failures = []
    files = index_files(ROOT) if args.index else worktree_files(ROOT)
    count = 0
    for name, data in files:
        count += 1
        errors = ["symlink, submodule or unresolved merge"] if data is None else check_file(name, data)
        failures.extend(f"{name}: {error}" for error in errors)
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Checked {count} source files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
