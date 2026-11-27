"""Allocate and check one writable node-local MIOpen home per inference rank."""

import os
from pathlib import Path
import stat
import sys


PRIVATE = (
    ".config", ".config/miopen", ".config/miopen/user-db", ".config/miopen/miopen-lockfiles",
    ".cache", ".cache/miopen", ".cache/huggingface",
)
UNSET = (
    "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME",
    "MIOPEN_SYSTEM_DB_PATH", "MIOPEN_LOCKFILE_PATH", "MIOPEN_LOCKFILE_DIR",
)


def environment(home, inherited):
    result = {key: value for key, value in inherited.items() if key not in UNSET}
    result.update(HOME=str(home), MIOPEN_USER_DB_PATH=str(home / ".config/miopen/user-db"),
                  MIOPEN_CUSTOM_CACHE_DIR=str(home / ".cache/miopen"),
                  HF_HOME=str(home / ".cache/huggingface"))
    return result


def prepare(home, inherited):
    home.mkdir(mode=0o700)
    for name in PRIVATE:
        (home / name).mkdir(mode=0o700)
    return environment(home, inherited)


def verify(home):
    """Check the real child environment and acquire a write probe in each MIOpen directory."""
    expected = environment(home, {})
    if ("torch" in sys.modules or any(os.environ.get(key) != value for key, value in expected.items())
            or any(os.environ.get(key) for key in UNSET)):
        raise ValueError("Original inference requires an isolated MIOpen environment before Torch")
    root = home.stat()
    for path in (home, *(home / name for name in PRIVATE)):
        actual = path.lstat()
        if (path.resolve(strict=True) != path or not stat.S_ISDIR(actual.st_mode)
                or actual.st_dev != root.st_dev or actual.st_uid != os.getuid()
                or stat.S_IMODE(actual.st_mode) != 0o700):
            raise ValueError("MIOpen directory is not a private directory in the rank scratch filesystem")
    dirs = [home / name for name in (
        ".config/miopen/user-db", ".config/miopen/miopen-lockfiles", ".cache/miopen",
    )]
    for directory in dirs:
        if any(directory.iterdir()):
            raise ValueError("Rank MIOpen directory must start cold")
        marker = directory / ".write-probe"
        with marker.open("xb") as file:
            file.write(b"probe")
        marker.unlink()
    return {"cold": True, "writer": "rank_local", "write_probes": len(dirs),
            "filesystem_device": root.st_dev, "directory_inode": root.st_ino}
