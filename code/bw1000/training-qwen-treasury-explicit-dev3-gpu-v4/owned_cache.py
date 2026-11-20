"""Own named cache directories and verify the container temp mounts."""
import json
import os
from pathlib import Path

DIRECTORIES = {
    "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "HOME": "home",
    "XDG_CACHE_HOME": "xdg", "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "AMD_COMGR_CACHE_DIR": "comgr", "MIOPEN_USER_DB_PATH": "miopen",
    "MIOPEN_CUSTOM_CACHE_DIR": "miopen", "MIOPEN_CACHE_DIR": "miopen",
    "HF_HOME": "hf_home", "HF_MODULES_CACHE": "hf_modules",
}


def prepare(run):
    cache = run / "cache"
    cache.mkdir(mode=0o700)
    for child in set(DIRECTORIES.values()) | {"var_tmp"}:
        (cache / child).mkdir(mode=0o700)
    return {key: str(cache / value) for key, value in DIRECTORIES.items()}


def observe(run):
    cache = run / "cache"
    if Path.cwd().resolve() != run or not os.path.samefile("/tmp", cache/"tmp") or not os.path.samefile("/var/tmp", cache/"var_tmp"):
        raise ValueError("container temporary mounts or working directory differ")
    paths = sorted({cache/name for name in DIRECTORIES.values()} | {cache/"var_tmp"})
    for key, name in DIRECTORIES.items():
        if os.environ.get(key) != str(cache/name):
            raise ValueError("owned cache environment differs")
    samples = []
    for path in paths:
        if not path.is_dir() or path.is_symlink() or path.stat().st_uid != os.getuid():
            raise ValueError("owned cache directory differs")
        s = os.statvfs(path)
        canary = path / "write-check"
        payload = bytes(range(256)) * 16
        with canary.open("xb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        try:
            if canary.read_bytes() != payload:
                raise ValueError("owned cache write/read mismatch")
        finally:
            canary.unlink()
        samples.append(dict(name=path.name, canary_bytes=len(payload), filesystem_available_bytes=s.f_bavail*s.f_frsize,
                            filesystem_free_inodes=s.f_favail))
    with (run/"cache-startup.json").open("x", encoding="utf-8") as f:
        json.dump(dict(temporary_bind_mounts_verified=True, paths=samples),f)
