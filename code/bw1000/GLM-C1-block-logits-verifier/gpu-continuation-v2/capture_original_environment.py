"""Recover the original GLM Python dependency path in its environment-only branch."""
from hashlib import sha256
from pathlib import Path
import os
import subprocess

ORIGINAL_SHA256 = "418dc5466f3f91c14c8eecba753f45340bf85d7ba4d4f0c57f8c2a8df66d701b"
ANCHOR = "LD_PRELOAD LD_LIBRARY_PATH INFINILM_OWNER_CPU_WIDTH_R2392 USE_WEIGHT_CACHE_R3604;"


def capture(original, inherited):
    raw = Path(original).read_bytes()
    if sha256(raw).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("original GLM environment source differs")
    source = raw.decode("utf-8")
    if source.count(ANCHOR) != 1:
        raise ValueError("original GLM environment export anchor differs")
    source = source.replace(ANCHOR, ANCHOR.replace("LD_LIBRARY_PATH ", "LD_LIBRARY_PATH PYTHONPATH "), 1)
    child = dict(inherited, GLM_R444_ENV_ONLY="1", GLM_EXACT_BANKS_R4188="1",
                 GLM_CHANNELS_R4363="0", GLM_BOUNDED_ROPE_R4279="1", GLM_SHARED_CONTROL_R4279="1",
                 GLM_NETWORK_TRIAL="shca", INFINILM_OWNER_CPU_WIDTH_R2392="4")
    run = subprocess.run(["/bin/bash", "--noprofile", "--norc", "-ec", source],
                         env=child, capture_output=True, check=True, timeout=20)
    marker = b"GLM_R444_ENV_BEGIN\0"
    if run.stdout.count(marker) != 1:
        raise ValueError("original environment capture marker differs")
    fields = run.stdout.split(marker, 1)[1].split(b"\0")
    if fields[-1] != b"" or len(fields) % 2 != 1:
        raise ValueError("original environment capture is incomplete")
    exported = {key.decode(): value.decode()
                for key, value in zip(fields[:-1:2], fields[1:-1:2], strict=True)}
    if "PYTHONPATH" not in exported or "/old/offline-deps" not in exported["PYTHONPATH"].split(":"):
        raise ValueError("original Python dependency path was not captured")
    return {**inherited, **exported}
