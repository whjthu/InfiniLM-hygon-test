"""Reuse the original production environment without importing Torch or a DSO."""
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
run = subprocess.run(["/bin/bash", "--noprofile", "--norc",
    "/refactor/incoming/GLM-W4-remapped-final-slots-r469/deployment-v1/inside.sh"],
    capture_output=True, timeout=20, check=True)
prefix = b"GLM_R444_ENV_BEGIN\0"
if run.stdout.count(prefix) != 1:
    raise ValueError("original environment capture marker differs")
entries = run.stdout.split(prefix, 1)[1].split(b"\0")
if entries[-1] != b"" or len(entries) % 2 != 1:
    raise ValueError("original environment capture is incomplete")
env = dict(os.environ)
for key, value in zip(entries[:-1:2], entries[1:-1:2], strict=True):
    env[key.decode()] = value.decode()
env.update(CUDA_VISIBLE_DEVICES="-1", HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1",
           PYTHONDONTWRITEBYTECODE="1")
claim = root / "entry-cpu-v1/CPU_RESOURCE_CLAIM.json"
argv = ["/usr/bin/python3", "-B", str(root / "entry-cpu-v1/check_prefix_entry_cpu.py"),
        str(root / "provider"), str(claim), sha256(claim.read_bytes()).hexdigest()]
os.execve(argv[0], argv, env)
