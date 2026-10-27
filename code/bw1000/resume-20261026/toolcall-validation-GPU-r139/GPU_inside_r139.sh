#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
TORCH_LIB_DIR="$(python -c 'import pathlib,importlib.util;print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib")')"
export LD_LIBRARY_PATH="$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:$TORCH_LIB_DIR:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/refactor/candidates/toolcall-Qwen06-r138/python:$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset LD_PRELOAD
timeout --kill-after=20s 1400 python -B -u /refactor/incoming/toolcall-validation-GPU-r139/run_GPU_r139.py
