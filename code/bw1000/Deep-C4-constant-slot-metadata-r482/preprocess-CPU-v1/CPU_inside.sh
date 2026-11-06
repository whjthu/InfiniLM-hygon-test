#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset LD_PRELOAD LD_AUDIT GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH PYTHONPATH HIP_CLANG_PATH HIP_PLATFORM HIP_COMPILER
exec /usr/bin/timeout --kill-after=5s 270s /usr/bin/python3 -I -S -B -u /refactor/incoming/Deep-C4-constant-slot-metadata-r482/preprocess-CPU-v1/preprocess.py
