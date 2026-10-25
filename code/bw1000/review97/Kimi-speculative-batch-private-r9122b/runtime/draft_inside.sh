#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX=/local/pycache-batch-r9122b
export PATH=/root/.local/bin:/opt/dtk/llvm/bin:/opt/dtk/bin:${PATH}
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=8
export LD_LIBRARY_PATH=/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/refactor/incoming:/old/offline-deps:${PYTHONPATH:-}
timeout --kill-after=15s 5200 python -u /refactor/incoming/Kimi-speculative-batch-Model-r9122b/runtime/batch_draft_worker_r9128.py
