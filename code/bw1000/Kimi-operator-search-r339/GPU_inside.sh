#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export PATH=/root/.local/bin:/opt/dtk/llvm/bin:/opt/dtk/bin:$PATH
export PYTHONPATH=/old/offline-deps:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
D=/refactor/incoming/Kimi-operator-search-r339
export LOCAL_GPU=$SLURM_LOCALID ARM=$((SLURM_LOCALID+1)) SEED=$((339+SEED_OFFSET))
exec timeout --kill-after=10s 600 python -B "$D/bench.py"
