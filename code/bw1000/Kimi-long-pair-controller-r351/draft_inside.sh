#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/local/kimi-long-draft-r347
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export INFINILM_KIMI_LONG_RPC_R334=1
export KIMI_DRAFT_NONCE_R9046="${KIMI_LONG_SESSION_R347}"
exec python -B -u /refactor/incoming/Kimi-long-draft-worker-r334b/worker.py \
  --manifest-sha 49fa1d269f18f78820c7589496011300a7648fe340933e3c7a0e9105958b93c2 \
  --host "${KIMI_LONG_DRAFT_HOST_R347}" --port 46947 \
  --deadline-unix "${KIMI_LONG_NATIVE_DEADLINE_UNIX_R347}" \
  --output "/refactor/results/Kimi-long-draft-r334-${KIMI_LONG_RUN_R347}.json"
