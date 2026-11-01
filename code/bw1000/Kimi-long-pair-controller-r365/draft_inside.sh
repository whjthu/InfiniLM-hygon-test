#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/local/kimi-long-draft-r347
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
export INFINILM_KIMI_LONG_RPC_R334=1
export KIMI_DRAFT_NONCE_R9046="${KIMI_LONG_SESSION_R347}"
exec python -B -u /refactor/incoming/Kimi-long-draft-memory-floor-r362b/worker.py \
  --manifest-sha 61ac8b04af464d76b4ce46f0793e6d8a2a8e57f41224d6228c5b826bda6ffd31 \
  --host "${KIMI_LONG_DRAFT_HOST_R347}" --port 46947 \
  --deadline-unix "${KIMI_LONG_NATIVE_DEADLINE_UNIX_R347}" \
  --output "/refactor/results/Kimi-long-draft-r334-${KIMI_LONG_RUN_R347}.json"
