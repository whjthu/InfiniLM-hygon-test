#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/local/pageable-h2d-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
export LD_LIBRARY_PATH="/opt/dtk/hip/lib:${LD_LIBRARY_PATH:-}"
timeout --kill-after=15s 600 python -B -u /refactor/incoming/pageable_h2d_probe_r8418.py
