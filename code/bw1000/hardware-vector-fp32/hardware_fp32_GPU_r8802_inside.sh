#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/local/hardware-fp32-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
export LD_LIBRARY_PATH="/opt/dtk/hip/lib:${LD_LIBRARY_PATH:-}"
timeout --kill-after=10s 150 python -B -u /refactor/incoming/hardware_fp32_fma_r8802.py
