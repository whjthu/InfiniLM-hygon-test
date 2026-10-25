#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=/local/dflash2-r9019
timeout --kill-after=5s 450 python -B -u /refactor/incoming/run_DFlash2_trained_r9019.py
