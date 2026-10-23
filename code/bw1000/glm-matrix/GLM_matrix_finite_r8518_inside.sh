#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/opt/dtk/cuda/cuda/lib64:/opt/dtk/lib:${LD_LIBRARY_PATH:-}"
python -u /refactor/incoming/GLM_matrix_finite_r8518.py
