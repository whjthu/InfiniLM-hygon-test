#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset LD_PRELOAD
export PYTHONPATH=/refactor/incoming/GLM-V-on-demand-Model-r404/owner_v2
exec timeout --kill-after=5s 270 python3 -B -u /refactor/incoming/GLM-V-on-demand-Model-r404/cpu-layout-v4/check_layout_CPU.py
