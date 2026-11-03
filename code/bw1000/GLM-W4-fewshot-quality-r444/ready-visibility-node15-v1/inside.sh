#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset LD_PRELOAD
exec timeout --kill-after=5s 45 python3 -B -u /refactor/incoming/GLM444-ready-visibility-node15-v1/probe.py
