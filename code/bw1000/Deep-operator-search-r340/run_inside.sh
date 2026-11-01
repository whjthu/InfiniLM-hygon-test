#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export HIP_VISIBLE_DEVICES="$SLURM_LOCALID"
timeout --kill-after=5s 600 python -u /refactor/incoming/Deep-operator-search-r340/bench.py
