#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
ulimit -c 0
unset LD_PRELOAD BASH_ENV ENV PYTHONSTARTUP PYTHONINSPECT
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export USE_TORCH=0 USE_TF=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/refactor/incoming/Deep-a314-fixed-cohort-C4-r458/deployment-v1:/refactor/incoming/Deep-a314-fixed-cohort-C4-r458:/refactor/incoming/public-Deep-native-provider-r427/source-v3:/old/offline-deps
exec timeout --kill-after=10s 270 python3 -B -u /refactor/incoming/Deep-a314-fixed-cohort-C4-r458/deployment-v1/run_glue_CPU.py
