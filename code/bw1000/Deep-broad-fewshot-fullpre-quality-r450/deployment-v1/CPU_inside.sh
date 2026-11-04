#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
ulimit -c 0
unset LD_PRELOAD BASH_ENV ENV PYTHONSTARTUP PYTHONINSPECT
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/refactor/incoming/Deep-broad-fewshot-fullpre-quality-r450/deployment-v1:/refactor/incoming/Deep-broad-fewshot-fullpre-quality-r450:/refactor/incoming/public-Deep-native-provider-r427/source-v3:/old/offline-deps
exec timeout --kill-after=10s 270 python3 -B -u /refactor/incoming/Deep-broad-fewshot-fullpre-quality-r450/deployment-v1/run_CPU.py
