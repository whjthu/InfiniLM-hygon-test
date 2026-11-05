#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
unset LD_PRELOAD BASH_ENV ENV PYTHONSTARTUP PYTHONINSPECT
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec timeout --kill-after=5s 100 python3 -B /refactor/incoming/Deep-C4-compact-bank-pair/entry-cpu-v1/check_deployment_CPU.py
