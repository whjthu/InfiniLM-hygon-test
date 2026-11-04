#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
ulimit -c 0
unset LD_PRELOAD BASH_ENV ENV PYTHONSTARTUP PYTHONINSPECT
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec timeout --kill-after=10s 270 python3 -B /refactor/incoming/Deep-C1-gap-enclosure-r467/profile-source-v1/run_CPU.py
