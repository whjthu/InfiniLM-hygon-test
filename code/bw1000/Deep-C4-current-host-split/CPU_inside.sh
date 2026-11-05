#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
unset LD_PRELOAD BASH_ENV ENV PYTHONSTARTUP PYTHONINSPECT
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
d=/refactor/incoming/Deep-C4-current-host-split/cpu-v1
exec timeout --kill-after=8s 275 python3 -B -u "$d/build_cpu.py" "$DEEP_C4_SPLIT_CLAIM_SHA"
