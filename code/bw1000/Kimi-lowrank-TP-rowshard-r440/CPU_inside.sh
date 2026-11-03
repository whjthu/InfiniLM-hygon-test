#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset LD_PRELOAD INFINILM_OWNER_CPU_WIDTH_R2392
exec timeout --kill-after=10s 270 python3 -B /refactor/incoming/Kimi-lowrank-TP-rowshard-r440/run_CPU.py
