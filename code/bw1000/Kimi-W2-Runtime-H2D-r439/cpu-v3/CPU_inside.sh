#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset LD_PRELOAD INFINILM_OWNER_CPU_WIDTH_R2392
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/refactor/install/hygon-kimi-5f91ad7-r3672/lib:/usr/local/lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
exec timeout --kill-after=10s 270 python3 -B /refactor/incoming/Kimi-W2-Runtime-H2D-r439/cpu-v3/run_cpu.py
