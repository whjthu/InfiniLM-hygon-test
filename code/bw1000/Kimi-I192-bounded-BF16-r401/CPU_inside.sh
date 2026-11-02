#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset LD_PRELOAD
export LD_LIBRARY_PATH="/refactor/install/hygon-kimi-5f91ad7-r3672/lib:/refactor/candidates/kimi-helper-bounded-bf16-r6054:/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib:/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python:/usr/local/lib/python3.10/dist-packages/torch/lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
exec timeout --kill-after=10s 270 python3 -B /refactor/incoming/Kimi-I192-bounded-BF16-r401/CPU_build.py
