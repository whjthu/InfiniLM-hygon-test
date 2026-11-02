#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset LD_PRELOAD
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
export LD_LIBRARY_PATH="/refactor/install/hygon-deep-5f91ad7-r3672/lib:$LM/python/infinilm/_lib:$LM/python:/usr/local/lib/python3.10/dist-packages/torch/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
exec timeout --kill-after=10s 270 python3 -B /refactor/incoming/Deep-fixed-Gemm-Cast-graph-r399/v3/CPU_build.py
