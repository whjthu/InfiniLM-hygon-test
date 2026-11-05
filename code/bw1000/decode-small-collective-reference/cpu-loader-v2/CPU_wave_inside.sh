#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/refactor/install/hygon-kimi-5f91ad7-r3672/lib:/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/lib:/opt/dtk/cuda/lib64"
unset LD_PRELOAD LD_AUDIT GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH
root=/refactor/incoming/Deep-decode-chain-reference/cpu-loader-v2
exec /usr/bin/timeout --kill-after=3s 75s /usr/bin/python3 -B -u "$root/loader.py"
