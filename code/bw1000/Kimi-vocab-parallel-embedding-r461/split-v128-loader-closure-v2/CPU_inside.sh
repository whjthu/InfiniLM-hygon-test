#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
source /refactor/incoming/Kimi-vocab-parallel-embedding-r461/split-v128-loader-closure-v2/ENV.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export KIMI_LOADER_EXPECTED_PRELOAD="$LD_PRELOAD"
unset LD_PRELOAD LD_AUDIT
exec timeout --kill-after=5s 150s python -B -u /refactor/incoming/Kimi-vocab-parallel-embedding-r461/split-v128-loader-closure-v2/loader_check.py
