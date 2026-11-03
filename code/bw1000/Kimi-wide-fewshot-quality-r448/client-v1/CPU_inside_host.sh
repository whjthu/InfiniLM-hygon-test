#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
unset LD_PRELOAD BASH_ENV ENV PYTHONSTARTUP PYTHONINSPECT
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TASK_ROOT=/public/home/aclsylqidf/hygon-inference-refactor-20261014
export PYTHONPATH="$TASK_ROOT/incoming/Kimi-public-incremental-text-r434:$TASK_ROOT/incoming/public-Kimi-native-provider-r428/runtime-v2"
exec timeout --kill-after=10s 270 python3 -B -u "$TASK_ROOT/incoming/Kimi-wide-fewshot-quality-r448/client-v1/CPU_new.py"
