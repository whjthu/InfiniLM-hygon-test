#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/refactor/candidates/models-5f91ad7-r3675/Deep:/refactor/candidates/models-5f91ad7-r3675/Deep/helpers:/refactor/install/hygon-deep-5f91ad7-r3672/lib:/refactor/candidates/latest-family-native-r269/Deep/python:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD LD_AUDIT GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH PYTHONPATH HIP_CLANG_PATH HIP_PLATFORM HIP_COMPILER
export LD_LIBRARY_PATH="/refactor/candidates/Deep-C4-causal-consumer-r483-CPU-v2:$LD_LIBRARY_PATH"
budget=$(/usr/bin/python3 -I -S -B -c 'import math,os,time; s=min(105.0,float(os.environ["RUNTIME_OWNER_DEADLINE"])-time.time()-60.0); assert math.isfinite(s) and s>2; print(s)')
exec /usr/bin/timeout --kill-after=5s "${budget}s" /usr/bin/python3 -B -u /refactor/incoming/Deep-C4-causal-consumer-r483/runtime-owner-v2/device_owner.py worker
