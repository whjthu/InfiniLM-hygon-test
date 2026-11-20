#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/refactor/candidates/latest-family-native-r269/GLM/python:/refactor/install/hygon-glm-5f91ad7-r3672/lib:/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python/infinilm/_lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch.libs:/opt/dtk/cuda/cuda/lib64:/opt/dtk/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD LD_AUDIT GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH
remaining=$(/usr/bin/python3 -I -S -B -c 'import os,time,math; x=min(270.0,float(os.environ["TUNING_CPU_DEADLINE"])-time.time()-7.0); assert math.isfinite(x) and x>2; print(x)')
exec /usr/bin/timeout --kill-after=5s "${remaining}s" /usr/bin/python3 -I -S -B -u /refactor/incoming/GLM-I128-down-rows-gfx936-CPU-v1/build_once.py
