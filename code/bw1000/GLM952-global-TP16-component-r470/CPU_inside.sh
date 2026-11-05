#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/refactor/install/hygon-glm-5f91ad7-r3672/lib:/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python/infinilm/_lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch.libs:/opt/dtk/cuda/cuda/lib64:/opt/dtk/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD LD_AUDIT GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH
exec /usr/bin/timeout --kill-after=5s 270s /usr/bin/python3 -B -u /refactor/incoming/GLM952-global-TP16-component-r470/build_CPU.py
