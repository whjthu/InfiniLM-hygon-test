#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/refactor/candidates/latest-family-native-r269/GLM/python:/refactor/install/hygon-glm-5f91ad7-r3672/lib:/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python/infinilm/_lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch.libs:/opt/dtk/cuda/cuda/lib64:/opt/dtk/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD LD_AUDIT GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH
