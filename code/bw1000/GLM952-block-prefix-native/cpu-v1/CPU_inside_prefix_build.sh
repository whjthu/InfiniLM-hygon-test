#!/bin/bash
set -euo pipefail
ulimit -c 0
test "$#" -eq 2
test "$2" = "${GLM952_PREFIX_CPU_CLAIM_SHA:?missing published CPU claim}"
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/refactor/install/hygon-glm-5f91ad7-r3672/lib:/refactor/candidates/latest-family-native-r269/GLM/python:/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python/infinilm/_lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
here="$(cd -- "$(dirname -- "$0")" && pwd)"
exec /usr/bin/timeout --kill-after=5s 620s /usr/bin/python3 -B -u "$here/build_prefix_native_cpu.py" "$1" "$2"
