#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
unset LD_PRELOAD LD_AUDIT BASH_ENV ENV PYTHONSTARTUP PYTHONINSPECT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export GLM_GLOBAL_TP16_R470=1
export LD_LIBRARY_PATH="/refactor/install/hygon-glm-5f91ad7-r3672/lib:/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python/infinilm/_lib:/refactor/candidates/latest-family-native-r269/GLM/python:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch.libs:/opt/dtk/cuda/cuda/lib64:/opt/dtk/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
exec /usr/bin/python3 -B -u /refactor/incoming/GLM952-global-TP16-component-r470/device-v1/worker.py
