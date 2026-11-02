#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export USE_TORCH=0 USE_TF=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
unset LD_PRELOAD
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/refactor/install/hygon-glm-5f91ad7-r3672/lib:/refactor/candidates/latest-family-native-r269/GLM/python:/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python/infinilm/_lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/refactor/incoming/GLM-V-on-demand-Model-r404/owner:/refactor/candidates/latest-family-service-r270/GLM/python:/old/offline-deps:${PYTHONPATH:-}"
exec timeout --kill-after=5s 270 python -B -u /refactor/incoming/GLM-V-on-demand-Model-r404/cpu-v2/CPU_main.py
