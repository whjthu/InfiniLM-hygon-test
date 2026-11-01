#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export USE_TORCH=0 USE_TF=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="/refactor/install/hygon-glm-5f91ad7-r3672/lib:/refactor/candidates/latest-family-native-r269/GLM/python:/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python/infinilm/_lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/refactor/incoming/GLM-two-cohort-Model-r355/owner:/refactor/candidates/latest-family-service-r270/GLM/python:/old/offline-deps:${PYTHONPATH:-}"
python -B /refactor/incoming/GLM-two-cohort-Model-r355/check_controller_CPU.py
exec timeout --kill-after=5s 300 python -B -u /refactor/incoming/GLM-two-cohort-Model-r355/CPU-repair-r2/CPU_whole.py
