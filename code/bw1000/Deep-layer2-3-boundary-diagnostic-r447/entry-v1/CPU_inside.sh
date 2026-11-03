#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export USE_TORCH=0 USE_TF=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset LD_PRELOAD
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}"
exec timeout --kill-after=10s 270 python3 -B -u /refactor/incoming/Deep-layer2-3-boundary-diagnostic-r447/entry-v1/run_CPU.py
