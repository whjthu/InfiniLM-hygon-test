#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export USE_TORCH=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}"
timeout --kill-after=5s 90 python -B -u /refactor/incoming/GLM-context-count-only-r9392/count.py
