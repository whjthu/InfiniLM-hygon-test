#!/bin/bash
set -euo pipefail
export PYTHONPATH=/refactor/incoming:/old/offline-deps:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
python -B -u /refactor/incoming/GLM-real-concurrency-r9132/CPU_prepare.py
