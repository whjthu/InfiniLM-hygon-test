#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
unset LD_PRELOAD
exec python3 -B -u /refactor/incoming/GLM-W4-fewshot-quality-r444-eight-slot-v1/deployment-v2/client_dispatch.py
