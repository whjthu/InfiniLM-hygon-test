#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONPATH=/refactor/incoming/GLM444-verified-ready-socket-CPU-v1/provider:/refactor/incoming/GLM-W4-fewshot-quality-r444-client-v1/client-v1
exec python3 -B /refactor/incoming/GLM-W4-remaining-three-replicas-r462/deployment-v1/check_CPU.py
