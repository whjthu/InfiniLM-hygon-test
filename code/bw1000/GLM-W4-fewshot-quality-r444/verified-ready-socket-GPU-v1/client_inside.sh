#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
unset LD_PRELOAD
exec python3 -B -u /refactor/incoming/GLM444-verified-ready-socket-singleton-v1/GPU-final-v1/client_dispatch.py
