#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
exec python3 -B /refactor/incoming/Deep-HC-epilogue-r358-affinity-r2/CPU_affinity_contract.py
