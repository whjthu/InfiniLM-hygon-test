#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
python3 -B -u /refactor/incoming/Kimi-loader-bindings-cache-r9130/CPU_run.py
