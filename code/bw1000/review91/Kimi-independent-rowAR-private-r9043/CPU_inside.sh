#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export INFINILM_OWNER_CPU_WIDTH_R2392=4
python /refactor/incoming/Kimi_layer0_driver_r9043.py --cpu-worker-contract /refactor/incoming/Kimi-layer0-manifest-r9043.json
