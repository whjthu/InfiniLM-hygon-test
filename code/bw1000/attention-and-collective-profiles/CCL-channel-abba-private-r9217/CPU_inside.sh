#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
export PYTHONPATH=/refactor/incoming:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/refactor/install/hygon-kimi-5f91ad7-r3672/lib:/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:${LD_LIBRARY_PATH:-}
cd /refactor/incoming/CCL-channel-abba-r9217
timeout --kill-after=5s 100 python -B -u CPU_preflight.py
