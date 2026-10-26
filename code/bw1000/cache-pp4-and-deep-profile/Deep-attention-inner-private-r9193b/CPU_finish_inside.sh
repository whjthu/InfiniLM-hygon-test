#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-5f91ad7-r3672
export LD_LIBRARY_PATH="$SDK/lib:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH=/refactor/incoming/Deep-attention-inner-r9193:/refactor/incoming:/old/offline-deps:${PYTHONPATH:-}
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH"
python -B -u /refactor/incoming/Deep-attention-inner-r9193/CPU_finish.py
python -B -u /refactor/incoming/Deep-attention-inner-r9193/verify_consumer_CPU_r9193.py
