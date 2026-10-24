#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM/python
SDK=/refactor/install/hygon-deep-5f91ad7-r3672
TORCH=/usr/local/lib/python3.10/dist-packages/torch/lib
export LD_LIBRARY_PATH="$SDK/lib:$COMMON:$COMMON/infinilm/_lib:$TORCH:/refactor/candidates/models-5f91ad7-r3675/Deep/helpers:/opt/dtk/cuda/cuda/lib64:${LD_LIBRARY_PATH:-}"
python3 -B /refactor/incoming/build_Deep_HC_RMS_Model_r7383.py
