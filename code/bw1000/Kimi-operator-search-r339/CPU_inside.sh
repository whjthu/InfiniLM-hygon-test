#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
D=/refactor/incoming/Kimi-operator-search-r339
timeout --kill-after=5s 570 /opt/dtk/cuda/cuda/bin/nvcc -std=c++17 -O3 -arch=gfx936 -Xcompiler=-fPIC -shared "$D/search.cu" -o "$D/search.so" >"$D/compile.log" 2>&1
ldd "$D/search.so"
sha256sum "$D/search.so"
