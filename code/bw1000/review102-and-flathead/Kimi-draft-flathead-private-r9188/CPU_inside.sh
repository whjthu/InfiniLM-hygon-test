#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=/local/draft-kmax-cpu-r9182
cd /refactor/incoming/Kimi-draft-flathead-r9188
timeout --kill-after=5s 160 python -B -u CPU_contract.py
timeout --kill-after=5s 100 python -B -u CPU_entry_imports.py
