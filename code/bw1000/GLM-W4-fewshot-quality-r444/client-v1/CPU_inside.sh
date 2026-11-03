#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset LD_PRELOAD
exec timeout --kill-after=5s 270 python -B -u /refactor/incoming/GLM-W4-fewshot-quality-r444-client-v1/client-v1/check_main_CPU.py --fixtures /refactor/incoming/GLM-W4-fewshot-quality-r444/fixtures-r1 --encoding /refactor/candidates/GLM-W4-fewshot-quality-r444-CPU-v1/encoding.json --output /refactor/candidates/GLM-W4-fewshot-quality-r444-client-CPU-v1/whole-main-result.json
