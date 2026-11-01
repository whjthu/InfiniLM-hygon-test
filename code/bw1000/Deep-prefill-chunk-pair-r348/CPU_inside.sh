#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec timeout --kill-after=5s 540 python3 -B /refactor/incoming/Deep-prefill-chunk-pair-r348/CPU_entry.py --owner-sha 40f69be372f89c4ed5345a4d326bbd1936423d5330f40da3dcf435a9b2f80d4e
