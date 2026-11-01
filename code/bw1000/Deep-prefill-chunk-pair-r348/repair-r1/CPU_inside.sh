#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec timeout --kill-after=5s 540 python3 -B /refactor/incoming/Deep-prefill-chunk-pair-r348/repair-r1/CPU_entry.py --owner-sha 3b54c63930f6edbc18627f2faa0fcfb5ace0756c7633e7df445fc7e05b15ccd3
