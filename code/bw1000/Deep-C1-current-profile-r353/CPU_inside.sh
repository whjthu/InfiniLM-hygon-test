#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec timeout --kill-after=5s 540 python3 -B /refactor/incoming/Deep-C1-current-profile-r353/CPU_entry.py --owner-sha 3164db9bbc42018a61b422cc902b0c8a16409b57649f2236ad8d45bc4512e431
