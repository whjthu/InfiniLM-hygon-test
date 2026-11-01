#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec timeout --kill-after=5s 540 python3 -B /refactor/incoming/Deep-C1-current-profile-r353/repair-r1/CPU_entry.py --owner-sha 60bb0a4341ccb574fd8d14a5971cbf651cade93f8d9e5be78ffbe29fc7033c99
