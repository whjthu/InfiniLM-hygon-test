#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec timeout --kill-after=5s 540 python3 -B /refactor/incoming/Deep-prefill-chunk-pair-r348/repair-r2/CPU_entry.py --owner-sha 26cc8ec24564a6912ca85f3c88076c1607763e0983910321eee0e14ef5852db7
