#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
unset LD_PRELOAD BASH_ENV ENV PYTHONSTARTUP PYTHONINSPECT
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
deadline="${1:?original CPU deadline required}"
left=$(python3 -B -c '
import sys,time
left=float(sys.argv[1])-time.time()
if not 18<left<=300:raise SystemExit("invalid build deadline")
print(int(left)-10)
' "$deadline")
exec timeout --kill-after=8s "$left" python3 -B -u /build/source/build_cpu.py "$DEEP_HISTORY_CLAIM_SHA" "$deadline"
