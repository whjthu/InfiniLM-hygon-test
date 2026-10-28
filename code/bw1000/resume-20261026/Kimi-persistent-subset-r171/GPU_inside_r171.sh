#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
: "${MANIFEST_R171:?root release manifest required; GPU_GO=false by default}"
CPU_LD=$(python3 -B -c 'import json,os;m=json.load(open(os.environ["MANIFEST_R171"]));assert m["GPU_authorized"] and m["source_published"];print(m["CPU_LD_LIBRARY_PATH"])')
export LD_LIBRARY_PATH="$CPU_LD:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset LD_PRELOAD
exec timeout --kill-after=15s 180s python3 -B -u /refactor/incoming/Kimi-persistent-subset-r171/run_GPU_r171.py "$MANIFEST_R171"
