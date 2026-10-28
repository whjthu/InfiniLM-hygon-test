#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
: "${MANIFEST_R168A:?root-generated manifest required}"
COMMON=/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM
SDK=/refactor/install/hygon-kimi-5f91ad7-r3672
# Read only the locked CPU receipt's dependency path; no Torch import or evaluation.
CPU_LD=$(python3 -B -c 'import os,json;print(json.load(open(os.environ["MANIFEST_R168A"]))["CPU_LD_LIBRARY_PATH"])')
export LD_LIBRARY_PATH="$CPU_LD:$SDK/lib:$COMMON/python:$COMMON/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$COMMON/python:/old/offline-deps:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset LD_PRELOAD
mapfile -t PROBE_ARGS < <(python3 -B -c 'import os,json;m=json.load(open(os.environ["MANIFEST_R168A"]));assert m["GPU_authorized"] and m["source_published"];print(m["arm"]);print(m["CPU_build_SHA256"]);print(m["publication_commit"])')
test "${#PROBE_ARGS[@]}" = 3
exec timeout --kill-after=15s 180s python3 -B -u /refactor/incoming/Kimi-I96-shared-storage-probe-r168/run_probe_r168.py "${PROBE_ARGS[@]}"
