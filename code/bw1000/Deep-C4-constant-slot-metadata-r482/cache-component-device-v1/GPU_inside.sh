#!/bin/bash
set -euo pipefail
ulimit -c 0
test "$SLURM_JOB_ID" = 910374
test "$(hostname -s)" = a02r4n01
source /refactor/incoming/Deep-C4-constant-slot-metadata-r482/cache-component-device-v1/ENV.sh
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL INFINILM_OWNER_CPU_WIDTH_R2392
case "${SLURM_STEP_GPUS:?Actual GPU allocation ID required}" in [0-7]) export HIP_VISIBLE_DEVICES="$SLURM_STEP_GPUS";; *) exit 2;; esac
export PYTHONPYCACHEPREFIX="/local/infini-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
LEFT=$(python3 -B -c 'import os,time;print(max(0,int(float(os.environ["DEEP_SLOT_GPU_START"])+560-time.time())))')
test "$LEFT" -gt 30
exec timeout --kill-after=10s "$LEFT" python3 -B -u /refactor/incoming/Deep-C4-constant-slot-metadata-r482/cache-component-device-v1/GPU_run.py
