#!/bin/bash
set -euo pipefail
ulimit -c 0
source /refactor/incoming/GLM952-global-TP16-component-r470/full-model-entry-v1/ENV.sh
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL INFINILM_OWNER_CPU_WIDTH_R2392
export GLM_GLOBAL_TP16_R470=1 GLM_GLOBAL_TP16_MODEL_R470=1 GLM_V_ON_DEMAND_R403=1 GLM_BOUNDED_ROPE_R4279=1 GLM_SHARED_CONTROL_R4279=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="/local/infini-pycache-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID}"
LEFT=$(/usr/bin/python3 -B -c 'import os,time;print(max(0,int(float(os.environ["GLM470_EPOCH"])+14100-time.time())))')
test "$LEFT" -gt 60
exec /usr/bin/timeout --kill-after=15s "$LEFT" /usr/bin/python3 -B -u /refactor/incoming/GLM952-global-TP16-component-r470/full-model-entry-v1/control.py worker
