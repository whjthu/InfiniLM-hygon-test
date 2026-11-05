#!/bin/bash
set -euo pipefail
ulimit -c 0
test "$SLURM_JOB_ID" = 910374
test "$(hostname -s)" = a02r4n05
source /opt/dtk/env.sh
unset USE_TORCH USE_TF USE_FLAX BASH_ENV ENV PYTHONSTARTUP PYTHONINSPECT
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 INFINILM_OWNER_CPU_WIDTH_R2392=4
export ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu5
LM=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-5f91ad7-r3672
export LD_PRELOAD=/refactor/candidates/blas-policy-r271/libblas_policy_r271.so
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/refactor/candidates/models-5f91ad7-r3675/Deep:/refactor/candidates/models-5f91ad7-r3675/Deep/helpers:$SDK/lib:/refactor/candidates/latest-family-native-r269/Deep/python:$LM/python:$LM/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/old/offline-deps"
D=/refactor/incoming/Deep-C1-routed-down-W16/device-v1
O=/refactor/results/Deep-C1-routed-down-W16-device-v1
exec python3 -B -u "$D/device_probe.py" "$O/GPU_ACTUAL_CLAIM.json" "$O/fixtures" "$O/component.json"
