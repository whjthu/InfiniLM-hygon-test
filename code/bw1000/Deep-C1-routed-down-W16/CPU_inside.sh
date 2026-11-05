#!/bin/bash
set -euo pipefail
ulimit -c 0
test "$SLURM_JOB_ID" = 910374
test "$(hostname -s)" = a02r4n05
source /opt/dtk/env.sh
unset LD_PRELOAD BASH_ENV ENV PYTHONSTARTUP PYTHONINSPECT
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
root=/refactor/incoming/Deep-C1-routed-down-W16/cpu-build-v1
actual=$(sha256sum "$root/CPU_RESOURCE_CLAIM.json" | cut -d' ' -f1)
test "$actual" = "$DEEP_W16_CPU_CLAIM_SHA"
exec timeout --kill-after=8s 275 python3 -B -u "$root/build_component.py" \
  "$root/derived-source" /refactor/candidates/Deep-C1-routed-down-W16-CPU-v1 \
  /refactor/candidates/Deep-HC-fullpre-Native-r377-cpu-v1/CPU-build.json
