#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export PYTHONPYCACHEPREFIX="/local/Kimi-PMC-tool-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset LD_PRELOAD INFINILM_KIMI_PACKED_W2_R6122 INFINILM_KIMI_PACKED_AUTO_R6136
target="/refactor/results/Kimi-PMC-tool-r8669f-${SLURM_JOB_ID}.${SLURM_STEP_ID}-raw"
mkdir -m700 "$target"
# CLI-native wholeprocess PMC; no assumption hipProfilerStart controls PMC.
timeout --kill-after=5s 60 /opt/dtk/hip/bin/hipprof --pmc --pmc-type 2 --kernel-name elementwise_kernel -d "$target" -o "$target/pmc" python /refactor/incoming/Kimi_PMC_tiny_r8669f.py "/refactor/results/Kimi-PMC-tool-r8669f-${SLURM_JOB_ID}.${SLURM_STEP_ID}.json"
