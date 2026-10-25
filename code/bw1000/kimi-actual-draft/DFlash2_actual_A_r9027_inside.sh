#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=/local/dflash2-actual-A-r9027
export PYTHONPATH="/refactor/incoming/Kimi-target-taps-r9022:/refactor/incoming:${PYTHONPATH:-}"
: "${DFLASH_A_OUT_R9027:=/refactor/results/dflash2-actual-A-r9027-${SLURM_JOB_ID}.${SLURM_STEP_ID}.json}"
timeout --kill-after=5s 450 python -B -u /refactor/incoming/score_actual_A_draft_r9027.py --target-result "$TARGET_A_RESULT_R9027" --out "$DFLASH_A_OUT_R9027"
