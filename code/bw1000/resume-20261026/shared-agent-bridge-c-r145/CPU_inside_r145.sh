#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}"
export BRIDGE_CPU_OUTPUT_R145="/refactor/results/bridge-C-timeout-contract-r145-${SLURM_JOB_ID:-manual}.${SLURM_STEP_ID:-cpu}"
mkdir -p "$BRIDGE_CPU_OUTPUT_R145"
timeout --kill-after=5s 90 python -B -u /refactor/incoming/shared-agent-bridge-c-r145/CPU_timeout_contract_r145.py
timeout --kill-after=5s 90 python -B -u /refactor/incoming/shared-agent-bridge-c-r145/CPU_HTTP_timeout_contract_r145.py
