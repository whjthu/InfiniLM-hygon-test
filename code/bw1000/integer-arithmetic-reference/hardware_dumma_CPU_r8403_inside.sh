#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export PYTHONPYCACHEPREFIX="/local/hardware-dumma-cpu-${SLURM_JOB_ID}-${SLURM_STEP_ID}"
python -B -u /refactor/incoming/build_hardware_dumma_r8403.py
