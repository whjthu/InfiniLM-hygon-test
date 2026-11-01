#!/bin/bash
set -euo pipefail
source /opt/dtk/env.sh
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}"
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL LD_PRELOAD LD_AUDIT
export HIP_VISIBLE_DEVICES="$SLURM_LOCALID"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
SOURCE=/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM
SDK=/refactor/install/hygon-deep-5f91ad7-r3672
export LD_LIBRARY_PATH="$SDK/lib:$SOURCE/python:$SOURCE/python/infinilm/_lib:/opt/dtk/cuda/cuda/lib64:/opt/dtk/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/old/offline-deps:${PYTHONPATH:-}"
exec timeout --kill-after=5s 600 python -B -u /refactor/incoming/Deep-Runtime-boundary-fusion-r344/adjacent-r1/GPU.py
