#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset LD_PRELOAD LD_AUDIT NCCL_NET NCCL_NET_PLUGIN NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
root=/refactor/incoming/Deep-decode-chain-reference/gpu-chain-v1
exec /usr/bin/timeout --kill-after=5s 90s /usr/bin/python3 -B -u "$root/run_inside.py"
