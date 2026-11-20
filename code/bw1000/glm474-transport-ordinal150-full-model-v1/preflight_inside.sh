#!/bin/bash
set -euo pipefail
ulimit -c 0
source /paired-worker/ENV.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
exec /usr/bin/python3 -B -u /paired-worker/preflight.py "$1"
