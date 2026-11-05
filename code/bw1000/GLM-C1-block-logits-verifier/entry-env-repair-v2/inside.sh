#!/bin/bash
set -euo pipefail
ulimit -c 0
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
exec /usr/bin/timeout --kill-after=4s 45s /usr/bin/python3 -B /refactor/incoming/GLM952-prefix-continuation/entry-env-repair-v2/check.py
