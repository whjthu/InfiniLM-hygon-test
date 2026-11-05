#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
unset LD_PRELOAD LD_AUDIT
exec /usr/bin/timeout --kill-after=4s 48s /usr/bin/python3 -B /refactor/incoming/Deep-decode-chain-reference/runtime-identity-v2/probe.py
