#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
root=/refactor/incoming/GLM952-prefix-continuation/gpu-continuation-v2
left=$(python3 -B -c 'import os,time;print(max(0,int(float(os.environ["GLM_PREFIX_EPOCH"])+3480-time.time())))')
test "$left" -gt 120
exec /usr/bin/timeout --kill-after=15s "$left" /usr/bin/python3 -B "$root/environment.py"
