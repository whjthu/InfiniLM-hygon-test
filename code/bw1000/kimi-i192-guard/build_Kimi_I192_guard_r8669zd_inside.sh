#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
python /refactor/incoming/build_Kimi_I192_guard_r8669zd.py
