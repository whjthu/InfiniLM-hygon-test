#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
python /refactor/incoming/build_Kimi_cold_BF16_r8668.py
