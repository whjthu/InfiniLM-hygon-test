#!/bin/bash
set -euo pipefail
ulimit -c 0
source /opt/dtk/env.sh
python /refactor/incoming/build_Kimi_cyclic_EP2_r8669zp.py
